"""Round-trip tests: build -> query -> save_index -> load_index -> same results; stale indexes refused.

Run with ``python test_index_persistence.py`` or ``pytest``.
"""

import contextlib
import io
import json
import math
import tempfile
from unittest import mock

import main
from main import DOCS, EDGES, EmbeddingGraphRAG, GraphRAG

QUERIES = [
    "Which downstream processes were affected by the failed nightly settlement load?",
    "How is the EVE outlier test calculated?",
    "What is the reconciliation tolerance for payment breaks?",
    "Who owns the settled transactions schema?",
    "zzz words that are not in the corpus",
]
GROUP_SETS = [["sre"], ["finance"], ["risk"], ["data-eng"], ["treasury", "risk", "finance", "sre", "data-eng"]]


def run_queries(rag):
    """Return chunk scores, rankings, ACL result, provenance and evidence per query."""
    results = []
    for q in QUERIES:
        with contextlib.redirect_stdout(io.StringIO()):  # _expand_graph prints debug lines
            out = rag.fit(q).predict()
        results.append({
            "allowed": rag.allowed, "lex_scores": rag.lex_scores, "dense_scores": rag.dense_scores,
            "lex": rag.lex, "dense": rag.dense, "fused": rag.fused,
            "provenance": out["provenance"], "evidence": rag.evidence, "answer": out["answer"],
        })
    return results


def assert_same_scores(a, b):
    assert list(a) == list(b)
    for key in a:
        assert math.isclose(a[key], b[key], rel_tol=1e-12, abs_tol=1e-12), (key, a[key], b[key])


def assert_same_results(before, after):
    for b, a in zip(before, after):
        assert_same_scores(b["lex_scores"], a["lex_scores"])
        assert_same_scores(b["dense_scores"], a["dense_scores"])
        for key in ("lex", "dense", "fused", "allowed", "provenance", "evidence", "answer"):
            assert b[key] == a[key], key


def test_graph_round_trip_is_identical():
    with tempfile.TemporaryDirectory() as vdb:
        for groups in GROUP_SETS:
            built = GraphRAG(DOCS, EDGES, groups, use_llm=False)
            before = run_queries(built)
            built.save_index(vdb)

            loaded = GraphRAG(None, None, groups, use_llm=False, index_path=vdb)
            assert not hasattr(loaded, "tokens")  # nothing was rechunked or retokenized
            for attr in ("docs", "chunks", "doc_chunks", "vocab", "chunk_vectors", "tfidf_idf",
                         "bm25_tf", "bm25_chunk_len", "bm25_idf", "avg_len"):
                assert getattr(loaded, attr) == getattr(built, attr), attr
            assert loaded.edges == list(built.edges) and dict(loaded.adj) == dict(built.adj)
            assert_same_results(before, run_queries(loaded))


def test_stale_index_is_refused():
    with tempfile.TemporaryDirectory() as vdb:
        GraphRAG(DOCS, EDGES, ["sre"], use_llm=False).save_index(vdb)
        GraphRAG(DOCS, EDGES, ["sre"], use_llm=False, index_path=vdb)  # same sources: loads

        edited = {d: dict(v) for d, v in DOCS.items()}
        edited["general_ledger"]["text"] += "\n\nNew paragraph."
        new_acl = {d: dict(v) for d, v in DOCS.items()}
        new_acl["payments_platform"]["acl"] = {"sre"}
        extra = {**DOCS, "new_doc": {"text": "A new document.", "acl": {"sre"}}}
        cases = [
            (edited, EDGES, None, "changed documents ['general_ledger']"),
            (new_acl, EDGES, None, "changed documents ['payments_platform']"),  # ACL change counts
            (extra, EDGES, None, "new documents ['new_doc']"),
            (DOCS, EDGES[:-1], None, "graph edges changed"),
            (DOCS, EDGES, {"chunk_size": 200}, "chunking settings"),
        ]
        for docs, edges, chunking, message in cases:
            try:
                GraphRAG(docs, edges, ["sre"], use_llm=False, index_path=vdb, chunking=chunking)
            except ValueError as exc:
                assert message in str(exc), (message, str(exc))
                continue
            raise AssertionError(f"stale index was loaded: {message}")


def test_old_index_version_is_refused():
    with tempfile.TemporaryDirectory() as vdb:
        GraphRAG(DOCS, EDGES, ["sre"], use_llm=False).save_index(vdb)
        meta_path = f"{vdb}/metadata.json"
        metadata = json.load(open(meta_path, encoding="utf-8"))
        metadata["config"]["index_version"] = 1
        json.dump(metadata, open(meta_path, "w", encoding="utf-8"))
        try:
            GraphRAG(None, None, ["sre"], use_llm=False, index_path=vdb)
        except ValueError as exc:
            assert "version" in str(exc)
            return
        raise AssertionError("old index version was loaded")


class FakeOllama:
    """Stands in for the HTTP opener: letter-count vectors, records every request."""

    def __init__(self):
        self.requests = []

    def open(self, request, timeout=None):
        texts = json.loads(request.data)["input"]
        self.requests.append(texts)
        vectors = [[t.lower().count(c) + 1.0 for c in "abcdefghijklmnopqrstuvwxyz0123456789"] for t in texts]
        return io.BytesIO(json.dumps({"embeddings": vectors}).encode())


def test_embedding_round_trip_reuses_faiss_rows():
    if main.faiss is None:
        return  # FAISS not installed; nothing to check
    with tempfile.TemporaryDirectory() as vdb:
        fake = FakeOllama()
        with mock.patch.object(main.urllib.request, "build_opener", return_value=fake):
            built = EmbeddingGraphRAG(DOCS, EDGES, ["sre"])
        assert fake.requests == [[c["text"] for c in built.chunks.values()]]  # one request: every chunk
        before = run_queries(built)
        built.save_index(vdb)

        fake_after = FakeOllama()
        with mock.patch.object(main.urllib.request, "build_opener", return_value=fake_after):
            loaded = EmbeddingGraphRAG(None, None, ["sre"], index_path=vdb)
            after = run_queries(loaded)
        embedded = [t for batch in fake_after.requests for t in batch]
        assert embedded == QUERIES  # queries only: no chunk was re-embedded on load
        assert loaded.faiss_chunk_ids == list(built.chunks) and loaded.faiss_index.ntotal == len(built.chunks)
        assert_same_results(before, after)

        # Row i of FAISS is chunk faiss_chunk_ids[i]: each chunk's own vector finds itself first.
        texts = [loaded.chunks[c]["text"] for c in loaded.faiss_chunk_ids]
        vectors = loaded._normalized(loaded._embed(texts))
        _, rows = loaded.faiss_index.search(vectors, 1)
        assert [loaded.faiss_chunk_ids[r] for r in rows[:, 0]] == loaded.faiss_chunk_ids


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} persistence tests passed")
