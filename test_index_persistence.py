"""Round-trip test: build -> query -> save_index -> load_index -> same results.

Run with ``python test_index_persistence.py`` or ``pytest``.
"""

import contextlib
import io
import math
import tempfile

from main import DOCS, EDGES, GraphRAG

QUERIES = [
    "Which services are affected by the payments table failure?",
    "ledger service reconcile balances",
    "postmortem reconciliation gap audit",
    "search product catalog",
    "zzz words that are not in the corpus",
]
GROUP_SETS = [["sre"], ["finance"], ["data-eng"], ["sre", "finance", "data-eng"], ["nobody"]]


def run_queries(rag):
    """Return BM25 + TF-IDF scores, rankings, ACL result and evidence per query."""
    results = []
    for q in QUERIES:
        with contextlib.redirect_stdout(io.StringIO()):  # _expand_graph prints debug lines
            out = rag.fit(q).predict()
        results.append({
            "allowed": rag.allowed,
            "lex_scores": rag.lex_scores,
            "dense_scores": rag.dense_scores,
            "lex": rag.lex,
            "dense": rag.dense,
            "provenance": out["provenance"],
            "evidence": rag.evidence,
            "answer": out["answer"],
        })
    return results


def assert_same_scores(a, b):
    """Same keys, same order, values equal within floating-point tolerance."""
    assert list(a) == list(b)
    for d in a:
        assert math.isclose(a[d], b[d], rel_tol=1e-12, abs_tol=1e-12), (d, a[d], b[d])


def test_save_load_round_trip():
    with tempfile.TemporaryDirectory() as vdb:
        for groups in GROUP_SETS:
            built = GraphRAG(DOCS, EDGES, groups, use_llm=False)
            before = run_queries(built)
            built.save_index(vdb)

            # Load without docs/edges: nothing is retokenized or reindexed.
            loaded = GraphRAG(None, None, groups, use_llm=False, index_path=vdb)
            assert not hasattr(loaded, "tokens")
            after = run_queries(loaded)

            # Restored index structures match the built ones.
            assert loaded.docs == built.docs
            assert loaded.vocab == built.vocab
            assert loaded.doc_vectors == built.doc_vectors
            assert loaded.tfidf_idf == built.tfidf_idf
            assert loaded.bm25_tf == built.bm25_tf
            assert loaded.bm25_doc_len == built.bm25_doc_len
            assert loaded.bm25_idf == built.bm25_idf
            assert loaded.avg_len == built.avg_len
            assert loaded.edges == list(built.edges)
            assert dict(loaded.adj) == dict(built.adj)

            # Query results match: scores, rankings, ACL filter, graph evidence.
            for b, a in zip(before, after):
                assert_same_scores(b["lex_scores"], a["lex_scores"])
                assert_same_scores(b["dense_scores"], a["dense_scores"])
                assert b["lex"] == a["lex"]
                assert b["dense"] == a["dense"]
                assert b["allowed"] == a["allowed"]
                assert b["provenance"] == a["provenance"]
                assert b["evidence"] == a["evidence"]
                assert b["answer"] == a["answer"]


if __name__ == "__main__":
    test_save_load_round_trip()
    print(f"OK: {len(GROUP_SETS)} group sets x {len(QUERIES)} queries identical after save/load")
