"""Tests for calc_metric(): hand-computed metric values, ACL handling, no LLM calls.

Run with ``python test_metrics.py`` or ``pytest``.
"""

import contextlib
import io
import json
import math
from unittest import mock

import main
from main import DOCS, EDGES, EmbeddingGraphRAG, GraphRAG, TraditionalRAG

score = TraditionalRAG.score_retrieval
LOG2_3 = math.log2(3)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12)


def test_ranked_metrics_by_hand():
    # Top 3 = [a, b, c]; only b is a hit (rank 2). Target {b, d, e}; c is blocked.
    m = score(["a", "b", "c", "d"], {"b", "d", "e"}, {"c"}, k=3)
    assert m["retrieved"] == ["a", "b", "c"]
    assert close(m["precision"], 1 / 3)        # 1 hit / 3 retrieved
    assert close(m["recall"], 1 / 3)           # 1 hit / 3 relevant
    assert close(m["mrr"], 1 / 2)              # first hit at rank 2
    idcg = 1 + 1 / LOG2_3 + 1 / 2              # ideal: 3 hits at ranks 1..3
    assert close(m["ndcg"], (1 / LOG2_3) / idcg)
    assert m["leaked"] == ["c"]


def test_perfect_and_partial_rankings():
    m = score(["b", "d"], {"b", "d"}, set(), k=3)
    assert (m["precision"], m["recall"], m["mrr"], m["ndcg"]) == (1.0, 1.0, 1.0, 1.0)

    # Hit at rank 3 only, single-document target.
    m = score(["x", "y", "b"], {"b"}, set(), k=3)
    assert close(m["precision"], 1 / 3)
    assert close(m["mrr"], 1 / 3)
    assert close(m["ndcg"], (1 / 2) / 1)       # 1/log2(4) over ideal 1/log2(2)


def test_mrr_looks_past_k():
    # Relevant doc at rank 4 is outside the top 3 but still sets MRR.
    m = score(["a", "b", "c", "d"], {"d"}, set(), k=3)
    assert m["recall"] == 0.0 and m["ndcg"] == 0.0
    assert close(m["mrr"], 1 / 4)


def test_unranked_evidence():
    m = score(["a", "b", "d"], {"b", "d", "e"}, set())
    assert close(m["precision"], 2 / 3) and close(m["recall"], 2 / 3)
    assert m["mrr"] is None and m["ndcg"] is None


def test_empty_and_undefined_cases():
    m = score([], {"b"}, set(), k=3)            # nothing retrieved
    assert m["precision"] is None
    assert (m["recall"], m["mrr"], m["ndcg"]) == (0.0, 0.0, 0.0)

    m = score(["a", "c"], set(), {"c"}, k=3)    # nothing relevant is readable
    assert m["precision"] == 0.0
    assert (m["recall"], m["mrr"], m["ndcg"]) == (None, None, None)
    assert m["leaked"] == ["c"]


def test_acl_target_and_leaks():
    q = "Which services are affected by the payments table failure?"
    relevant = ["tbl_payments", "svc_ledger", "svc_reporting", "pm_payments"]

    trad = TraditionalRAG(DOCS, EDGES, ["sre"], use_llm=False).calc_metric(q, relevant, ["sre"])
    assert trad["entitled"] == ["svc_ledger", "svc_reporting", "tbl_payments"]
    assert trad["blocked"] == ["pm_payments"]
    assert "pm_payments" in trad["stages"]["dense"]["leaked"]  # no ACL filter
    assert list(trad["stages"]) == ["dense"]

    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m = graph.calc_metric(q, relevant, ["sre"])
    assert list(m["stages"]) == ["dense", "bm25", "rrf", "evidence"]
    assert all(not s["leaked"] for s in m["stages"].values())
    assert m["latency_ms"] > 0


def test_bad_inputs():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    for kwargs in ({"relevant": ["nope"], "groups": ["sre"]},
                   {"relevant": ["svc_search"], "groups": ["finance"]},
                   {"relevant": ["svc_search"], "groups": ["sre"], "k": 0}):
        try:
            graph.calc_metric("search", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_no_llm_calls():
    def boom(*args, **kwargs):
        raise AssertionError("calc_metric must not call the LLM")

    for model in (TraditionalRAG(DOCS, EDGES, ["sre"], use_llm=False),
                  GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)):
        model._answer = boom
        model.llm_answer = boom
        with contextlib.redirect_stdout(io.StringIO()):
            model.calc_metric("payments table", ["tbl_payments"], ["sre"])


class FakeOllama:
    """Stands in for the HTTP opener: letter-count vectors, records every request."""

    def __init__(self):
        self.requests = []

    def open(self, request, timeout=None):
        texts = json.loads(request.data)["input"]
        self.requests.append(texts)
        vectors = [[t.lower().count(c) + 1.0 for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]
        return io.BytesIO(json.dumps({"embeddings": vectors}).encode())


def test_embedding_docs_not_re_embedded():
    if main.faiss is None:
        return  # FAISS not installed; nothing to check
    fake = FakeOllama()
    with mock.patch.object(main.urllib.request, "build_opener", return_value=fake):
        model = EmbeddingGraphRAG(DOCS, EDGES, ["sre"])
    assert len(fake.requests) == 1  # all documents embedded once, at build time
    with contextlib.redirect_stdout(io.StringIO()):
        model.calc_metric("payments failure", ["tbl_payments"], ["sre"])
        model.calc_metric("payments failure", ["tbl_payments"], ["sre"])
        model.calc_metric("search service", ["svc_search"], ["sre"])
    later = [t for batch in fake.requests[1:] for t in batch]
    assert later == ["payments failure", "search service"]  # queries only, cached on repeat


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} tests passed")
