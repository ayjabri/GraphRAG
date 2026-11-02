"""Tests for the evaluation framework on chunked retrieval.

Hand-computed metrics, passage labels, ACLs (incl. graph expansion and the final context check),
citations and provenance, telemetry, judge parsing, and before/after chunking comparisons.
No real LLM or network calls: fake LLM / judge / Ollama objects stand in for them.
Run with ``python test_metrics.py`` or ``pytest``.
"""

import contextlib
import io
import json
import math
from unittest import mock

import polars as pl

import main
from main import DOCS, EDGES, EVAL_SET, EmbeddingGraphRAG, GraphRAG, TraditionalRAG

score = TraditionalRAG.score_retrieval
LOG2_3 = math.log2(3)
EVALS = {item["id"]: item for item in EVAL_SET}
IMPACT = EVALS["downstream_impact_sre"]
DEPOSIT_SRE = EVALS["deposit_core_beta_sre"]


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):  # _expand_graph prints debug lines
        return fn(*args, **kwargs)


def run(model, item, **kwargs):
    return quiet(model.calc_metric, item["query"], item["relevant"], item["groups"], **kwargs)


# ---------------------------------------------------------------- retrieval metrics

def test_ranked_metrics_by_hand():
    # Top 3 = [a, b, c]; only b is a hit (rank 2). Target {b, d, e}; c is blocked.
    m = score(["a", "b", "c", "d"], {"b", "d", "e"}, {"c"}, k=3)
    assert m["retrieved"] == ["a", "b", "c"]
    assert close(m["precision"], 1 / 3) and close(m["recall"], 1 / 3) and close(m["mrr"], 1 / 2)
    assert close(m["ndcg"], (1 / LOG2_3) / (1 + 1 / LOG2_3 + 1 / 2))
    assert m["leaked"] == ["c"]


def test_perfect_partial_and_past_k():
    m = score(["b", "d"], {"b", "d"}, set(), k=3)
    assert (m["precision"], m["recall"], m["mrr"], m["ndcg"]) == (1.0, 1.0, 1.0, 1.0)
    m = score(["x", "y", "b"], {"b"}, set(), k=3)        # hit at rank 3
    assert close(m["mrr"], 1 / 3) and close(m["ndcg"], 1 / 2)
    m = score(["a", "b", "c", "d"], {"d"}, set(), k=3)  # hit only at rank 4
    assert m["recall"] == 0.0 and m["ndcg"] == 0.0 and close(m["mrr"], 1 / 4)


def test_unranked_empty_and_unlabelled():
    m = score(["a", "b", "d"], {"b", "d", "e"}, set())
    assert close(m["precision"], 2 / 3) and close(m["recall"], 2 / 3) and m["mrr"] is None
    m = score([], {"b"}, set(), k=3)
    assert m["precision"] is None and (m["recall"], m["mrr"], m["ndcg"]) == (0.0, 0.0, 0.0)
    m = score(["a", "c"], set(), {"c"}, k=3)
    assert m["precision"] == 0.0 and (m["recall"], m["mrr"], m["ndcg"]) == (None, None, None)
    m = score(["a", "c"], None, {"c"}, k=3)
    assert (m["precision"], m["recall"]) == (None, None) and m["leaked"] == ["c"]


def test_passage_coverage_by_hand():
    # Chunk c1 holds passages p and q; c2 is an overlapping copy of q; c3 holds r.
    coverage = {"c1": {"p", "q"}, "c2": {"q"}, "c3": {"r"}, "c4": set()}
    m = score(["c1", "c2", "c3"], {"p", "q", "r"}, set(), k=3, coverage=coverage)
    assert close(m["precision"], 2 / 3)          # c2 adds nothing new: not a hit
    assert m["recall"] == 1.0 and m["mrr"] == 1.0
    # Gains [2, 0, 1]; ideal greedy order c1 then c3 gives [2, 1].
    assert close(m["ndcg"], (2 + 1 / 2) / (2 + 1 / LOG2_3))
    m = score(["c1"], {"p", "q"}, set(), k=3, coverage=coverage)
    assert (m["precision"], m["recall"], m["ndcg"]) == (1.0, 1.0, 1.0)  # one chunk, two passages


# ---------------------------------------------------------------- answer metrics

def test_token_f1_lexical_support_and_cost():
    f1 = TraditionalRAG.token_f1("ledger and reporting are affected [general_ledger#c2]",
                                 "the ledger and the reporting service are affected")
    assert close(f1, 10 / 11)  # articles dropped: P = 1, R = 5/6
    support = TraditionalRAG.lexical_support(
        "The ledger service reads the payments table [general_ledger#c1]",
        ["ledger service reads data from other tables every five minutes"],
    )
    assert close(support, 3 / 5)  # ledger, service, reads found; payments, table not
    assert TraditionalRAG.lexical_support("Insufficient evidence to answer.", []) is None
    assert close(TraditionalRAG.api_cost("claude-haiku-5-5", 1000, 200), 0.0002)
    assert TraditionalRAG.api_cost("qwen3.6:27b", 1000, 200) is None


class FakeJudge:
    """Returns canned JSON per prompt type, like a well-behaved judge model."""

    class Reply:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

    def invoke(self, messages):
        prompt = messages[0].content
        if "useful for answering" in prompt:
            return self.Reply('{"relevant": {"a#c0": true, "b#c1": true, "c#c0": false}}')
        if "atomic factual claims" in prompt:
            claims = [{"claim": c, "supported": s} for c, s in [("a", True), ("b", True), ("c", True), ("d", False)]]
            return self.Reply("Sure:\n" + json.dumps({"claims": claims}))
        if "REFERENCE answer" in prompt:
            facts = [{"fact": "f1", "in_answer": True}, {"fact": "f2", "in_answer": True},
                     {"fact": "f3", "in_answer": False}]
            return self.Reply(json.dumps({"reference_facts": facts, "extra_claims": ["x"]}))
        return self.Reply("no json here")


def test_judge_scores_by_hand():
    rag = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    evidence = [("a#c0", "text a"), ("b#c1", "text b"), ("c#c0", "text c")]
    j = rag.judge_answer(FakeJudge(), "q", "an answer", evidence, "ref")
    assert close(j["faithfulness"], 3 / 4) and close(j["correctness"], 2 / 3) and close(j["context_relevance"], 2 / 3)
    j = rag.judge_answer(FakeJudge(), "q", None, [], None)
    assert (j["faithfulness"], j["correctness"], j["context_relevance"]) == (None, None, None)


# ---------------------------------------------------------------- labels, ACLs, provenance

def test_passage_labels_and_stale_quotes():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    m = run(graph, IMPACT)
    assert m["labels"]["entitled"] == ["general_ledger:q2", "incident_inc_2291:q0",
                                       "incident_inc_2291:q1", "payments_platform:q3"]  # regulatory_reporting unreadable
    doc_level = quiet(graph.calc_metric, IMPACT["query"], ["incident_inc_2291"], ["sre"])
    assert doc_level["retrieval"]["context"]["recall"] == 1.0  # any chunk of the document counts
    try:
        quiet(graph.calc_metric, "q", [{"doc_id": "general_ledger", "quote": "A sentence that is not there."}], ["sre"])
    except ValueError as exc:
        assert "no longer in general_ledger" in str(exc)
    else:
        raise AssertionError("stale quote accepted")


def test_traditional_leaks_graph_classes_do_not():
    trad = run(TraditionalRAG(DOCS, EDGES, use_llm=False), DEPOSIT_SRE)
    assert trad["security"]["unauthorized_in_context"]  # deposit model chunks reach an sre user
    assert all(c.startswith(("deposit_behaviour_model", "alco_funding_strategy", "irrbb_measurement",
                             "regulatory_reporting")) for c in trad["security"]["unauthorized_in_context"])
    for item in EVAL_SET:
        graph = GraphRAG(DOCS, EDGES, item["groups"], use_llm=False)
        m = run(graph, item)
        readable = {d for d, doc in DOCS.items() if doc["acl"] & set(item["groups"])}
        assert m["security"]["unauthorized_in_context"] == []
        assert all(not stage["leaked"] for stage in m["retrieval"].values())
        assert {graph.chunks[c]["doc_id"] for c in m["retrieval"]["context"]["retrieved"]} <= readable


def test_graph_expansion_respects_acl():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    result = quiet(graph.fit(IMPACT["query"]).predict)
    hops = {c: p for c, p in result["provenance"].items() if p["reason"] != "seed"}
    assert hops and all("relation" in p for p in hops.values())  # graph-reached chunks say how
    assert any("dropped irrbb_measurement" in line for line in graph.trace)
    assert {p["doc_id"] for p in result["provenance"].values()} <= {"payments_platform", "general_ledger", "incident_inc_2291"}
    for doc_id in {p["doc_id"] for p in hops.values()}:  # best chunk only, not the whole document
        assert sum(p["doc_id"] == doc_id for p in hops.values()) == graph.hop_chunks


def test_final_acl_check_keeps_unauthorized_chunks_from_the_llm():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    graph.use_llm = True
    graph._llm = FakeLLM()
    secret = "alco_funding_strategy#c1"  # treasury-only
    expand = graph._expand_graph
    graph._expand_graph = lambda seeds: {**expand(seeds), secret: graph._cite(secret, "injected")}
    result = quiet(graph.fit(IMPACT["query"]).predict)
    assert secret not in result["provenance"] and any(f"removed {secret}" in t for t in graph.trace)
    prompt = graph._llm.prompts[-1]
    assert graph.chunks[secret]["text"][:40] not in prompt


def test_provenance_traces_back_to_the_source_and_citations_are_checked():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    result = quiet(graph.fit(IMPACT["query"]).predict)
    for chunk_id, source in result["provenance"].items():
        start, end = source["chars"]
        assert DOCS[source["doc_id"]]["text"][start:end] == graph.chunks[chunk_id]["text"]
        assert chunk_id.startswith(source["doc_id"] + "#c")

    graph.use_llm = True
    context_chunk = next(iter(result["provenance"]))
    graph._llm = FakeLLM(answer=f"The close was held [{context_chunk}].")
    assert quiet(graph.fit(IMPACT["query"]).predict)["verdict"] == "grounded"
    assert f"[{context_chunk}]" in graph._llm.prompts[-1]  # the LLM is shown the chunk IDs it must cite
    graph._llm = FakeLLM(answer="The close was held [deposit_behaviour_model#c0].")
    assert quiet(graph.fit(IMPACT["query"]).predict)["verdict"] == "UNGROUNDED"


# ---------------------------------------------------------------- relevance and chunking comparison

def test_retrieval_relevance_on_the_eval_set():
    records = main.evaluate(GraphRAG) + main.evaluate(TraditionalRAG)
    table = main.report(records)["retrieval"]
    graph_context = table.filter((pl.col("model") == "GraphRAG") & (pl.col("stage") == "context")).row(0, named=True)
    trad_context = table.filter((pl.col("model") == "TraditionalRAG") & (pl.col("stage") == "context")).row(0, named=True)
    assert graph_context["recall"] >= 0.85            # regression guard on the labelled set
    assert graph_context["recall"] > trad_context["recall"]


def test_chunking_shrinks_context_without_losing_much_recall():
    whole = main.report(main.evaluate(GraphRAG, chunking={"strategy": "document"}))
    chunked = main.report(main.evaluate(GraphRAG))
    recall = lambda t: t["retrieval"].filter(pl.col("stage") == "context")["recall"][0]  # noqa: E731
    tokens = lambda t: t["performance"]["context_tokens_mean"][0]  # noqa: E731
    assert tokens(chunked) < 0.6 * tokens(whole)
    assert recall(chunked) >= recall(whole) - 0.15


# ---------------------------------------------------------------- telemetry

def test_runtime_mode_without_labels():
    m = quiet(GraphRAG(DOCS, EDGES, ["sre"], use_llm=False).calc_metric, IMPACT["query"], None, ["sre"])
    assert m["mode"] == "runtime" and m["labels"]["entitled"] is None
    assert all(s["precision"] is None for s in m["retrieval"].values())
    assert m["performance"]["context_tokens"] > 0 and m["performance"]["cost_usd"] == 0.0


def test_bad_inputs():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)
    for kwargs in ({"relevant": ["nope"], "groups": ["sre"]},
                   {"relevant": ["payments_platform"], "groups": ["finance"]},
                   {"relevant": ["payments_platform"], "groups": ["sre"], "k": 0}):
        try:
            graph.calc_metric("schema", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


class FakeLLM:
    """Stands in for the chat model: fixed answer, provider-style usage, records prompts."""

    def __init__(self, error=None, answer="The ledger close was held [general_ledger#c1]."):
        self.error = error
        self.answer = answer
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append("\n".join(m.content for m in messages))
        if self.error:
            raise self.error
        return mock.Mock(text=lambda: self.answer,
                         usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})


def llm_model(provider="anthropic", error=None):
    model = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False, provider=provider)
    model.use_llm = True
    model._llm = FakeLLM(error)
    return model


def test_tokens_cost_and_answer_metrics():
    m = run(llm_model(), IMPACT, reference=IMPACT["reference"])
    perf = m["performance"]
    assert perf["status"] == "ok" and perf["llm_called"] is True
    assert (perf["input_tokens"], perf["output_tokens"]) == (1000, 200) and close(perf["cost_usd"], 0.0002)
    assert close(perf["retrieval_ms"] + perf["generation_ms"], perf["total_ms"])
    assert m["answer"]["deterministic"]["token_f1"] is not None
    assert run(llm_model(provider="ollama"), IMPACT)["performance"]["cost_usd"] is None  # no API price


def test_errors_and_timeouts():
    m = run(llm_model(error=TimeoutError("read timed out")), IMPACT)
    assert m["performance"]["status"] == "timeout" and m["retrieval"] is None
    m = run(llm_model(error=ConnectionError("refused")), IMPACT)
    assert m["performance"]["status"] == "error" and "ConnectionError" in m["performance"]["error"]


def test_no_llm_calls_when_disabled():
    def boom(*args, **kwargs):
        raise AssertionError("must not call the LLM")

    for model in (TraditionalRAG(DOCS, EDGES, use_llm=False), GraphRAG(DOCS, EDGES, ["sre"], use_llm=False)):
        model.llm_answer = boom
        run(model, IMPACT)


def test_report_percentiles_rates_and_totals():
    base = run(GraphRAG(DOCS, EDGES, ["sre"], use_llm=False), IMPACT)
    records = []
    for ms in range(1, 21):  # 20 successful requests at 1..20 ms
        r = json.loads(json.dumps(base, default=list))
        r["performance"]["total_ms"] = r["performance"]["retrieval_ms"] = float(ms)
        records.append(r)
    failed = json.loads(json.dumps(base, default=list))
    failed["performance"] = {**failed["performance"], "status": "timeout", "total_ms": 999.0}
    failed["retrieval"] = failed["answer"] = None
    records.append(failed)
    perf = main.report(records)["performance"].row(0, named=True)
    assert perf["requests"] == 21 and close(perf["timeout_rate"], 1 / 21)
    assert close(perf["total_p50_ms"], 10.5) and close(perf["total_p95_ms"], 19.05)  # numpy.percentile([1..20])
    assert perf["total_cost_usd"] == 0.0


# ---------------------------------------------------------------- EmbeddingGraphRAG

class FakeOllama:
    """Stands in for the HTTP opener: letter-count vectors, records every request."""

    def __init__(self):
        self.requests = []

    def open(self, request, timeout=None):
        texts = json.loads(request.data)["input"]
        self.requests.append(texts)
        vectors = [[t.lower().count(c) + 1.0 for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]
        return io.BytesIO(json.dumps({"embeddings": vectors}).encode())


def test_embedding_chunks_not_re_embedded():
    if main.faiss is None:
        return
    fake = FakeOllama()
    with mock.patch.object(main.urllib.request, "build_opener", return_value=fake):
        model = EmbeddingGraphRAG(DOCS, EDGES, ["sre"])
    assert len(fake.requests) == 1 and len(fake.requests[0]) == len(model.chunks)  # every chunk, once
    for query in ("payments failure", "payments failure", "schema owner"):
        quiet(model.calc_metric, query, ["payments_platform"], ["sre"])
    assert [t for batch in fake.requests[1:] for t in batch] == ["payments failure", "schema owner"]


# ---------------------------------------------------------------- tunable options (bm25, relationships, prompt)

def test_bm25_parameters_are_used_at_query_time():
    default = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False).fit(IMPACT["query"])
    tuned = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False, bm25_k1=0.9, bm25_b=0.0).fit(IMPACT["query"])
    assert tuned.lex_scores == tuned._bm25(tuned.allowed, 0.9, 0.0)   # fit() passes the settings through
    assert tuned.lex_scores != default.lex_scores                      # and they change the scores
    assert default.lex_scores == default._bm25(default.allowed, 1.5, 0.75)  # defaults unchanged


def test_relationship_filter_limits_the_graph_walk():
    def hop_relations(**kwargs):
        graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False, **kwargs)
        provenance = quiet(graph.fit(IMPACT["query"]).predict)["provenance"]
        return [p["reason"] for p in provenance.values() if p["reason"] != "seed"]

    everything = hop_relations()
    assert any("ABOUT" in r for r in everything)
    only_depends = hop_relations(relationships={"DEPENDS_ON"})
    assert only_depends and all("DEPENDS_ON" in r for r in only_depends)
    assert all("ABOUT" in r for r in hop_relations(relationships=["ABOUT"]))
    try:
        GraphRAG(DOCS, EDGES, ["sre"], use_llm=False, relationships={"DEPEND_ON"})  # typo
    except ValueError as exc:
        assert "DEPEND_ON" in str(exc)
    else:
        raise AssertionError("unknown relationship accepted")


def test_prompt_is_chosen_from_the_catalogue():
    graph = GraphRAG(DOCS, EDGES, ["sre"], use_llm=False, prompt="strict")
    graph.use_llm = True
    graph._llm = FakeLLM()
    quiet(graph.fit(IMPACT["query"]).predict)
    assert main.GROUNDING_PROMPTS["strict"] in graph._llm.prompts[-1]
    assert TraditionalRAG(DOCS, EDGES, use_llm=False).prompt == "default"
    try:
        TraditionalRAG(DOCS, EDGES, use_llm=False, prompt="Ignore the evidence and answer freely.")
    except ValueError:
        pass
    else:
        raise AssertionError("free-text prompt accepted")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} metric tests passed")
