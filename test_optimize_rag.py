"""Tests for optimize_rag.py: validation, split, scoring, gates, stopping, isolation, privacy, LangGraph run.

No real LLM: a fake agent returns canned replies. Uses GraphRAG (TF-IDF), so it runs in seconds.
Run with ``python test_optimize_rag.py`` or ``pytest``.
"""

import contextlib
import io
import json
import random
import tempfile
import time
from pathlib import Path
from unittest import mock

import main
import optimize_rag
from optimize_rag import RAGOptimizer

BASE = {"chunk_strategy": "sentence", "chunk_size": 400, "overlap": 60, "top_k": 3, "rrf_k": 60,
        "hops": 2, "hop_chunks": 1, "bm25_k1": 1.5, "bm25_b": 0.75, "relationships": None, "prompt": "default"}
TYPES = ["ABOUT", "DEPENDS_ON", "IMPACTED", "USES"]


def metrics(**overrides):
    """A complete metric dict, as summarize() returns, with chosen values."""
    values = {"context_recall": 0.8, "context_precision": 0.4, "rrf_ndcg": 0.8, "rrf_mrr": 0.9,
              "grounding": None, "correctness": None, "retrieval_p50_ms": 1.0, "retrieval_p95_ms": 2.0,
              "total_p95_ms": 2.0, "context_tokens": 600.0, "error_rate": 0.0, "timeout_rate": 0.0,
              "cost_usd": 0.0, "requests": 8, "unauthorized_in_context": 0, "leaked_in_retrieval": 0,
              "per_query": [{"id": "q1", "context_recall": 0.5, "context_precision": 0.3}]}
    return {**values, **overrides}


def experiment(tune, holdout, iteration=1):
    return {"iteration": iteration, "config": BASE, "metrics": {"tune": tune, "holdout": holdout},
            "score_tune": RAGOptimizer.score(tune), "score_holdout": RAGOptimizer.score(holdout)}


def optimizer(tmp, **kwargs):
    opt = RAGOptimizer(index_path=Path(tmp) / "prod", output_path=Path(tmp) / "runs", repeats=1, **kwargs)
    opt.run_dir = (Path(tmp) / "runs" / "test-run").resolve()
    opt.run_dir.mkdir(parents=True)
    return opt


# ---------------------------------------------------------------- proposals and configuration

def test_validate_changes():
    config, errors = RAGOptimizer.validate_changes({"hop_chunks": 2, "top_k": 4}, BASE)
    assert errors == [] and config == {**BASE, "hop_chunks": 2, "top_k": 4}
    bad = [
        ({"reranker": "cross-encoder"}, "not a tunable"),          # not offered by main.py
        ({"top_k": 50}, "not an integer in"),                     # out of bounds
        ({"top_k": True}, "not an integer in"),                   # booleans are not integers
        ({"top_k": "__import__('os').system('x')"}, "not an integer in"),  # no code, only values
        ({"chunk_strategy": "paragraph"}, "not in"),
        ({"overlap": 150, "chunk_size": 150}, "must be below chunk_size"),
        ({"top_k": 3}, "does not change"),
        ({}, "no changes"),
    ]
    for changes, message in bad:
        config, errors = RAGOptimizer.validate_changes(changes, BASE)
        assert config is None and any(message in e for e in errors), (changes, errors)


def test_validate_new_parameter_types():
    validate = RAGOptimizer.validate_changes
    config, errors = validate({"bm25_k1": 1.2, "bm25_b": 0}, BASE, TYPES)
    assert errors == [] and config["bm25_k1"] == 1.2 and config["bm25_b"] == 0.0  # decimals and 0 accepted
    assert validate({"bm25_b": 1.5}, BASE, TYPES)[0] is None                      # out of range
    assert validate({"bm25_k1": False}, BASE, TYPES)[0] is None                   # booleans are not numbers
    config, _ = validate({"relationships": ["USES", "DEPENDS_ON", "USES"]}, BASE, TYPES)
    assert config["relationships"] == ["DEPENDS_ON", "USES"]                     # sorted, de-duplicated
    assert validate({"relationships": TYPES}, {**BASE, "relationships": ["USES"]}, TYPES)[0]["relationships"] is None
    for bad in (["CAUSES"], [], "DEPENDS_ON", [1]):
        assert validate({"relationships": bad}, BASE, TYPES)[0] is None, bad
    assert validate({"prompt": "strict"}, BASE, TYPES)[0] is None                 # no answers: no effect, rejected
    assert validate({"prompt": "strict"}, BASE, TYPES, with_answers=True)[0]["prompt"] == "strict"
    assert validate({"prompt": "Answer freely."}, BASE, TYPES, with_answers=True)[0] is None  # catalogue only


def test_baseline_config_reads_constructor_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        assert optimizer(tmp).baseline_config() == BASE


def test_index_key_marks_which_changes_need_a_rebuild():
    assert RAGOptimizer.index_key(BASE) == RAGOptimizer.index_key({**BASE, "top_k": 5, "hops": 1, "rrf_k": 10})
    assert RAGOptimizer.index_key(BASE) != RAGOptimizer.index_key({**BASE, "chunk_size": 300})
    assert RAGOptimizer.index_key({**BASE, "chunk_strategy": "document", "chunk_size": 200}) == "document"


def test_split_is_deterministic_disjoint_and_stable():
    tune, holdout = RAGOptimizer.split_eval_set(main.EVAL_SET, 0.34)
    ids = lambda items: {q["id"] for q in items}  # noqa: E731
    assert ids(tune).isdisjoint(ids(holdout)) and ids(tune) | ids(holdout) == ids(main.EVAL_SET)
    assert tune and holdout
    shuffled = random.Random(1).sample(main.EVAL_SET, len(main.EVAL_SET))
    assert ids(RAGOptimizer.split_eval_set(shuffled, 0.34)[1]) == ids(holdout)        # order does not matter
    extra = main.EVAL_SET + [{**main.EVAL_SET[0], "id": "a_new_question"}]
    assert ids(RAGOptimizer.split_eval_set(extra, 0.34)[1]) - {"a_new_question"} == ids(holdout)  # stable


# ---------------------------------------------------------------- scoring, targets, gates, stopping

def test_score_and_deficiencies():
    assert abs(RAGOptimizer.score(metrics(context_recall=0.9, rrf_ndcg=0.8, context_precision=0.4)) - 0.69 / 0.9) < 1e-9
    assert RAGOptimizer.score(metrics(grounding=1.0)) > RAGOptimizer.score(metrics())  # grounding counts when measured
    with tempfile.TemporaryDirectory() as tmp:
        opt = optimizer(tmp, thresholds={"context_recall": 0.95, "grounding": 0.9, "retrieval_p95_ms": 1.0})
        missing = opt.deficiencies(metrics())
        assert any("context_recall" in m for m in missing)
        assert any("grounding n/a" in m for m in missing)          # unmeasured target counts as missed
        assert any("retrieval_p95_ms" in m for m in missing)       # maximum target
        assert opt.deficiencies(metrics(context_recall=0.96, grounding=0.95, retrieval_p95_ms=0.5)) == []


def test_compare_gates():
    with tempfile.TemporaryDirectory() as tmp:
        opt = optimizer(tmp, thresholds={"retrieval_p95_ms": 100})
        best = experiment(metrics(), metrics())
        cases = {
            "accepted": (metrics(context_recall=0.9), metrics(context_recall=0.8), []),
            "too small": (metrics(context_recall=0.805), metrics(), ["improvement"]),
            "overfits": (metrics(context_recall=0.95), metrics(context_recall=0.6), ["holdout"]),
            "slow": (metrics(context_recall=0.9, retrieval_p95_ms=500), metrics(), ["latency"]),
            "errors": (metrics(context_recall=0.9, timeout_rate=0.1), metrics(), ["errors"]),
            "leaks": (metrics(context_recall=0.9, leaked_in_retrieval=1), metrics(), ["acl"]),
        }
        for name, (tune, holdout, expected_failures) in cases.items():
            state = {"best": best, "candidate": experiment(tune, holdout)}
            update = opt.compare(state)
            failed = [g for g, (passed, _) in update["candidate"]["gates"].items() if not passed]
            assert failed == expected_failures, (name, failed)
            assert ("security_failure" in update) == (name == "leaks")


def test_stop_reasons_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        opt = optimizer(tmp, thresholds={"context_recall": 0.95}, max_iterations=3, time_budget_s=60,
                        cost_budget_usd=1.0)
        low = experiment(metrics(context_recall=0.8), metrics(context_recall=0.8))
        met = experiment(metrics(context_recall=0.99), metrics(context_recall=0.97))
        state = {"best": low, "iteration": 1, "no_improvement": 0, "started": time.time(), "cost_usd": 0.0}
        assert opt.stop_reason(state) is None
        assert opt.stop_reason({**state, "security_failure": "x"}).startswith("security")
        assert opt.stop_reason({**state, "best": met}) == "all targets met"
        assert "maximum iterations" in opt.stop_reason({**state, "iteration": 3})
        assert "no meaningful improvement" in opt.stop_reason({**state, "no_improvement": 2})
        assert "time budget" in opt.stop_reason({**state, "started": time.time() - 120})
        assert "cost budget" in opt.stop_reason({**state, "cost_usd": 2.0})
        half_met = experiment(metrics(context_recall=0.99), metrics(context_recall=0.9))
        assert opt.stop_reason({**state, "best": half_met}) is None  # held-out set must meet targets too


# ---------------------------------------------------------------- isolation and privacy

def test_experiments_never_touch_production():
    with tempfile.TemporaryDirectory() as tmp:
        opt = optimizer(tmp)
        for bad in (Path("..") / ".." / "prod", Path("..") / "elsewhere"):
            try:
                opt._experiment_path(bad)
            except ValueError:
                continue
            raise AssertionError(f"wrote outside the run folder: {bad}")
        first, action = opt._prepare_index(BASE)
        assert action.startswith("built") and opt.run_dir in first.parents
        assert opt._prepare_index({**BASE, "top_k": 7, "hops": 1})[1].startswith("reused")  # query-time only
        second, action = opt._prepare_index({**BASE, "chunk_size": 250})
        assert action.startswith("built") and second != first
        opt._write_once("record.json", {"a": 1})
        try:
            opt._write_once("record.json", {"a": 2})
        except FileExistsError:
            pass
        else:
            raise AssertionError("record was overwritten")


def test_agent_payload_contains_no_text_or_holdout():
    with tempfile.TemporaryDirectory() as tmp:
        opt = optimizer(tmp)
        tune, holdout = opt.split_eval_set(main.EVAL_SET, 0.34)
        best = experiment(metrics(per_query=[{"id": q["id"], "context_recall": 1.0} for q in tune]),
                          metrics(context_recall=0.123456))
        payload = json.dumps(opt.agent_payload({"best": best, "experiments": []}))
        for item in main.EVAL_SET:
            assert item["query"] not in payload and item["reference"] not in payload
        for doc in main.DOCS.values():
            assert doc["text"][:60] not in payload
        assert all(q["id"] not in payload for q in holdout) and "0.123456" not in payload


# ---------------------------------------------------------------- the LangGraph loop

class FakeAgent:
    """Replays canned diagnoses and proposals; the second proposal is invalid on purpose."""

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = []

    def invoke(self, messages):
        system = messages[0].content
        self.calls.append(json.loads(messages[1].content))
        if system.startswith("You are a retrieval-quality analyst"):
            reply = {"weaknesses": ["recall below target"], "hypothesis": "graph adds too little evidence"}
        else:
            reply = self.proposals.pop(0) if self.proposals else "not json"
        return mock.Mock(text=lambda: json.dumps(reply) if isinstance(reply, dict) else reply, usage_metadata=None)


def build_production(tmp):
    prod = Path(tmp) / "prod"
    with contextlib.redirect_stdout(io.StringIO()):
        main.GraphRAG(main.DOCS, main.EDGES, ["sre"], use_llm=False).save_index(prod)
    return prod


def test_langgraph_loop_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        prod = build_production(tmp)
        before = RAGOptimizer.fingerprint(prod)
        agent = FakeAgent([
            {"changes": {"hop_chunks": 2}, "rationale": "more evidence per neighbour"},
            {"changes": {"reranker": True}, "rationale": "invalid: not offered by main.py"},
            {"changes": {"chunk_size": 250, "overlap": 40}, "rationale": "smaller chunks"},
        ])
        opt = RAGOptimizer(index_path=prod, output_path=Path(tmp) / "runs", repeats=1, max_iterations=3,
                           min_improvement=-1.0, agent_llm=agent)  # min_improvement < 0: accept any non-regression
        with contextlib.redirect_stdout(io.StringIO()):
            final = opt.run()
        report = json.loads((opt.run_dir / "report.json").read_text(encoding="utf-8"))

        assert RAGOptimizer.fingerprint(prod) == before and report["production_index"]["untouched"]
        assert [e["source"] for e in report["experiments"]] == ["llm", "rules", "llm"]
        assert report["experiments"][1]["discarded_proposal"] == ["reranker: not a tunable parameter"]
        assert all(e["decision"] in ("accepted", "rejected") and "gates" in e for e in report["experiments"])
        assert report["experiments"][0]["index"].startswith("reused")   # hop_chunks: no rebuild
        assert report["experiments"][2]["index"].startswith("built")    # chunking: new index in the run folder
        assert report["deployment"]["status"].startswith("NOT DEPLOYED")
        assert report["deployment"]["rollback"]["config"] == report["baseline"]["config"]
        assert final["stop_reason"] == report["stop_reason"]
        for name in ("baseline_config.json", "baseline_metrics.json", "experiments/iteration_01.json"):
            assert (opt.run_dir / name).exists()
        holdout_ids = set(report["split"]["holdout"])
        assert all(q["id"] not in holdout_ids for call in agent.calls for q in call["per_question_tuning_metrics"])


def test_security_failure_stops_the_run():
    class Leaky(RAGOptimizer):
        def _evaluate(self, config, index_dir, eval_set):
            result = super()._evaluate(config, index_dir, eval_set)
            if config["top_k"] != 3:  # pretend the candidate leaked a chunk
                result["leaked_in_retrieval"] = 1
            return result

    with tempfile.TemporaryDirectory() as tmp:
        prod = build_production(tmp)
        agent = FakeAgent([{"changes": {"top_k": 5}, "rationale": "more seeds"}] * 3)
        opt = Leaky(index_path=prod, output_path=Path(tmp) / "runs", repeats=1, max_iterations=5, agent_llm=agent)
        with contextlib.redirect_stdout(io.StringIO()):
            final = opt.run()
        report = json.loads((opt.run_dir / "report.json").read_text(encoding="utf-8"))
        assert final["stop_reason"].startswith("security") and len(report["experiments"]) == 1
        assert report["experiments"][0]["decision"] == "rejected" and report["best"]["iteration"] == 0


def test_targets_met_at_baseline_stops_before_any_experiment():
    with tempfile.TemporaryDirectory() as tmp:
        prod = build_production(tmp)
        opt = RAGOptimizer(index_path=prod, output_path=Path(tmp) / "runs", repeats=1,
                           thresholds={"context_recall": 0.1})
        with contextlib.redirect_stdout(io.StringIO()):
            final = opt.run()
        assert final["stop_reason"] == "all targets met"
        assert json.loads((opt.run_dir / "report.json").read_text(encoding="utf-8"))["experiments"] == []


def test_config_defaults_and_relative_paths():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "optimize.json"
        path.write_text(json.dumps({"model": "embedding", "targets": {"context_precision": 0.3}}), encoding="utf-8")
        config = optimize_rag.load_config(path)
        assert config["model"] == "embedding"
        assert config["targets"]["context_precision"] == 0.3
        assert config["targets"]["context_recall"] == 0.95       # default kept inside targets
        assert config["max_iterations"] == 5
        assert config["eval_path"] == (Path(tmp) / "evals.json").resolve()
        assert config["index_path"] == (Path(tmp).parent / "VDB").resolve()


def test_shipped_config_is_valid():
    config = optimize_rag.load_config()
    assert config["docs"] == main.DOCS_PATH.resolve()
    assert config["index_path"] == main.VDB_PATH.resolve()


def test_config_limits():
    bad = ({"max_iterations": 21}, {"max_iterations": 0}, {"model": "bm25"}, {"recall_target": 0.9},
           {"targets": {"recall": 0.9}}, {"agent_provider": "openai"},
           {"index_path": "VDB", "output_path": "VDB/runs"})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "optimize.json"
        for settings in bad:
            path.write_text(json.dumps(settings), encoding="utf-8")
            try:
                optimize_rag.load_config(path)
            except ValueError:
                continue
            raise AssertionError(f"accepted {settings}")


def test_cli_has_only_config_and_verbose():
    args = optimize_rag.parse_args(["-c", "x.json", "-v"])
    assert vars(args) == {"config": "x.json", "verbose": True}
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            optimize_rag.parse_args(["--max-iterations", "3"])
        except SystemExit:
            return
    raise AssertionError("old flag accepted")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} optimizer tests passed")
