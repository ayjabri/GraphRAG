"""Self-improving retrieval optimizer for GraphRAG / EmbeddingGraphRAG, orchestrated with LangGraph.

The loop is: evaluate -> diagnose weaknesses (LLM) -> propose a configuration change (LLM, validated)
-> build or load the candidate index -> re-evaluate -> compare with the best accepted configuration
-> accept or reject (deterministic rules) -> repeat or stop.

The LLM only suggests. Every proposal is checked against PARAM_SPACE, every decision is made by
code, experiments live in a versioned folder under output_path, and the production index in
index_path is opened read-only. Nothing is deployed: the run ends with a plan for a human.

All settings (model, targets, budgets, paths, agent provider) live in one JSON file, by default
docs/optimize.json next to the documents. See DEFAULT_CONFIG for every key.

Run:
  python optimize_rag.py                                   # settings from docs/optimize.json
  python optimize_rag.py --config my_settings.json -v    # another settings file
"""

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import polars as pl
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import main

MODELS = {"graph": main.GraphRAG, "embedding": main.EmbeddingGraphRAG}
OUTPUT_PATH = Path(__file__).resolve().parent / "optimization_runs"
MAX_ITERATIONS = 20
CONFIG_PATH = main.DOCS_PATH / "optimize.json"

# Settings read from the config file; a key missing from the file takes this value. Relative
# paths are resolved against the config file's folder. agent_provider null = rule-based
# proposals only (no agent LLM calls). A grounding target generates and judges answers (slow).
DEFAULT_CONFIG = {
    "model": "graph",                  # "graph" or "embedding"
    "max_iterations": 5,               # candidate experiments, 1-MAX_ITERATIONS
    "docs": ".",                       # source document folder
    "eval_path": "evals.json",         # evaluation questions
    "index_path": "../VDB",            # production index, read-only
    "output_path": "../optimization_runs",  # parent folder for run folders
    "targets": {
        "context_recall": 0.95,        # minimum context recall
        "context_precision": None,     # minimum context precision
        "rrf_ndcg": None,              # minimum RRF NDCG@K
        "grounding": None,             # minimum judge faithfulness
        "retrieval_p95_ms": None,      # maximum retrieval P95 latency
    },
    "min_improvement": 0.005,          # smallest meaningful score gain
    "holdout_fraction": 0.34,          # share of questions held out
    "holdout_tolerance": 0.02,         # allowed held-out drop
    "repeats": 3,                      # runs per question per evaluation (latency)
    "time_budget_s": 3600,             # wall-clock budget
    "cost_budget_usd": None,           # budget for known API spend
    "agent_provider": main.PROVIDER,   # LLM for diagnosis/proposals, or null
    "allow_external_llm": False,       # let a non-Ollama provider generate/judge answers
}
NO_IMPROVEMENT_LIMIT = 2

# Everything the agent may change, with bounds. These map 1:1 to existing constructor arguments
# (chunking=..., top_k, rrf_k, hops, hop_chunks, bm25_k1, bm25_b, relationships, prompt);
# "reindex" marks changes that need a new index. Integer parameters take whole numbers, "float"
# ones any number in range, "subset" takes a list of relationship types from the corpus edges
# (or null for all), and "needs_answers" parameters only matter when answers are generated.
PARAM_SPACE = {
    "chunk_strategy": {"choices": ["sentence", "document"], "reindex": True,
                       "help": "sentence-aware chunks, or one chunk per document"},
    "chunk_size": {"min": 100, "max": 800, "reindex": True, "help": "max chunk length in index tokens"},
    "overlap": {"min": 0, "max": 200, "reindex": True, "help": "tokens repeated between chunks (< chunk_size)"},
    "top_k": {"min": 1, "max": 10, "reindex": False, "help": "RRF seed chunks"},
    "rrf_k": {"min": 1, "max": 200, "reindex": False, "help": "RRF smoothing constant"},
    "hops": {"min": 0, "max": 3, "reindex": False, "help": "graph traversal depth"},
    "hop_chunks": {"min": 0, "max": 3, "reindex": False, "help": "chunks taken per document reached by the graph"},
    "bm25_k1": {"min": 0.5, "max": 3.0, "float": True, "reindex": False, "help": "BM25 term-frequency saturation"},
    "bm25_b": {"min": 0.0, "max": 1.0, "float": True, "reindex": False, "help": "BM25 length normalization"},
    "relationships": {"subset": True, "reindex": False,
                      "help": "relationship types the graph walk follows; null = all"},
    "prompt": {"choices": sorted(main.GROUNDING_PROMPTS), "needs_answers": True, "reindex": False,
               "help": "grounding system prompt from main.GROUNDING_PROMPTS (only with generated answers)"},
}

# Requested tuning knobs that main.py does not offer, so the agent cannot tune them.
UNSUPPORTED_PARAMS = {
    "reranking": "there is no reranker; evidence is the RRF seeds plus hop_chunks per graph neighbour.",
    "embedding model": "EmbeddingGraphRAG accepts embed_model, but a change re-embeds every chunk; left to humans.",
}

# Weighted objective used to compare configurations (see RAGOptimizer.score).
OBJECTIVE_WEIGHTS = {"context_recall": 0.5, "rrf_ndcg": 0.2, "context_precision": 0.2, "grounding": 0.1}

DIAGNOSE_PROMPT = (
    "You are a retrieval-quality analyst for a permission-aware Graph RAG system. You receive only "
    "aggregate metrics, per-question metrics keyed by question ID, the current configuration and the "
    "experiment history. Identify the main weaknesses that block the targets and what is most likely "
    "causing them. Be specific and brief. Reply with JSON only: "
    '{"weaknesses": ["..."], "hypothesis": "..."}'
)
PROPOSE_PROMPT = (
    "You tune a permission-aware Graph RAG retriever. Propose ONE controlled experiment: change one or "
    "two parameters from the allowed parameter space, within the bounds given, that best addresses the "
    "diagnosis. Integer parameters take whole numbers; bm25_k1 and bm25_b take decimals; relationships "
    "takes a list chosen from relationship_types, or null for all; prompt is only allowed when "
    "answers_measured is true. Never repeat a configuration in the history. Prefer changes that do not "
    "need a reindex when they could plausibly fix the weakness. Reply with JSON only: "
    '{"changes": {"<parameter>": <value>}, "rationale": "..."}'
)


class OptimizerState(TypedDict, total=False):
    """LangGraph state carried across iterations.

    Keys: ``iteration``, ``baseline`` and ``best`` (experiment records),
    ``candidate`` (experiment in progress), ``diagnosis``, ``experiments``
    (every candidate with its decision), ``no_improvement`` (consecutive
    non-accepted iterations), ``security_failure``, ``stop_reason``,
    ``started`` and ``cost_usd`` (known spend so far).
    """

    iteration: int
    baseline: dict
    best: dict
    candidate: dict
    diagnosis: dict
    experiments: list
    no_improvement: int
    security_failure: str
    stop_reason: str
    started: float
    cost_usd: float


class RAGOptimizer:
    """Controlled, auditable optimization of a Graph RAG configuration.

    Parameters
    ----------
    model : str, default="graph"
        "graph" (GraphRAG, TF-IDF) or "embedding" (EmbeddingGraphRAG, Ollama + FAISS).
    corpus : dict, optional
        main.load_corpus() result; defaults to main.CORPUS (docs/).
    eval_set : list of dict, optional
        Labelled questions; defaults to the corpus's evals.json.
    index_path : str or Path, default=main.VDB_PATH
        Production index. Opened read-only for the baseline; never written.
    output_path : str or Path, default=OUTPUT_PATH
        Parent folder for versioned run folders.
    thresholds : dict, optional
        Targets: ``context_recall``, ``context_precision``, ``rrf_ndcg``,
        ``grounding`` (minimums) and ``retrieval_p95_ms`` (maximum). None
        entries are not required.
    max_iterations : int, default=5
        Candidate experiments to run at most (capped at MAX_ITERATIONS).
    min_improvement : float, default=0.005
        Smallest tuning-score gain that counts as a meaningful improvement.
    holdout_fraction : float, default=0.34
        Share of questions held out for regression checks; never shown to the LLM.
    holdout_tolerance : float, default=0.02
        Largest allowed drop of the held-out score or recall versus the best.
    repeats : int, default=3
        Times each question is run per evaluation, for latency percentiles.
    k : int, default=3
        Fixed cutoff for ranked metrics, so candidates with a different
        ``top_k`` are still scored on the same scale.
    time_budget_s : float, default=3600
        Wall-clock budget for the whole run.
    cost_budget_usd : float, optional
        Budget for known API spend (agent and answer LLMs). Self-hosted models
        have no price, so their spend is not counted.
    agent_llm : BaseChatModel, optional
        LLM for diagnosis and proposals; None uses rule-based fallbacks only.
    agent_model : str, optional
        The agent's model name, used only to price its calls (LLM_PRICES_PER_MTOK).
    with_answers : bool, default=False
        Generate answers and judge them, which is needed for ``grounding``.
    judge : BaseChatModel, optional
        Judge LLM for answer metrics (required when ``with_answers``).
    verbose : bool, default=False
        Print diagnoses, proposals and per-question metrics.

    Notes
    -----
    Safeguards, all enforced in code:

    - The agent sees aggregate and per-question metrics keyed by question
      ID, configurations and history. It never sees question text, answers
      or document text, so nothing sensitive leaves the machine through the
      agent even if its provider is external.
    - Proposals are validated against PARAM_SPACE; anything else (unknown
      keys, out-of-bounds values, code) is discarded and logged.
    - Indexes are built only inside the run folder (_experiment_path), and
      the production index's files are hashed before and after the run.
    - Acceptance is deterministic (see compare()); the LLM cannot override it.
    - The run never deploys; finalize() writes a plan for human approval.
    """

    def __init__(self, model="graph", corpus=None, eval_set=None, index_path=main.VDB_PATH,
                 output_path=OUTPUT_PATH, thresholds=None, max_iterations=5, min_improvement=0.005,
                 holdout_fraction=0.34, holdout_tolerance=0.02, repeats=3, k=3, time_budget_s=3600,
                 cost_budget_usd=None, agent_llm=None, agent_model=None, with_answers=False, judge=None,
                 verbose=False):
        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}")
        if not 1 <= max_iterations <= MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1 and {MAX_ITERATIONS}")
        if with_answers and judge is None:
            raise ValueError("with_answers needs a judge LLM to measure grounding")
        self.model_name = model
        self.model_cls = MODELS[model]
        self.corpus = main.CORPUS if corpus is None else corpus
        self.eval_set = self.corpus["evals"] if eval_set is None else eval_set
        self.index_path = Path(index_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.thresholds = {"context_recall": 0.95, **(thresholds or {})}
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.holdout_fraction = holdout_fraction
        self.holdout_tolerance = holdout_tolerance
        self.repeats = repeats
        self.k = k
        self.time_budget_s = time_budget_s
        self.cost_budget_usd = cost_budget_usd
        self.agent_llm = agent_llm
        self.agent_model = agent_model
        self.with_answers = with_answers
        self.judge = judge
        self.verbose = verbose
        self.groups = sorted(set().union(*(doc["acl"] for doc in self.corpus["docs"].values())))
        self.relationship_types = sorted({relationship for _, relationship, _ in self.corpus["edges"]})
        self.index_dirs = {}  # index key -> folder holding that index
        self.agent_calls = []  # one entry per agent LLM call: node, seconds, tokens, cost
        self.graph = self.build_graph()  # compiled LangGraph; view with self.graph.get_graph().draw_mermaid()

    # ------------------------------------------------------------ configuration

    @staticmethod
    def split_eval_set(eval_set, holdout_fraction):
        """Split questions into tuning and held-out sets, deterministically.

        Each question's ``id`` is hashed (SHA-256) to a number in [0, 1); it
        is held out when that number is below ``holdout_fraction``. The
        assignment depends only on the question's own ID, so it does not
        change between runs, when evals.json is reordered, or when new
        questions are added. If that leaves a set empty (tiny eval sets), the
        question with the lowest (or highest) hash is moved over.

        Returns
        -------
        tuple of list
            (tuning, holdout), both non-empty.
        """
        if len(eval_set) < 2:
            raise ValueError("Need at least two labelled questions to keep a held-out set")
        position = {item["id"]: int(hashlib.sha256(item["id"].encode("utf-8")).hexdigest()[:12], 16) / 16 ** 12
                    for item in eval_set}
        holdout_ids = {qid for qid, value in position.items() if value < holdout_fraction}
        ranked = sorted(position, key=position.get)
        if not holdout_ids:
            holdout_ids = {ranked[0]}
        if len(holdout_ids) == len(eval_set):
            holdout_ids.discard(ranked[-1])
        tuning = [item for item in eval_set if item["id"] not in holdout_ids]
        holdout = [item for item in eval_set if item["id"] in holdout_ids]
        return tuning, holdout

    def baseline_config(self):
        """Read the production configuration: chunking from the index (or docs/), the rest from defaults.

        Returns
        -------
        dict
            Flat configuration with every key of PARAM_SPACE.
        """
        chunking = self.corpus["chunking"]
        metadata_path = self.index_path / "metadata.json"
        if metadata_path.exists():
            chunking = json.loads(metadata_path.read_text(encoding="utf-8"))["config"]["chunking"]
        defaults = inspect.signature(self.model_cls.__init__).parameters
        relationships = defaults["relationships"].default
        return {
            "chunk_strategy": chunking["strategy"], "chunk_size": chunking["chunk_size"],
            "overlap": chunking["overlap"], "top_k": defaults["top_k"].default,
            "rrf_k": defaults["rrf_k"].default, "hops": defaults["hops"].default,
            "hop_chunks": defaults["hop_chunks"].default, "bm25_k1": defaults["bm25_k1"].default,
            "bm25_b": defaults["bm25_b"].default,
            "relationships": None if relationships is None else sorted(relationships),
            "prompt": defaults["prompt"].default,
        }

    @staticmethod
    def validate_changes(changes, config, relationship_types=(), with_answers=False):
        """Apply proposed changes to ``config`` after checking them against PARAM_SPACE.

        Parameters
        ----------
        changes : dict
            Parameter -> proposed value, as returned by the agent.
        config : dict
            Configuration the changes start from (the best accepted one).
        relationship_types : iterable of str, default=()
            Relationship types present in the corpus edges; the only values
            ``relationships`` may contain.
        with_answers : bool, default=False
            Whether answers are generated. ``prompt`` changes are rejected
            without answers, because they could not change any metric.

        Returns
        -------
        tuple
            (new_config or None, errors). A proposal is rejected as a whole
            if any key is unknown, a value has the wrong type or is out of
            bounds, ``overlap`` is not below ``chunk_size``, or nothing
            changes. Booleans are never accepted as numbers. A relationship
            list is stored sorted, and a list naming every type is stored as
            None (all), so equivalent settings compare equal.
        """
        errors = []
        if not isinstance(changes, dict) or not changes:
            return None, ["proposal has no changes"]
        all_types = sorted(relationship_types)
        new_config = dict(config)
        for name, value in changes.items():
            spec = PARAM_SPACE.get(name)
            if spec is None:
                errors.append(f"{name}: not a tunable parameter")
                continue
            if spec.get("needs_answers") and not with_answers:
                errors.append(f"{name}: only affects generated answers; set a grounding target")
                continue
            if "choices" in spec:
                if value not in spec["choices"]:
                    errors.append(f"{name}: {value!r} not in {spec['choices']}")
                    continue
            elif spec.get("subset"):
                if value is not None and (not isinstance(value, list) or not value
                                          or not all(isinstance(v, str) for v in value)
                                          or not set(value) <= set(all_types)):
                    errors.append(f"{name}: {value!r} is not null or a non-empty list from {all_types}")
                    continue
                value = None if value is None or sorted(set(value)) == all_types else sorted(set(value))
            elif spec.get("float"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not spec["min"] <= value <= spec["max"]:
                    errors.append(f"{name}: {value!r} is not a number in [{spec['min']}, {spec['max']}]")
                    continue
                value = float(value)
            elif isinstance(value, bool) or not isinstance(value, int) or not spec["min"] <= value <= spec["max"]:
                errors.append(f"{name}: {value!r} is not an integer in [{spec['min']}, {spec['max']}]")
                continue
            new_config[name] = value
        if new_config["overlap"] >= new_config["chunk_size"]:
            errors.append(f"overlap {new_config['overlap']} must be below chunk_size {new_config['chunk_size']}")
        if not errors and new_config == config:
            errors.append("proposal does not change the configuration")
        return (None, errors) if errors else (new_config, [])

    @staticmethod
    def index_key(config):
        """Name the index a configuration needs; equal keys share one index (no rebuild)."""
        if config["chunk_strategy"] == "document":
            return "document"
        return f"sentence-{config['chunk_size']}-{config['overlap']}"

    @staticmethod
    def chunking_of(config):
        """The ``chunking`` constructor argument for a flat configuration."""
        return {"strategy": config["chunk_strategy"], "chunk_size": config["chunk_size"],
                "overlap": config["overlap"]}

    @staticmethod
    def changed(before, after):
        """Parameters whose value differs: name -> [before, after]."""
        return {name: [before[name], after[name]] for name in after if before.get(name) != after[name]}

    # ------------------------------------------------------------ isolation

    def _experiment_path(self, *parts):
        """Return a path inside the run folder; refuse anything else (production safeguard)."""
        path = self.run_dir.joinpath(*parts).resolve()
        if self.run_dir not in path.parents and path != self.run_dir:
            raise ValueError(f"Refusing to write outside the run folder: {path}")
        if path == self.index_path or self.index_path in path.parents:
            raise ValueError(f"Refusing to write into the production index: {path}")
        return path

    def _write_once(self, path, payload):
        """Write a JSON file that must not exist yet, so records cannot be silently rewritten."""
        target = self._experiment_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    @staticmethod
    def fingerprint(path):
        """SHA-256 of every file under ``path`` (relative name -> digest); None if it does not exist."""
        path = Path(path)
        if not path.exists():
            return None
        return {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(path.rglob("*")) if f.is_file()}

    def _prepare_index(self, config):
        """Return (index folder, action) for ``config``, building an index only if none matches.

        Query-time parameters (top_k, rrf_k, hops, hop_chunks) never trigger
        a rebuild. A new chunking builds once into
        ``<run>/indexes/<model>-<key>/`` and is reused by later candidates.
        """
        key = self.index_key(config)
        if key in self.index_dirs:
            return self.index_dirs[key], f"reused index {key}"
        index_dir = self._experiment_path("indexes", f"{self.model_name}-{key}")
        with contextlib.redirect_stdout(io.StringIO()):
            builder = self.model_cls(self.corpus["docs"], self.corpus["edges"], groups=self.groups,
                                     use_llm=False, chunking=self.chunking_of(config))
            builder.save_index(index_dir)
        self.index_dirs[key] = index_dir
        return index_dir, f"built index {key}"

    # ------------------------------------------------------------ evaluation

    def _evaluate(self, config, index_dir, eval_set):
        """Run main.evaluate() for one configuration and question set; return summarize()."""
        records = main.evaluate(
            self.model_cls, eval_set=eval_set, k=self.k, repeats=self.repeats,
            use_llm=self.with_answers, judge=self.judge if self.with_answers else None, corpus=self.corpus,
            chunking=self.chunking_of(config), index_path=index_dir, top_k=config["top_k"],
            rrf_k=config["rrf_k"], hops=config["hops"], hop_chunks=config["hop_chunks"],
            bm25_k1=config["bm25_k1"], bm25_b=config["bm25_b"], relationships=config["relationships"],
            prompt=config["prompt"],
        )
        return self.summarize(records, eval_set)

    @staticmethod
    def summarize(records, eval_set):
        """Reduce calc_metric() records to the metrics the optimizer decides on.

        Returns
        -------
        dict
            ``context_recall`` / ``context_precision`` (what the LLM receives),
            ``rrf_ndcg`` / ``rrf_mrr`` (ranking quality at the fixed cutoff),
            ``grounding`` and ``correctness`` (judge estimates, None without
            answers), ``retrieval_p50_ms`` / ``retrieval_p95_ms`` /
            ``total_p95_ms``, ``context_tokens``, ``error_rate``,
            ``timeout_rate``, ``cost_usd`` (None when unpriced),
            ``unauthorized_in_context``, ``leaked_in_retrieval``,
            ``requests`` and ``per_query`` (first round, keyed by question ID).
        """
        tables = main.report(records)
        retrieval = tables["retrieval"]
        perf = tables["performance"].row(0, named=True)
        security = tables["security"].row(0, named=True)
        answer = tables["answer"].row(0, named=True)

        def stage(name, column):
            rows = retrieval.filter(pl.col("stage") == name)
            return rows[column][0] if rows.height else None

        per_query = []
        for item, record in zip(eval_set, records):
            stages = record["retrieval"] or {}
            context = stages.get("context") or {}
            rrf = stages.get("rrf") or {}
            entitled = record["labels"]["entitled"]
            per_query.append({
                "id": item["id"], "status": record["performance"]["status"],
                "readable_passages": None if entitled is None else len(entitled),
                "context_recall": context.get("recall"), "context_precision": context.get("precision"),
                "rrf_ndcg": rrf.get("ndcg"), "context_chunks": len(context.get("retrieved", [])),
            })
        return {
            "context_recall": stage("context", "recall"), "context_precision": stage("context", "precision"),
            "rrf_ndcg": stage("rrf", "ndcg"), "rrf_mrr": stage("rrf", "mrr"),
            "grounding": answer["judge_faithfulness"], "correctness": answer["judge_correctness"],
            "retrieval_p50_ms": perf["retrieval_p50_ms"], "retrieval_p95_ms": perf["retrieval_p95_ms"],
            "total_p95_ms": perf["total_p95_ms"], "context_tokens": perf["context_tokens_mean"],
            "error_rate": perf["error_rate"], "timeout_rate": perf["timeout_rate"],
            "cost_usd": perf["total_cost_usd"], "requests": perf["requests"],
            "unauthorized_in_context": security["unauthorized_in_context"] or 0,
            "leaked_in_retrieval": security["leaked_in_retrieval"] or 0,
            "per_query": per_query,
        }

    @staticmethod
    def score(metrics):
        r"""Weighted objective used to compare configurations.

        .. math::

            S = \frac{\sum_m w_m \, x_m}{\sum_m w_m}

        over the metrics :math:`m` in OBJECTIVE_WEIGHTS that were measured:
        context recall (0.5), RRF NDCG@K (0.2), context precision (0.2) and
        judge grounding (0.1, only with answers). Recall dominates because
        evidence the LLM never sees cannot be recovered; precision and NDCG
        keep the context small and well ordered. Latency, errors and ACLs are
        hard gates, not part of the score. Example: recall 0.9, NDCG 0.8,
        precision 0.4 -> (0.45 + 0.16 + 0.08) / 0.9 = 0.767.
        """
        measured = {name: metrics.get(name) for name in OBJECTIVE_WEIGHTS if metrics.get(name) is not None}
        total_weight = sum(OBJECTIVE_WEIGHTS[name] for name in measured)
        if not total_weight:
            return 0.0
        return sum(OBJECTIVE_WEIGHTS[name] * value for name, value in measured.items()) / total_weight

    def deficiencies(self, metrics):
        """List every configured target that ``metrics`` misses (empty when all are met).

        Minimum targets: context_recall, context_precision, rrf_ndcg,
        grounding. Maximum target: retrieval_p95_ms. A target whose metric
        was not measured counts as missed.
        """
        missing = []
        for name, target in self.thresholds.items():
            if target is None:
                continue
            value = metrics.get(name)
            if name == "retrieval_p95_ms":
                if value is None or value > target:
                    missing.append(f"{name} {self._fmt(value)} > {target}")
            elif value is None or value < target:
                missing.append(f"{name} {self._fmt(value)} < {target}")
        return missing

    def acl_violations(self, *metric_sets):
        """Unauthorized chunks in any retrieval stage or LLM context, summed over metric sets."""
        return sum(m["unauthorized_in_context"] + m["leaked_in_retrieval"] for m in metric_sets)

    # ------------------------------------------------------------ agent

    def _ask_json(self, node, system, payload, required_key):
        """Ask the agent LLM and parse its JSON reply; None on any failure (callers fall back to rules)."""
        if self.agent_llm is None:
            return None
        messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(payload, default=str))]
        start = time.perf_counter()
        try:
            response = self.agent_llm.invoke(messages)
            text = response.text() if callable(getattr(response, "text", None)) else str(response.content)
            usage = getattr(response, "usage_metadata", None) or {}
            match = re.search(r"\{.*\}", text, re.DOTALL)
            reply = json.loads(match.group(0)) if match else None
            error = None if reply and required_key in reply else f"reply has no {required_key!r}"
        except Exception as exc:
            reply, usage, error = None, {}, f"{type(exc).__name__}: {exc}"
        cost = main.TraditionalRAG.api_cost(self.agent_model, usage.get("input_tokens"), usage.get("output_tokens"))
        self.agent_calls.append({
            "node": node, "seconds": round(time.perf_counter() - start, 2), "error": error,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "cost_usd": cost,
        })
        return None if error else reply

    def agent_payload(self, state):
        """What the agent is allowed to see: tuning metrics, configs and history; no text.

        Question text, answers, document text and held-out results are never
        included, so the agent cannot overfit the held-out set and sensitive
        content is not sent to its provider.
        """
        best = state["best"]
        history = [
            {"iteration": e["iteration"], "changes": e["changes"], "decision": e["decision"],
             "tuning_score": e.get("score_tune"), "tuning_recall": (e.get("metrics") or {}).get("tune", {}).get("context_recall")}
            for e in state.get("experiments", [])
        ]
        tune = {k: v for k, v in best["metrics"]["tune"].items() if k != "per_query"}
        per_question = [
            {**q, "note": "user may read no labelled passage; empty context is correct, not a failure"}
            if q.get("readable_passages") == 0 else q
            for q in best["metrics"]["tune"]["per_query"]
        ]
        return {
            "model": self.model_name,
            "targets": {k: v for k, v in self.thresholds.items() if v is not None},
            "objective_weights": OBJECTIVE_WEIGHTS,
            "metric_notes": "Averages skip null values. context_recall is null when the user may read none of "
                            "the labelled passages (permission-restricted question); those are not failures.",
            "best_config": best["config"],
            "best_tuning_metrics": tune,
            "per_question_tuning_metrics": per_question,
            "parameter_space": PARAM_SPACE,
            "relationship_types": self.relationship_types,
            "answers_measured": self.with_answers,
            "not_tunable": list(UNSUPPORTED_PARAMS),
            "history": history,
        }

    def fallback_diagnosis(self, metrics):
        """Rule-based diagnosis used when the agent is disabled or its reply is unusable."""
        weaknesses = list(self.deficiencies(metrics))
        if metrics.get("context_precision") is not None and metrics["context_precision"] < 0.4:
            weaknesses.append("low context precision: many context chunks hold no labelled passage")
        missed = [q["id"] for q in metrics["per_query"] if q["context_recall"] is not None and q["context_recall"] < 1]
        if missed:
            weaknesses.append(f"questions missing passages: {missed}")
        return {"weaknesses": weaknesses or ["no target missed"], "hypothesis": "rule-based", "source": "rules"}

    def fallback_proposal(self, config, metrics, tried):
        """Next untried single-parameter move, chosen by rule from the current weaknesses.

        Low recall tries more evidence first (hop_chunks, top_k, hops), then
        larger chunks and more overlap; low precision tries less evidence,
        smaller chunks, and a graph walk that follows only one relationship
        type at a time; finally rrf_k and the BM25 parameters are varied.
        Returns (config, rationale) or (None, reason) when every move has
        been tried or is invalid.
        """
        recall_low = (metrics.get("context_recall") or 0) < (self.thresholds.get("context_recall") or 1.0)
        precision_low = (metrics.get("context_precision") or 0) < (self.thresholds.get("context_precision") or 0.4)
        moves = []
        if recall_low:
            moves += [("hop_chunks", config["hop_chunks"] + 1), ("top_k", config["top_k"] + 1),
                      ("hops", config["hops"] + 1), ("chunk_size", round(config["chunk_size"] * 1.5 / 50) * 50),
                      ("overlap", config["overlap"] + 30)]
        if precision_low:
            moves += [("top_k", config["top_k"] - 1), ("hop_chunks", config["hop_chunks"] - 1),
                      ("chunk_size", round(config["chunk_size"] * 0.75 / 50) * 50)]
            moves += [("relationships", [relationship]) for relationship in self.relationship_types]
        moves += [("rrf_k", 30), ("rrf_k", 120), ("bm25_b", 0.5), ("bm25_b", 0.9), ("bm25_k1", 1.2), ("bm25_k1", 2.0)]
        for name, value in moves:
            candidate, errors = self.validate_changes({name: value}, config, self.relationship_types, self.with_answers)
            if candidate is not None and candidate not in tried:
                return candidate, f"rule: set {name} from {config[name]} to {value}"
        return None, "every rule-based move has been tried"

    # ------------------------------------------------------------ graph nodes

    def load_baseline(self, state):
        """Node 1: create the run folder, split the questions, freeze the baseline configuration."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = (self.output_path / f"{stamp}-{self.model_name}").resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.tuning_set, self.holdout_set = self.split_eval_set(self.eval_set, self.holdout_fraction)
        self.production_fingerprint = self.fingerprint(self.index_path)
        config = self.baseline_config()
        self._write_once("baseline_config.json", {
            "config": config, "index_path": self.index_path,
            "tuning_ids": [q["id"] for q in self.tuning_set], "holdout_ids": [q["id"] for q in self.holdout_set],
        })
        print(f"Run folder: {self.run_dir}")
        print(f"Questions: {len(self.tuning_set)} tuning, {len(self.holdout_set)} held out (never shown to the agent)")
        return {"iteration": 0, "experiments": [], "no_improvement": 0, "started": time.time(),
                "cost_usd": 0.0, "baseline": {"iteration": 0, "config": config}}

    def evaluate_baseline(self, state):
        """Node 2: evaluate the baseline on both sets, reading the production index if it matches."""
        config = state["baseline"]["config"]
        action = "loaded production index (read-only)"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.model_cls(self.corpus["docs"], self.corpus["edges"], groups=self.groups, use_llm=False,
                               chunking=self.chunking_of(config), index_path=self.index_path)
            self.index_dirs[self.index_key(config)] = self.index_path
        except (ValueError, FileNotFoundError, KeyError) as exc:
            action = f"production index unusable ({exc}); built a copy in the run folder"
            self._prepare_index(config)
        index_dir = self.index_dirs[self.index_key(config)]
        metrics = {"tune": self._evaluate(config, index_dir, self.tuning_set),
                   "holdout": self._evaluate(config, index_dir, self.holdout_set)}
        baseline = {**state["baseline"], "metrics": metrics, "index": action, "decision": "baseline",
                    "score_tune": self.score(metrics["tune"]), "score_holdout": self.score(metrics["holdout"])}
        self._write_once("baseline_metrics.json", baseline)
        update = {"baseline": baseline, "best": baseline,
                  "cost_usd": state["cost_usd"] + (metrics["tune"]["cost_usd"] or 0) + (metrics["holdout"]["cost_usd"] or 0)}
        violations = self.acl_violations(metrics["tune"], metrics["holdout"])
        if violations:
            update["security_failure"] = f"baseline has {violations} unauthorized chunks"
        self._print_baseline(baseline)
        return update

    def diagnose(self, state):
        """Node 3: LLM diagnosis of the best configuration's weaknesses (tuning metrics only)."""
        payload = self.agent_payload(state)
        reply = self._ask_json("diagnose", DIAGNOSE_PROMPT, payload, "weaknesses")
        diagnosis = ({**reply, "source": "llm"} if reply
                     else self.fallback_diagnosis(state["best"]["metrics"]["tune"]))
        if self.verbose:
            print(f"\n  diagnosis ({diagnosis['source']}): {diagnosis['weaknesses']}")
        return {"diagnosis": diagnosis, "iteration": state["iteration"] + 1}

    def propose(self, state):
        """Node 4: LLM proposal of one controlled change, validated; rules take over if it is unusable."""
        best_config = state["best"]["config"]
        tried = [state["baseline"]["config"]] + [e["config"] for e in state["experiments"]]
        payload = {**self.agent_payload(state), "diagnosis": state["diagnosis"]}
        reply = self._ask_json("propose", PROPOSE_PROMPT, payload, "changes")
        config, errors = (self.validate_changes(reply.get("changes"), best_config, self.relationship_types,
                                                self.with_answers) if reply else (None, []))
        if config is not None and config in tried:
            config, errors = None, ["configuration already tried"]
        source = "llm"
        rationale = (reply or {}).get("rationale", "")
        if config is None:
            source = "rules"
            config, rationale = self.fallback_proposal(best_config, state["best"]["metrics"]["tune"], tried)
        candidate = None
        if config is not None:
            candidate = {"iteration": state["iteration"], "config": config, "source": source,
                         "changes": self.changed(best_config, config), "rationale": rationale,
                         "discarded_proposal": errors, "diagnosis": state["diagnosis"]}
        if self.verbose and errors:
            print(f"  discarded LLM proposal: {errors}")
        update = {"candidate": candidate}
        if candidate is None:
            update["stop_reason"] = f"no untried candidate ({rationale})"
        return update

    def prepare_index(self, state):
        """Node 5: build or reuse the candidate's index inside the run folder."""
        candidate = state["candidate"]
        index_dir, action = self._prepare_index(candidate["config"])
        return {"candidate": {**candidate, "index": action, "index_dir": index_dir}}

    def evaluate_candidate(self, state):
        """Node 6: evaluate on the tuning set and run the regression checks on the held-out set."""
        candidate = state["candidate"]
        start = time.perf_counter()
        try:
            metrics = {"tune": self._evaluate(candidate["config"], candidate["index_dir"], self.tuning_set),
                       "holdout": self._evaluate(candidate["config"], candidate["index_dir"], self.holdout_set)}
            error = None
        except Exception as exc:
            metrics, error = None, f"{type(exc).__name__}: {exc}"
        cost = 0.0 if metrics is None else (metrics["tune"]["cost_usd"] or 0) + (metrics["holdout"]["cost_usd"] or 0)
        evaluated = {**candidate, "metrics": metrics, "evaluation_error": error,
                     "evaluation_seconds": round(time.perf_counter() - start, 2)}
        if metrics is not None:
            evaluated["score_tune"] = self.score(metrics["tune"])
            evaluated["score_holdout"] = self.score(metrics["holdout"])
        return {"candidate": evaluated, "cost_usd": state["cost_usd"] + cost}

    def compare(self, state):
        """Node 7: check the candidate against the best accepted configuration.

        Gates, all required for acceptance:

        - ``evaluation``: the evaluation ran without an exception.
        - ``acl``: zero unauthorized chunks in any stage or context, on both
          sets. A failure also stops the run (security failure).
        - ``errors``: no request errors or timeouts.
        - ``latency``: retrieval P95 within ``retrieval_p95_ms`` (if set).
        - ``improvement``: tuning score >= best tuning score +
          ``min_improvement``.
        - ``holdout``: held-out score and held-out context recall each drop
          by at most ``holdout_tolerance`` versus the best. This is the
          overfitting guard: a change that only helps the questions the agent
          saw is rejected.
        """
        candidate = state["candidate"]
        best = state["best"]
        metrics = candidate["metrics"]
        if metrics is None:
            gates = {"evaluation": [False, candidate["evaluation_error"]]}
            return {"candidate": {**candidate, "gates": gates}}
        tune, holdout = metrics["tune"], metrics["holdout"]
        best_holdout = best["metrics"]["holdout"]
        violations = self.acl_violations(tune, holdout)
        failures = max(tune["error_rate"], tune["timeout_rate"], holdout["error_rate"], holdout["timeout_rate"])
        p95_limit = self.thresholds.get("retrieval_p95_ms")
        p95 = max(tune["retrieval_p95_ms"] or 0, holdout["retrieval_p95_ms"] or 0)
        gain = candidate["score_tune"] - best["score_tune"]
        holdout_drop = best["score_holdout"] - candidate["score_holdout"]
        recall_drop = (best_holdout["context_recall"] or 0) - (holdout["context_recall"] or 0)
        gates = {
            "evaluation": [True, "ok"],
            "acl": [violations == 0, f"{violations} unauthorized chunks"],
            "errors": [failures == 0, f"error/timeout rate {failures:.3f}"],
            "latency": [p95_limit is None or p95 <= p95_limit, f"retrieval p95 {p95:.2f} ms (limit {p95_limit})"],
            "improvement": [gain >= self.min_improvement, f"tuning score {gain:+.4f} (need >= {self.min_improvement})"],
            "holdout": [holdout_drop <= self.holdout_tolerance and recall_drop <= self.holdout_tolerance,
                        f"held-out score {-holdout_drop:+.4f}, recall {-recall_drop:+.4f} (tolerance {self.holdout_tolerance})"],
        }
        update = {"candidate": {**candidate, "gates": gates}}
        if violations:
            update["security_failure"] = f"iteration {candidate['iteration']}: {violations} unauthorized chunks"
        return update

    def decide(self, state):
        """Node 8: accept the candidate only if every gate passed; otherwise keep (roll back to) the best."""
        candidate = state["candidate"]
        accepted = all(passed for passed, _ in candidate["gates"].values())
        record = {**candidate, "decision": "accepted" if accepted else "rejected",
                  "failed_gates": [name for name, (passed, _) in candidate["gates"].items() if not passed]}
        record.pop("index_dir", None)
        self._write_once(Path("experiments") / f"iteration_{record['iteration']:02d}.json", record)
        previous_best = state["best"]
        best = record if accepted else previous_best
        self._print_iteration(record, previous_best)
        return {"experiments": state["experiments"] + [record], "best": best,
                "no_improvement": 0 if accepted else state["no_improvement"] + 1}

    def stop_reason(self, state):
        """Return why the loop must stop now, or None to continue.

        Checked in order: security / ACL failure; every target met by the
        best configuration on both the tuning and held-out sets; maximum
        iterations; NO_IMPROVEMENT_LIMIT consecutive iterations without an
        accepted candidate; time budget; known-cost budget.
        """
        if state.get("security_failure"):
            return f"security: {state['security_failure']}"
        if state.get("stop_reason"):
            return state["stop_reason"]
        best = state["best"]["metrics"]
        if not self.deficiencies(best["tune"]) and not self.deficiencies(best["holdout"]):
            return "all targets met"
        if state["iteration"] >= self.max_iterations:
            return f"maximum iterations ({self.max_iterations}) reached"
        if state["no_improvement"] >= NO_IMPROVEMENT_LIMIT:
            return f"no meaningful improvement for {NO_IMPROVEMENT_LIMIT} consecutive iterations"
        if time.time() - state["started"] > self.time_budget_s:
            return f"time budget of {self.time_budget_s} s exceeded"
        known_cost = state["cost_usd"] + sum(c["cost_usd"] or 0 for c in self.agent_calls)
        if self.cost_budget_usd is not None and known_cost > self.cost_budget_usd:
            return f"cost budget of ${self.cost_budget_usd} exceeded (${known_cost:.4f})"
        return None

    def route(self, state):
        """Conditional edge: "finalize" when stop_reason() has a reason, else "diagnose"."""
        return "finalize" if self.stop_reason(state) else "diagnose"

    def route_after_proposal(self, state):
        """Conditional edge: no valid untried candidate -> "finalize", else "prepare_index"."""
        return "prepare_index" if state.get("candidate") else "finalize"

    def finalize(self, state):
        """Node 9: write the report, check production was untouched, print baseline vs best.

        Nothing is deployed. The report holds a deployment plan and the
        baseline configuration for rollback; a human applies it.
        """
        reason = self.stop_reason(state) or "stopped"
        baseline = state["baseline"]
        best = state["best"]
        production_untouched = self.fingerprint(self.index_path) == self.production_fingerprint
        valid = self.acl_violations(best["metrics"]["tune"], best["metrics"]["holdout"]) == 0
        report = {
            "run_dir": self.run_dir, "model": self.model_name, "stop_reason": reason,
            "thresholds": self.thresholds, "objective_weights": OBJECTIVE_WEIGHTS,
            "split": {"tuning": [q["id"] for q in self.tuning_set], "holdout": [q["id"] for q in self.holdout_set]},
            "baseline": baseline, "best": best, "best_is_valid": valid,
            "improved": best["iteration"] != 0,
            "remaining_deficiencies": {"tune": self.deficiencies(best["metrics"]["tune"]),
                                       "holdout": self.deficiencies(best["metrics"]["holdout"])},
            "experiments": state["experiments"], "agent_calls": self.agent_calls,
            "known_cost_usd": state["cost_usd"] + sum(c["cost_usd"] or 0 for c in self.agent_calls),
            "elapsed_s": round(time.time() - state["started"], 1),
            "production_index": {"path": self.index_path, "untouched": production_untouched},
            "unsupported_parameters": UNSUPPORTED_PARAMS,
            "deployment": self.deployment_plan(baseline["config"], best["config"]),
        }
        self._write_once("report.json", report)
        self._print_final(report)
        return {"stop_reason": reason}

    @staticmethod
    def deployment_plan(baseline, best):
        """Human-applied steps to deploy ``best``, and to roll back to ``baseline``. Never executed here."""
        changes = RAGOptimizer.changed(baseline, best)
        steps = []
        if any(PARAM_SPACE[name]["reindex"] for name in changes):
            steps.append("Set docs/metadata.json \"chunking\" to "
                         f"{json.dumps(RAGOptimizer.chunking_of(best))}, then rebuild: python main.py --save "
                         "(and python main.py -m ollama --save).")
        query_time = {name: best[name] for name in changes if not PARAM_SPACE[name]["reindex"]}
        if query_time:
            steps.append(f"Construct GraphRAG / EmbeddingGraphRAG with {query_time}. main.py's CLI does not "
                         "expose these yet, so this is a code or configuration change for a reviewer.")
        return {
            "status": "NOT DEPLOYED - requires explicit human approval",
            "changes": changes, "steps": steps or ["No change: the baseline is still the best configuration."],
            "rollback": {"config": baseline, "note": "Production was never modified by this run; to roll "
                                                       "back a deployment, restore these values and rebuild."},
        }

    # ------------------------------------------------------------ orchestration

    def build_graph(self):
        """Wire the nodes into a LangGraph StateGraph and compile it.

        Flow::

            load_baseline -> evaluate_baseline -> (stop? finalize : diagnose)
            diagnose -> propose -> (candidate? prepare_index : finalize)
            prepare_index -> evaluate_candidate -> compare -> decide -> (stop? finalize : diagnose)
            finalize -> END
        """
        graph = StateGraph(OptimizerState)
        graph.add_node("load_baseline", self.load_baseline)
        graph.add_node("evaluate_baseline", self.evaluate_baseline)
        graph.add_node("diagnose", self.diagnose)
        graph.add_node("propose", self.propose)
        graph.add_node("prepare_index", self.prepare_index)
        graph.add_node("evaluate_candidate", self.evaluate_candidate)
        graph.add_node("compare", self.compare)
        graph.add_node("decide", self.decide)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "load_baseline")
        graph.add_edge("load_baseline", "evaluate_baseline")
        graph.add_conditional_edges("evaluate_baseline", self.route, {"diagnose": "diagnose", "finalize": "finalize"})
        graph.add_edge("diagnose", "propose")
        graph.add_conditional_edges("propose", self.route_after_proposal,
                                    {"prepare_index": "prepare_index", "finalize": "finalize"})
        graph.add_edge("prepare_index", "evaluate_candidate")
        graph.add_edge("evaluate_candidate", "compare")
        graph.add_edge("compare", "decide")
        graph.add_conditional_edges("decide", self.route, {"diagnose": "diagnose", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(self):
        """Run the optimization loop (self.graph); return the final LangGraph state."""
        return self.graph.invoke({}, config={"recursion_limit": 20 + 8 * self.max_iterations})

    # ------------------------------------------------------------ reporting

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    def _comparison(self, left, right, left_name, right_name):
        """Polars table of the decision metrics for two metric dicts."""
        names = ["context_recall", "context_precision", "rrf_ndcg", "rrf_mrr", "grounding", "correctness",
                 "context_tokens", "retrieval_p95_ms", "total_p95_ms", "cost_usd", "unauthorized_in_context"]
        rows = [{"metric": n, left_name: self._fmt(left.get(n)), right_name: self._fmt(right.get(n)),
                 "delta": self._fmt(right[n] - left[n]) if isinstance(left.get(n), (int, float))
                 and isinstance(right.get(n), (int, float)) else ""} for n in names]
        return pl.DataFrame(rows)

    def _print_baseline(self, baseline):
        print(f"\nBaseline {baseline['config']}")
        print(f"  index: {baseline['index']}")
        print(f"  tuning score {baseline['score_tune']:.4f}, held-out score {baseline['score_holdout']:.4f}")
        print(f"  deficiencies: {self.deficiencies(baseline['metrics']['tune']) or 'none'}")

    def _print_iteration(self, record, previous_best):
        print(f"\n=== Iteration {record['iteration']} ({record['source']} proposal) ===")
        print(f"  changes: {record['changes']}")
        print(f"  why: {record['rationale']}")
        print(f"  index: {record.get('index')}")
        if record["metrics"] is not None:
            with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
                print(self._comparison(previous_best["metrics"]["tune"], record["metrics"]["tune"],
                                       "best (tuning)", "candidate (tuning)"))
            print(f"  score: tuning {previous_best['score_tune']:.4f} -> {record['score_tune']:.4f}, "
                  f"held-out {previous_best['score_holdout']:.4f} -> {record['score_holdout']:.4f}")
        for name, (passed, detail) in record["gates"].items():
            print(f"  gate {name:11s} {'pass' if passed else 'FAIL'}  {detail}")
        print(f"  decision: {record['decision'].upper()}")
        if self.verbose and record["metrics"] is not None:
            print(pl.DataFrame(record["metrics"]["tune"]["per_query"]))
        best = record if record["decision"] == "accepted" else previous_best
        print(f"  remaining deficiencies: {self.deficiencies(best['metrics']['tune']) or 'none'}")

    def _print_final(self, report):
        baseline, best = report["baseline"], report["best"]
        print(f"\n=== Finished: {report['stop_reason']} ===")
        print(f"Baseline config: {baseline['config']}")
        print(f"Best config:     {best['config']}  (iteration {best['iteration']})")
        with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
            print("Tuning set:")
            print(self._comparison(baseline["metrics"]["tune"], best["metrics"]["tune"], "baseline", "best"))
            print("Held-out set:")
            print(self._comparison(baseline["metrics"]["holdout"], best["metrics"]["holdout"], "baseline", "best"))
        accepted = sum(e["decision"] == "accepted" for e in report["experiments"])
        print(f"Experiments: {len(report['experiments'])} ({accepted} accepted); "
              f"elapsed {report['elapsed_s']} s; known cost ${report['known_cost_usd']:.4f}")
        deficiencies = report["remaining_deficiencies"]
        if deficiencies["tune"] or deficiencies["holdout"]:
            print(f"Remaining deficiencies: tuning {deficiencies['tune']}, held-out {deficiencies['holdout']}")
        print(f"Production index untouched: {report['production_index']['untouched']}")
        print(f"Deployment: {report['deployment']['status']}")
        for step in report["deployment"]["steps"]:
            print(f"  - {step}")
        print(f"Report: {Path(report['run_dir']) / 'report.json'}")


def load_eval_set(path, corpus):
    """Load and validate an evaluation file against the corpus (same rules as load_corpus()).

    Raises
    ------
    ValueError
        On a missing ``id``, a duplicate ``id``, an unknown document or a
        quote that is no longer in its document.
    """
    evals = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for item in evals:
        if not item.get("id") or item["id"] in seen:
            raise ValueError(f"Every question needs a unique id: {item.get('id')!r}")
        seen.add(item["id"])
        for label in item["relevant"]:
            doc_id = label if isinstance(label, str) else label["doc_id"]
            if doc_id not in corpus["docs"]:
                raise ValueError(f"Eval {item['id']}: unknown document {doc_id}")
            quote = None if isinstance(label, str) else label.get("quote")
            if quote and main.normalize_space(quote) not in main.normalize_space(corpus["docs"][doc_id]["text"]):
                raise ValueError(f"Eval {item['id']}: quote no longer in {doc_id}: {quote[:70]}")
    return evals


def load_config(path=CONFIG_PATH):
    """Read and validate the optimizer settings file.

    Every key is optional; a missing key takes its value from
    DEFAULT_CONFIG. Relative paths are resolved against the folder that
    holds the settings file, so the file can live next to the documents.

    Parameters
    ----------
    path : str or Path, default=CONFIG_PATH
        JSON settings file (default: docs/optimize.json).

    Returns
    -------
    dict
        DEFAULT_CONFIG merged with the file, with absolute paths and a
        complete ``targets`` block.

    Raises
    ------
    ValueError
        On an unknown key or target, a value out of range, a non-Ollama
        provider judging answers without ``allow_external_llm``, or an
        output folder inside the production index.
    """
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(raw.keys() - DEFAULT_CONFIG.keys())
    if unknown:
        raise ValueError(f"{path}: unknown settings {unknown}; allowed: {sorted(DEFAULT_CONFIG)}")
    targets = raw.get("targets") or {}
    unknown_targets = sorted(targets.keys() - DEFAULT_CONFIG["targets"].keys())
    if unknown_targets:
        raise ValueError(f"{path}: unknown targets {unknown_targets}; allowed: {sorted(DEFAULT_CONFIG['targets'])}")
    config = {**DEFAULT_CONFIG, **raw}
    config["targets"] = {**DEFAULT_CONFIG["targets"], **targets}
    for key in ("docs", "eval_path", "index_path", "output_path"):
        config[key] = (path.parent / config[key]).resolve()

    if config["model"] not in MODELS:
        raise ValueError(f"{path}: model must be one of {sorted(MODELS)}, not {config['model']!r}")
    if not 1 <= config["max_iterations"] <= MAX_ITERATIONS:
        raise ValueError(f"{path}: max_iterations must be between 1 and {MAX_ITERATIONS}")
    if not 0 < config["holdout_fraction"] < 1:
        raise ValueError(f"{path}: holdout_fraction must be between 0 and 1")
    if config["repeats"] < 1:
        raise ValueError(f"{path}: repeats must be at least 1")
    if config["agent_provider"] is not None and config["agent_provider"] not in main.LLM_MODELS:
        raise ValueError(f"{path}: agent_provider must be one of {sorted(main.LLM_MODELS)} or null")
    grounding = config["targets"]["grounding"]
    if grounding is not None and main.PROVIDER != "ollama" and not config["allow_external_llm"]:
        raise ValueError(f"{path}: a grounding target sends questions and chunks to {main.PROVIDER}; "
                         "set allow_external_llm to true")
    index_path = config["index_path"]
    output_path = config["output_path"]
    if output_path == index_path or index_path in output_path.parents:
        raise ValueError(f"{path}: output_path must not be inside index_path")
    return config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Self-improving retrieval optimizer for GraphRAG (LangGraph). "
                                            "All settings live in the JSON config file.")
    p.add_argument("-c", "--config", default=CONFIG_PATH, help="settings file (default: docs/optimize.json)")
    p.add_argument("-v", "--verbose", action="store_true", help="print diagnoses, proposals, per-question metrics")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    corpus = main.load_corpus(config["docs"])
    eval_set = load_eval_set(config["eval_path"], corpus)
    provider = config["agent_provider"]
    agent_llm = None if provider is None else main.TraditionalRAG.get_llm(provider)
    agent_model = None if provider is None else main.LLM_MODELS[provider]
    with_answers = config["targets"]["grounding"] is not None
    judge = main.TraditionalRAG.get_llm(main.PROVIDER) if with_answers else None
    optimizer = RAGOptimizer(
        model=config["model"], corpus=corpus, eval_set=eval_set, index_path=config["index_path"],
        output_path=config["output_path"], thresholds=config["targets"], max_iterations=config["max_iterations"],
        min_improvement=config["min_improvement"], holdout_fraction=config["holdout_fraction"],
        holdout_tolerance=config["holdout_tolerance"], repeats=config["repeats"],
        time_budget_s=config["time_budget_s"], cost_budget_usd=config["cost_budget_usd"],
        agent_llm=agent_llm, agent_model=agent_model,
        with_answers=with_answers, judge=judge, verbose=args.verbose,
    )
    optimizer.run()
