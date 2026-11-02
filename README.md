# GraphRAG

A small, self-contained comparison of **Traditional RAG** and **Graph RAG** on a toy
corpus of services and tables, implemented as three classes in [main.py](main.py):

| Class               | What it does                                                               |
| ------------------- | -------------------------------------------------------------------------- |
| `TraditionalRAG`    | TF-IDF top-k over everything. No permissions, no graph, no citation check. |
| `GraphRAG`          | ACL filter, BM25 + TF-IDF fused with RRF, graph walk, citation check.      |
| `EmbeddingGraphRAG` | `GraphRAG`, but the dense step uses an Ollama embedding model.             |

## How `TraditionalRAG` works

From construction to answer. Each box inside `fit` and `predict` is one line of that method,
labelled with the attribute it sets. Nested boxes are the steps inside a called function.
The dotted path is the optional LLM route, taken only when `use_llm=True`; otherwise the
retrieved evidence is returned as the answer.

```mermaid
flowchart TD
    subgraph INIT["Initiation (once)"]
        direction TB
        A(["TraditionalRAG(docs, top_k, use_llm, provider)"]) --> I1["__init__<br/>store docs, top_k, use_llm, provider"]
        I1 --> LAUNCH_IN

        subgraph LAUNCH["__launch__"]
            direction TB
            LAUNCH_IN[" "]:::hidden
            LLMB["get_llm(provider)<br/>-> self._llm"]

            subgraph BUILD["_build_index"]
                direction TB
                T1["tokenize each document<br/>-> self.tokens"] --> T2["build_tfidf_idf<br/>-> self.tfidf_idf"]
                T2 --> TV["list of IDF words<br/>-> self.vocab"] --> TD["vec for each document<br/>-> self.doc_vectors"]
            end

            LAUNCH_IN -.->|"use_llm only"| LLMB
            LAUNCH_IN --> T1
        end
    end

    subgraph QUERY["Query: fit_predict(query)"]
        direction TB
        Q(["query"]) --> F1

        subgraph FIT["fit"]
            direction TB
            F1["store query<br/>-> self.query"] --> F2["tokenize<br/>-> self.q_tokens"]

            subgraph DENSE["_dense(all docs) -> self.dense_scores"]
                direction TB
                subgraph TS["tfidf_scores"]
                    direction TB
                    D1["vec: query vector only"] --> D2["look up precomputed<br/>self.doc_vectors"] --> D3["cosine: query vs document"]
                end
            end

            F3["_rank(self.dense_scores)<br/>sort, drop scores <= 0<br/>-> self.dense"]

            F2 --> D1
            D3 --> F3
        end

        subgraph PREDICT["predict"]
            direction TB
            P1["_select_top_k<br/>-> self.top"] --> P2["_get_evidence<br/>-> self.evidence"]

            subgraph ANSWER["_answer -> answer"]
                direction TB
                A0{"use_llm?"}
                A1["join evidence text with [doc_id]"]

                subgraph LLMA["llm_answer"]
                    direction TB
                    L1["build system + user messages"] --> L2["self._llm.invoke"]
                end

                A0 -->|"no"| A1
                A0 -.->|"yes"| L1
            end

            P3["result: answer, verdict 'not checked', provenance, trace"]

            P2 --> A0
            A1 --> P3
            L2 -.-> P3
        end

        F3 --> P1
        P3 --> S1["show: print result"]
    end

    T2 --> Q
    LLMB -.-> L2

    classDef hidden fill:none,stroke:none,color:none,width:0px,height:0px;
```

## How `GraphRAG` works

`GraphRAG` extends `TraditionalRAG`. At initiation it adds BM25 statistics and the graph. In `fit`
it adds an ACL filter and a second ranking: `dense` (TF-IDF cosine) and `lex` (BM25), both over the
allowed documents only. In `predict` it fuses the two rankings, walks the graph and checks citations.
In `GraphRAG` both rankings are keyword-based; only `EmbeddingGraphRAG` makes `dense` semantic.
`EmbeddingGraphRAG` follows exactly this flow, except that `_dense` calls Ollama embeddings
instead of `tfidf_scores`. The dotted path is the optional LLM route, taken only when `use_llm=True`.

```mermaid
flowchart TD
    subgraph INIT["Initiation (once)"]
        direction TB
        A(["GraphRAG(docs, edges, groups, top_k, hops, rrf_k, use_llm, provider)"]) --> I0["__init__<br/>store edges, groups, hops, rrf_k"]
        I0 --> I1["TraditionalRAG.__init__<br/>store docs, top_k, use_llm, provider"]
        I1 --> LAUNCH_IN

        subgraph LAUNCH["__launch__ (GraphRAG's own)"]
            direction TB
            LAUNCH_IN[" "]:::hidden
            LLMB["get_llm(provider)<br/>-> self._llm"]

            subgraph BUILD["_build_index"]
                direction TB
                T1["tokenize each document<br/>-> self.tokens"] --> T2["build_tfidf_idf<br/>-> self.tfidf_idf"]
                T2 --> TV["list of IDF words<br/>-> self.vocab"] --> TD["vec for each document<br/>-> self.doc_vectors"]
                TD --> T3["build_bm25_idf<br/>-> self.bm25_idf"]
            end

            AVG["_build_avg_len<br/>-> self.avg_len"]

            subgraph GRAPH["_build_graph -> self.adj"]
                direction TB
                G1["validate edge endpoints exist"] --> G2["add forward edge and reverse edge ~"]
            end

            LAUNCH_IN -.->|"use_llm only"| LLMB
            LAUNCH_IN --> T1
            T3 --> AVG --> G1
        end
    end

    subgraph QUERY["Query: fit_predict(query)"]
        direction TB
        Q(["query"]) --> F1

        subgraph FIT["fit"]
            direction TB
            F1["store query<br/>-> self.query"] --> F2["tokenize<br/>-> self.q_tokens"] --> F3["start audit trace<br/>-> self.trace"]

            subgraph ACL["_filter_acl -> self.allowed, self.trace"]
                direction TB
                C1["_allowed: user groups intersect document ACL"] --> C2["keep allowed docs, log counts in trace"]
            end

            F4["_dense(self.allowed)<br/>TF-IDF cosine (embeddings in EmbeddingGraphRAG)<br/>-> self.dense_scores"]
            F5["_rank(self.dense_scores)<br/>-> self.dense"]
            F6["_bm25(self.allowed)<br/>-> self.lex_scores"]
            F7["_rank(self.lex_scores)<br/>-> self.lex"]

            F3 --> C1
            C2 --> F4 --> F5 --> F6 --> F7
        end

        subgraph PREDICT["predict"]
            direction TB
            R1["Reciprocal Rank Fusion of self.lex + self.dense<br/>score = sum of 1 / (rrf_k + position + 1)"]
            R2["seeds = top_k fused docs"]

            subgraph WALK["_expand_graph(seeds) -> context, up to hops steps"]
                direction TB
                W1["for each frontier node, follow edges"] --> W2{"neighbor already in context?"}
                W2 -->|"no"| W3{"_allowed(neighbor)?"}
                W2 -->|"yes"| W5["skip"]
                W3 -->|"yes"| W4["add to context, next frontier"]
                W3 -->|"no"| W6["drop, log in trace"]
            end

            R3["evidence = text of every context document<br/>-> self.evidence"]

            subgraph ANSWER["_answer -> answer"]
                direction TB
                A0{"use_llm?"}
                A1["join evidence text with [doc_id]"]

                subgraph LLMA["llm_answer"]
                    direction TB
                    L1["build system + user messages"] --> L2["self._llm.invoke"]
                end

                A0 -->|"no"| A1
                A0 -.->|"yes"| L1
            end

            subgraph CITE["Citation check -> verdict"]
                direction TB
                K1["extract cited [doc_id] from answer"] --> K2{"cited and all in context?"}
                K2 -->|"yes"| K3["grounded"]
                K2 -->|"no"| K4["UNGROUNDED"]
            end

            P1["result: answer, verdict, provenance, trace"]

            R1 --> R2 --> W1
            W4 --> R3
            W5 --> R3
            W6 --> R3
            R3 --> A0
            A1 --> K1
            L2 -.-> K1
            K3 --> P1
            K4 --> P1
        end

        F7 --> R1
        P1 --> S1["show: print result"]
    end

    G2 --> Q
    LLMB -.-> L2

    classDef hidden fill:none,stroke:none,color:none,width:0px,height:0px;
```

### Graph expansion: `_expand_graph(seeds)`

`predict` hands the fused seeds to `_expand_graph`, which walks `self.adj` hop by hop (up to `hops`),
skips documents it has already reached, and drops any neighbor the user's groups may not read
(logging the drop in the trace). It returns a dictionary of document ID to how that document was
reached, in discovery order: seeds map to `"seed"`, and every discovered document maps to
`"hop{n}:{from}-{relationship}->{doc}"`, where `~` marks an edge followed in reverse.
`predict` uses its keys as the evidence and returns it as `provenance`.

Example with group `sre`, `hops=2` and seed `tbl_payments`:

```python
{
    "tbl_payments":  "seed",
    "svc_ledger":    "hop1:tbl_payments-~DEPENDS_ON->svc_ledger",
    "svc_reporting": "hop2:svc_ledger-~DEPENDS_ON->svc_reporting",
}
# trace: "hop1: dropped pm_payments via ~ABOUT (not entitled)"  (pm_payments is finance-only)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use `source .venv/bin/activate` elsewhere)
pip install -e ".[ollama]"      # or ".[anthropic]" / ".[ollama,anthropic,dev]"
```

The LLM provider is set by `PROVIDER` at the top of `main.py` (`"ollama"` or `"anthropic"`).
Copy `.env.example` to `.env` and fill in your settings (`.env` is gitignored):

| Variable             | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| `ANTHROPIC_API_KEY`  | API key, needed only for the `anthropic` provider   |
| `OLLAMA_BASE_URL`    | Ollama server URL (default `http://localhost:11434`) |
| `OLLAMA_EMBED_MODEL` | Embedding model (default `nomic-embed-text`)        |

## Usage

```bash
python main.py                                    # GraphRAG as user group 'sre'
python main.py --model traditional                # TraditionalRAG
python main.py --model ollama                     # host and embed model come from .env
python main.py --model graph --groups sre finance --query "your question"
python main.py -h                                 # all options
```

## Persisted index (`VDB/`)

`GraphRAG` can save its query-independent indexes and load them back without retokenizing
anything. Each part uses a format that suits it, much like a search engine keeps its inverted
index, vectors, metadata and document store apart.

```python
from main import DOCS, EDGES, GraphRAG, VDB_PATH

GraphRAG(DOCS, EDGES, ["sre"], use_llm=False).save_index()               # writes VDB/
rag = GraphRAG(None, None, ["sre"], use_llm=False, index_path=VDB_PATH)  # loads, no rebuild
rag.fit("payments table failure").predict()
```

| File                     | Format  | Contents                                             | Restores                     |
| ------------------------ | ------- | ---------------------------------------------------- | ---------------------------- |
| `bm25_postings.parquet`  | Parquet | `term`, `doc_id`, `tf` (inverted index)              | `bm25_tf`                    |
| `bm25_documents.parquet` | Parquet | `doc_id`, `doc_len`                                  | `bm25_doc_len`               |
| `bm25_stats.parquet`     | Parquet | `term`, `df`, `n_docs`, `avg_doc_len`                | `bm25_idf`, `avg_len`        |
| `tfidf.npz`              | SciPy   | CSR matrix, rows = documents, columns = vocabulary   | `doc_vectors`                |
| `metadata.json`          | JSON    | `doc_ids`, `vocab` (term -> column), `tfidf_idf`, `config` | `vocab`, `tfidf_idf`, ordering |
| `documents.json`         | JSON    | `doc_id` -> `text`, `acl` (list of groups)          | `docs`                       |
| `graph.json`             | JSON    | directed edges as `[source, relationship, target]`   | `edges`, `adj`               |

BM25 IDF is not stored: it is recomputed from `df` and `n_docs`. Reverse graph links are
rebuilt from the directed edges.

Inspect the BM25 files with Polars or DuckDB:

```python
import polars as pl

df = pl.read_parquet("VDB/bm25_postings.parquet")
```

```python
import duckdb

df = duckdb.sql("""
    SELECT *
    FROM 'VDB/bm25_postings.parquet'
""").pl()
```

`python test_index_persistence.py` builds, queries, saves, reloads and checks that scores,
rankings, ACL results and graph evidence are identical.

## Project layout

```
main.py         # corpus, the three RAG classes, and the CLI
test_index_persistence.py  # save_index / load_index round-trip test
VDB/            # index written by GraphRAG.save_index() (Parquet, NPZ, JSON)
pyproject.toml  # packaging, dependencies, ruff config
.env.example    # template for local settings (copy to .env)
.vscode/        # shared editor settings (loads .env into the integrated terminal)
LICENSE         # MIT
```

## Branches

| Branch | Purpose                                  |
| ------ | ---------------------------------------- |
| `main` | Stable baseline (protected)              |
| `Dev`  | Active development                       |
| `UAT`  | User acceptance testing before `main`    |

All three branches are on GitHub. `main` is protected, so work is committed and pushed to `Dev`
and reaches `main` through a pull request.

## Status

Working prototype on a five-document toy corpus. `GraphRAG.fit_predict` runs end to end with
`use_llm=False` and returns a grounded answer for the default query; the LLM path and
`EmbeddingGraphRAG` have not been re-checked since the latest refactors. Recent structural changes:

- Pure helpers (`tokenize`, `build_tfidf_idf`, `vec`, `cosine`, `_rank`, `build_bm25_idf`, `get_llm`) are `@staticmethod`s.
- The LLM is built once in `__launch__` as `self._llm` and used by `llm_answer`.
- Ollama host and embedding model come from `.env`; the `--host` and `--embed-model` flags were removed.
- `OllamaGraphRAG` was renamed `EmbeddingGraphRAG` (still selected with `--model ollama`).
- In `GraphRAG`, both rankings (`self.lex`, `self.dense`) are computed in `fit`; `predict` only fuses, walks the graph and answers.
- `GraphRAG.fit`, `__launch__` and `_build_index` no longer call `super()`; they repeat the inherited steps inline.
- Document TF-IDF vectors are computed once in `_build_index` (`self.vocab`, `self.doc_vectors`); at query time `tfidf_scores` builds only the query vector.
- The graph walk moved out of `GraphRAG.predict` into `_expand_graph(seeds)`; behavior is unchanged.
- Each step in `fit`/`predict` is one line with a named intermediate (e.g. `self.dense_scores`, then `self.dense`), and a single shared `_rank` sorts both rankings.

Known gaps:

- **No tests or CI.** The ACL filter, graph hops, RRF fusion and both IDF functions are the first candidates.
- **Provider selection.** The CLI has no `--provider` flag; the provider comes from `PROVIDER` in `main.py`, and the `use_llm` default differs for `EmbeddingGraphRAG` (off) versus the other two (on).
- **Permission leaks in the audit trace.** It lists IDs of documents the user may not see, and IDF statistics include restricted documents.
- **Weak citation check.** It verifies that cited IDs exist in the context, not that claims are supported, and its regex ignores IDs containing `-` or `.`.
- **`GraphRAG` requires `edges`.** Passing `None` raises an error.
- **Performance.** Document vectors are dense lists over the whole vocabulary, and the graph walk has no context cap. `EmbeddingGraphRAG` still builds TF-IDF document vectors it never uses.
- **No retrieval evaluation** (e.g. recall@k) to compare the three models.

## License

MIT. See [LICENSE](LICENSE).
