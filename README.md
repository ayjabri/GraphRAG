# GraphRAG

A small, self-contained comparison of **Traditional RAG** and **Graph RAG** on a toy
corpus of services and tables, implemented as three classes in [main.py](main.py):

| Class               | What it does                                                               |
| ------------------- | -------------------------------------------------------------------------- |
| `TraditionalRAG`    | TF-IDF top-k over everything. No permissions, no graph, no citation check. |
| `GraphRAG`          | ACL filter, BM25 + TF-IDF fused with RRF, graph walk, citation check.      |
| `EmbeddingGraphRAG` | `GraphRAG`, but the dense step uses an Ollama embedding model.             |

## How `TraditionalRAG` works

From construction to answer. The dotted path is the optional LLM route, taken only when `use_llm=True`;
otherwise the retrieved evidence is returned as the answer.

```mermaid
flowchart TD
    subgraph INIT["Initiation (once)"]
        A(["TraditionalRAG(docs, top_k, use_llm, provider)"]) --> LAUNCH_IN

        subgraph LAUNCH["__launch__"]
            direction TB
            LAUNCH_IN[" "]:::hidden
            LLMB["get_llm(provider)<br/>stored as self._llm"]

            subgraph BUILD["_build_index"]
                direction TB
                T1["tokenize each document"] --> T2["build_tfidf_idf<br/>IDF weight per word"]
            end

            LAUNCH_IN -.->|"use_llm only"| LLMB
            LAUNCH_IN --> T1
        end
    end

    subgraph QUERY["Query: fit_predict(query)"]
        direction TB
        G(["query"]) --> F1

        subgraph FIT["fit"]
            direction TB
            F1["store query"] --> F2["tokenize query"]

            subgraph RD["dense ranking: _rank(_dense(all docs))"]
                direction TB
                subgraph DENSE["_dense"]
                    direction TB
                    subgraph TS["tfidf_scores"]
                        direction TB
                        D1["vec: query vector"] --> D2["vec: document vector"] --> D3["cosine: query vs document"]
                    end
                end
                D3 --> R1["_rank<br/>sort by score, drop scores <= 0"]
            end

            F2 --> D1
        end

        R1 --> R2

        subgraph PREDICT["predict"]
            direction TB
            R2["_select_top_k<br/>keep top_k document IDs"] --> R3["_get_evidence<br/>pair each ID with its text"]
            R3 --> A0

            subgraph ANSWER["_answer"]
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

            A1 --> P1["answer, verdict, provenance, trace"]
            L2 -.-> P1
        end

        P1 --> S1["show: print result"]
    end

    T2 --> G
    LLMB -.-> L2

    classDef hidden fill:none,stroke:none,color:none,width:0px,height:0px;
```

## How `GraphRAG` works

`GraphRAG` extends `TraditionalRAG`: it adds BM25 statistics and the graph at initiation, then
an ACL filter, BM25 ranking (`lex`) and similarity ranking (`dense`) in `fit`, then rank fusion,
a graph walk and a citation check in `predict`. In `GraphRAG` both rankings are keyword-based
(BM25 versus TF-IDF cosine); only `EmbeddingGraphRAG` makes `dense` semantic.
The dotted path is the optional LLM route, taken only when `use_llm=True`.
`EmbeddingGraphRAG` is identical except that `_dense` uses Ollama embeddings instead of TF-IDF.

```mermaid
flowchart TD
    subgraph INIT["Initiation (once)"]
        A(["GraphRAG(docs, edges, groups, top_k, hops, rrf_k, use_llm, provider)"]) --> I0["store edges, groups, hops, rrf_k"]
        I0 --> LAUNCH_IN

        subgraph LAUNCH["__launch__"]
            direction TB
            LAUNCH_IN[" "]:::hidden
            LLMB["get_llm(provider)<br/>stored as self._llm"]

            subgraph BUILD["_build_index"]
                direction TB
                T1["tokenize each document"] --> T2["build_tfidf_idf<br/>IDF weight per word"] --> T3["build_bm25_idf<br/>BM25 IDF weight per word"]
            end

            AVG["_build_avg_len<br/>average document length"]

            subgraph GRAPH["_build_graph"]
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

            F1["store query"] --> F2["tokenize query"]
            F3["start audit trace"]

            subgraph ACL["_filter_acl"]
                direction TB
                C1["_allowed: user groups intersect document ACL"] --> C2["keep allowed docs, log counts in trace"]
            end

            subgraph BM["lexical ranking: _rank(_bm25(allowed))"]
                direction TB
                B1["_bm25: score allowed docs"] --> B2["_rank: sort, drop scores <= 0"]
            end

            subgraph DR["dense ranking: _rank(_dense(allowed))"]
                direction TB
                D1["_dense on allowed docs<br/>TF-IDF cosine (embeddings in EmbeddingGraphRAG)"] --> D2["_rank: sort, drop scores <= 0"]
            end

            F2 --> F3 --> C1
            C2 --> B1
            B2 --> D1
        end

        subgraph PREDICT["predict"]
            direction TB

            R2["Reciprocal Rank Fusion of self.lex + self.dense<br/>score = sum of 1 / rrf_k + position + 1"]

            R3["seeds = top_k fused docs"]

            subgraph WALK["Graph walk, up to hops steps"]
                direction TB
                W1["for each frontier node, follow edges"] --> W2{"neighbor already in context?"}
                W2 -->|"no"| W3{"_allowed(neighbor)?"}
                W2 -->|"yes"| W5["skip"]
                W3 -->|"yes"| W4["add to context, next frontier"]
                W3 -->|"no"| W6["drop, log in trace"]
            end

            R4["evidence = all context documents"]

            subgraph ANSWER["_answer"]
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

            subgraph CITE["Citation check"]
                direction TB
                K1["extract cited [doc_id] from answer"] --> K2{"cited and all in context?"}
                K2 -->|"yes"| K3["verdict: grounded"]
                K2 -->|"no"| K4["verdict: UNGROUNDED"]
            end

            R2 --> R3 --> W1
            W4 --> R4
            W5 --> R4
            W6 --> R4
            R4 --> A0
            A1 --> K1
            L2 -.-> K1
            K3 --> P1["result: answer, verdict, provenance, trace"]
            K4 --> P1
        end

        D2 --> R2
        P1 --> S1["show: print result"]
    end

    G2 --> Q
    LLMB -.-> L2

    classDef hidden fill:none,stroke:none,color:none,width:0px,height:0px;
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

## Project layout

```
main.py         # corpus, the three RAG classes, and the CLI
pyproject.toml  # packaging, dependencies, ruff config
.env.example    # template for local settings (copy to .env)
.vscode/        # shared editor settings (loads .env into the integrated terminal)
LICENSE         # MIT
```

## Branches

| Branch | Purpose                                  |
| ------ | ---------------------------------------- |
| `main` | Stable baseline                          |
| `Dev`  | Active development                       |
| `UAT`  | User acceptance testing before `main`    |

`Dev` and `UAT` currently exist locally only; push them with `git push -u origin Dev UAT`.

## Status

Working prototype on a five-document toy corpus. Not yet verified end to end in a
clean environment after the latest refactors. Recent structural changes:

- Pure helpers (`tokenize`, `build_tfidf_idf`, `vec`, `cosine`, `_rank`, `build_bm25_idf`, `get_llm`) are `@staticmethod`s.
- The LLM is built once in `__launch__` as `self._llm` and used by `llm_answer`.
- Ollama host and embedding model come from `.env`; the `--host` and `--embed-model` flags were removed.
- `OllamaGraphRAG` was renamed `EmbeddingGraphRAG` (still selected with `--model ollama`).
- In `GraphRAG`, both rankings (`self.lex`, `self.dense`) are computed in `fit`; `predict` only fuses, walks the graph and answers.
- `GraphRAG.fit` and `__launch__` no longer call `super()`; they repeat the inherited steps inline.

Known gaps:

- **No tests or CI.** The ACL filter, graph hops, RRF fusion and both IDF functions are the first candidates.
- **Provider selection.** The CLI has no `--provider` flag; the provider comes from `PROVIDER` in `main.py`, and the `use_llm` default differs for `EmbeddingGraphRAG` (off) versus the other two (on).
- **Permission leaks in the audit trace.** It lists IDs of documents the user may not see, and IDF statistics include restricted documents.
- **Weak citation check.** It verifies that cited IDs exist in the context, not that claims are supported, and its regex ignores IDs containing `-` or `.`.
- **`GraphRAG` requires `edges`.** Passing `None` raises an error.
- **Performance.** TF-IDF vectors are rebuilt per query and the graph walk has no context cap.
- **No retrieval evaluation** (e.g. recall@k) to compare the three models.

## License

MIT. See [LICENSE](LICENSE).
