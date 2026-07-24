# Contract: Query-Time Reranking

**Feature**: 065-composable-retrieval

## `rerank` argument

| Value | Meaning |
|---|---|
| `None` / `False` | No reranking (default) — behavior identical to pre-feature. |
| `True` | Default cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). |
| `str` | Named registered strategy. |
| `Callable[[str, list[Document]], list[tuple[Document, float]]]` | Caller-supplied reranker, used directly. |

## Behavioral contract

| ID | Given | When | Then |
|---|---|---|---|
| C-R1 | `basic` pipeline | `query("q", rerank=True)` | Results reordered by cross-encoder; `rerank_score` in `metadata` (FR-008). |
| C-R2 | Any registered pipeline | `query("q", rerank=True)` | Reranking applies (universal post-retrieval step) (FR-007 parity). |
| C-R3 | `rerank=<callable>` | `query("q", rerank=fn)` | `fn` is invoked with `(query, docs)` and its order is honored. |
| C-R4 | `rerank=False`/omitted | `query("q")` | Identical to non-reranked path (FR-013). |
| C-R5 | Reranker raises at runtime | `query("q", rerank=True)` | Falls back to pre-rerank order; `metadata["rerank_degraded"]=True`; no exception surfaced (FR-009). |
| C-R6 | Rerank + fusion | `query("q", retrieval="rrf", rerank=True)` | Rerank runs AFTER fusion (ordering per spec). |
| C-R7 | Repeated reranked queries in one process | N calls | Cross-encoder model loaded at most once per config (FR-015, Principle VI); assert via load counter/benchmark. |

## Backward-compatibility

`create_pipeline("basic_rerank")` continues to work and is equivalent to `basic` + `rerank=True` with the default cross-encoder.
