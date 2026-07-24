# Phase 1 Data Model: Composable Query-Time Retrieval

**Feature**: 065-composable-retrieval | **Date**: 2026-07-22

These are in-process Python data structures (not persisted schema). They define the shape of the composable query layer.

---

## QueryOptions

Normalized representation of the inputs to `pipeline.query()`. Produced by `normalize_query_params(**kwargs)` which resolves aliases and applies defaults.

| Field | Type | Default | Validation |
|---|---|---|---|
| `query` | `str` | — (required) | Non-empty after strip; `query_text` alias accepted (FR-005). If both given, `query` wins and a warning is logged. |
| `top_k` | `int` | `5` | 1 ≤ `top_k` ≤ 100 (existing rule). |
| `generate_answer` | `bool` | `True` | — |
| `include_sources` | `bool` | `True` | Consistent default across pipelines (FR-004). |
| `metadata_filter` | `dict[str, scalar] \| None` | `None` | Keys must pass `MetadataFilterManager` whitelist; values scalar (FR-003). |
| `similarity_threshold` | `float` | `0.0` | 0.0 ≤ x ≤ 1.0; applied post-retrieval (research U2). |
| `retrieval` | `str` | pipeline's current default | One of `vector`, `text`, `hybrid`, `rrf`, or a pipeline-declared extension (FR-010). |
| `weights` | `dict[str, float] \| None` | `None` | Keys ⊆ sources produced by the chosen mode; values ≥ 0 (FR-011). |
| `rerank` | `bool \| str \| Callable \| None` | `None`/`False` | `True`→default; `str`→named strategy; callable→used directly (FR-007). |
| `custom_prompt` | `str \| None` | `None` | Existing param, preserved. |

**Invariants**:
- When `retrieval`, `weights`, and `rerank` are all unset, `QueryOptions` MUST drive the pipeline's pre-existing behavior (FR-013).
- `weights` without a fusion mode (`hybrid`/`rrf`) is a validation error.

---

## RetrievalResult

The internal per-document result carried between retrieval, fusion, rerank, and response assembly.

| Field | Type | Notes |
|---|---|---|
| `document` | `Document` | Existing core model. |
| `vector_score` | `float \| None` | Present when vector mode contributed. |
| `text_score` | `float \| None` | BM25 score when text mode contributed. |
| `fusion_score` | `float \| None` | Weighted-score or RRF score when fused. |
| `rerank_score` | `float \| None` | Cross-encoder score when reranked. |
| `rank` | `int` | Final position after all stages. |

All non-null scores are echoed into `Document.metadata` in the standardized response (FR-008, FR-011), so callers can inspect provenance — mirroring MongoDB `scoreDetails`.

---

## RerankerStrategy

Resolved by `resolve_reranker(spec)`; cached per config (research U4).

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | `"cross-encoder"` (default) or a registered strategy name. |
| `model_name` | `str \| None` | e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `rerank_factor` | `int` | Over-fetch multiplier (existing default 2). |
| `impl` | `Callable[[str, list[Document]], list[tuple[Document, float]]]` | The reranking function. |

**Cache key**: `(name, model_name, config_hash)`. Callables passed directly are not cached.

**Degradation**: on runtime failure, return the pre-rerank ordering and set `metadata["rerank_degraded"] = True` (FR-009).

---

## RetrievalMode

A registry entry declaring a mode and its prerequisites (research U5).

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | `vector` / `text` / `hybrid` / `rrf` / extensions. |
| `sources` | `list[str]` | e.g. `["vector"]`, `["vector", "text"]`. |
| `requires` | `list[str]` | Prerequisites, e.g. `["iris_graph_core_bm25"]`, `["knowledge_graph"]`. |
| `fusion` | `str \| None` | `None`, `weighted_score`, or `rrf`. |

**Prerequisite check**: before executing, the engine verifies each `requires` item; a missing one raises a clear, named error (FR-012) — never a silent fallback.

**Standardized response (unchanged shape, FR-006)**: `{query, answer, retrieved_documents, contexts, metadata, execution_time}` plus `sources` when `include_sources`. `metadata` gains `retrieval_mode`, `weights`, `rerank_strategy`, and any degradation flags.
