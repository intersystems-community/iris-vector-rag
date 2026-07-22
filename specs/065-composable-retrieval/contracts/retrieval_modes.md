# Contract: Retrieval Modes & Fusion

**Feature**: 065-composable-retrieval

## Modes

| Mode | Sources | Fusion | Requires |
|---|---|---|---|
| `vector` | vector | — | IRIS native vector search |
| `text` | BM25 | — | iris-vector-graph BM25 index |
| `hybrid` | vector + BM25 | weighted relative-score (`$scoreFusion`-like) | vector + iris-vector-graph BM25 |
| `rrf` | vector + BM25 | reciprocal rank fusion (`$rankFusion`-like) | vector + iris-vector-graph BM25 |

## Behavioral contract

| ID | Given | When | Then |
|---|---|---|---|
| C-M1 | Vector + BM25 available | `query("q", retrieval="hybrid", weights={"vector":0.7,"text":0.3})` | Weighted relative-score fusion; per-source scores in `metadata` (FR-011). |
| C-M2 | Vector + BM25 available | `query("q", retrieval="rrf")` | Reciprocal rank fusion; weights (if given) scale each source's RRF contribution. |
| C-M3 | `retrieval="text"` | BM25 index present | Returns BM25-ranked results; `text_score` populated. |
| C-M4 | Mode prerequisite absent (e.g. no BM25 index, or graph mode without KG) | `query("q", retrieval=...)` | Raises a clear error naming the missing prerequisite — NO silent fallback (FR-012). |
| C-M5 | `weights` given without a fusion mode | `query("q", retrieval="vector", weights={...})` | Validation error. |
| C-M6 | No `retrieval` arg | `query("q")` | Uses the pipeline's existing default mode (FR-013). |
| C-M7 | `filter` supplied with any mode | `query("q", retrieval="hybrid", metadata_filter={"source":"A"})` | 100% of results match the filter; injection-safe parameterized SQL (FR-001, Principle VIII). |

## Parity contract

Every registered pipeline (`basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`, `multi_query_rrf`) MUST accept every mode argument and either serve it or raise the C-M4 prerequisite error. A pipeline MUST NOT reject a mode argument with `TypeError`/unknown-kwarg.
