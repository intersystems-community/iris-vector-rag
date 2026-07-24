# Changelog

## [Unreleased] — 065-composable-retrieval

### Added

- **Composable query-time options** on all pipelines via unified `query()` signature:
  - `retrieval=` — `"vector"` (default) | `"text"` | `"hybrid"` | `"rrf"` — switch retrieval mode per query without changing pipeline type
  - `weights=` — `{"vector": 0.7, "text": 0.3}` — tune score fusion weights at call time
  - `rerank=` — `True` | `"cross-encoder"` | `callable` — apply reranking on any pipeline
  - `metadata_filter=` — `{"source": "..."}` — actually filter results (fixes silent-drop bug)
  - `similarity_threshold=` — post-retrieval threshold applied correctly
  - `generate_answer=` — skip LLM generation for retrieval-only use
  - `include_sources=` — control source metadata in response

- **`IRISVectorStore.search_by_text(query, top_k)`** → `List[Document]` — explicit text-search entry point
- **`IRISVectorStore.search_by_vector(embedding, top_k)`** → `List[Tuple[Document, float]]` — explicit vector entry point; legacy `similarity_search` unchanged

- **`iris_vector_rag.retrieval.rerank.resolve_reranker(spec, model_name=)`** — public reranker resolver; process-level cache ensures the cross-encoder model loads once per `(strategy, model_name)` tuple per process; thread-safe

- **`iris_vector_rag.retrieval.engine.RetrievalEngine`** — routes `vector`/`text`/`hybrid`/`rrf` modes; emits `vector_score`/`text_score`/`fusion_score` into `Document.metadata`

- **`iris_vector_rag.retrieval.modes.RetrievalMode`** — prerequisite checks with `RetrievalPrerequisiteError` (named, non-silent) when a mode requires iris-vector-graph

- **Structured log fields** on all pipeline completion logs: `retrieval_mode=`, `weights=`, `rerank_strategy=`, `rerank_degraded=`

### Fixed

- `metadata_filter` and `similarity_threshold` were silently discarded in `BasicRAGPipeline.query()` — now forwarded to the vector store and enforced
- `RetrievalPrerequisiteError` was swallowed by generic `except Exception` — now re-raised with full context
- `tests/unit/conftest.py` injected `sentence_transformers` into `sys.modules` without restoring it — now uses `monkeypatch.setitem` so benchmarks run clean after unit tests in the same process

### Changed

- `query_text=` retained as an alias for `query=` (backward-compatible)
- `basic_rerank` pipeline is now equivalent to `basic` + `rerank=True`; both patterns are supported
- All six registered pipelines (`basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`, `multi_query_rrf`) accept the full unified `query()` signature
