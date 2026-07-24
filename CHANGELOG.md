# Changelog

## [0.12.0] — 2026-07-24

### Added

- Composable query-time options on all pipelines via unified `query()` signature:
  - `retrieval=` — `"vector"` (default) | `"text"` | `"hybrid"` | `"rrf"` — switch retrieval mode per query without changing pipeline type
  - `weights=` — `{"vector": 0.7, "text": 0.3}` — tune score fusion weights at call time
  - `rerank=` — `True` | `"cross-encoder"` | `callable` — apply reranking on any pipeline
  - `metadata_filter=` — `{"source": "..."}` — filter results (was silently dropped)
  - `similarity_threshold=` — post-retrieval threshold applied correctly
  - `generate_answer=` — skip LLM generation for retrieval-only use
  - `include_sources=` — control source metadata in response
- `IRISVectorStore.search_by_text(query, top_k)` → `List[Document]` — explicit text-search entry point
- `IRISVectorStore.search_by_vector(embedding, top_k)` → `List[Tuple[Document, float]]` — explicit vector entry point; legacy `similarity_search` unchanged
- `iris_vector_rag.retrieval.rerank.resolve_reranker(spec, model_name=)` — public reranker resolver; process-level cache ensures cross-encoder model loads once per `(strategy, model_name)` tuple per process; thread-safe
- `iris_vector_rag.retrieval.engine.RetrievalEngine` — routes `vector`/`text`/`hybrid`/`rrf` modes; emits `vector_score`/`text_score`/`fusion_score` into `Document.metadata`
- `iris_vector_rag.retrieval.modes.RetrievalMode` — prerequisite checks with `RetrievalPrerequisiteError` (named, non-silent) when a mode requires iris-vector-graph
- Structured log fields on all pipeline completion logs: `retrieval_mode=`, `weights=`, `rerank_strategy=`, `rerank_degraded=`

### Fixed

- `metadata_filter` and `similarity_threshold` were silently discarded in `BasicRAGPipeline.query()` — now forwarded to the vector store and enforced
- `RetrievalPrerequisiteError` was swallowed by generic `except Exception` — now re-raised with full context
- `tests/unit/conftest.py` injected `sentence_transformers` into `sys.modules` without restoring it — now uses `monkeypatch.setitem` so benchmarks run clean after unit tests in the same process

### Changed

- `query_text=` retained as an alias for `query=` (backward-compatible)
- `basic_rerank` pipeline is now equivalent to `basic` + `rerank=True`; both patterns are supported
- All six registered pipelines (`basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`, `multi_query_rrf`) accept the full unified `query()` signature

## v0.11.4

- Consolidated all IRIS connection logic through `get_iris_connection()` — removes duplicate `get_iris_connector_for_embedded()` from `common/utils.py` and the 3-way `hasattr` fan-out in `hybrid_graphrag.py`.
- Routed `colbert_iris/plaid.py` and `vecindex_phase2.py` through `get_iris_connection()`; removed direct `intersystems_iris.createConnection` bypass.
- Fixed top-level `import iris.dbapi` in `common/connection_pool.py` that raised `ImportError` at import time — now lazy-loaded inside `_create_connection()`.
- Added embedded-mode support to `get_iris_connection()`: detects `embedded-kernel` / `embedded-local` runtime via `iris.runtime`, auto-configures embedded-local when `IRISINSTALLDIR` is set, skips TCP probe in both cases.
- Added `module.xml` IPM manifest and pip install hook for OpenExchange listing.
- Bumped Dependabot PRs: `actions/checkout` 6→7, `gitleaks/gitleaks-action` 2→3.

## v0.11.3

- Upgraded to IVG 1.96.2 — fixed gref MagicMock hang, switched to `iris.dbapi.connect()`.
- Fixed IVG 1.88–1.96 compatibility: connection fail-fast, thread handling.

## v0.11.2

- Moved `iris-vector-graph` from extras-only to base dependencies.
- Removed stale `iris_rag/common` imports in CI conftest files.
- Fixed unit tests failing in CI without `OPENAI_API_KEY`.
- Fixed CI failure and resolved 28 npm security vulnerabilities.

## v0.11.1

- Updated dependency bounds from PR triage.
- Updated `iris-vector-graph` requirement to `>=2.1.0`.

## v0.11.0

- Major cleanup for OpenExchange resubmission.
- Breaking: `create_pipeline()` default `validate_requirements=False` (was `True`).
- Removed modules: `memory/`, `optimization/`, `monitoring/`, `security/`, `plugins/`.
- Changed `docker-compose.yml` to standard port 1972 (was 21972).
- Added `iris-vector-graph` to base dependencies.
- 864 tests passing, 0 failures.

## v0.10.2

- Updated `iris-vector-graph` pin to `>=1.80.5`.
- Synced new `.cls` files from IVG 1.80.5: `NKGAccel.cls`, `Snapshot.cls` (new); `Algorithms`, `PageRank`, `Subgraph`, `Traversal`, `VecIndex`, `PLAIDSearch`, `BM25Index`, `IVFIndex`, `EdgeScan` (updated).
- 51/51 tests pass.

## v0.10.1

- Added BM25 + IVFFlat search paths in `RAG.SDK.Search`/`Bridge`.
- Updated `iris-vector-graph` to `>=1.55.3`.

## v0.10.0

- IVG 1.55.3 upgrade: SQL views, LangChain Neo4jGraph compatibility, list comprehensions, ALL/ANY/NONE/SINGLE, reduce(), range(), keys(n).

## v0.9.0

- IVG 1.50.1 upgrade: BM25 lexical search (`BM25Index.cls`), IVFFlat index, `shortestPath`, unified edge store.

## v0.8.0

- ObjectScript SDK (feat 070): `RAG.SDK.Schema`, `Pipeline`, `Search`, `Bridge`, `Evaluate` — 15 ClassMethods, pure ObjectScript + SQL, overlay pattern via `SetDefaultTable`, shared DDL in `sql/schema.sql`.

## v0.7.0

- Fixed version mismatch (`__init__.py` was 0.5.19, `pyproject.toml` was 0.6.0) — both now 0.7.0.
- Added `attach_existing_corpus` (feat 069).
- Added PLAID native engine (IVG 1.27+).
- Added VecIndex two-means split.
- Improved benchmark warmup methodology.

## v0.6.0

- Initial public release on PyPI.
- Six production RAG pipelines: `basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`, `multi_query_rrf`.
- Unified `query()` API, RAGAS evaluation, LangChain compatibility.

## v0.5.4

- Fixed IRIS connection API usage (replace unsupported `iris.connect()` with supported APIs).
- Required `iris-vector-graph` for GraphRAG pipelines; fail fast with clear `ImportError` when missing.
- Auto-initialized GraphRAG schema tables with validation, logging, and per-table error reporting.
