# Changelog

## v0.14.0 — delete_node + delete_documents fix

### New: `HybridGraphRAGPipeline.delete_node`

- `delete_node(node_id: str) -> None` — symmetric teardown counterpart to `index_node()`.
  Removes a node from all stores managed by the pipeline:
  - Knowledge graph tables via `iris_engine.delete_node(node_id)`
  - Vector document store via `vector_store.delete_documents([node_id])`
  - BM25: no per-document delete API in iris_vector_graph 2.3.1; KG deletion makes BM25 results inert
- Raises `ValueError` for empty/None `node_id`. Idempotent for nonexistent nodes.
- Partial store failures log `WARNING` and propagate.
- 8 contract tests + live IRIS integration test.

### Fix: `IRISVectorStore.delete_documents` column detection

- Hard-coded `WHERE doc_id` caused `SQLCODE -29` on tables where the id column is named `id`.
- Added `_get_id_column()` to detect the correct column name at runtime via `INFORMATION_SCHEMA`.

### Fix: `auto_setup=True` on a fresh IRIS produced an unusable schema

- `SetupOrchestrator` created `RAG.SourceDocuments` from a legacy inline DDL
  (`id INTEGER IDENTITY`, `filename`, ...) that conflicted with `SchemaManager`'s
  canonical layout (`doc_id VARCHAR PRIMARY KEY`, `embedding VECTOR`). First-run
  ingest then failed with `SQLCODE -104` and vector search with `SQLCODE -254`.
  The orchestrator now delegates the standard RAG tables to `SchemaManager`.
- `DocumentChunks` foreign keys referenced `SourceDocuments(id)`; now `doc_id`.
- Smoke test (`tests/contract/test_smoke.py`) updated to current pipeline API and
  verified against a pristine IRIS 2026.1 container.

### Fix: IRIS connection robustness

- `get_iris_connection()` bounds the native DBAPI handshake with
  `IRIS_CONNECT_TIMEOUT` (default 10s) on a daemon thread and raises
  `ConnectionError` instead of blocking forever; a hung endpoint is remembered for
  `IRIS_CONNECT_BACKOFF` seconds (default 30) so callers fail fast.
- `dbapi.connect` is forced to TCP (`sharedmemory=False`).
- Expired-password recovery now uses `Security.Users.UnExpireUserPasswords("*")`,
  which works on fresh Community containers (2026.1 included).

### Packaging / CI

- `email-validator` added to the `api` and `all` extras (required by
  `iris_vector_rag.api.models.auth`).
- Release workflow runs against a pristine `intersystemsdc/iris-community:2026.1`
  started by `scripts/ci/start-iris.sh`; pre-existing contract failures that need a
  pre-populated corpus are tracked in `tests/contract/ci_known_failures.txt`.

## v0.13.0 — Dual-Level (Global/Mix) Retrieval

### New retrieval modes: `global` and `mix`

- `retrieval="global"` — theme-level retrieval via relation embeddings (LightRAG-inspired):
  extracts `high_level_keywords` via `KeywordExtractor`, embeds them, searches
  `RAG.EntityRelationships.relation_embedding` HNSW index for semantically similar relationships.
  Graceful degradation (FR-009) when index empty: `metadata["degraded"]=True` with reason.
  Hard error (FR-008) via `RetrievalPrerequisiteError` when KG tables absent.

- `retrieval="mix"` — comprehensive RRF-fused retrieval across three sources:
  high-level relation embeddings, low-level entity vector search, naive chunk vector search.
  Default fusion: RRF. Optional: pass `weights={"relation":0.6,"vector":0.4}` for weighted-score.
  Response `metadata["fusion_method"]` is `"rrf"` or `"weighted_score"`.

### New components

- `RelationEmbeddingStore` (`iris_vector_rag/storage/relation_embedding_store.py`):
  manages `relation_embedding VECTOR(FLOAT,384) NULL` column on `RAG.EntityRelationships`.
  Methods: `_ensure_schema()` (idempotent ALTER TABLE + HNSW index),
  `embed_and_store(relationship_id, type, src, tgt, description)` (UPDATE via `TO_VECTOR`),
  `search(query_embedding, top_k)` → `List[Dict]` with float `score`,
  `count_embedded()` → int.

- `KeywordExtractor` (`iris_vector_rag/retrieval/keyword_extractor.py`):
  LLM-backed dual-level keyword extraction. Accepts any `llm_func(prompt) -> str` callable.
  `extract(query)` → `(high_level_keywords, low_level_keywords)` tuple of lists.
  Returns `([], [])` on any error (LLM exception, bad JSON, timeout).
  `parse_keywords(raw)` module-level function strips markdown fences before JSON parse.

### New `QueryOptions` fields

- `high_level_keywords: Optional[List[str]]` — pre-supply to skip LLM keyword extraction
- `low_level_keywords: Optional[List[str]]` — pre-supply to skip LLM keyword extraction

### Response metadata keys added

All `global` and `mix` results include:

- `metadata["high_level_keywords"]` — extracted or pre-supplied high-level themes
- `metadata["low_level_keywords"]` — extracted or pre-supplied entity-level terms
- `metadata["degraded"]` — bool; True when index empty or extraction failed
- `metadata["degradation_reason"]` — string; explains why degraded
- `metadata["retrieval_mode"]` — `"global"` or `"mix"`
- `metadata["extraction_model"]` — model name of configured `KeywordExtractor`
- `metadata["fusion_method"]` — (`mix` only) `"rrf"` or `"weighted_score"`
- `metadata["low_level_count"]`, `metadata["high_level_count"]`, `metadata["naive_count"]` — (`mix` only) per-source doc counts

### Tunability

- `pipeline.keyword_extractor = KeywordExtractor(cheap_llm, model_name="gpt-4o-mini")`
  overrides the default extractor for `global` and `mix` modes on any pipeline.
- Pre-supplying `high_level_keywords=` or `low_level_keywords=` in `pipeline.query(...)`
  skips the LLM extraction call entirely.

### Schema changes

- `RAG.EntityRelationships`: new column `relation_embedding VECTOR(FLOAT,384) NULL`
- `RAG.EntityRelationships`: new HNSW index `idx_hnsw_rel_embedding` (COSINE, M=16, efConstruction=200)
- `db_init_complete.sql` updated to include above (fresh installs only; existing instances use `_ensure_schema()`)

## v0.12.1

### Composable retrieval (065) — restored and wired end-to-end

- `IRISVectorStore.search_by_text(query, top_k, metadata_filter=)` → `List[Document]` — explicit text-search entry point; embeds query and delegates to `similarity_search_by_embedding`
- `IRISVectorStore.search_by_vector(embedding, top_k)` → `List[Tuple[Document, float]]` — explicit vector entry point with scores; legacy `similarity_search` unchanged
- `IRISVectorStore._embed_query(query)` → `List[float]` — shared embed helper used by both wrappers
- All six pipelines (`basic`, `basic_rerank`, `crag`, `graphrag`, `hybrid_graphrag`, `pylate_colbert`) accept composable query-time params: `retrieval=`, `weights=`, `rerank=`, `metadata_filter=`, `similarity_threshold=` via `normalize_query_params()`

### AUD-002 error handling

- `BasicRAGPipeline.query()` response now always includes `"error"` key: `None` on success, `{"type": ..., "message": ..., "error_class": ...}` on retrieval or generation failure
- `answer=None` on failure (was `"Error generating answer"`)
- Generation failures captured as `"GenerationError"` type; retrieval failures as `"RetrievalError"`

### BM25 / GraphRAG

- Added `IVGTextSearchBackend` — pluggable BM25 text search via `IVG Graph.KG.BM25Index`, injectable into `HybridRetrievalMethods.text_engine`
- Fixed `graphrag.py` `_execute_sql` dispatch: checks `self._executor` before falling back to direct connection
- Fixed `_validate_knowledge_graph` raising `KnowledgeGraphNotPopulatedException` instead of returning `False` when count is 0
- `hybrid_graphrag.py` response now includes `"error": None` and `"context_count"` metadata

### Fixes

- `validators.py` canonical query param corrected from `query_text` to `query`; `query_text` marked deprecated
- `load_documents()` param order fixed (`documents_path` first) to match positional call sites
- `sources` removed from `metadata` dict (kept at top level only) per interface conformance
- Fixed black formatting across 40 files; pinned `black==26.5.1` in CI

## v0.12.0

### Composable query-time retrieval (065)

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
- `iris_vector_rag.retrieval.rerank.resolve_reranker(spec, model_name=)` — process-level cache; cross-encoder model loads once per `(strategy, model_name)` tuple per process; thread-safe
- `iris_vector_rag.retrieval.engine.RetrievalEngine` — routes `vector`/`text`/`hybrid`/`rrf` modes; emits `vector_score`/`text_score`/`fusion_score` into `Document.metadata`
- `iris_vector_rag.retrieval.modes.RetrievalMode` — prerequisite checks with `RetrievalPrerequisiteError` when a mode requires iris-vector-graph
- Structured log fields on all pipeline completion logs: `retrieval_mode=`, `weights=`, `rerank_strategy=`, `rerank_degraded=`
- Fixed `metadata_filter` and `similarity_threshold` silently discarded in `BasicRAGPipeline.query()`
- Fixed `RetrievalPrerequisiteError` swallowed by generic `except Exception`
- Fixed `tests/unit/conftest.py` leaking mock `sentence_transformers` into `sys.modules` across test processes

### IRISVectorEngine unified engine object (080)

- Add `IRISVectorEngine` — unified engine object that collapses `(connection_manager, config_manager)` pair into one entry point. `IRISVectorEngine.from_config()` constructs from env/YAML in one call; accepts raw DBAPI connection or `ConnectionManager` as first arg.
- Export `IRISVectorEngine` from top-level `iris_vector_rag`.
- Add `engine=` kwarg to `create_pipeline()` and `create_validated_pipeline()`.
- `RAGPipeline.__init__` accepts `IRISVectorEngine` as first positional arg.
- Fully lazy — no DB connection until `.connection` or `.vector_store` accessed.
- 313 unit tests pass; 8 new E2E tests for engine in `tests/e2e/test_engine_e2e.py`.
- Rewrite AGENTS.md from 4-line stub to full 370-line agent reference.

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
