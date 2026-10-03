# Changelog

## Unreleased

### iris-vector-graph 4.1

IVR now requires `iris-vector-graph>=4.1.1,<5`. ivg 4.0 re-keyed `Graph_KG`
storage on `(graph_id, node_id)`, so an install created under 2.x has to be
migrated once. See [docs/IVG_4X_UPGRADE.md](docs/IVG_4X_UPGRADE.md).

- **Old layout is reported, not migrated.** `validate_graph_prerequisites()` and
  `ensure_iris_vector_graph_tables()` list every `Graph_KG` table still in the 2.x
  layout in a new `legacy_layout_tables` field, and validation fails with the call
  that fixes it.
- **`SchemaManager.upgrade_iris_vector_graph_storage(dry_run=False, **options)`**
  runs ivg's `initialize_schema()` and then `upgrade_to_4_0_0`. The dry run writes
  nothing and skips `initialize_schema()`; a step ivg cannot predict on a 2.x install
  comes back with `blocked_because`.
- **Graph-scoped search.** The RRF, text and HNSW retrieval methods accept
  `graph=`, and HNSW accepts `model_key=`. Both are passed to ivg only when given.
- **PLAID fix.** `PLAIDSearcher` read `nCentroids` from `plaid_info()`, which ivg
  returns as `nlist`, so every search raised `PLAIDNotBuiltError`.
- **CI pin.** `pr-integration.yml` installed ivg with an unquoted `>=`, which the
  shell treated as a redirect. The pins are now quoted.
- **`delete_node` reaches named graphs.** With ivg 4.1.1 the engine's `delete_node`
  removes a node's vectors from routed `kg_emb_<hash>` tables too, so
  `HybridGraphRAGPipeline.delete_node` deletes the node from every graph that holds it.
- **No more staging ObjectScript by hand.** ivg 4.1.1 ships its `.cls` files in the
  wheel and `initialize_schema()` deploys them over the connection. See
  [docs/IVG_4X_UPGRADE.md](docs/IVG_4X_UPGRADE.md) and `docs/DEBT.md` entry 10 for a
  container whose server cannot compile them that way.
- `MIN_IVG_VERSION` is `4.0.0`; an unversioned ivg checkout is accepted only if it
  ships `iris_vector_graph.migrations.upgrade_to_4_0_0`.

### Storing documents no longer risks the document table, and every creation path declares `FLOAT`

Two things a user can observe.

- **`IRISStorage.initialize_schema` no longer creates, drops or empties the document
  table.** It asks `SchemaManager.ensure_table_schema` for the table and raises if that
  fails. The probe that reached this code — a `SELECT COUNT(*) ... WHERE 1=0` in
  `store_documents` — cannot tell an absent table from an unreachable one, so the
  `DROP TABLE` and `DELETE FROM` it used to trigger could empty a populated table on
  nothing worse than a closed connection. `initialize_schema` also no longer reassigns
  `self.table_name` on failure, so a storage object cannot silently start writing to a
  different table than the one it was configured with.
- **New tables get `VECTOR(FLOAT, n)` from every creation path.** `RAG.SDK.Schema`,
  `sql/schema.sql` and `iris_vector_rag/common/db_init_complete.sql` all declared
  `DOUBLE` in places; IRIS refuses to compare vectors of different element types
  (SQLCODE -259), so a `DOUBLE` table was unsearchable from a library that converts to
  the canonical `FLOAT`. Tables created earlier are unaffected and stay searchable: the
  SDK's write path resolves the element type from the column it is writing to, and only
  falls back to `FLOAT` when the column cannot be read.

### IRIS session: one object owns the connection (0.15)

Before this release there were eleven ways to get a connection to IRIS and each
carried its own subset of the protections — one had the bounded handshake, the
others had none of it, so a caller that picked the
wrong one could hang on an instance that was still starting. `IRISSession` owns one
instance-and-namespace relationship: the identity, the driver choice, the connection,
its health cadence and the connect back-off. Everything in the library goes through
it, so a protection added once applies everywhere.

```python
from iris_vector_rag.session import session_for

session = session_for(config_manager)          # does not connect
rows = session.execute("SELECT TOP 5 doc_id FROM RAG.SourceDocuments")
```

New in `iris_vector_rag.session`: `session_for`, `IRISSession`, `SessionIdentity`,
`Rows`, `Driver` / `EmbeddedDriver` / `NetworkDriver` / `select_driver`,
`open_sessions` / `close_all_sessions`, and the error taxonomy — `SessionError`,
`IRISUnreachable`, `HandshakeTimeout`, `ConnectFailed`, `InBackoff`,
`NoUsableDriver`, `PasswordExpired`, `ConnectionLost`, `CallerConnectionBroken`.
Every error names the instance and namespace; the password appears in no log, `repr`
or message. `iris_vector_rag.testing.fake_driver.fake_session()` gives a working
session with no driver and no network call.

Tuning is environment-only: `IRIS_HEALTH_INTERVAL` (30s), `IRIS_CONNECT_TIMEOUT`
(10s), `IRIS_CONNECT_BACKOFF` (30s).

See [docs/api/session.md](docs/api/session.md).

#### Behaviour changes

- **Acquiring a session costs nothing.** The first statement opens the connection,
  so constructing an engine no longer reaches the database.
- **One probe per interval.** A connection is trusted for `IRIS_HEALTH_INTERVAL`
  seconds; ordinary operations inside that window do not probe. `is_connected`
  reports recorded state and never probes.
- **A dead connection is replaced transparently**, at most once per operation, when
  the session owns it. A connection supplied by the caller is never closed or
  replaced — the session raises `CallerConnectionBroken` instead. A caller that
  closes a handed-out connection is noticed (`is_closed`, an optional driver
  operation) and given another.
- **A failed or timed-out handshake puts that identity into a back-off window**, and
  further attempts inside it fail immediately with `InBackoff` rather than queueing
  behind a socket timeout.
- **An expired password is reported, not fixed.** A fresh container's "password change
  required" state raises `PasswordExpired` carrying the driver's error and a hint:
  preconfigure the container (`ChangePassword=0`) or create it with iris-devtester. The
  library runs no `docker exec` and no `UnExpireUserPasswords`.
- **Rows outlive their cursor.** `session.execute()` copies values out and releases
  the cursor in a `finally`, on success and on failure, so the `DataRow is
inaccessible` failure that follows `cursor.close()` cannot happen.
- **Connections are per thread**, with a fork guard, and a finished thread's
  connection is closed when the thread exits — a thread-per-request server no longer
  exhausts a Community licence.
- **Driver selection is a decision, not an import side effect.** The session prefers
  the embedded driver when it is usable and reports which one it chose;
  `NoUsableDriver` names every driver tried.
- **`RAGEngine` owns exactly one session** and pipelines sharing an engine share it.
  Its embedding-dimension attribute reports the configured dimension instead of a
  fixed 1536.
- **The HTTP API serves concurrent requests from sessions**; `common/connection_pool.py`
  is a pool over sessions with no direct driver connect of its own.
- **A direct driver connect anywhere else in the library fails the quality gates**
  (`unguarded-dbapi-connect` in `scripts/gates/antipatterns.py`: only the drivers
  module may call a driver's `connect`).

#### Deprecated

Each of these keeps its 0.14 signature and behaviour, warns once naming its
replacement, and **is removed in 0.16**:

- `common.iris_connection.get_iris_connection()` →
  `session_for().connection_for_adapter()`
- `common.iris_dbapi_connector.get_iris_dbapi_connection()` → same
- `common.iris_connection_manager.IRISConnectionManager` (and its module function) →
  `session_for()`
- `common.connection_manager.ConnectionManager` → `session_for()`

#### Removed

Gone in 0.15, not deprecated: nothing imported them.

- `common/iris_connector.py` (whole module) and its re-exports.
- `IRISConnectionPool` and `_PooledConnection` from `common/iris_connection.py`, and
  `get_pool()` / `initialize_pool()` / `close_pool()` — hold a session instead.
- `get_iris_connection_old` from `common/iris_connection_manager.py`.

#### Renamed

Two library classes shared the name `ConnectionManager`:

- `common.connection_manager.ConnectionManager` → `LegacyConnectionManager`
- `api.websocket.connection.ConnectionManager` → `WebSocketConnectionManager`

### Table shape: the database decides which column fills which role (0.15)

`TableShape` reads a document or chunk table out of the IRIS catalog once and
answers the only question the pipelines ever asked of it: which column is the
identifier, the text, the metadata, the embedding. Every column name in the read,
write, delete and join paths now comes from that one answer, so the writer and the
reader can no longer disagree about the same table.

New in `iris_vector_rag.storage`:

- `table_shape.TableShape` — `column_for(role)`, `optional_column_for(role)`,
  `has(role)`, `legacy_column_for(role)`, `vector_spec()`, `writable_identifier()`,
  plus `qualified_name`, `columns`, `kind`, `is_canonical`.
- `catalog.IRISCatalogReader` / `catalog.RecordedCatalog` — the only place in the
  package that queries `INFORMATION_SCHEMA` or `%Dictionary.*`.
- `shape_resolver.TableShapeResolver` — one catalog read per table per instance.

`IRISVectorStore`, `EntityStorageAdapter`, `SetupOrchestrator` and
`HybridGraphRAGPipeline` all take an optional `shapes=` resolver, so a caller (or a
test) can describe a table without touching a database.

See [docs/api/table_shape.md](docs/api/table_shape.md).

#### Behaviour changes

- **Mixed tables.** A table carrying both a canonical and a legacy column (for
  example `text_content` _and_ `content`) now resolves to the canonical one
  everywhere. Previously the write path chose one column and the read path chose
  another, so rows written through the library were invisible to search on such a
  table. Reads still fall back per row to the legacy column when the canonical one
  is empty, so existing data stays readable.
- **`IRIS_VECTOR_DATA_TYPE` no longer decides the vector element type.** `TO_VECTOR`
  takes its element type and length from the embedding column itself, and the
  canonical `CREATE TABLE` takes them from the shape and the configured embedding
  dimension. `is_canonical` is true only of a `VECTOR(FLOAT, n)` embedding, so a
  creation path reading the environment could emit a table the library then refused
  to call canonical.
- **Table names are matched without regard to case, in one place.** The catalog reader
  resolves a table's identity against `INFORMATION_SCHEMA.TABLES` with both sides
  upper-cased, then describes it using the spelling the catalog reported. Feature 074
  had given `table_exists` that tolerance; routing the check through the shape catalog
  briefly dropped it, so a configuration naming `rag.sourcedocuments` read as "no such
  table". A table that does not exist now costs one catalog query instead of two.
- **`SchemaManager.verify_table_structure` describes a table in this manager's own
  schema.** Its `schema_name` default was the literal `RAG`, so every deployment that
  configured a different schema prefix was told the table had no columns. An explicit
  `schema_name` still wins, and `{}` still means "cannot tell" rather than "no
  columns".
- **A table with no embedding column raises.** Vector search on such a table used
  to fall back silently to `FLOAT` and return nothing; it now raises
  `NotVectorSearchable` naming the table.
- **A query embedding of the wrong length raises** before any SQL is sent, naming
  both the length given and the length the column declares.
- **A caller-supplied identifier on a generated key raises**
  `IdentifierNotWritable` naming the schema, table and column, instead of letting
  IRIS reject the insert.
- **The row-count diagnostic (LOG-004) is read only when a search finds nothing.**
  The two `COUNT(*)` statements exist to tell an empty table apart from a table whose
  embeddings were never written, so a search that returned rows has nothing for them
  to explain. A search is two statements again: rank the vectors, fetch their
  metadata.
- **`attach_existing_corpus`** (`HybridGraphRAGPipeline`) no longer requires
  `id_col`, `text_col` and `embedding_col`: omitted arguments come from the
  attached table's shape, and `graph_label` defaults to the unqualified table name.
  Explicit arguments still win.

#### Deprecated

- Column-name overrides in the `configuration` JSON of `RAG.SchemaMetadata`
  (`id_column`, `text_column`, `title_column`, `metadata_column`,
  `embedding_column`) are ignored and now raise a `DeprecationWarning` naming the
  keys found. The table's own columns are the single source of that fact. Remove
  the keys from the stored configuration.

#### Removed

- `iris_vector_rag/pipelines/graphrag_merged.py` (1163 lines). Nothing imported it,
  nothing exported it, and it could not be imported: it read from a
  `iris_vector_rag.visualization` package that is not part of the library. Every name
  it defined — `GraphRAGPipeline`, `GraphRAGException`,
  `KnowledgeGraphNotPopulatedException`, `EntityExtractionFailedException` — is also
  defined in `pipelines/graphrag.py`, which is what `pipelines/__init__.py` has always
  exported. No call can have been affected.

#### Test infrastructure: the order dependence is fixed

`tests/utils/schema_models.py` carried a second, hand-written description of
`RAG.SourceDocuments` (`id`, `source`, `content`, no embedding column at all).
`SchemaValidator` therefore rejected a healthy canonical table, the
`database_with_clean_schema` fixture called `SchemaResetter`, and the resetter
replaced the table with one that cannot answer a vector search — for the rest of
the session. Validation could not pass even against the resetter's own output,
because it demanded `JSON` where `VARCHAR(MAX)` reads back as `LONGVARCHAR`, so
the reset fired for every test class that asked for a clean schema.

- The document and chunk tables are now derived from `CANONICAL_DOCUMENT_SHAPE` /
  `CANONICAL_CHUNK_SHAPE` and created with the library's own
  `create_table_statement`, so the test infrastructure cannot disagree with the
  code under test.
- `SchemaValidator.satisfies(expected, actual)` accepts the types IRIS actually
  reports: `varchar`/`varbinary` for a `VECTOR` column, `longvarchar` for `JSON`.
- `test_graphrag_handler_execute_hybrid_search` passes in the full CI-equivalent
  contract run and has been removed from
  `tests/contract/ci_known_failures.txt` (the 0.14.2 note below is what this
  supersedes).

The `LOG-004` row-count diagnostics (`Total documents in <table>`, `Documents with
embeddings`) are kept, now counting through the embedding column the shape names
rather than a hard-coded `embedding`.

Two shapes were added to the recorded fixtures (`tests/fixtures/shapes/`), captured
from real DDL like the six before them: `no_metadata` and `no_title`. All six of the
original shapes carry a `metadata` column and all six carry a column that fills the
TITLE role, so the two absent-optional-column paths the code has could only be
exercised by deleting a column from a recorded snapshot — a test asserting against
its own idea of a table. One consequence is now pinned rather than assumed: a
metadata filter against a table that has no metadata column matches nothing and says
nothing, because the documents were stored without their metadata.

### Dependencies / CI (Dependabot PRs #81–#91, #95–#98, #100, #101)

- GitHub Actions: `actions/checkout` v7 everywhere, `codecov/codecov-action` v7
  (`file` input renamed `files`), `gitleaks/gitleaks-action` v3,
  `softprops/action-gh-release` v3, `docker/setup-buildx-action` v4.
- `evaluation_framework/requirements.txt` floors: datasets 5.0.1, markdown 3.10.3,
  nltk 3.10.3, spacy 3.8.16, scikit-learn 1.9.0, jinja2 3.1.6, pingouin 0.6.1.
- dev group: gitpython 3.1.61.
- `transformers` upper bound widened to <5.17 (PR #102) and `sentence-transformers`
  to <6.1 (PR #99) after a compatibility check under transformers 5.16.1 /
  sentence-transformers 6.0.1: all-MiniLM-L6-v2 embeddings identical (384-d, same
  cosines), ms-marco cross-encoder scores correct, unit suite green.
- Tooling: black pinned to 26.5.1 in pre-commit to match the CI Format Check
  (three different black versions had been in play).
- Public default branch `main` fast-forwarded to `master` (it had been 50 commits
  behind since 2026-07-25); Dependabot, CI and OpenExchange all target `main`.

## v0.14.2 — iris-vector-graph 2.18 pin

### Dependencies: iris-vector-graph >= 2.18.2, unified `iris` module pinned

- `iris-vector-graph` is now a declared base dependency at `>=2.18.2` (was only in
  extras at `>=1.6.0`; CI installed `>=2.1.0`). 2.18.2 fixes `kg_TXT` failing with
  `SQLCODE -51` on every call (the text leg of hybrid search was silently
  vector-only) and makes a failing text leg raise instead of degrading. Lockfile
  moved from 2.3.1 to 2.18.5.
- `iris-embedded-python-wrapper>=0.5.20` declared explicitly: both it and the bare
  driver ship a top-level `iris` package, so install order decided which one won.
- Vendored ObjectScript snapshots under `iris_vector_rag/pipelines/colbert_iris/sp/`
  re-synced to IVG 2.18.5 (`ColBERTSearch.cls` and `UserExec.cls` are IVR-owned and
  unchanged). Only `ColBERTSearch.cls` is deployed by the scripts; the rest are
  reference copies of classes IVG installs itself.
- `test_graphrag_handler_execute_hybrid_search` passes again on a pristine database
  with the kg_TXT fix, but inside the full suite it inherits a legacy-shaped
  `RAG.SourceDocuments` from earlier tests (no `embedding` column), so it stays in
  `tests/contract/ci_known_failures.txt` until that order dependence is fixed.
- CI: `tests/contract/test_ragas_validation_contract.py` is ignored in the release
  gate like the CRAG suites. It runs real `ragas.evaluate` (needs an LLM) and only
  ever skipped before because earlier tests failed to load documents under IVG 2.3.1.
- `HybridGraphRAGPipeline.delete_node`: BM25 note corrected. IVG 2.18 has
  `bm25_delete(name, doc_id)`; the pipeline owns no BM25 index, so nothing changes.

## v0.14.1 — metadata filter pushdown

### Fix: metadata `filter` is applied in SQL, not as a post-ranking Python pass

- `IRISVectorStore.similarity_search_by_embedding(..., filter=...)` built a LIKE
  predicate for the filter and then discarded it, ranking the whole corpus,
  fetching `top_k * 5` and filtering in Python. On a shared table a small
  tenant's documents fell outside the global page and were silently lost
  (reported by opsreview: `docs/upstream/ivr-metadata-filter-pushdown.md`).
- The predicate now narrows the query (`WHERE ... LIKE '%"key":<json literal>%'`,
  with and without the space `json.dumps` emits), the page is exactly `top_k`,
  and the exact-match check remains only as a backstop for LIKE approximations,
  fetching one wider page if it actually drops rows.
- Stream (`LONGVARCHAR`) metadata columns are wrapped in `SUBSTRING`: IRIS
  `LIKE` on a stream column silently matches nothing.
- Non-string values match their JSON literal (`"page_number": 3`, `"flag": true`).
- `JSON_VALUE` was evaluated and rejected: a single non-JSON row anywhere in the
  table fails the whole query (SQLCODE -400) and it cannot run on stream columns.
- Tests: `tests/unit/test_metadata_filter_pushdown.py` (SQL shape, page size,
  backstop, stream column) and `tests/e2e/test_metadata_filter_pushdown_e2e.py`
  (60 noisy-tenant documents; the small tenant's two documents are returned).

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
