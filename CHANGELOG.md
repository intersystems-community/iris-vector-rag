# Changelog

## v0.15.0 — correctness release (2026-10-10)

### Fixed: embedding width probe on sentence-transformers 6

sentence-transformers 6 renamed `get_sentence_embedding_dimension()` to
`get_embedding_dimension()` and warns on the old name. The package allows 6.x, and under the
test suite's `filterwarnings = error` the warning made the dimension probe fail, which
surfaced as "Cannot determine the embedding dimension". `EmbeddingManager` now reads the
width through `embeddings.dimension.embedding_dimension_of`, which prefers the new name and
falls back to the old one (5.x), and `TestEmbedder` offers both.

### Changed: iris-vector-graph 4.2.0 (BM25 schema 2, no default superuser login)

IVR now requires `iris-vector-graph>=4.2.0,<5` (4.2.0 is on PyPI). See
[docs/IVG_4X_UPGRADE.md](docs/IVG_4X_UPGRADE.md), "From 4.1.1 to 4.2.0". There is no
storage migration between 4.1.1 and 4.2.0.

- **BM25 indexes must be upgraded once before they are written to.** ivg 4.2.0 refuses
  `Insert` and `Delete` on an index built earlier until `engine.bm25_upgrade(name)` has run
  (`Search` is unaffected, but scores change). New: `SchemaManager.upgrade_bm25_indexes()`
  (arguments `names`, `dry_run`, `continue_on_error`) converts only the indexes still on
  schema 1, reports `upgraded` / `pending` / `current` / `missing` / `failed`, and can be repeated;
  `SchemaManager.upgrade_iris_vector_graph()` runs the layout step and then the BM25 step.
  The first failure raises `Bm25UpgradeError` unless `continue_on_error=True`. IVR owns no
  BM25 index, so nothing is upgraded implicitly.
- **The engine's class-method connection is IVR's account, not `_SYSTEM`.** ivg 4.2.0 stopped
  logging in as `_SYSTEM`/`SYS`; an engine built from a bare connection would have run its
  BM25, vector, PLAID and ledger calls on the session's connection. IVR records the session's
  credentials on each engine it builds (`integrations.ivg.remember_credentials`;
  `create_graph_engine(..., credentials=)`). A restricted account's role now decides what
  those calls may do: without `%Native_ClassExecution` they are refused, where 4.1.1 ran
  them as `_SYSTEM`. A superuser account sees no change.
- **`retrieval.bm25.max_df_fraction`** (default `0`, off) is passed to `bm25_search` by
  `IVGTextSearchBackend` (argument `max_df_fraction`, or `IVGTextSearchBackend.from_config`).
  `IVGTextSearchBackend` also logs one warning when it searches a schema-1 index.
- No IVR code changed for the Arno reader routing, the algorithm result cache or the
  ObjectScript fixes: IVR calls none of the affected algorithms and had no workaround for the
  fixed defects. `docs/DEBT.md` entry 10 records what was re-checked about the embedded-Python
  probe.

### Added: note search, first slice (retrieval mode `phrase`, sections, polarity, roll-up)

For long documents such as clinical notes (spec 088, `specs/088-note-search-first-slice/`).

- **`iris_vector_rag.note_search`**: `split_sections(text, detector, label_map=None)` cuts a
  document into labelled sections with a heading detector you supply (a default for
  `UPPER CASE:` lines is provided and never applied implicitly). `SectionStore.store()`
  (over an `IRISVectorStore` on a chunk-shaped table) writes one row per section
  (`doc_id = <parent>#<n>`, `document_id` = the parent, the label in `metadata["section"]`),
  replacing the parent's earlier sections in one transaction.
- **`retrieval="phrase"`** (`RetrievalEngine`, `IRISVectorStore.phrase_search`): an exact
  phrase (words adjacent, in order, no stemming, stop words kept) over the table's iFind
  index, optionally limited with `QueryOptions(section="allergies")`, applied in SQL before
  ranking. Each candidate is re-scanned and dropped if the phrase is not really there:
  IRIS 2026.1 returns reverse-order false positives for two-word phrases at the start of a
  text.
- **Polarity.** Every hit says `affirmed`, `negated` or `unknown` (`metadata["polarity"]`),
  from a pluggable `PolarityDetector`; the default is a small NegEx-style reference
  detector that is **not validated on clinical text**. Negated hits are never dropped.
- **Roll-up.** Section hits come back as one hit per parent document, scored by its best
  section, with `metadata["matched_sections"]`.
- **`AllowlistHeadingDetector(labels, accept=, colon=, standalone=)`** finds only headings
  named in a list you supply, in the casings you accept, with the colon optional, required
  or forbidden. Use it where a casing pattern alone over-catches.
- No schema change: the section label lives in the existing `metadata` column.
- Query semantics and shared test vectors for other implementations (fhir-017, ADR 0003):
  `specs/088-note-search-first-slice/contracts/query-semantics.md` and
  `tests/fixtures/note_search/vectors.json`.

### Fixed: the text leg of `text`, `hybrid` and `rrf` retrieval searches the real document table (existing tables gain an index)

On a real deployment the text leg found nothing, so `hybrid` and `rrf` returned the
vector leg alone. It read an iris-vector-graph `docs(id, text)` table that nothing
created; its `%FIND(text, ?)` is not IRIS SQL (SQLCODE -359) and its `LIKE` fallback
addressed columns `RAG.SourceDocuments` does not have (and `LIKE` on the long text
column matches nothing anyway).

- **`SchemaManager` now keeps a full-text index, `SourceDocumentsTextIdx`, on the document
  table's text column** (`CREATE INDEX ... AS %iFind.Index.Basic(LANGUAGE='en', LOWER=1)`).
  The column is the one the catalog says fills the text role (`text_content`, `content`,
  ...). `ensure_table_schema("SourceDocuments")` creates it with a new table and adds it
  to an existing one; `migrate_table("SourceDocuments")` recreates it.
- **DDL change on existing tables.** The first `ensure_table_schema("SourceDocuments")`
  after upgrading builds the index over the rows already there. No row is read out,
  dropped or rewritten, but the build is synchronous and its time grows with the table.
  Call `SchemaManager.ensure_text_index()` yourself to choose when (idempotent;
  `has_text_index()` reports whether it exists). It raises `SchemaIndexError` if the
  database refuses the index; `ensure_table_schema` propagates that error rather than
  returning a table that would silently search nothing.
- **`IRISVectorStore.text_search()`** runs the search with `%ID %FIND search_index(...)`,
  ranks by the index's TF-IDF, honours `metadata_filter` in SQL like the vector leg, and
  returns complete documents with `metadata["text_score"]`. A question is searched as an
  `OR` of its content words.
- **A table without the index raises `TextIndexMissingError`** (a `SchemaIndexError`)
  naming the index and `ensure_text_index()`, instead of returning nothing;
  `BasicRAGPipeline.query` re-raises it.
- iris-vector-graph's `TextSearchEngine` is no longer used by `RetrievalEngine`, and the
  retrieval modes now `require` `text_index` instead of `iris_vector_graph_bm25`.

### Fixed: the MCP server holds one IRIS connection, not one per query in flight

Every call that touches IRIS (queries, the reachability check before each query, pipeline
construction in the health check) now runs on one worker thread. Before, each call ran on
whichever executor thread was free and each thread kept its own connection, so five
concurrent queries held five IRIS licences (a Community IRIS has eight) even though the
handlers run one query at a time. A refused connect then put the session into its 30 s
back-off and every tool call in the window failed with "IRIS is not reachable".

### Added: `IRIS_LOCK_TIMEOUT` bounds lock waits on network connections

When `IRIS_LOCK_TIMEOUT=<seconds>` is set, every network connection the session opens
runs `SET OPTION LOCK_TIMEOUT=<seconds>`, so a statement that waits on another
connection's lock fails with SQLCODE -110 after that long. Unset, nothing changes.
ColBERT ingest now rolls back before it restores autocommit when a failure that is
not an `Exception` (an interrupt, a timeout) escapes it.

### Changed: entity and relationship types keep their case (existing tables are altered)

- **`entity_type` and `relationship_type` are `COLLATE EXACT`.** IRIS gives a string
  column the case-folding `SQLUPPER` collation by default, so
  `SELECT entity_type, COUNT(*) FROM RAG.Entities GROUP BY entity_type` returned
  `DISEASE` for a stored `Disease`, and `= 'disease'` matched it. The canonical DDL
  (`SchemaManager`, `sql/schema.sql`, `db_init_complete.sql`, the SDK `Schema.cls`)
  now declares the two type columns EXACT; no other column changes.
- **An existing table is altered in place.** When `ensure_table_schema` finds
  `Entities` or `EntityRelationships` with the default collation it reads the collation
  from `%Dictionary.CompiledProperty` and runs
  `ALTER TABLE ... ALTER COLUMN c VARCHAR(n) COLLATE EXACT`. Rows, nullability and
  indexes are kept (IRIS rebuilds the column's indexes). The ALTER runs on its own
  connection with a lock timeout; if another connection holds a lock on the table it
  gives up, logs why, changes nothing and retries on the next ensure. Queries that group
  or compare these columns are now case-sensitive.

### Changed: LLM calls are bounded by a timeout and a retry count

- **`get_llm_func` gives the OpenAI client `timeout=60` (seconds per attempt) and
  `max_retries=2`.** Before, the client used the SDK defaults (ten minutes, two
  retries), so a stalled network hung a caller for half an hour. Set `llm.timeout` and
  `llm.max_retries` in the configuration, the environment variables `RAG_LLM__TIMEOUT`
  and `RAG_LLM__MAX_RETRIES`, or pass `timeout=` / `max_retries=` to `get_llm_func`.
  A call that times out raises the new `LLMTimeoutError` (a `GenerationError`) naming
  the provider, the model and the limit. The worst case is
  `timeout * (max_retries + 1)` seconds. Other providers are not limited by this.
- **`load_documents` accepts dict documents** (`content`/`page_content`/`text`,
  `id`/`doc_id`) on the basic, rerank, CRAG, PyLate and GraphRAG pipelines, as its
  documentation always said.
- **Retrieved documents carry `score` and `similarity`** on the basic pipeline when
  the store defines `search_by_text_with_score` (`IRISVectorStore` does), and the MCP
  tools report `score`. Multi-query RRF documents carry `rrf_score`, `rrf_rank` and the
  fused `score`.
- **Re-ingesting a stored document no longer logs an ERROR** from
  iris-vector-graph: the store checks for the row and updates it instead of trying the
  INSERT first.

### Changed: an entity table from an older release is upgraded in place

- **The entity writers add a missing `confidence` column.** `EntityStorageAdapter` and
  `BatchEntityProcessor` always INSERT `confidence`; a `RAG.Entities` (or
  `RAG.EntityRelationships`) created by an older release has none, and every write
  failed with SQLCODE -29. They now ask the catalog which columns exist and run
  `ALTER TABLE ... ADD confidence DOUBLE DEFAULT 1.0` when it is absent, logging a
  warning. Nothing is dropped or rewritten: rows written by the old release keep a NULL
  `confidence` (IRIS applies a column DEFAULT to new rows only), and a failed ALTER
  raises.

### Changed: only the ObjectScript that exists nowhere else ships

- **The wheel now ships `RAG.ColBERTSearch`** (`pipelines/colbert_iris/sp/ColBERTSearch.cls`)
  through package data. `package-data` listed no `*.cls`, so no release ever contained
  it (OS-13).
- **Removed thirteen vendored copies** of iris-vector-graph classes from
  `iris_vector_rag/pipelines/colbert_iris/sp/` (`Algorithms`, `BM25Index`, `EdgeScan`,
  `IVFIndex`, `Meta`, `NKGAccel`, `PLAIDSearch`, `PageRank`, `Snapshot`, `Subgraph`,
  `Traversal`, `VecIndex`, `UserExec`). ivg 4.1.1 ships every one of them, several of
  the copies were stale, and none was packaged or referenced (OS-12, OS-14). `User.Exec.Run`,
  which runs `Xecute` on its argument, is therefore not in this repository or its wheel;
  ivg still provides it.
- **`RAG.ColBERTSearch.Search` no longer builds SQL from document ids**: candidate ids are
  bound parameters and `topK` / `nProbe` are coerced to integers, so an apostrophe in a
  `doc_id` is searched rather than breaking the query (OS-11).

### Removed

Dead code with no production caller and a hazard attached. No deprecation release:
none of it was documented, and waiting would ship a known data-loss path for one more
cycle.

- **`IRISStorage`** (`iris_vector_rag.storage.enterprise_storage`) and its re-export from
  `iris_vector_rag.storage`. It duplicated `IRISVectorStore`, could drop a populated
  table, and formatted metadata filter keys into SQL. Use `IRISVectorStore`.
- **`IrisDevToolsBridge`** (`iris_vector_rag.testing.iris_devtools_bridge`) and its
  re-export from `iris_vector_rag.testing`. It ran `DROP NAMESPACE` with no container
  guard and imported `iris_devtools`, which is not a declared dependency. Tests that need
  a container use `iris-devtester` directly. The `iris_devtools_bridge` pytest fixture is
  gone with it.
- **`iris_vector_rag.embeddings.entity_extractor`**, the regex entity extractor that
  labelled any two capitalised words a `Person`. The LLM-backed services in
  `iris_vector_rag.services` are unchanged.
- **The REST API** (`iris_vector_rag.api`, the FastAPI service), its tests (`tests_api/`,
  `tests/integration/api/`), `config/api_config.yaml`, the `api` extra (`fastapi`,
  `uvicorn`, `python-multipart`, `websockets`, `redis`, `bcrypt`, `python-jose`,
  `email-validator`) and the `bcrypt` and `email-validator` installs in `release.yml`.
  Deprecated in 0.15 ([ADR 0001](docs/adr/0001-retire-the-rest-api.md)); it could not
  serve a request as built and its upload endpoint stored nothing. Use the MCP server
  (`iris-vector-rag[mcp]`) or call the pipelines from Python. If you used `[api]`, pin
  `iris-vector-rag<0.16` or take the code from the git tag `archive/rest-api-v1`. The
  design is kept in [docs/archived/rest-api/](docs/archived/rest-api/README.md). Findings
  AP-01 to AP-24 are closed by deletion.
- **The ontology package** (`iris_vector_rag.ontology`: loader, reasoner, models and the
  general plugin), `tests/test_ontology_support.py`, `tests/test_ontology_loader.py` and
  the `ontology:` section (and `graphrag.ontology_integration`) of `default_config.yaml`.
  Deprecated in 0.15; entity extraction stopped consulting it then, so the config section
  had no effect, and a leftover `ontology:` section in your own config still loads and is
  ignored. `OntologyAwareEntityExtractor` keeps its name and its ignored
  `ontology_sources` argument, and `EntityExtractionService.extract_with_ontology()` stays
  as a deprecated alias of `extract_entities_basic()`. If you used the package, take it
  from the git tag `archive/ontology-v1`. The guide and the architecture analysis moved
  to [docs/archived/ontology/](docs/archived/ontology/README.md). Findings OE-08, OE-09,
  OE-11 and OE-17 are closed by deletion.
- **`IRISSQLUtils.check_table_exists`**, which returned `False` for an unreachable
  database as readily as for a missing table. Nothing called it.
- **The pure-SQL fallbacks in `iris_vector_rag.common.db_vector_utils`.** The module now
  only re-exports `iris_vector_graph.dbapi_utils`, which is a hard dependency; the
  fallback `insert_vector` took different arguments than its call sites passed.
- **Both copies of `StubSentenceTransformer`** (in `embeddings/manager.py` and
  `embeddings/iris_embedding.py`), the `unittest.mock` detection branch that swapped a
  stub in when a cached model looked like a Mock, the MD5 "simple fallback" embeddings
  (`EmbeddingManager._create_fallback_function`), the `[0.0] * 384` zero vector in
  `RetrievalEngine._embed_text_for_search`, and the Mock sniffing in
  `IRISVectorStore.similarity_search`. A model that will not load, or every backend
  failing, used to produce random or hash-derived vectors that were stored and searched
  as if real (OE-01, OE-02, CR-09).
- **`EntityExtractionService.extract_batch_with_dspy`.** It loaded a DSPy module and
  never used it; it was `extract_entities` in a loop. The GraphRAG pipeline now calls
  `extract_batch`. Spec 073 may add a real opt-in. The `entity_types` and batch-logging
  tests that only exercised this method are deleted with it, and so are the example
  scripts built on it (`examples/optimized_dspy_entity_extraction.py`,
  `examples/scripts/test_batch_extraction.py`, `examples/scripts/test_quality_comparison.py`).
- **`OntologyAwareEntityExtractor.extract_with_ontology`.** On the base class it called an
  `extract_entities_basic` that was never defined, so it raised `AttributeError`. It
  remains on `EntityExtractionService` as a deprecated alias of LLM extraction and emits
  a `DeprecationWarning`.
- **`HybridGraphRAGPipeline._ivg_embedding_dimension()`** and the `cloud_config`
  vector-dimension default as the source of the schema's width. See the dimension entry
  below.

### Changed: validation and auto-setup report what is true

- **`.required` is honoured.** `PreConditionValidator` decides what gates a pipeline
  from each requirement's `required` flag, not from which list it sits in. GraphRAG is
  valid without the `Communities` table it flags `required=False`; it is reported under
  `Communities_optional`. Summaries and the "run the setup orchestrator" suggestion no
  longer blame optional components. `ValidationReport` gains `required_names` and
  `setup_errors`.
- **A failed count is not a missing table.** When `SELECT COUNT(*)` fails the validator
  asks the catalog: `details["error_kind"]` is `missing_table` (suggest creating it),
  `count_failed` (a dropped connection, a permission; `details["table_exists"]` says
  what the catalog answered) or `below_min_rows`. The schema of each requirement is
  honoured.
- **No rows invented.** Auto-setup no longer inserts `test_entity_1`, `test_rel_1` or a
  `test_document_marker.txt` document so that a minimum is met.
  `SetupOrchestrator._perform_entity_extraction` built `EntityExtractionService` with
  arguments it does not take (`TypeError`), read the file name instead of the text, and
  on failure fell back to the fake rows. It now extracts from each document's text with
  the LLM given to `SetupOrchestrator(llm_func=...)`, `setup_pipeline(llm_func=...)` or
  `create_pipeline(..., llm_func=...)`; with none it records a setup error and writes
  nothing. Each document's most confident ten entities are linked by `co_occurrence`
  relationships.
- **Setup raises or records what it could not do.** `_validate_embeddings_after_generation`
  raises `SetupError` for zero, incomplete (below the validator's 95%) or
  wrongly-formatted embeddings instead of logging and returning; an unverifiable count
  is no longer assumed to be success; a table that cannot be created, a requirement setup
  has no generator for and a chunk that cannot be stored are listed in
  `ValidationReport.setup_errors`. `print()` is gone from library code.
- **Standard tables only through `SchemaManager`.** The inline DDL and index lists in
  the orchestrator (which disagreed with the canonical layout) are removed, the chunk
  table is created and filled in the canonical layout at the configured width (the old
  path hard-coded 384 and inserted non-canonical columns, so every chunk failed), and
  the `filename` fallback that chunked and embedded file names is gone. CRAG setup now
  creates missing tables like the other document pipelines.
- **Requirements name real columns.** `RAG.DocumentChunks.embedding` is
  `chunk_embedding`; the never-satisfiable `RAG.Entities.entity_embeddings` requirement
  (the standard table has no vector column) is removed; the suggestion names
  `SetupOrchestrator.setup_pipeline(...)`, which exists.
- **Factory.** `hybrid_graphrag` builds (it was "Unknown pipeline type") and both graph
  types receive the schema manager; `list_available_pipelines` lists every registered
  type, not two.
- **`SchemaManager.migrate_table("Entities")`** keeps the `confidence` column the
  storage adapter writes.
- **`SetupOrchestrator.heal_token_embeddings`** raises `NotImplementedError`: it
  embedded each word of a document into columns `RAG.DocumentTokenEmbeddings` does not
  have and logged every failure at debug level. Token embeddings are written by the
  ColBERT ingest.

### Changed: the MCP server is Python over stdio, the Node server is removed

Feature `mcp-stdio-server` of the 2026-10 review
([ADR 0002](docs/adr/0002-mcp-server-is-python-over-stdio.md); VM-01, VM-03 to VM-06,
VM-13, VM-16 (the `mcp/` items), VM-17, OS-25).

- **Changed: one entry point.** `iris-vector-rag-mcp`, or
  `python -m iris_vector_rag.mcp`, starts a stdio MCP server built on the official `mcp`
  Python SDK. No subcommands, no port. The IRIS connection comes from the connection
  settings resolver (`IRIS_HOST`, `IRIS_PORT`, `IRIS_NAMESPACE`, `IRIS_USERNAME`,
  `IRIS_PASSWORD`, `IRIS_CONTAINER_NAME`).
- **Changed: six tools.** `rag_basic`, `rag_basic_rerank`, `rag_crag`, `rag_graphrag`,
  `rag_pylate_colbert` and `rag_health_check`. Input schemas set
  `additionalProperties: false`, and unknown arguments are rejected instead of being
  passed to `pipeline.query(**params)` (VM-13). A failing tool returns an error that
  names the exception type and nothing else; the detail goes to the log on stderr.
- **Changed: the health check measures.** It times `SELECT 1` against the resolved
  IRIS, constructs each pipeline and embeds one string, and reports `healthy`,
  `degraded` or `unavailable`. The hard-coded "connected, 10 ms, pool 1/10, 2% errors"
  values are gone (VM-06). `include_performance_metrics` is removed from its schema.
- **Changed: a query with IRIS down is an error.** The server checks the connection
  before running a pipeline, because `BasicRAGPipeline` logs a retrieval failure and
  still returns an answer.
- **Changed: the `mcp` extra** is the official `mcp` SDK only (`mcp>=1.9.0,<2`);
  `fastapi`, `uvicorn` and `websockets` left it and the `all` extra (the `api` extra
  still has them). The wheel now ships `mcp/mcp_tool_schema.json`, which the installed
  package needed and did not contain.
- **Removed: the Node.js server** (`tools/nodejs/`), `Dockerfile.mcp`,
  `scripts/docker/docker-entrypoint-mcp.sh` and `config/mcp_config.yaml`. The Node server
  only worked through an HTTP bridge at `localhost:8001` that was never served. The
  last commit that contains it is tagged `archive/mcp-node-v1`.
- **Removed: the HTTP bridge and the authentication** (`mcp/bridge.py`, `MCP_AUTH_MODE`,
  the `api_key` check that never validated anything), `mcp/cli.py`, `mcp/config.py`, the
  `python`/`both` modes stub in `mcp/__main__.py` and the dead `api.services` import.
  Stdio makes the process boundary the trust boundary.
- **Removed: the tools `rag_metrics`** (no counters existed) **and
  `rag_iris_global_graphrag`** (it depends on an external application), the unused range
  validators in `mcp/validation.py`, and `mcp/iris_llm_handler.py` (its only consumer was
  the bridge).
- **Fixed: `docs/MCP_INTEGRATION.md`** is rewritten with a working Claude Desktop
  configuration and uses `IRIS_USERNAME`, not `IRIS_USER`.

### Changed: the entity services stop reporting success they did not have

Feature `entity-services-fix` of the 2026-10 review (CR-06, CR-10, CR-11, CR-13, CR-17).

- **A failed document is reported, the batch continues.** `EntityExtractionService.extract_batch`
  returns a `ServiceBatchResult` whose `failed` lists `{"document_id", "reason"}` for each
  document whose extraction raised; `success_status` is `False` and `error_message` names
  them. It raises `EntityExtractionError` only if every document failed.
  `GraphRAGPipeline.load_documents` returns the same `failed` list (extraction or entity
  storage failures; the documents themselves are stored) and raises
  `EntityExtractionError` if every document failed. `documents_failed` still counts
  documents that failed to load.
- **Batch extraction is concurrent, retries, records metrics and asks for JSON.** Up to
  `batch_processing.max_workers` (8) documents of a batch are extracted at once; 1 is sequential. `extract_batch` retries a
  batch in which every document failed (2s, 4s, 8s; `batch_processing.retry_delays`), splits it
  and retries the halves, updates `get_batch_metrics()`, and the service logs its `llm`
  configuration at startup (or warns that none is set). The extraction prompt now asks the LLM
  for a JSON array of the configured entity types; it used to leave the format out, so a chat
  model's prose parsed as one entity.
- **No swallowed errors in extraction.** `_call_llm` raises when no `llm_func` is set or
  the LLM raises; `_parse_llm_response` raises on a reply it cannot parse instead of
  returning `[]`. `confidence_threshold` (default 0.3 for the service), a configured
  `entity_types` and `max_entities` are now applied. `process_document` reports
  `stored: False` without a storage adapter and raises if the adapter refuses a write.
  `token_budget` on `extract_batch` is advisory (a batch over it is logged).
- **`BatchEntityProcessor.store_entities_batch` is correct.** One tuple width per
  statement (a mix of entities with and without embeddings no longer fails the batch; an
  empty embedding counts as none), `confidence` is written, existing entity ids are
  updated and new ones inserted, the source document is resolved as `store_entity` does,
  and the vector width is the embedding's own, not a constant 384. `batch_size` chunks the
  work; a failing chunk is rolled back and retried in halves, and rows that still fail are
  returned in `failed`. `validate_count` re-counts the table. Storage is no longer
  all-or-nothing across a call (each chunk commits), and the "atomic" claim is withdrawn.
- **`EntityStorageAdapter`.** `store_entities_batch` and `store_relationships_batch` raise
  `EntityStorageError` (with `stored` and `failed`) when anything was not stored; they
  used to log and return a smaller count. `get_entities_by_document`,
  `get_entities_by_type`, `get_relationships_by_document` and `get_storage_stats` run
  real queries and let errors propagate (they returned `[]` or zeros). `create_tables_if_not_exist`
  ensures the tables in the entities table's schema and raises on failure (it logged DDL
  and returned `True`). `store_entity` caches only after the commit, so a failed write no
  longer appears in `search_entities`; the catalog is no longer re-read on every call.
  `search_entities` documents what it does: type and confidence filters run in SQL, the
  name match runs in Python over the remaining rows.
- **Entity embeddings need a column.** The standard `Entities` table has no vector column.
  An entity carrying `metadata["embedding"]` written to such a table is reported in
  `failed` rather than stored without its vector.

### Changed: storage stops losing data and reporting what did not happen

Feature `storage-correctness` of the 2026-10 review (ST-04 to ST-07, ST-10 to ST-12,
ST-16 to ST-18).

- **`SchemaManager.migrate_table` refuses to drop a populated table.** It raises
  `DestructiveMigrationError` unless `allow_destructive=True` is passed; `preserve_data`
  never did anything and does not unlock it. Migrating `SourceDocuments` also checks
  `DocumentChunks`. Empty and absent tables migrate as before. A migration that fails
  returns `False` instead of retrying on a bare table name, and the `rdf_*` and
  `kg_NodeEmbeddings_optimized` migrations now create and drop their tables in the
  manager's schema, not the connecting user's.
- **Index ensuring reports what happened.** `IRISCatalogReader.index_names` takes the
  schema, so an index in one schema no longer hides a missing one in another.
  `ensure_all_vector_indexes` raises `SchemaIndexError` naming every failed index (it used
  to log, roll back and return `None`) and returns the names it ensured; a table without
  the vector column (the standard `Entities`) is left out. `_ensure_index` creates plain
  indexes, sends `HNSW` to `ensure_vector_hnsw_index`, and raises on a configuration it
  cannot act on; `ensure_pipeline_schema` returns `False` for a required index that was
  not made.
- **DDL conforms.** New `Entities.description` and `EntityRelationships.metadata` are
  `VARCHAR(4000)`, not `TEXT`. Existing tables are not altered.
- **`table_exists("X")` looks in the manager's schema**, not the literal `RAG`, and
  `SchemaMetadata` is only ever created there (no copy in the connecting user's schema).
- **Valid names are accepted.** Table names such as `RAG.Descriptions` and `RAG.Forecasts`
  and filter keys such as `category`, `priority`, `organization` and `brand` were rejected
  for containing "script", "cast", "or" or "and". The identifier pattern stays; only a
  whole keyword is refused.
- **`process_document_row` keeps types.** Only streams and bytes become strings;
  `chunk_index` 3 stays 3 in metadata dicts.
- **`RelationEmbeddingStore.embed_and_store` raises `RelationNotFoundError`** when the
  relationship ID matches no row, and uses the configured schema (`schema_prefix`) rather
  than a literal `RAG.`.

### Changed: GraphRAG and ColBERT pipelines stop swallowing failures

- **GraphRAG graph expansion raises.** `_find_seed_entities` and `_expand_neighborhood`
  no longer catch every exception: a failed lookup used to look like "no seed entities"
  and the query fell back to vector search with no trace, and an expansion error quietly
  returned a smaller neighbourhood. The `IN (...)` list is now split into chunks of 200
  nodes, so a large frontier no longer hits the driver limit.
- **`HybridGraphRAGPipeline`**: the entity-to-content lookup uses `SELECT TOP 1` and the
  vector store's table shape instead of `LIMIT 1` and a hard-coded `RAG.SourceDocuments`,
  returns `None` for an entity no document holds, and raises on a database error. It no
  longer returns a made-up `"Entity: <id>"` string as if it were content.
  `delete_node` raises one error naming both failures when the graph delete and the
  vector delete that follows it both fail, chained to the graph error. Hybrid
  `get_performance_statistics` closes its cursor on every path.
- **`IRISGlobalGraphRAGPipeline.load_documents` works.** It passed `?` placeholders and a
  positional list to `sqlalchemy.text()`, which rejects that before any SQL is sent, so
  every load failed. It now inserts through the DBAPI cursor with a typed
  `TO_VECTOR(?, FLOAT, <dimension>)` and commits or rolls back.
- **ColBERT ingest is atomic per document.** A document row and its token rows commit
  together; a failed token insert rolls the document back instead of leaving an orphan
  `ColBERTDocuments` row. The driver connection autocommits, so ingest turns autocommit
  off for its duration and restores the caller's setting.
- **`IRISColBERTPipeline.query` with `use_hnsw=True` no longer runs the dense candidate
  retrieval** whose result it discarded; `rerank_factor` now sizes the candidate pool
  only on the bulk-fetch path (`use_hnsw=False`), where candidates are matched by id.
- **`ColBERTSchema.drop_tables` reports failures.** A table that is already gone is still
  fine; any other DROP error is raised after the rest were tried, and the cursor is
  closed on every path. The unused `token_dim` argument of `create_hnsw_index` is gone.
- **`PyLateColBERTPipeline.load_documents("path")`** treats a string or path as the
  documents path, as the parent's first positional parameter does, instead of iterating
  it character by character.

### Changed: `core/` contracts match what they promise

- **`RAGPipeline.retrieve()` returns `List[Document]`**, as annotated. It called
  `query()` and returned its whole response dict. It now calls
  `query(..., generate_answer=False)` and returns `retrieved_documents`, which is what
  `BasicRAGPipeline.retrieve()` already did. A subclass that relied on the dict should
  call `query()`.
- **`RAGPipeline.clear()` clears.** It looked for `vector_store.clear`, which no store
  has, logged a warning and deleted nothing. It calls `clear_documents()`; a store
  without it raises `NotImplementedError`.
- **`RAGPipeline.query()` names its first parameter `query`**, not `query_text`, and
  `load_documents()` makes `documents_path` optional. Both match what every pipeline
  implements and what `PipelineValidator` requires; a subclass following the base used
  to get a validator ERROR, and `ingest()` could not reach a `load_documents` that
  required a path.
- **`PipelineValidator(strict_mode=True)` reports every WARNING as an ERROR**, as its
  docstring says. The flag was stored and never read.
- **`ProcessingMetrics` averages cover successful batches only.** A failed batch no
  longer counts in the divisor (one failure plus one 10 s success reported 5 s).
  `total_batches_processed` still counts every batch.
- **`Document.__hash__` hashes `(id, page_content)`.** It hashed the metadata, so a list
  or dict value raised `TypeError` and fusion's metadata writes changed the hash of a
  document already in a set.

### Changed: fusion keeps real content and sane scores

`retrieval="text"`, `"hybrid"` and `"rrf"` (and the weighted form of `"mix"`) return
different results from before. There is no flag: the old results were wrong. (CR-01,
CR-02, CR-03, CR-09)

- **Content is never blank.** A text-index hit carries an id and a score; the engine now
  fetches the document by id from the vector store, so text-only hits come back with
  content and metadata. A hit found by both legs is one document, with content. A text
  hit the store does not hold is dropped with a warning.
- **New weighted score.** A hit at 1-based rank `r` in a list of `n` hits earns
  `weight * (n - r + 1) / n`: the top hit earns the whole weight, the last earns
  `weight / n`, a lone hit earns its whole weight. Before, it was `weight * (1 - r / n)`,
  so the last hit, and any one-hit list, earned 0. A top-1 vector hit in `hybrid` mode now
  scores 0.7 (the default vector weight), not 0. `rrf` scoring (`1 / (60 + rank)`) is
  unchanged. Rankings and `fusion_score` values change for existing callers.
- **`mix` with `weights` keeps all three lists.** The naive vector list was dropped and
  the score keys mislabelled; the keys are now `high_level_score`, `low_level_score` and
  `naive_score`, as under RRF. Fusion raises `ValueError` if lists, names and weights do
  not pair up.
- **`metadata_filter` and `similarity_threshold` are honoured in every mode.** The filter
  reaches the vector leg, and scopes text hits on their stored metadata. The threshold is
  a similarity cut-off, so it applies to the vector leg (a fused text-only hit has no
  similarity and is kept) and to `mix` relation hits. It also now works on the plain
  vector path, where it used to drop every document because the store returns none of
  their scores. Not applicable, so refused loudly: `similarity_threshold` with
  `retrieval="text"` and `metadata_filter` with `retrieval="global"` raise `ValueError`;
  `mix` with a filter skips the relation leg (relations carry no document metadata) and
  reports `degraded` with the reason in the result metadata.

### Changed: RAG pipelines and the LLM factory

Feature `pipelines-fix` of the 2026-10 review, the basic, rerank and CRAG pipelines and
`get_llm_func` (PL-01, PL-02, PL-11, PL-12, PL-13, OS-10, OS-23).

- **`get_llm_func` caches one client per argument set.** It kept a single client and
  ignored its keyword arguments, so a second call with other arguments reused the first
  client, and a function returned earlier called whichever client was built last. The
  cache key is now provider, model, cache flag and the keyword arguments; each returned
  function keeps its own client.
- **`create_pipeline` passes `get_llm_func` only LLM arguments** (provider and model
  name). `create_pipeline("basic", embedding_func=f)` with `OPENAI_API_KEY` set used to
  hand `embedding_func` to the OpenAI client and fail.
- **A failed rerank raises.** `BasicRAGRerankingPipeline._rerank_documents` swallowed its
  error and returned the input order while the caller reported `reranked=True`. Pass
  `allow_rerank_fallback=True` to keep the retrieval order instead; the result then
  carries `reranked=False` and the error in `metadata["rerank_error"]`.
- **`OPENAI_API_KEY` is required only without an `llm_func`.** Basic, rerank and CRAG
  demanded the key whenever an answer was asked for, even with a local model injected.
  With neither a key nor an `llm_func` the query still raises `ValueError`.
- **CRAG reads its tables through the table shape.** The chunk search takes the table,
  identifier, text and embedding column names, and the vector width, from the catalog, so
  a legacy-shaped chunk table works and a query of the wrong width raises
  `DimensionMismatchError`. The knowledge-base expansion goes through the vector store
  instead of its own `RAG.SourceDocuments` SQL; the debug `SELECT TOP 3 ... text_content`
  that ran on every call is gone, and so is the fallback that returned any rows with a
  made-up score of 0.5 when embedding failed.
- **CRAG calls the embedding function one way**: a list of texts in, a list of vectors
  out. A function that answers otherwise raises `TypeError` naming it.
- **`multi_query_rrf` ids of id-less documents are stable** across processes (a SHA-256 of
  the content, not the salted `hash()` of its first 100 characters).

### Changed: no fake vectors, and one embedding dimension

Failures that used to produce fake vectors now raise. This follows the connection-settings
decision in the same release and is not a major bump. Decision record:
`docs/remediation/2026-10-review/tickets/G6-fake-embedding-policy.md`.

- **A model that will not load raises.** `EmbeddingManager` tries the primary backend,
  then each configured `fallback_backends` entry in order, then raises `EmbeddingError`
  naming every backend and why it failed, with a hint about offline mode. Nothing failed
  is cached. If you depended on the old behaviour on a host that cannot reach the model
  hub, set `HF_HUB_OFFLINE=1` and point `embeddings.sentence_transformers.model_name` at
  a model on disk. (OE-01, OE-02)
- **The test embedder is reached only by name:** `embeddings: {provider: stub, dimension:
N}` (or `primary_backend: stub`). It lives in `iris_vector_rag.testing.embedder`,
  produces normalised deterministic vectors of the configured dimension, logs one warning
  per process, and cannot be a fallback. Its embedding identity is `stub`. Tests that
  relied on the implicit stub must ask for it (the `use_test_embedder` fixture does).
  `get_embedding_func(provider="stub")` returns it too.
- **A query-side embedding failure raises** instead of searching with a zero vector
  (relation search, `IRISVectorStore.similarity_search`) or falling back to text-only
  retrieval (HybridGraphRAG `hybrid_fusion` and `rrf`). CRAG no longer falls back to a
  stub embedder when the real one cannot be built. (CR-09, CR-11)
- **One dimension.** `EmbeddingManager.get_embedding_dimension()` is the active backend's
  real output width, computed once (sentence-transformers from the model, HuggingFace from
  its config, OpenAI from a model table or a one-time probe). `embeddings.dimension` (or
  `VECTOR_DIMENSION`) is an override that is checked against it and raises on
  disagreement. `SchemaManager`, the relation-embedding table, the ivg engine and
  `IRISVectorEngine.embedding_dimension` all read that value; `EntityRelationships` and
  the relation embedding column are no longer fixed at 384, and the
  `base_embedding_dimension` constructor argument is honoured by the first `SchemaManager`
  too. Resolution happens on first use, so constructing a `SchemaManager` or a store still
  loads no model. An OpenAI primary now reports 1536, not 384. (OE-03, ST-08, ST-09,
  ST-16)
- **A wrong-sized vector raises `DimensionMismatchError`** (a `VectorStoreDataError` and a
  `ValueError`, in ivg's wording) on insert and on query. `normalize_vector` and
  `insert_vector` in `common.db_vector_utils` no longer truncate or zero-pad. A failed
  `insert_vector` aborts the batch instead of being logged, skipped and committed with a
  partial id list; every vector's width is checked before the first row is written, since
  the session connection autocommits and a rollback cannot take rows back. Re-adding a
  document that exists now updates it: the store asks IRIS whether the row exists instead
  of relying on ivg matching `UNIQUE` in the driver's error text, which this driver can
  replace with `<LIST ERROR> ...` (such a re-add used to be logged as a failure and the
  old row kept). (ST-13, ST-23)
- **Writes through ivg carry the model.** `integrations.ivg.store_embedding` and
  `store_embeddings` pass the active model's name as `model_key`, so ivg refuses to mix
  stub and real vectors, or two models, without `force`. IVR's own tables do not record
  an embedding identity yet.
- **Embedding manager fixes.** The HuggingFace backend masks padding and normalises, so a
  text's vector no longer depends on its batch neighbours: **HuggingFace vectors change
  value, and vectors stored with the old pooling are stale; re-embed them.** (OE-04) The
  `batch_size` setting is applied by `iris_embedding.embed_texts` (and `embeddings.batch_size`
  by the manager). (OE-10) `clear_cache(config)` clears whatever device key the model was
  cached under, and the average embedding time divides by timed calls. (OE-13) The OpenAI
  client has a timeout (`embeddings.openai.timeout`, default 60 s), chunks its input
  (`embeddings.openai.batch_size`, default 256) and rejects a short response. (OE-14)

### Connection settings: one resolver, no guessing

- **Changed:** every connection entry point reads one resolver,
  `iris_vector_rag.config.connection_settings`, with iris-devtester's variable names
  and precedence: explicit arguments, the environment, a `.env` in the working
  directory, the pinned container, the `database.iris` section of the configuration
  file, then defaults. The variables are `IRIS_HOST`, `IRIS_PORT`, `IRIS_NAMESPACE`,
  `IRIS_USERNAME`, `IRIS_PASSWORD`, `IRIS_TIMEOUT` and `IRIS_CONTAINER_NAME`.
  `IRIS_TIMEOUT` is resolved and exposed on the settings; the session's handshake budget
  is still `IRIS_CONNECT_TIMEOUT`.
- **Changed:** the defaults are `localhost:1972`, namespace `USER`, user `_SYSTEM` and
  password `SYS`, stated once (not in `default_config.yaml`, `ConfigurationManager`,
  the session and the discovery helper). One log line names the `host:port` and user
  in use. When nothing was configured and the connection is refused or times out, the
  error says so and points at `IRIS_PORT` and `IRIS_CONTAINER_NAME`.
- **Changed:** `IRIS_USERNAME` beats `IRIS_USER` everywhere. `IRIS_USER` still works
  and raises a `DeprecationWarning`; it is removed after 0.16.
- **Changed:** `LOG_LEVEL` and `LOG_PATH` now beat the `logging` section of the YAML
  file, as every other environment variable already did.
- **Changed:** a configuration file no longer needs `database.iris.host`;
  `ConfigurationManager` no longer raises `ConfigValidationError` for its absence.
  A port that is not a port raises in every getter, naming `IRIS_PORT`.
- **Changed:** importing `iris_vector_rag` no longer loads a `.env` into the
  environment. The resolver reads `.env` from the working directory for `IRIS_*`
  settings; a script that wants other variables (an API key) from a file calls
  `dotenv.load_dotenv()` itself.
- **Changed:** `get_iris_connector()` no longer prints the database URL, which
  normally carries the password.
- **Removed:** Docker scanning. `auto_detect_iris_port()` ran `docker ps` and took the
  first container mentioning "iris" and "1972", any project's. The only Docker lookup
  left is `docker port <name> 1972/tcp` for the exact container named by
  `IRIS_CONTAINER_NAME`. Anyone who relied on the scan to find a non-default port sets
  `IRIS_PORT` or `IRIS_CONTAINER_NAME`.
- **Removed:** the password-fix helper (`_hard_fix_iris_passwords`, and 083's
  `recover_password`), which ran `UnExpireUserPasswords("*")` through `docker exec` on
  a container found by port substring. An expired password raises the session's
  `PasswordExpired` with a hint: preconfigure the container (`ChangePassword=0`) or
  create it with iris-devtester.
- **Fixed:** `default_config.yaml` had two `embeddings` keys, so the YAML parser kept
  the second and silently dropped `mode` and `text_in`. They are one key now, with every
  setting.

### CI gates for the first time

- **Changed:** `ci.yml` now triggers on pushes to `master` and on every pull
  request (it listed `main` and `develop`, so it never ran). The unit step no
  longer ends in `|| true` with `continue-on-error`; a unit failure turns the
  job red. The matrix is Python 3.11 and 3.12 (3.10 is below `requires-python`).
  Failures listed in `tests/contract/ci_known_failures.txt` are deselected; any
  other failure blocks. (OS-07, OS-08, OS-26 for the unit tests)
- **Removed:** the Docker job (no `Dockerfile` exists), the coverage jobs and
  Codecov upload, and the duplicate Trivy job (`security.yml` scans).

### CI gates, packaging and release (stage 2)

- **Added:** an `integration-tests` job in `ci.yml` runs `tests/integration/` on a real
  IRIS (CI pins its throwaway container with `IRIS_CONTAINER_NAME=iris`). No `|| true`;
  failures in `tests/contract/ci_known_failures.txt` are deselected, any other failure
  blocks. The file now carries the first honest integration run (about 70 failures that
  were hidden behind setup errors), grouped under a comment naming the feature expected
  to fix each group. `test_ivg_4x_upgrade_e2e.py` is `--ignore`d: it imports a
  `ConnectionManager` and a test module that no longer exist. (OS-26)
- **Fixed:** `pipelines.yaml` ships inside the wheel. It lived in a top-level `config/`
  directory outside the package, so an installed `iris_vector_rag` could not find it and
  `get_pipeline_requirements` silently returned `enabled: True` with no schema
  requirements. It is now `iris_vector_rag/config/pipelines.yaml`, and a missing file
  raises `FileNotFoundError`. (CM-12)
- **Removed:** the `rag-templates` console script, which pointed at
  `iris_vector_rag.cli`, a module that does not exist. The only script is
  `iris-vector-rag-mcp`. (OS-24)
- **Fixed:** `pytest-timeout` is in the `dev` and `all` extras, so `make test-unit`
  (which passes `--timeout`) works after `make install`; the Python 3.10 classifier is
  gone (`requires-python` is `>=3.11`); `pr-integration.yml` no longer installs an
  `evaluation` extra that does not exist. (OS-06, OS-09, OS-18)
- **Changed:** `release.yml` checks that the tag equals `v<pyproject version>` before
  anything else runs; `trufflehog` and `gh-action-pypi-publish` are pinned to release
  tags instead of `main` and `release/v1`. `release.yml` no longer ignores
  `test_mcp_tool_schemas.py` (it passes) nor the four CRAG contract files: their four
  failures are quarantined in `ci_known_failures.txt` instead, so the other CRAG tests
  gate. (OS-19)
- **Added:** `make setup-env`, `make test-community` and `make test-enterprise`, the
  targets `CLAUDE.md` names. `make install` uses `uv pip install`. (OS-20)
- **Added:** a `swallowed-error` detector in `scripts/gates/antipatterns.py`. A broad
  `except` (bare, `Exception`, `BaseException`) in `iris_vector_rag/` that never raises
  fails the gate unless `antipatterns-baseline.txt` lists it. The baseline holds the 302
  instances that exist today; fixing one means deleting its line.

### Changed

- **DSPy support marked experimental.** The `dspy_modules` package is now available via
  the optional `dspy` extra (`pip install iris-vector-rag[dspy]`), with its experimental
  status clearly documented. This module is under active development (spec 073) and not
  recommended for production. A `FutureWarning` is issued on import.
- **Documentation corrected.** Removed false claims that DSPy integration is supported or
  production-ready from README, AGENTS, and Makefile. Updated all references to note spec
  073 as the tracking issue and direct users to the core entity services for production use.
- **`RAG.SDK` (ObjectScript) reports real status and checks what it concatenates into
  SQL** (review findings OS-03, OS-04, OS-05, OS-15, OS-16; OS-01 and OS-02 were fixed by
  086 and are now covered by tests that compile the classes).
  - A new class, `RAG.SDK.Identifier`, is the one place table and column names and row
    counts are checked. `SetDefaultTable`, `AttachTable`, `AddDocument`,
    `AddDocumentBatch`, `AddDocumentWithEmbed`, `BuildBM25Index` and every search refuse an
    unsafe name; `topK` must be a positive integer (a numeric string is accepted). The
    refusal is `{"status":"error","message":...}` (searches: `{"error":...}`), and
    `SetDefaultTable` writes nothing when it refuses.
  - Stream text columns are read to the end, not cut at 2,000 characters: search hits
    carry the whole text and `BuildBM25Index` indexes whole documents. A value over
    3,000,000 characters is cut, and the hit (or the `BuildBM25Index` result, as a
    `truncated` count) says so. Stream columns are recognised in the current OID format;
    the old code returned the raw OID as the text for them.
  - `BuildBM25Index` indexes the configured text column by default (its `textCol` default
    was the literal `text_content`), returns an error instead of `ok` with 0 documents when
    its query does not prepare or run, and leaves the old index alone in that case.
  - `TextSearch` returns `{"error":...}` when its query fails, scoped or not; unscoped it
    used to return `[]`.
  - `VectorSearch` no longer uses `SELECT TOP n ... ORDER BY`: it orders in SQL and stops
    at `n` rows in the loop. An HNSW index can no longer be chosen by the optimizer for
    this query.
  - `Schema.Initialize` stops at and names the first failing statement (it used to return
    `ok` whatever happened), no longer splits its DDL on the text "GO" (a schema named
    `GOLD` broke), and refuses a non-positive dimension. `Schema.Drop` reports failures.
    `Schema.Status` takes an optional schema and reads the dimension from the class
    dictionary; a missing table reports dimension 0 instead of a made-up 384.
  - `Schema.Install` (run by `zpm install`) installs the latest release instead of the
    hardcoded `iris-vector-rag==0.11.4`; `Install("x.y.z")` pins one.
  - Every ObjectScript method follows the workspace rules (`pName` parameters, `tName`
    locals, `Set tSC = $$$OK` first, `Quit` last). Parameters were renamed; calls are
    positional, so callers are unaffected. Return shapes are unchanged. `Embed` and
    `Evaluate` (embedded Python) are not converted.

### iris-vector-graph 4.1

IVR now requires `iris-vector-graph>=4.1.1,<5`. ivg 4.0 re-keyed `Graph_KG`
storage on `(graph_id, node_id)`, so an install created under 2.x has to be
migrated once. See [docs/IVG_4X_UPGRADE.md](docs/IVG_4X_UPGRADE.md).

- **Old layout is reported, not migrated.** `validate_graph_prerequisites()` and
  `ensure_iris_vector_graph_tables()` list every `Graph_KG` table still in the 2.x
  layout in a new `legacy_layout_tables` field, and validation fails with the call
  that fixes it.
- **`SchemaManager.upgrade_iris_vector_graph_storage(dry_run=False, **options)`**
  runs ivg's `initialize_schema()`and then`upgrade_to_4_0_0`. The dry run writes
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

### Test environment: one real IRIS, no mock, no skip (0.15)

- **Removed: `iris_vector_rag.testing`.** The backend-mode harness (`backend_manager`,
  `connection_pool`, `validators`, `exceptions`), the fake session driver and the devtools
  bridge no longer ship in the wheel; they are `tests/support/` in the repository.
  `iris_vector_rag.config.backend_modes` moved with them. Nothing in the package imported
  them. The package `iris_vector_rag.testing` stays, empty, for the test embedder.
- **Changed: the test suite's `iris_connection` is a live IRIS connection.** It was a
  `MockConnection` from the connection pool, so tests that believed they used IRIS used a
  dictionary. `IRIS_CONTAINER_NAME` pins this project's container;
  without it the suite starts an ephemeral container and removes it. Another project's
  container is refused. An unreachable IRIS raises `IrisUnreachable` with a hint; it never
  becomes a mock and the test is never skipped.
- **Removed: the `los-iris` fixtures.** `iris_database_config` and `iris_test_session`
  attached to another project's container, ran `DROP TABLE` there, and returned a `Mock`
  on any failure. The session-start probe of ports 11972 and 21972 went with them.
- **New: `make iris-reset`** stops IRIS cleanly, restarts the project container through
  `idt container restart`, waits until it answers and makes the warm-up connection.
- Eleven tests that only passed against the mock now fail for the reasons they always
  had (the medical-20 graph fixture is not loaded; one inserts into a column that does not
  exist). They are listed in `tests/contract/ci_known_failures.txt`.

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
