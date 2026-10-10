# Release Notes: iris-vector-rag v0.15.0

**DRAFT** for review. Not tagged, not published.

0.15.0 is a correctness release. The October 2026 code review found that several
parts of the library reported success they did not have: fake vectors stored as
real ones, a migration that could drop a populated table, a connection layer
with eleven entry points, a test suite that passed against a mock. This release
fixes those, and because the old behaviour was wrong, **several results
change**. Read "Upgrading" before you install.

Requires `iris-vector-graph>=4.2.0,<5`. Python 3.11 and 3.12.

## Upgrading: read this first

These are the changes that can break a working install or change what you get
back. Each is announced in the [CHANGELOG](CHANGELOG.md) with the detail.

### Failures now raise instead of returning something plausible

- **No fake vectors.** If the embedding model will not load, `EmbeddingManager`
  tries the configured real fallbacks and then raises `EmbeddingError`, naming
  each backend and why it failed. The cached stub, the MD5 pseudo-embeddings and
  the zero query vector are gone. The test embedder exists only by naming it
  (`embeddings: {provider: stub, dimension: N}`). **Action:** if you ran offline
  and relied on the silent stub, set a real provider or opt in to the test
  embedder explicitly.
- **Wrong-sized vectors raise `DimensionMismatchError`** on insert and on query.
  One dimension now comes from the active embedding backend and is handed to
  iris-vector-graph; `embeddings.dimension` is an override that must agree.
- **HuggingFace vectors change value** (padding is masked and the output is
  normalised). **Action:** re-embed anything stored through that backend.
- **Entity extraction and storage report failures.** `extract_batch` returns a
  `failed` list and carries on; it raises only if every document fails. No LLM
  configured, an LLM error or an unparseable reply raises instead of returning
  `[]`. The batch processor is no longer all-or-nothing.
- **Pipelines stop swallowing errors.** GraphRAG graph expansion, the ColBERT
  ingest and `drop_tables`, CRAG's chunk search, rerank failures and several
  storage calls raise. A failed rerank raises unless you pass
  `allow_rerank_fallback=True`.
- **`migrate_table` refuses to drop a populated table** unless you pass
  `allow_destructive=True`.

### Connection settings

- One resolver reads explicit arguments, the environment, `.env`, a container
  pinned by `IRIS_CONTAINER_NAME`, the config file, then defaults
  (`localhost:1972`, namespace `USER`, `_SYSTEM`/`SYS`), in that order, with
  iris-devtester's variable names. The defaults are logged once, and a refused
  connection on all-default settings tells you which variables to set.
- `IRIS_USERNAME` beats `IRIS_USER`; `IRIS_USER` still works and warns.
- Importing the package no longer loads `.env` into `os.environ`.
- Docker is consulted only to find the port of the container named by
  `IRIS_CONTAINER_NAME`. The port scanner and the password-fix helper are gone.

### Scores and results change

- **Fusion scoring.** `retrieval="text"`, `"hybrid"` and `"rrf"` (and the
  weighted `"mix"`) return different results: text-only hits carry their
  content, and a hit at rank `r` of `n` scores `weight*(n-r+1)/n`. There is no
  flag. A score threshold on `retrieval="text"` and a filter on
  `retrieval="global"` raise `ValueError`.
- **Entity and relationship types keep their case.** `entity_type` and
  `relationship_type` are now `COLLATE EXACT` (`GROUP BY` used to return
  `DISEASE` for `Disease`). **Existing tables are altered the first time
  `ensure_table_schema` runs**, without losing rows or indexes, and are
  case-sensitive from then on.

### iris-vector-graph 4.2.0

- **BM25 indexes need one upgrade before they are written to.** ivg 4.2.0
  refuses `Insert` and `Delete` on an index built earlier until
  `engine.bm25_upgrade(name)` has run; searching still works and scores change.
  IVR owns no BM25 index, so nothing runs for you. **Action:** if you built a
  BM25 index, call `SchemaManager.upgrade_bm25_indexes(dry_run=True)` to preview
  and then without `dry_run`, as a job: writers are refused while it runs.
- **The engine's class-method calls run as the account IVR connected with**, not
  `_SYSTEM`. An account without `%Native_ClassExecution` is now refused where
  4.1.1 ran the call as `_SYSTEM`. **Action:** none for a superuser account; for a
  restricted one, grant the role.

### Schema changes made on first use

- **Text search.** The text leg of `text`, `hybrid` and `rrf` used to find
  nothing on a real deployment. It now searches the document table through an
  iFind index, `SourceDocumentsTextIdx`. The first
  `ensure_table_schema("SourceDocuments")` builds it synchronously over existing
  rows, so the time grows with table size; call
  `SchemaManager.ensure_text_index()` to choose when. Until the index exists
  those modes raise `TextIndexMissingError`, which names the index.
- **Entity tables from an older release** gain a `confidence` column
  (`DOUBLE DEFAULT 1.0`), added in place. Rows written before the upgrade keep a
  NULL `confidence`.

### Removed

- **The Node.js MCP server**, `Dockerfile.mcp`, the HTTP bridge and its
  authentication. The MCP server is Python over stdio: run `iris-vector-rag-mcp`
  (or `python -m iris_vector_rag.mcp`). Six tools: `rag_basic`,
  `rag_basic_rerank`, `rag_crag`, `rag_graphrag`, `rag_pylate_colbert` and
  `rag_health_check`. The tools `rag_metrics` and `rag_iris_global_graphrag` are
  removed, unknown tool arguments are rejected, and the health check measures
  the connection and no longer takes `include_performance_metrics`. See
  [docs/MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md) for a working Claude
  Desktop configuration.
- **The REST API** (`iris_vector_rag.api`, the `api` extra). It could not serve
  a request as built. The code stays at the git tag `archive/rest-api-v1` and
  the design is kept in
  [docs/archived/rest-api/](docs/archived/rest-api/README.md).
- **The ontology package** (`iris_vector_rag.ontology`) and the `ontology:`
  config section. The tag is `archive/ontology-v1`.
- **`IRISStorage`**, `IrisDevToolsBridge`, the regex entity extractor,
  `extract_batch_with_dspy`, both copies of `StubSentenceTransformer`, the
  pure-SQL fallbacks in `db_vector_utils` and `IRISSQLUtils.check_table_exists`
  (it returned `False` for an unreachable database).
- **`iris_vector_rag.testing`** no longer ships the backend-mode harness, the
  fake session driver or the connection-pool mock; they are `tests/support/` in
  the repository. The package keeps the test embedder.
- **Vendored ObjectScript.** Only `RAG.ColBERTSearch` ships; the 13 classes
  copied from iris-vector-graph are deleted.

## Highlights

- **One owner for the IRIS connection.** Eleven ways to get a connection, each
  with a different subset of the protections, became one session object with a
  bounded handshake, backoff, health checks and a clear error when the password
  has expired.
- **Table shape from the catalog.** Pipelines ask the database which column
  fills the content role instead of guessing names, so documents and chunks work
  on tables created by other versions.
- **iris-vector-graph 4.2.** `validate_graph_prerequisites()` reports a 2.x
  `Graph_KG` layout and `SchemaManager.upgrade_iris_vector_graph_storage()`
  migrates it; `SchemaManager.upgrade_bm25_indexes()` brings BM25 indexes to
  schema 2. See [docs/IVG_4X_UPGRADE.md](docs/IVG_4X_UPGRADE.md).
- **LLM calls are bounded.** `get_llm_func` gives the OpenAI client a 60 second
  timeout and 2 retries (`llm.timeout`, `llm.max_retries`, or `RAG_LLM__TIMEOUT`
  and `RAG_LLM__MAX_RETRIES`), and a timeout raises `LLMTimeoutError` naming the
  provider and limit. Before, a stalled network hung for ten minutes.
- **Batch entity extraction is concurrent.** `extract_batch` runs up to
  `entity_extraction.batch_processing.max_workers` documents at once (default 8;
  1 is sequential). Measured on gpt-4o-mini it was 9x to 24x faster than one
  document at a time.
- **The MCP server holds one IRIS connection.** Queries, the reachability check
  and the health probe share one worker thread. Before, five concurrent queries
  held five connections and exhausted the eight licence units of the Community
  image.
- **CI runs real tests.** `ci.yml` triggers on `master` and every pull request;
  the `|| true` guards are gone; an integration job runs on a real IRIS; the
  release workflow checks the tag against the package version; and a
  `swallowed-error` detector fails CI on a new broad `except` that does not
  raise.
- **`make iris-reset`** recovers a wedged project container.

## Fixed

By area of the review. The CHANGELOG lists each finding.

- **Storage.** Migration retries no longer run on bare table names; index
  ensuring is schema-scoped and reports what it did; new `Entities.description`
  and `EntityRelationships.metadata` columns are `VARCHAR(4000)`, not TEXT;
  valid table names such as `RAG.Descriptions` and filter keys such as
  `category` are accepted; CLOB dicts keep their types; a zero-row `UPDATE` in
  `RelationEmbeddingStore` raises `RelationNotFoundError`.
- **Validation and auto-setup** report what is true: `.required` is honoured, a
  failed `COUNT` is reported as a count failure and not a missing table, fake
  test rows are gone, and entity extraction in setup needs an `llm_func`.
- **Core contracts.** `retrieve()` returns `List[Document]`; `clear()` clears;
  `strict_mode` escalates warnings; `Document.__hash__` no longer raises on list
  metadata; metrics average over successful batches.
- **RAG pipelines.** Dict documents are accepted by `load_documents`; CRAG
  builds a default connection and reports its correction strategy; the basic
  pipeline returns scores; multi-query RRF sets `rrf_score`; `get_llm_func`
  keeps one client per argument set.
- **ObjectScript SDK.** `RAG.SDK` reports real status, validates identifiers and
  `topK`, reads the whole text of a stream, and builds its BM25 index from the
  configured column.
- **Packaging.** `pipelines.yaml` and the ColBERT class now ship in the wheel;
  `pytest-timeout` is a dev dependency; a dead console script is removed.

## Experimental

- **DSPy.** `iris_vector_rag.dspy_modules` is experimental (spec 073) and
  available through the `dspy` extra, which is no longer part of `all`.

## Known limitations

- **Text search over a table that older releases wrote to `content`** indexes
  `text_content` only; such rows are not searchable until copied.
- **GraphRAG ingest** through `load_documents` needs `documents=`; a path with
  no documents raises.
- **Legacy entity tables whose `id` is a required column with no default** still
  reject inserts.
- **`IRIS_TIMEOUT`** is resolved but nothing reads it yet.
- **Some findings are deferred** to [docs/DEBT.md](docs/DEBT.md): structural
  smells, test doubles left in production code, and the hardcoded vector type at
  21 baselined sites.

## Verification

On `remediation/integrated` after the iris-vector-graph 4.2.0 merge (branch head
`42937278`, merged unchanged), against IRIS 2026.1 and the real OpenAI API: 1800 unit tests pass;
the contract suite as `release.yml` runs it passes (761 passed, 28 skipped, 0
failed); the integration, end-to-end and root suites pass in one session (738
passed, 26 skipped, 6 xfailed, 0 failed, 0 errors); the quality gates and the
swallowed-error detector are clean. These ran against a wheel built from the
`v4.2.0` tag; `iris-vector-graph` 4.2.0 reached PyPI on 2026-10-10, and the
install from PyPI is re-run before tagging.
