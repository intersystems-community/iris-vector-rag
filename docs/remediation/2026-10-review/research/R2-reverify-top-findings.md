# R2: Re-verification of the severity-A, status-C findings

Code read at `/Users/tdyar/ws/iris-vector-rag-private`, commit `0cbb39b7` (branch
`087-ivg-4.1`), read-only. Nothing touched IRIS or Docker. Where a reproduction is
quoted it ran in pure Python with mocks or stubs, using `.venv/bin/python` and
`IRIS_BACKEND_MODE=community`. Paths below are relative to that repo. "Reachable today"
means a caller in the shipped package or a documented entry point reaches the code, not
just a test.

## Summary table

| ID    | Verdict     | Reachable today?                                                                 | Severity | One-line reason                                                                                             |
| ----- | ----------- | -------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| ST-01 | CONFIRMED   | No. `IRISStorage` has no caller outside tests; it is only re-exported            | B        | Probe failure runs `DROP TABLE` (and `DELETE FROM` if the drop fails); reproduced                           |
| ST-03 | CONFIRMED   | No (same class)                                                                  | B        | `'$.{key}'` is formatted into SQL at two sites (lines 464, 534); reproduced                                 |
| ST-04 | CONFIRMED   | Only via explicit `migrate_table`; no caller in the package                      | same (A) | Every `_migrate_*` drops; the chunks migrator drops too unless `preserve_data=True`                         |
| AP-01 | CONFIRMED   | Only if someone runs `uvicorn iris_vector_rag.api.main:app`; no in-repo launcher | same (A) | Routers and middleware are built with `None` at import; reproduced                                          |
| AP-04 | CONFIRMED   | Dormant behind AP-01                                                             | B        | No ownership check in WS, but it needs a valid key plus a random-UUID `operation_id`                        |
| AP-05 | CONFIRMED   | Dormant behind AP-01                                                             | same (A) | WS path never checks permission or limits; `str(e)` echoed                                                  |
| AP-06 | CONFIRMED   | Dormant behind AP-01                                                             | same (A) | File is never read; `_process_upload` sleeps and marks COMPLETED                                            |
| AP-08 | PARTLY TRUE | Only the Redis/degraded leg; the DB-down leg is masked by AP-07                  | B        | `/pipelines` is not public, and `PipelineInstance` has no `error_message` field                             |
| CM-01 | CONFIRMED   | Yes when `IRIS_PORT` is unset (the repo's `.env` masks it in-tree)               | same (A) | `docker ps` scrape picks the first "iris" container exposing 1972; reproduced                               |
| CM-02 | CONFIRMED   | Conditional: connect error containing "expired" plus `docker`                    | same (A) | `str(port) in line` matches every IRIS container when port is 1972; reproduced                              |
| CM-03 | PARTLY TRUE | Yes                                                                              | B        | Stock-IRIS defaults in four places; a convention, not a defect, outside this workspace                      |
| CM-08 | CONFIRMED   | No. The only call is the module's own `test_iris_sql_utils`                      | C        | Swallow is real; nothing calls it                                                                           |
| CM-09 | PARTLY TRUE | Narrow: cache fallback with `IRIS_CONNECTION_URL` set                            | B        | `print` is real but sits on a rare fallback path                                                            |
| CR-01 | CONFIRMED   | Yes: `BasicRAG.query(retrieval="rrf"\|"hybrid"\|"text")`                         | same (A) | Last list wins and text hits are always blank; worse than reported; reproduced                              |
| CR-06 | PARTLY TRUE | Mixed tuples: no in-repo producer. Lost `confidence` on the GraphRAG path: yes   | B        | No code sets `metadata["embedding"]`; DDL default means NULL confidence is wrong, but the failure is silent |
| CR-09 | PARTLY TRUE | Yes for `BasicRAG` with `retrieval="global"\|"mix"`                              | same (A) | Zero-vector swallow is real and always hit; `metadata_filter` is dropped by every non-vector mode           |
| VM-01 | CONFIRMED   | No. Nothing serves `MCPBridge` over HTTP; callers are tests only                 | B        | Check is a stub; reproduced                                                                                 |
| VM-02 | PARTLY TRUE | Only `create_pipeline(..., auto_setup=True)` for graphrag                        | B        | On the canonical schema the fake-row insert fails too, so the "fake data passes validation" claim is wrong  |
| VM-10 | PARTLY TRUE | No. `reset_schema` has no caller and the bridge cannot be built                  | C        | `DROP NAMESPACE` is not SQL I believe IRIS accepts (unverified); needs `../iris-devtools`                   |
| OE-01 | CONFIRMED   | Yes (any model-load failure, e.g. offline)                                       | same (A) | Two stubs, both cached permanently; reproduced                                                              |
| OE-02 | CONFIRMED   | Yes (primary fails at call time and all fallbacks fail)                          | same (A) | MD5 pseudo-embeddings returned with a warning                                                               |
| OE-03 | PARTLY TRUE | Yes                                                                              | B        | Returns 384 for an OpenAI primary (reproduced), but it feeds a dimension check, not table creation          |
| OS-02 | CONFIRMED   | Yes: documented public ObjectScript SDK                                          | same (A) | Any SQLCODE other than 0 and -119 returns `"ok"`                                                            |
| OS-14 | PARTLY TRUE | Only to a caller who already has IRIS access and deployed the class              | C        | Identical to the class IVG ships; no caller in IVR or in ivg 4.1.1 Python; not packaged                     |

Counts: 15 CONFIRMED, 9 PARTLY TRUE, 0 REFUTED.

## ST-01: `store_documents` drops a populated table

**Verdict:** CONFIRMED in code; no caller.

**Path.**

- `storage/enterprise_storage.py:241` runs `SELECT COUNT(*) FROM RAG.SourceDocuments WHERE 1=0`.
- `except Exception` (line 242) catches anything, including a transient connection reset.
- Line 248 calls `initialize_schema()`.
- `initialize_schema` (lines 100-200) tries `DROP TABLE {table}` at line 137, then `CREATE TABLE`.
- If the drop fails it runs `DELETE FROM {table}` at line 146, which wipes every row. The ledger misses this.
- On the second attempt the table name is the bare `SourceDocuments`, so a copy is created in the user's schema.

**Reproduction.** A mock cursor whose probe raises once, and `store_documents` then executes:

```text
SELECT COUNT(*) FROM RAG.SourceDocuments WHERE 1=0
DROP TABLE RAG.SourceDocuments
CREATE TABLE RAG.SourceDocuments ( doc_id VARCHAR(255) PRIMARY KEY, ti...
```

**Callers.** `grep` finds `IRISStorage` only in `storage/__init__.py` (export) and
`tests/test_enterprise_storage.py`. No pipeline uses it (ledger ST-20 says the same).

**Severity.** B while the class has no caller. It stays a trap for anyone importing
`iris_vector_rag.storage.IRISStorage` (public export).

**Fix note.** The cheapest fix is deleting the class with ST-20. Otherwise `initialize_schema` must never run `DROP` or `DELETE` from a read-path failure. Also ST-02: reads use `id`, the DDL uses `doc_id`, and `TO_VECTOR(?, FLOAT, 384)` is hardcoded.

## ST-03: metadata filter key injection

**Verdict:** CONFIRMED in code; no caller.

**Path.** `vector_search` builds `JSON_EXTRACT(metadata, '$.{key}') = ?` at
`storage/enterprise_storage.py:464`. `_fallback_text_search` (reached when `vector_search`
raises, line 503) repeats it at **line 534**. The ledger names only one site; both need the
fix.

**Reproduction.** `metadata_filter={"a') OR 1=1 --": "v"}` produces:

```sql
JSON_EXTRACT(metadata, '$.a') OR 1=1 --') = ? ORDER BY similarity_score DESC
```

Note `top_k` is also f-string interpolated into `SELECT TOP {top_k}`.

**Callers.** Same as ST-01: none outside tests. `IRISVectorStore` is the whitelisted path that pipelines use.

**Severity.** B while unreachable.

## ST-04: `migrate_table` drops unconditionally

**Verdict:** CONFIRMED, with one correction.

**Path.** `migrate_table` (`storage/schema_manager.py:1083`) dispatches to the `_migrate_*`
methods, which run, in order:

- `_migrate_source_documents_table`: on a failed FK drop, drops `DocumentChunks` entirely (1251), then `DROP TABLE IF EXISTS` on SourceDocuments (1261).
- `_migrate_entities_table` (1610) and `_migrate_entity_relationships_table` (1688): loop over `[qualified, bare]`. If the first `CREATE` fails after the qualified `DROP` succeeded, the second pass drops and creates the bare name and returns `True`. `RAG.Entities` is gone and a stray table exists.
- `_migrate_rdf_labels/props/edges` (1903, 1949, 1996) and `_migrate_kg_node_embeddings_optimized_table` (2052): unqualified names.

**Correction.** `_migrate_document_chunks_table` (1353) is not exempt: it honours
`preserve_data=True` only. With the default `preserve_data=False` it drops the table at
line 1381. The docstring of `migrate_table` says `preserve_data` is "not implemented yet".

**Callers.** No call in `iris_vector_rag/`. Callers are `scripts/test_graphrag_validation.py`,
`tests/integration/conftest.py` and `tests/integration/test_table_shape_six_shapes.py`.
Reach is explicit-call only, as the ledger says. "DDL autocommits so `rollback()` restores
nothing" is standard IRIS behaviour but I did not test it.

## AP-01: `create_app()` at import time

**Verdict:** CONFIRMED.

**Path.** `api/main.py:391` runs `app = create_app()`. At that moment `app_state` is the
empty `AppState()`. `create_app` (305-385) therefore passes `None` into
`AuthenticationMiddleware`, `RequestLoggingMiddleware`, `ApiKeyAuth`, and the four routers.
`lifespan` only fills `app_state` afterwards, so nothing rebinds them. The rate-limit
block is skipped (`redis_client` is `None`). CORS falls back to `allowed_origins ["*"]`
with `allow_credentials=True`.

**Reproduction.** Importing the module works (no connection is made). It shows:

```text
config {} pool None pm None redis None
MW AuthenticationMiddleware {'connection_pool': None}
MW RequestLoggingMiddleware {'connection_pool': None}
MW CORSMiddleware {'allow_origins': ['*'], 'allow_credentials': True, ...}
```

A `TestClient` request (lifespan not run) with a well-formed UUID key returns 500,
`AttributeError: 'NoneType' object has no attribute 'get_connection'` from
`ApiKeyAuth.verify_api_key`. An unauthenticated `GET /api/v1/health` (a public path) also
returns 500.

**Callers.** No in-repo launcher. `main.py:__main__` uses `uvicorn.run("iris_rag.api.main:app")`,
a module that does not exist (AP-16). `api/README.md` documents `python -m iris_rag.api.cli`,
which does not exist either. The only route in is a hand-typed
`uvicorn iris_vector_rag.api.main:app`, which the `api` extra supports.

**Omissions that change the fix.**

- `PipelineManager._initialize_pipelines` (`api/services/pipeline_manager.py:76,96`) builds `PipelineInstance(name=..., type=..., ...)`. The model (`api/models/pipeline.py`) requires `pipeline_id`, `pipeline_type`, `initialized_at`, and has no `type`, `created_at`, or `error_message`. For any enabled pipeline the `try` body raises `ValidationError`, and the `except` block builds the same invalid object and raises again. The lifespan therefore fails at startup with the default config (five enabled pipelines). Moving `create_app` alone would not make the API start.
- `lifespan` shutdown calls `connection_pool.dispose()`, but `common.connection_pool.IRISConnectionPool` has no `dispose` (only `close_all`).
- So AP-01 needs an app factory that builds routers from the populated state, plus this model repair.

## AP-04: WebSocket upload progress has no ownership check

**Verdict:** CONFIRMED in code; impact overstated.

**Path.** `websocket/routes.py` handles `watch_upload` by calling
`upload_handler.stream_upload_progress(session_id, UUID(message["operation_id"]), ...)`.
That calls `document_service.get_operation_status(operation_id)` (`handlers.py:225`), which
has no key filter. REST checks `operation.api_key_id != api_key.key_id` at
`routes/document.py:246`.

**Reachability and severity.** Dormant behind AP-01. The attacker also needs a valid key
and the victim's `operation_id`, which is a random UUID4. B. The leak includes
`error_message` on failed operations, as the ledger says.

## AP-05: WebSocket skips permission and limit checks

**Verdict:** CONFIRMED.

**Path.** `websocket/connection.py:91` calls `temp_auth.verify_api_key(...)` and never
`check_permission`. REST query routes require `Permission.READ` (`routes/query.py:52`).
`QueryStreamingHandler.stream_query` (`handlers.py:108`) runs `pipeline.query(query=query,
top_k=top_k)` on client-supplied values with no 10,000-character or `top_k <= 100` check.
Errors are echoed as `str(e)` (`handlers.py` except-block, and `routes.py` message loop).

**Reachability.** Dormant behind AP-01. Same severity: it is an authorisation bypass.

**Extra.** `message.get("query")` can be `None`. `asyncio.create_task` results are dropped (AP-15).

## AP-06: upload never reads or stores the file

**Verdict:** CONFIRMED.

**Path.** `routes/document.py:~135-152` never calls `await file.read()`. It estimates
`int(file.size / 10000)` documents (TypeError when `size` is `None`, AP-11) and calls
`initiate_upload`. `document_service._process_upload` (`services/document_service.py`)
loops `for i in range(total_documents): await asyncio.sleep(0.1)`, with the comment
"Process documents (simulated for now)", then sets `COMPLETED` and `progress 100.0`. The
pipeline is fetched only to check it exists.

**Reachability.** Dormant behind AP-01, but once the API runs this is a success message
for nothing. A 100 MB file would "process" for about 1000 s.

## AP-08: error text in unauthenticated responses

**Verdict:** PARTLY TRUE.

**Path, DB leg.** `routes/health.py:122` puts `str(e)` into `HealthStatus.error_message`,
but a down database makes the overall status UNAVAILABLE. The 503 branch then does
`JSONResponse(content=response.model_dump())`, which raises
`TypeError: Object of type datetime is not JSON serializable` (reproduced; this is AP-07).
The client gets the generic 500 body, so the DB string does not leak today. Fixing AP-07
would start leaking it.

**Path, Redis leg.** Line 150 puts `str(e)` into a DEGRADED `redis_cache` component. That
takes the normal 200 path, so the text reaches an unauthenticated `/api/v1/health` caller
(`/api/v1/health` is in `public_endpoints`). The pipeline health check does the same
(`pipeline_manager.py:~262`), capped at 500 characters by the model.

**Path, `/pipelines` leg.** Refuted for two reasons. `/api/v1/pipelines` is not in
`public_endpoints` (AP-03), so auth middleware requires a key. And `pipeline_manager.py:103`
passes `error_message=` to `PipelineInstance`, which has no such field, so it is never
serialised (see AP-01 omissions).

**Severity.** B.

## CM-01: auto-detect picks any project's container

**Verdict:** CONFIRMED.

**Path.** `common/iris_connection.py:115-118`: `_get_connection_params_from_env()` runs
on every `get_iris_connection()` call. If `IRIS_PORT` is unset it calls
`auto_detect_iris_port()` (line 68), which runs `docker ps` and returns the host port of
the first line that contains "iris" and "1972" and matches `0.0.0.0:(\d+)->1972/tcp`.
Only then does it fall back to 1972.

**Reproduction.** With `subprocess.run` patched to return `los-iris 0.0.0.0:11972->1972`,
`opsreview-iris 0.0.0.0:1972->1972`, `iris-vector-rag-iris 0.0.0.0:51972->1972`:

```text
CM-01 auto_detect -> 11972
```

That is `productivity-framework`'s container.

**Reachability.** Yes, for any installed-package user without `IRIS_PORT`. In this working
tree it is masked, because `iris_vector_rag/__init__.py` loads the repo `.env`
(`IRIS_PORT=51972`) at import (OS-17); the same script printed `51972` for the env
params after import.

**Omission.** The `docker ps` call also runs when the caller passes `port=` explicitly,
because `env = _get_connection_params_from_env()` is evaluated first. A fix must make
the scrape lazy or remove it.

## CM-02: `_hard_fix_iris_passwords` matches the wrong container

**Verdict:** CONFIRMED.

**Path.** On a connect error whose text contains "password change required" or
"expired" (lines 341-347) it calls `_hard_fix_iris_passwords(h, p)`. That runs `docker ps`
and picks the first line where `str(port) in line and "tcp" in line` (line 154). It then
runs `docker exec -i <name> iris session IRIS -U %SYS` with
`Do ##class(Security.Users).UnExpireUserPasswords("*")`.

**Reproduction.** With the same fake `docker ps` output:

```text
CM-02 _hard_fix(host, 1972) -> True
CM-02 exec target: los-iris | input: b'Do ##class(Security.Users).UnExpireUserPasswords("*")\nHalt'
CM-02 port 51972 target: iris-vector-rag-iris
```

For port 1972 every IRIS container matches, because each maps its internal port to 1972.
Port 51972 picked the right one.

**Reachability.** Conditional on the error text and docker being available. The caller
is the main `get_iris_connection` path, so it is live.

**Omission.** The target is also a wrong-project write even for an exact port match if
the connection is to a remote host: `host` is ignored, so a remote `localhost:1972` tunnel
can match a local container.

## CM-03: stock defaults `localhost:1972 / _SYSTEM / SYS`

**Verdict:** PARTLY TRUE.

**Path.** Confirmed in four places: `config/manager.py:518-521`, `config/default_config.yaml`
(host, port, user, password), `common/connection_pool.py:~406-411` (`os.getenv(..., "1972")`,
`"SYS"`), and `config/entities.py:71`. Also `common/iris_connection.py:120-124`.

**Assessment.** These are the stock InterSystems defaults, and the finding is a violation
of this workspace's exclusivity rule (1972 belongs to `opsreview`), not a library defect.
The real defect is that nothing warns when defaults are used and that a user's package
install silently connects to whatever is on 1972. B.

**Also noted.** Username precedence differs between `iris_connection.py` (`IRIS_USER`
first) and `connection_pool.py` (`IRIS_USERNAME` only), per CM-05.

## CM-08: `check_table_exists` swallows everything

**Verdict:** CONFIRMED, no caller.

**Path.** `common/iris_sql_utils.py:121-166` and `_check_table_exists_with_connection`
(180-205) catch `Exception` and return `False`. The code is as described.

**Callers.** The only call in the package is inside `test_iris_sql_utils()` in the same
module (line 562). `IRISSQLUtils` is otherwise used only for `optimize_vector_table`
(`enterprise_storage.py:172`). The branches "callers that branch on missing try to create
tables" do not exist; the real `table_exists` logic is in `schema_manager` and `validation`.
C unless an external caller is known.

## CM-09: `print` of the connection URL

**Verdict:** PARTLY TRUE.

**Path.** `common/utils.py:411` prints `f"Connecting to IRIS at: {db_url}"`. The only
caller is `common/llm_cache_manager.py:199`, the URL fallback after the DBAPI attempt
fails, and only when `IRIS_CONNECTION_URL` is set (otherwise it raises `ValueError` first,
line 408). A SQLAlchemy IRIS URL carries `user:password@`. B: a real credential leak to
stdout, but on a narrow fallback.

## CR-01: fusion keeps the last list's object, text hits are blank

**Verdict:** CONFIRMED, and worse than reported.

**Path.** `retrieval/engine.py:395` and `:425` do `seen[doc_id] = doc`, so the later list
wins. `_retrieve_fusion` (line 108) passes `[vector_docs, text_docs]`, so text wins.
`_text_search_to_docs` (line 96) builds `Document(page_content="")`.

**Reproduction.** Vector returns A and B with content; text returns A and C:

```text
rrf    [('A', "''"), ('B', "'full text of B'"), ('C', "''")]
hybrid [('A', "''"), ('B', "'full text of B'"), ('C', "''")]
```

**Omission.** Every text-only hit is always blank, not just overlaps, and `retrieval="text"`
returns all-blank documents. The fix has to hydrate content for text hits by id; merely
preferring the vector object in the overlap case is insufficient.

**Also.** The installed `iris_vector_graph.text_search.TextSearchEngine.search_documents`
defaults to `table_name="docs"` with columns `id, text`, not `RAG.SourceDocuments`, so
real overlap is unverified (I did not touch IRIS).

**Reachability.** Yes: `pipelines/basic.py:584-595` dispatches to `RetrievalEngine` whenever
`opts.retrieval` is set and not `"vector"`. It needs the BM25 prerequisite to pass.

## CR-06: `BatchEntityProcessor` tuple mix and missing confidence

**Verdict:** PARTLY TRUE.

**Confirmed.** `services/batch_entity_processor.py:~110-150`: tuples use
`embedding is not None`, the SQL choice uses truthiness. Reproduced with a mock cursor:

```text
mixed            | params in SQL: 6 | tuple lens: [6, 5] | has confidence col: False
empty-list only  | params in SQL: 5 | tuple lens: [6, 6] | has confidence col: False
```

`executemany` then fails, the `except` returns `{"entities_stored": 0, "error": ...}`, and
`EntityStorageAdapter.store_entities_batch` returns `0` with only a log line.

**Refuted or weakened.**

- No code in `iris_vector_rag/` sets `entity.metadata["embedding"]` (grep is empty), so the only in-repo caller, `pipelines/graphrag.py:176`, always builds 5-tuples. The mixed failure needs an external caller.
- "NULL confidence": `SchemaManager._create_entities_table` declares `confidence DOUBLE DEFAULT 1.0`, so a batch insert gets 1.0, not NULL. The extracted confidence is still lost (never inserted). `_migrate_entities_table`'s DDL has no `confidence` column at all.
- Re-ingest: `Entity.id` defaults to a generated id, so the PK collision needs a caller that reuses ids.

**Severity.** B. The real GraphRAG-path defects are: confidence not stored, no
`_resolve_source_document` (FK risk), hardcoded 384, errors swallowed into a count of 0.

## CR-09: zero-vector fallback and ignored filters

**Verdict:** PARTLY TRUE.

**Confirmed.** `retrieval/engine.py:163-173`: `except Exception: pass` then
`return [0.0] * 384`. Worse than reported: `pipelines/basic.py:586` builds
`RetrievalEngine(vector_store=..., connection=...)` without `config_manager`, so
`cfg_mgr is None` and the zero vector is returned **every time** in global and mix mode
through BasicRAG. `_retrieve_vector`, `_retrieve_text`, `_retrieve_fusion` never read
`opts.metadata_filter`. For any `retrieval` other than `"vector"`, BasicRAG routes to the
engine (`basic.py:577-595`), so a caller's `metadata_filter` (tenant isolation) is silently
dropped. The default `"vector"` path still passes it to the store.

**Weakened.** `similarity_threshold`: `basic.py` applies it after retrieval for all modes,
but reads only `metadata["score"]`/`["similarity"]`. Fused docs carry `fusion_score` and
text docs `text_score`, so a threshold drops everything in those modes.

**Also.** `global`/`mix` return a dict (CR-04), and `basic.py` then does
`[doc.page_content for doc in retrieved_documents]` on it, so BasicRAG with these modes
breaks on an attribute error instead of returning hits.

## VM-01: MCP API-key check is a stub

**Verdict:** CONFIRMED.

**Path.** `mcp/bridge.py:67`: `if self.config.auth_mode == "api_key" and api_key: pass`.
`load_config` (`mcp/config.py`) does not validate `auth_mode`.

**Reproduction.** With a stub handler:

```text
api_key None  -> success True, handler ran: True
api_key WRONG -> success True, handler ran: True
apikey (typo) -> success True, handler ran: True
```

**Reachability.** No. Nothing in Python exposes `MCPBridge.invoke_technique` over HTTP:
`mcp/__main__.py` builds the bridge and sleeps ("would start FastAPI server here"), and no
`/mcp/invoke_technique` route exists in `iris_vector_rag/`. The Node client
(`tools/nodejs/src/mcp/python_bridge_client.ts:55`) posts to an endpoint nothing serves. The
only callers are tests. B, until a server is wired.

## VM-02: `_perform_entity_extraction` legacy columns and fake rows

**Verdict:** PARTLY TRUE.

**Confirmed.** `validation/orchestrator.py:1621-1627` selects `id, filename, file_path`;
the canonical `SourceDocuments` (`storage/table_shape.py:511`) has `doc_id`, `title`,
`text_content`, and none of those three. Any failure lands at line 1691 and calls
`_insert_minimal_test_entities`. Extraction would also use the filename as text, build a
fresh `ConfigurationManager()`, and `extract_entities(content, doc_id)`.

**Refuted.** The claim that the fake rows satisfy `min_rows` on current schemas does not
hold. The fallback inserts into `RAG.Entities (..., source_document)`, but the canonical
`Entities` has `source_doc_id` (`schema_manager.py:1620`, `2355`), so that insert fails
too, is logged and rolled back. On canonical schemas no fake data lands and validation
stays invalid. The fake-data path only works against a legacy table that has both
`source_document` and a legacy `SourceDocuments`.

**Reachability.** Through `fulfill data requirement` in `SetupOrchestrator`, reached from
`create_pipeline(..., auto_setup=True)` for `graphrag` and `hybrid_graphrag`
(`validation/factory.py:95-100`). Default is `auto_setup=False`. The API `PipelineManager`
passes `auto_setup=False`. B.

## VM-10: `IrisDevToolsBridge.reset_schema`

**Verdict:** PARTLY TRUE.

**Confirmed.** `testing/iris_devtools_bridge.py:129` runs `DROP NAMESPACE {namespace}` on
the connection handed in, with no guard. `start_container` pins `:latest`
(`intersystemsdc/...`). `wait_for_ready` ignores `timeout` and returns `True` when the
container lacks `is_ready`.

**Weakened.** `reset_schema` has no caller anywhere (`tests/conftest.py:488` only
provides the bridge fixture; the `reset_schema` calls in tests are a different class,
`tests/fixtures/schema_reset.py`). The constructor raises `IrisDevtoolsMissingError`
unless `../iris-devtools` exists; it does not exist here. I believe IRIS SQL has no
`DROP NAMESPACE` statement, so the method would fail rather than destroy anything; I did
not verify this against IRIS. The module ships inside the wheel as
`iris_vector_rag.testing` and is exported in `testing/__init__.py`. C.

## OE-01: stub embedder cached permanently

**Verdict:** CONFIRMED.

**Path.** `embeddings/manager.py:129-144`: `except Exception` around
`SentenceTransformer(model_name, device=device)` falls to `StubSentenceTransformer()`,
and line 147 stores it in `_SENTENCE_TRANSFORMER_CACHE` under the real model's key. A
second stub with the same behaviour is in `embeddings/iris_embedding.py:332-357`, also
cached, with the comment "to keep contract tests working".

**Reproduction.** With `sys.modules["sentence_transformers"] = None`:

```text
StubSentenceTransformer 384 cached: ['sentence-transformers/all-mpnet-base-v2:cpu']
```

The vectors are seeded random from a SHA-256 of the text, so they are stable and
meaningless. The dimension is fixed at 384 whatever model was asked for.

**Reachability.** Any load failure (offline Hugging Face, bad model name, OOM), not only a
missing package. `sentence-transformers` is a hard dependency (`pyproject.toml:45`), so
the realistic trigger is a load error.

## OE-02: MD5 pseudo-embeddings when every backend fails

**Verdict:** CONFIRMED.

**Path.** `embeddings/manager.py:433-436` calls `_create_fallback_function()` (line 323),
which hashes with MD5, pads to `embeddings.dimension` (default 384), logs a warning and
returns.

**Reachability.** It needs the primary to fail at call time (not load time, because of
OE-01) and the fallback backends to fail or be uninitialisable. A primary of `openai` during
an outage satisfies that. The default primary is `sentence_transformers`, so the default
config reaches it less often than the ledger implies.

## OE-03: `get_embedding_dimension` ignores the backend

**Verdict:** PARTLY TRUE.

**Confirmed.** `embeddings/manager.py:450-468`. Reproduced with a fake config manager
(`primary_backend: openai`, `text-embedding-3-large`):

```text
primary openai | reported dim: 384
```

**Weakened.** Table creation does not use this method; `SchemaManager` has its own
dimension sources (ST-09). The only callers are `_validate_embedding_dimension` in
`pipelines/basic.py:733` and `pipelines/crag.py:528`, which compare it with the actual
embedding length. The effect is a false `ValueError("Embedding dimension mismatch")` for
an OpenAI or HuggingFace primary, which fails loudly. B. The hash fallback reads
`embeddings.dimension` (default 384) and has the same blind spot.

## OS-02: ObjectScript `AddDocument` reports success on any error

**Verdict:** CONFIRMED.

**Path.** `iris_src/src/RAG/SDK/Pipeline.cls:23-29`:

- SQLCODE 0 returns ok.
- SQLCODE -119 runs the UPDATE on `st2`, an undefined variable (OS-01), then returns ok.
- Every other outcome, including every negative SQLCODE, reaches line 29 and returns `{"status":"ok"}`.

`AddDocumentBatch` (lines 32-45) counts those as `inserted` and reports `errors: 0`.
`%Prepare` failures on the final branch are also unchecked.

**Reachability.** Yes: it is the documented public ObjectScript SDK
(`RELEASE_NOTES_v0.11.4.md`, `specs/070-objectscript-sdk/quickstart.md`). No Python caller
in the repo.

## OS-14: `User.Exec.Run` executes arbitrary strings

**Verdict:** PARTLY TRUE.

**Path.** `iris_vector_rag/pipelines/colbert_iris/sp/UserExec.cls` is a class with
`ClassMethod Run(cmd) { Xecute cmd }`.

**Findings on reach.**

- It is byte-identical to `User.Exec.cls` that iris-vector-graph ships in `.venv/.../iris_vector_graph/iris_src/src/`.
- No Python in `iris_vector_rag/` calls it, and grep of the installed ivg 4.1.1 Python finds no `_run_cos` or `User.Exec` reference either. The only mentions are specs and `CHANGELOG.md`.
- `pyproject.toml` package-data omits `*.cls` (OS-13), so it does not ship in the wheel.
- `scripts/deploy_vecindex.sh` loads it into a container when an operator runs it.
- `Xecute` runs with the calling user's privileges, so it is no privilege escalation beyond what that user could already do over the superserver. This is standard IRIS behaviour; I did not test it.

**Severity and fix.** C: a hygiene issue. The answer to the ledger's "is it needed?" is
no caller exists, so delete the vendored copy and the two lines in `deploy_vecindex.sh`.

## Summary

- Counts: 15 CONFIRMED, 9 PARTLY TRUE, 0 REFUTED (ST-01, ST-03, ST-04, AP-01, AP-04, AP-05, AP-06, CM-01, CM-02, CM-08, CR-01, VM-01, OE-01, OE-02, OS-02 confirmed).
- Should drop out of class A: ST-01, ST-03, AP-04, AP-08, CM-03, CM-09, CR-06, VM-01, VM-02, OE-03 go to B; CM-08, VM-10, OS-14 go to C.
- Missing from the ledger: the `PipelineInstance` schema mismatch that stops `PipelineManager` from starting (AP-01 omissions), `DELETE FROM` in `initialize_schema` (ST-01), the second `JSON_EXTRACT` site (ST-03), and text hits that are always blank (CR-01).
