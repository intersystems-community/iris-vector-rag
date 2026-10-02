# R1: verify the plausible findings

Code read at commit `0cbb39b7` (branch `087-ivg-4.1`), read-only. No IRIS was contacted. Repro snippets ran in `.venv/bin/python` with `IRIS_BACKEND_MODE=community`, using SQLite, Starlette's test client, `xml.etree`, and langchain-openai object construction only. Paths below are relative to the repo root; the package directory is `iris_vector_rag/` (the ledger's `iris_rag` spelling in CLAUDE.md is stale).

## Summary table

| ID    | Verdict                                                           | Severity                | Reason                                                                                                                                                            |
| ----- | ----------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PL-05 | CONFIRMED                                                         | same (B)                | `conn.execute(text(...), [list])` raises `ArgumentError` in SQLAlchemy 2.x before any SQL is sent, so `load_documents` always fails                               |
| ST-13 | CONFIRMED                                                         | same (B)                | a `False` from `insert_vector` is logged, skipped, then the batch commits and returns a shorter ID list; callers ignore the length                                |
| ST-14 | REFUTED                                                           | not a defect (latent C) | the only caller, `add_documents`, regenerates embeddings on any count mismatch, so the short-list branch is unreachable                                           |
| ST-15 | CONFIRMED                                                         | C (ledger B)            | fallback signature and return type both differ from the call site, but `iris-vector-graph` is a hard dependency so the fallback never runs on a supported install |
| ST-16 | CONFIRMED                                                         | same (C)                | `UPDATE` with no rowcount check; also hardcodes `RAG.` and `384`; no in-repo caller of `embed_and_store`                                                          |
| ST-17 | CONFIRMED                                                         | same (C)                | `table_exists` always passes `"RAG"` as the schema for unqualified names; `SchemaMetadata` fallback lands in the user schema and is never read                    |
| AP-14 | CONFIRMED                                                         | B (ledger C)            | all four claims hold, and the ledger missed that the 429 is raised inside `BaseHTTPMiddleware` and surfaces as a 500                                              |
| AP-15 | CONFIRMED                                                         | same (B)                | no timeout on the first `receive_json()`; `create_task` results are not retained                                                                                  |
| CM-11 | CONFIRMED (lock); leak UNDECIDABLE WITHOUT IRIS                   | same (C)                | plain `threading.Lock` held across `_t.join(connect_timeout)`; whether the late connection leaks depends on the driver's finalizer                                |
| CM-12 | CONFIRMED                                                         | B (ledger C)            | the released 0.14.0 wheel has no `config/pipelines.yaml`; GraphRAG schema requirements silently become empty                                                      |
| CR-14 | CONFIRMED                                                         | same (C)                | list value raises `TypeError`; mutation changes the hash; nothing in `iris_vector_rag/` hashes a `Document`                                                       |
| VM-13 | CONFIRMED                                                         | same (B)                | unknown keys pass `validate_params` and reach `pipeline.query(**params)`; any `KeyError` in a handler is reported as "Unknown technique"                          |
| OE-11 | CONFIRMED                                                         | same (B)                | `None` label crashes `get_all_synonyms`; `rdfs:altLabel` is read, `skos:altLabel` is not, for OWL; entity expansion is only partly exposed                        |
| OE-14 | CONFIRMED (location corrected); `OLlama` part UNDECIDABLE locally | same (B)                | the `[""]` return is in `entity_extraction_module.py:356`, not `embeddings/manager.py`; `dspy` is not installed and not declared                                  |
| OS-05 | CONFIRMED (code pattern)                                          | C (ledger A)            | concatenation is real, but nothing in the repo exposes these methods to an untrusted caller, and `whereClause` is a deliberate raw predicate                      |
| OS-10 | CONFIRMED                                                         | same (B)                | reproduced: the kwarg lands in the request payload and `Completions.create()` rejects it                                                                          |
| OS-12 | CONFIRMED                                                         | same (C)                | `TSTART` with no Try and no rollback, but the file is a stale, never-deployed, unreferenced copy of an IVG class                                                  |
| OS-13 | CONFIRMED                                                         | same (B)                | neither the 0.14.0 wheel nor sdist contains any `.cls`; two of the fourteen files exist nowhere else                                                              |

## Summary

- Counts: 17 CONFIRMED (CM-11, OE-14 each carry one UNDECIDABLE WITHOUT IRIS or locally undecidable sub-claim), 1 REFUTED (ST-14), 0 wholly UNDECIDABLE.
- Severity changes: OS-05 A to C, AP-14 C to B, CM-12 C to B, ST-15 B to C.
- Class changes: ST-14 defect to not a defect; OS-12 becomes "delete or sync the vendored copy" rather than a code fix; OS-05 is hardening, not an exploitable hole, until something exposes it.

## PL-05: `_store_document_sql` cannot execute

CONFIRMED.

- `iris_vector_rag/pipelines/iris_global_graphrag.py:317-336`: `sql = text("... VALUES (?, ?, ?, ?, ?, ?, ?, TO_VECTOR(?))")`, then `conn.execute(sql, [doc.id, title, ...])` with a list of 8 scalars.
- `self.iris_engine` is a real SQLAlchemy engine: it comes from `iris_db.get_sqlalchemy_engine()` (`~/ws/IRIS-Global-GraphRAG/app/iris_db.py:26-28`, `create_engine("iris://...")`).
- SQLAlchemy 2.0.45 (installed) rejects a positional list for `text()` in its parameter distillation, before the dialect is reached.
- Call order: `load_documents` (`:260`) loops documents, embeds, calls `_store_document_sql`; the exception is caught at `:299` and rethrown as `IRISGlobalGraphRAGException("Document loading failed: ...")`. The first document fails, so nothing is ever loaded.
- Reachable today: `MCP` registers the pipeline (`mcp/technique_handlers.py:196-203`) and `pipelines/__init__.py:24` exports it, but `_initialize_pipeline` needs the external `IRIS-Global-GraphRAG` checkout on disk, so only users with that project hit it.

Repro (the error is dialect independent):

```python
from sqlalchemy import create_engine, text
with create_engine("sqlite://").connect() as c:
    c.execute(text("create table t(a,b)"))
    c.execute(text("insert into t values (?, ?)"), [1, 2])
# ArgumentError: List argument must consist only of dictionaries
```

Omitted by the finding: `text()` also has no `?` positional support (needs `:name` binds); the fix is `:name` params with a dict, plus `TO_VECTOR(:vec, FLOAT, <dim>)`. The pipeline also has no test that runs `load_documents` against a real engine (`scripts/test_iris_global_graphrag.py` is a script, not a test).

## ST-13: a failed `insert_vector` is skipped, the batch still commits

CONFIRMED.

- `iris_vector_rag/storage/vector_store_iris.py:779-797`: `success = insert_vector(...)`; on `False` it logs `Failed to upsert document` and `continue`s. `:833-835` commits and returns `added_ids` (shorter than the input).
- `iris_vector_graph.dbapi_utils.insert_vector` (installed, `:101-180`) returns `False` on: `normalize_vector` returning `None` (non-numeric input), any non-UNIQUE SQL error (for example a value too long for `VARCHAR(n)`), and a failed upsert `UPDATE`.
- Callers do not compare lengths: `core/base.py:171` returns the list as is; `pipelines/basic.py:185` and `:371` discard it.
- Reachable: yes, through `add_documents` for any row the database rejects. A rejected row is lost, the rest commit, and the only trace is an `ERROR` log line.

Omitted by the finding: the same function `normalize_vector` silently truncates or zero-pads an embedding whose length differs from the column dimension (`dbapi_utils.py:92-97`), so a model/column dimension mismatch stores wrong vectors with no error at all. That is a separate silent-wrong-data path that the ST-13 fix should not assume away. Severity stays B because the returned list is honest; it becomes A only if the fix owner counts "callers never check" as silent loss.

## ST-14: a document is stored without a vector when `embeddings` is short

REFUTED as a reachable defect.

- The branch exists: `vector_store_iris.py:779` is `if embeddings and len(embeddings) > i:`, otherwise the code falls to the no-vector `INSERT` at `:816-828`.
- But `_store_documents` has exactly one caller in the package, `add_documents` (`:924`). Before it, `:905-916` regenerates embeddings whenever `len(embeddings) != len(processed_documents)` and `_generate_embeddings` (`:678-709`) returns one vector per document or raises. The other `_store_documents` hits in `core/base.py`, `basic.py`, `crag.py`, `graphrag.py` are pipeline methods, not this one.
- One residual: `embeddings=[]` passes both guards (`[]` is falsy at `:905` and `:911`), so `_store_documents` sees no embeddings and stores text-only rows. That is an empty-list edge, not the short-list claim.

Recommendation: reject, or fold the `embeddings=[]` edge into the ST-13 fix.

## ST-15: fallback `insert_vector` has a different signature

CONFIRMED, severity C.

- Fallback `common/db_vector_utils.py:31-41`: `insert_vector(cursor, table_name, id_column, vector_column, id_value, vector, dtype)` returning `None`. Call site `vector_store_iris.py:780-789`: keywords `vector_data`, `dimension`, `key_columns`, `additional_columns`, `dtype`. Calling it raises `TypeError: unexpected keyword argument 'vector_data'`.
- Even with matching arguments the fallback returns `None`, which the call site reads as failure (`if success`), and its `dtype` default is `FLOAT` where the real function defaults to `DOUBLE` in `vector_similarity_search` (`dbapi_utils.py:247`), the mismatch IRIS rejects with SQLCODE -259.
- Reachable: the `except ImportError` branch runs only if `iris_vector_graph.dbapi_utils` cannot import. `pyproject.toml:60` makes `iris-vector-graph>=4.1.1,<5` a hard dependency, and 4.1.1 ships the module. So on any supported install the fallback is dead code; this is a latent bug plus dead-code smell. Lower to C.

Fix shape: delete the fallbacks (the module docstring already says IVG 1.x is unsupported) rather than repair them.

## ST-16: `UPDATE` affects zero rows if the ID is missing

CONFIRMED, severity C.

- `storage/relation_embedding_store.py:22-26` defines `UPDATE RAG.EntityRelationships SET relation_embedding = TO_VECTOR(?, FLOAT, 384) WHERE relationship_id = ?`; `embed_and_store` (`:120-145`) executes it, commits, and logs `Embedded relationship_id=...` without reading `cursor.rowcount`. The comment at `:135-138` states the design choice (rows are written first).
- Reachable: `RelationEmbeddingStore` is constructed at `retrieval/engine.py:161`, but no code in the package calls `embed_and_store`; only its message at `engine.py:197` tells a user to.

Omitted: the SQL hardcodes the `RAG` schema (ignores `schema_prefix`, same class as ST-17) and the dimension `384` (a different embedding model fails or, via the IRIS vector cast, mismatches). Fix should take both from the schema manager.

## ST-17: `table_exists` ignores `schema_prefix`; `SchemaMetadata` in the wrong schema

CONFIRMED, severity C.

- `storage/schema_manager.py:253-278`: for an unqualified name, `schema, table = "RAG", table_name`, then `self._snapshot(table, schema=schema)`. `_snapshot` only applies `self.schema_prefix` when `schema` is falsy (`:278`: `schema or self.schema_prefix`), which never happens on this path. The docstring says "defaulting to the RAG schema", so this is intended for the default, wrong for a custom prefix.
- The only in-package caller passes a qualified name (`:478`, `f"{IVG_SCHEMA}.{table_name}"`); the other callers are tests with qualified names or the default. So nothing wrong happens today.
- `ensure_schema_metadata_table` (`:659-700`): if `CREATE TABLE IF NOT EXISTS <prefix>.SchemaMetadata` fails, it retries unqualified, which creates `SchemaMetadata` in the connecting user's schema. Every reader and writer uses `self._qn("SchemaMetadata")` (`:739, :1317, :1334`), so that table is never read, and `_initialized` is set to `True` so the failure is never retried. Reproducing the failure needs IRIS, but the code path is unconditional once the first attempt raises.

## AP-14: rate limiter races, collisions, outage, stale concurrency

CONFIRMED, severity B. All four claims hold, plus one the ledger missed.

- Check and add are separate calls: `middleware/rate_limit.py:92-93` (`zremrangebyscore`, `zcard`) then `:150` (`zadd`). No pipeline or Lua script, so concurrent requests all read the same count.
- Member collision: `:150` uses `str(now.timestamp())` as the member; two requests in the same timestamp tick overwrite each other and count once.
- Redis outage: no `try/except` around any Redis call in `check_rate_limit`, `check_concurrent_requests`, or `dispatch` (`:236-320`); a `redis.ConnectionError` propagates out of the middleware.
- Stale concurrency: `check_concurrent_requests` (`:170-200`) does `incr` and sets a 300 s expiry only when `current == 1`; a worker killed between `incr` and the `finally: decr` leaves the count up until the key expires.
- Missed, and worse: `HTTPException` raised inside `BaseHTTPMiddleware.dispatch` is not turned into a response by FastAPI's handler, because the middleware runs outside the router's exception handling. Reproduced with the installed Starlette: a `BaseHTTPMiddleware` raising `HTTPException(429)` makes `TestClient.get` re-raise `HTTPException`. In `api/main.py:308-340` the only handler is `@app.exception_handler(Exception)`, which answers 500, so the limiter's 429 (`:134`) and the concurrency 429 (`:289`) reach clients as 500 with no `Retry-After`. The limiter therefore "works" only in the sense that requests are refused.
- Also: the blocking `redis` client is called from `async def dispatch`, stalling the event loop. The note at `main.py:314` says middleware order is "outermost first", but `add_middleware` makes the last-added outermost, so authentication actually runs first (which is what the limiter needs); the comment is backwards, the behaviour is right.
- Reachable: wired when `app_state.redis_client` is set (`main.py:323`).

Raise to B because of the 429-to-500 behaviour. Fix should return a `JSONResponse` instead of raising.

## AP-15: websocket auth wait has no timeout; tasks not retained

CONFIRMED, severity B stays.

- `api/websocket/routes.py:105-108`: `await websocket.accept()` then `auth_data = await websocket.receive_json()` with no `asyncio.wait_for`. A client that connects and sends nothing holds the socket and handler coroutine indefinitely; transport-level pings do not end it because a plain client answers them.
- `:124-130, :154-170, :190-200`: `asyncio.create_task(...)` results are discarded. CPython keeps only a weak reference to running tasks, so an unreferenced task can be collected before it finishes; the documented fix is to hold them in a set with a done-callback. Exceptions in those tasks are also never retrieved.
- Reachable: every websocket connection takes this path.

Omitted: on a bad first message (`WebSocketAuthMessage(**auth_data)` raising) the code depends on the outer `except` for cleanup; a timeout fix should close with a 1008 code.

## CM-11: connection lock held across a 10 s connect

CONFIRMED for the lock; the leak claim is UNDECIDABLE WITHOUT IRIS. Severity C stays.

- `common/iris_connection.py:29` `_cache_lock = threading.Lock()`; `:258` `with _cache_lock:` encloses the cache validation (`SELECT 1`, `:261-266`), the thread start, and `_t.join(connect_timeout)` (`:323`, default 10 s). A second caller to any host/port/namespace waits on the lock for the whole join, so one hung handshake stalls every connection request in the process. After a timeout, `_handshake_backoff` makes later calls fail fast for 30 s, which bounds but does not remove the stall.
- Leak: after the timeout the daemon thread may finish and store the connection in a local `_result` dict that nothing reads. The dict and connection become garbage when the thread's frame ends; whether the IRIS driver closes a socket on finalization cannot be shown without connecting. Treat it as unproven.
- The `SELECT 1` health check on a cached connection also runs under the lock, so a hung cached connection blocks everyone.

## CM-12: `pipelines.yaml` not found in an installed wheel

CONFIRMED, severity B (ledger C).

- `config/manager.py:645-664`: `project_root = dirname(dirname(dirname(__file__)))` resolves to the repo root from a checkout (`<root>/config/pipelines.yaml` exists). In an installed wheel it resolves to `site-packages/`.
- `pyproject.toml:195-201` ships only `iris_vector_rag/config/*.yaml`; the real file lives in top-level `config/`, outside the package. Checked against the built artefact: `unzip -l dist/iris_vector_rag-0.14.0-py3-none-any.whl` lists `iris_vector_rag/config/default_config.yaml` and no `pipelines.yaml`.
- Effect: `pipeline_config = {}`, so the method returns empty `schema_requirements`, empty `dependencies`, and `enabled: True`. `enabled` is harmless (every entry in the file is `enabled: true`), but `SchemaManager.ensure_pipeline_schema` (`storage/schema_manager.py:2138-2160`) iterates `schema_requirements.tables` and `.indexes`, which are now empty, so it ensures nothing and returns `True`. For GraphRAG the entity and relationship tables the file lists are never created. Nothing is logged.
- Reachable: every installation from PyPI; developers working from a checkout never see it.

Raise to B. The file must move inside the package and be read with `importlib.resources`, and a missing file should raise or warn rather than return defaults.

## CR-14: `Document.__hash__` raises or drifts

CONFIRMED, severity C stays.

- `core/models.py:112-114`: `hash((page_content, tuple(sorted(self.metadata.items())), id))`.
- Repro on the installed package: `Document(page_content="x", metadata={"k": [1]}, id="1")` gives `TypeError: unhashable type: 'list'`; mixed key types (`{"a":1, 2:3}`) give `TypeError` from `sorted`; mutating `metadata["similarity_score"]` after `{doc}` changes `hash(doc)` and `doc in s` becomes `False`.
- Reachable: a search of `iris_vector_rag/` found no `set` of documents, no dict keyed by document, and no call to `hash(doc)`; deduplication uses `doc.id` or `hash(page_content[...])` (`pipelines/multi_query_rrf.py:238-258`, `hybrid_graphrag.py:828-839`). Metadata values that are lists (entities) are common, so any future set/dict use would fail. Latent.

Fix shape: hash on `id` (and content) only, or drop `__hash__` and `frozen`.

## VM-13: MCP parameter pass-through and `KeyError` masking

CONFIRMED, severity B stays.

- `mcp/tool_schemas.py:77-78` starts from `validated = params.copy()` and only checks declared properties, so unknown keys survive. Repro: `validate_params("rag_basic", {"foo": 1, "table_name": "X"}, skip_query=True)` returns `{'foo': 1, 'table_name': 'X', 'top_k': 5, ...}`.
- `mcp/technique_handlers.py:72` then calls `pipeline.query(query=query, **params)`. Pipelines read client-controllable keys from `**kwargs`: `custom_prompt`, `vector_query`, `method`, `label_filter`, `fusion_weights`, `generate_answer`, `min_confidence`, `vector_k`, `text_k`, `rrf_c` (grep of `pipelines/*.py`). JSON cannot supply callables such as `embedding_func`, so this is behaviour steering (including prompt override), not code execution. A client-supplied `query` key in `params` collides with `query=query` and raises `TypeError` (message returned to the client).
- `mcp/bridge.py:76` and `:96`: `handler = registry.get_handler(...)` is in its own `try/except KeyError`, but the whole invocation is inside the outer `try` whose first `except KeyError` (`:96`) answers `"Unknown technique: ..."`. A `KeyError` raised in `handler.execute` therefore masquerades as an unknown technique. `except Exception as e: return {"success": False, "error": str(e)}` (`:98`) returns raw exception text (database messages, paths) to the MCP client.
- Reachable: every MCP technique call.

Omitted: `bridge.py` `health_check` returns hardcoded values (`database.connected: True`, `error_rate: 0.02`, `average_response_time_ms: 500`), which is unrelated to this finding but will mislead anyone fixing error reporting.

## OE-11: ontology loader

CONFIRMED (three sub-claims, different strength), severity B stays.

- `None` labels: `ontology/loader.py:340-341` and `:547-548` add `alt_label.text` unconditionally. Repro with `SKOSLoader` on a file containing `<skos:altLabel/>`: `alternative_labels == {None}`, then `concept.get_all_synonyms()` raises `AttributeError: 'NoneType' object has no attribute 'lower'` (`ontology/models.py:95`). `prefLabel`/`label` with empty text have the same shape (`:324`, `:536`) and crash `self.label.lower()` at `:94`. `hiddenLabel` (`:551-554`) passes `None` to `add_synonym`.
- OWL alt labels: `:340` queries `rdfs:altLabel`, which RDFS does not define; OWL ontologies use `skos:altLabel`, so their alternative labels are silently ignored. Wrong data without an error.
- `ET.parse` on untrusted files (`:261, :488`): internal entity expansion works (repro: a two-level entity macro expanded into a 1000 character label). ElementTree does not fetch external entities, and the bundled Expat is 2.5.0, which has built-in amplification limits, so a billion-laughs file is stopped. The XML point is hardening (`defusedxml`), not an open hole; call it C.

## OE-14: DSPy fallback, embedding timeouts, `[""]`

CONFIRMED with a location correction; the `OLlama` sub-claim is UNDECIDABLE locally. Severity B stays (driven by the `[""]` path).

- `[""]`: it is `dspy_modules/entity_extraction_module.py:356` inside `DirectOpenAILM.__call__` (not `embeddings/manager.py`). When the JSON lacks `choices` it logs an error and returns `[""]`; DSPy then parses an empty completion, so entity extraction yields nothing with no exception. Its HTTP call does have `timeout=60`. The ledger's manager.py line 356 is in the hash-based fallback embedder and is unrelated.
- `OLlama`: `:441` `from dspy import OLlama` sits in the `except` of `dspy.LM(...)`. `dspy` is not installed in the venv and is not declared anywhere in `pyproject.toml` (grep finds no `dspy`; README lines 121 and 145 advertise `iris-vector-rag[dspy]`, an extra that does not exist). Whether the symbol exists in any DSPy release could not be checked offline. It is reached only if `dspy.LM(...)` raises at construction, which it rarely does, so the fallback is effectively dead and the `ImportError` it would raise hides the original failure.
- `embeddings/manager.py:263`: `openai.OpenAI(api_key=api_key)` with no `timeout` (library default 600 s, 2 retries); `embed_texts` (`:265-267`) sends the whole `texts` list in one request with no batching, so a large ingest hits the API's per-request input limit. Reachable when `openai` is the embedding backend.

Omitted: the undeclared `dspy` dependency and the non-existent `[dspy]` extra are the first thing a fix must settle; they are a packaging defect of their own.

## OS-05: string-built SQL in `RAG.SDK`

CONFIRMED as a code pattern; exploitability is not shown, so severity drops from A to C.

- Concatenation is as described: `iris_src/src/RAG/SDK/Search.cls:153` (`"SELECT TOP "_topK_" "_id_col_", "_text_col_...FROM "_table...`) and `:182`; `Bridge.cls:47-55` writes `^RAG.SDK.Config` from `SetDefaultTable` with no identifier check, while `AttachTable` (`:5-20`) does check that the named columns exist. `topK As %Integer` is a declaration, not coercion, when a method is called from ObjectScript or the Native API, so a string flows straight into the statement. `HybridSearch` doubles it (`topK * 2`), which coerces to a number and is safe.
- Trust boundary: `WhereFragment` and `CheckWhereClause` (`Search.cls:98-113`) show the design: `whereClause` is a caller-supplied raw SQL predicate, screened only for `;`, `--`, `/*`. A caller who can pass `whereClause` can already inject anything expressible as a predicate (subqueries included). No method carries `[ SqlProc ]`, no class is a REST or `%CSP` dispatcher (grep of `iris_src`), no Python in this repo calls them, and `module.xml` only installs them. So the methods are reachable only by code already running ObjectScript in that namespace, which can run arbitrary SQL anyway.
- Becomes A only if something downstream (an MCP tool, REST handler, or SQL procedure wrapper) forwards untrusted input to them.

Fix shape is cheap: `topK = +topK` (and a cap), validate identifiers against `^[A-Za-z_][A-Za-z0-9_.]*$` in `SetDefaultTable`, and document that `whereClause` is trusted. Flag the trust boundary in the plan rather than treating this as a security fix.

## OS-10: `embedding_func` flows into `ChatOpenAI`

CONFIRMED, severity B stays.

- `iris_vector_rag/__init__.py:143-147` (and the `engine=` twin at `:95-105`): when `llm_func` is None, `get_llm_func(provider=..., model_name=..., **kwargs)` receives all the pipeline kwargs, which include `embedding_func` (set at `:139-140`). `common/utils.py:331-333` passes them to `ChatOpenAI(model_name=..., openai_api_key=..., **kwargs)`.
- `OPENAI_API_KEY` set is what selects the `openai` provider (`default_provider = "openai" if os.environ.get("OPENAI_API_KEY") else "stub"`); otherwise the stub provider swallows the kwargs and the bug is invisible.
- Repro (no network): `get_llm_func(provider="openai", model_name="gpt-4.1-mini", embedding_func=f)` constructs with a "transferred to model_kwargs" `UserWarning`. Building the same model and calling `m._get_request_payload("hi")` shows `embedding_func` in the payload, and `m.client.create(**payload)` raises `TypeError: Completions.create() got an unexpected keyword argument 'embedding_func'`. So the pipeline builds, and the first answer generation fails.

Omitted: `_llm_instance` is cached under `llm_key = f"{provider}_{model_name}_{cache_enabled}"` (`utils.py:~281`), which excludes kwargs, so a second call with different kwargs silently reuses the first instance. The fix should filter kwargs to a known set rather than forward `**kwargs`.

## OS-12: `EdgeScan.cls` transactions

CONFIRMED in the file, but the file is dead code. Severity stays C.

- `iris_vector_rag/pipelines/colbert_iris/sp/EdgeScan.cls:139` and `:171`: `TSTART` in `BulkIngestEdgesSQL` and `BulkIngestNodesSQL`, `TCOMMIT` at the end, no `Try`, no `TRollback`. An exception inside the loop (for example from `WriteAdjacency`, or from the `$ZConvert` path that has its own empty `Catch`) leaves `$TLevel` raised in the calling process. Inline `&sql` does not throw, so the window is narrow. A failed `&sql` row `Continue`s without rollback, which is intended.
- Reachability: nothing in `iris_vector_rag/` references `EdgeScan`; none of `scripts/deploy_vecindex.sh`, `deploy_plaid.sh`, `deploy_colbert_sp.sh` deploys it; and it is not packaged (OS-13).
- Stale duplicate: IVG 4.1.1 ships a class of the same name, `Graph.KG.EdgeScan` (`.venv/.../iris_vector_graph/iris_src/src/Graph/KG/EdgeScan.cls`). The repo copy differs from it (it still documents `graph = ""` as the default sentinel, which the IVG header calls wrong). IVG's own copy also has `TSTART` at `:443` and `:502` with no rollback, so the same hazard exists upstream.

Fix shape: delete or resync the vendored copy and report the transaction pattern to IVG, not patch this file. See OS-13 for the other thirteen files.

## OS-13: `.cls` files not shipped; `sp/` not referenced

CONFIRMED, severity B stays.

- `pyproject.toml:195-201` `package-data` lists only `config/*.yaml`, `config/*.json`, `templates/*.yaml`, `templates/*.json`; there is no `MANIFEST.in`. In the built 0.14.0 artefacts, `unzip -l` on the wheel and `tar tzf` on the sdist show no `.cls` file at all, including nothing under `colbert_iris/sp/`.
- The ledger's "code does not reference `sp/`" is partly wrong: the Python never does, but three scripts do (`scripts/deploy_vecindex.sh:6`, `deploy_plaid.sh:6`, `deploy_colbert_sp.sh:6`, all via `${REPO_ROOT}/iris_vector_rag/pipelines/colbert_iris/sp`), and `colbert_iris/plaid.py` (`PLAIDSearcher.search_via_sp`) and `vecindex_phase2.py:88` depend on the classes being deployed. Those scripts work only from a checkout.
- Of the 14 vendored files, by comparison with IVG 4.1.1's shipped classes: 4 are identical (`IVFIndex`, `Meta`, `Snapshot`, `VecIndex`), 8 differ (`Algorithms`, `BM25Index`, `EdgeScan`, `NKGAccel`, `PageRank`, `PLAIDSearch`, `Subgraph`, `Traversal`), and 2 exist nowhere else (`ColBERTSearch.cls`, `UserExec.cls`). A pip-installed user therefore has no way to obtain `ColBERTSearch` and `UserExec`, and anyone who does deploy the vendored set over IVG 4.1.1 overwrites same-named IVG classes with older versions.
- `iris_src/` (the `RAG.SDK` classes) is shipped through `module.xml` for ZPM, not the wheel; that is a separate channel and was not changed by this finding.

Fix depends on intent: either ship the two unique classes in the package (and have Python load them), or stop vendoring and document that the ColBERT in-database path needs a checkout. The eight divergent copies should be removed either way.
