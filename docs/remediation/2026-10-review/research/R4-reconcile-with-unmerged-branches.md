# R4: reconcile the ledger with the unmerged branches

Read-only research, 2026-10-02. Source: `git log`, `git show`, `git diff`, `git merge-base`, `git cherry`
and the old-style `git merge-tree` (prints, writes nothing) against the local branches of
`/Users/tdyar/ws/iris-vector-rag-private`. Nothing was checked out, run or executed; "complete" below means
"every task box ticked and the last commit says so", not "tests pass". Finding IDs are those of
[findings.md](../findings.md); decisions are those of the closed tickets in `tickets/`.

Classes used per finding: **(a)** the branch already fixes it, **(b)** the branch changes the surrounding code
so the planned fix or deletion will conflict, **(c)** the branch touches the same file without affecting the
finding. Findings in class (d) (untouched) are not listed.

## Summary

| Branch                           | Ahead of merge base                                                                   | State                                                                                                                                                            | Ledger overlap                                                                                                                                       | Blocks or conflicts with                                                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `083-iris-session`               | 9 (base `0c033d60`; master is 6 ahead of that base, merges cleanly with master)       | Complete by its tasks (58 of 58 ticked), 2026-09-11, no upstream, unpushed. 117 files, +8182 -2170                                                               | (a) CM-04 CM-07 CM-10 CR-08 AP-24 OS-18 ST-21 part; (b) CM-01 CM-02 CM-03 CM-05 CM-16 OS-06; (c) many                                                | Connection settings policy, retire-rest-api (edits 14 `api/` files), the `embedding-integrity` and `mcp-stdio-server` designs (new gate, new dimension reader), 086 (`iris_sql_utils.py`, baseline) |
| `086-canonical-vector-type`      | 3 (base `95c1d558` = master tip)                                                      | Complete by its tasks (56 of 56), 2026-09-14 to 2026-09-16, no upstream, unpushed. 23 files, +2339 -250                                                          | (a) ST-01 ST-20 OS-01 OS-02; (b) ST-02 ST-03 ST-21 (deletion), OS-05; (c) OS-03 OS-15 OS-16 CM-08                                                    | Remove-dead-code decision (modify/delete on `enterprise_storage.py` and its tests, stale DEBT and baseline text), 083 (same `iris_sql_utils.py` hunks, 9 conflict regions)                          |
| `065-iris-llm-substrate`         | 5 by merge base, but 3 unique patches; content already on master as squash commit #21 | Merged in effect. Tip differs from the squash commit only by black/isort formatting plus one file master later deleted. Pushed (`origin/065-iris-llm-substrate`) | None beyond what the review already read (PL/OE/CR code in `graphrag.py`, `hybrid_graphrag.py`, `iris_llm_lm.py` is on master)                       | Nothing; merging it would only reintroduce formatting noise                                                                                                                                         |
| `084-query-plan`                 | 1                                                                                     | Spec and checklist only, no code, no plan or tasks, Draft, 2026-09-11, no upstream                                                                               | No code. Intent overlaps PL-01 PL-17 CR-04 CR-09 VM-13                                                                                               | Nothing now. Its FR-017 ("the HTTP API reads only the unified result") is moot once `api/` is retired                                                                                               |
| `067-colbert-plaid-sp`           | 1                                                                                     | Docs and a benchmark flag; spec marked "Complete, SC-001 not met, superseded by 068". 2026-03-28, no upstream                                                    | None in the package. Context for OS-11 to OS-14 (the `sp/*.cls` it built are declared not production-ready)                                          | Nothing                                                                                                                                                                                             |
| `001-fix-ci-security-failures`   | 5 (base `b24e84a1`, 2025-12-29; master is 266 ahead)                                  | Obsolete. Pushed (`origin/001-fix-ci-security-failures`). 16 files                                                                                               | Touches only `security.yml`, not `ci.yml`: no (a), (b) or (c) for OS-07, OS-08, OS-26; would reintroduce the unpinned `trivy-action@master` of OS-19 | Merging would regress `security.yml` (older action majors, jobs master removed) and resurrect four deleted Dockerfiles                                                                              |
| `060-fix-users-tdyar`            | 7 (base `9e268613`, 2026-01-13; master is 262 ahead)                                  | Fully merged. Tree is byte-identical to master's squash commit `8c41c2dd` (#20). No upstream                                                                     | None new. It is the origin of the stub embedder (OE-01) and `_normalize_vector_data`, already on master                                              | Nothing                                                                                                                                                                                             |
| `backup/083-pre-rebase-20260911` | 5 (base `0be6d1c1`)                                                                   | Context only: the first five commits of 083 before it was rebased and extended by four                                                                           | Superseded by `083-iris-session`                                                                                                                     | Nothing                                                                                                                                                                                             |

Why 065, 001 and 060 have strange bases: master was rewritten by squash merges, so git sees no ancestry. `git cherry`
marks every commit "+" (not found), but the trees say otherwise. For 060, `git diff 8c41c2dd 060-fix-users-tdyar` is empty.
For 065, `git diff 92090bde 065-iris-llm-substrate` is 14 files of formatting (`92090bde` is master's squash commit, #21).
That is why the diff from the merge base looks large (399 files for 060) and is meaningless.

Textual conflicts, measured with `git merge-tree`: 083 against master none; 083 against 087 one (`CHANGELOG.md`); 086 against 087
one (`CHANGELOG.md`); 083 against 086 twelve regions in `CHANGELOG.md`, `iris_vector_rag/common/iris_sql_utils.py` (9) and
`scripts/gates/antipatterns-baseline.txt` (2). The semantic conflicts below are more serious than the textual ones.

## 083-iris-session

**Purpose.** Spec 083: one `IRISSession` owns identity, driver choice, connection, health cadence and reconnect, so the
"eleven ways to get a connection" each lose their private subsets of protection. Commits: spec, plan, two feature commits
(`5c3524d2` session, `cbe393bd` per-thread reaping), docs, then `ca6fd368` (define each vector-store exception once) and
two test-config commits (`b81bb30a`, `727c7f4320`). All on 2026-09-11, local only, no upstream. New package
`iris_vector_rag/session/` (6 modules, about 1,260 lines), `testing/fake_driver.py`, 15 new unit test files, a contract and
an e2e test, CHANGELOG (+94), `docs/api/session.md`, 7 spec documents.

**Files by area.**

- `iris_vector_rag/`: `session/*` (new); `common/` connection modules rewritten (`connection_manager.py`, `connection_pool.py`
  506 lines down to 208, `iris_connection.py` 340 lines down to 190, `iris_connection_manager.py`, `iris_dbapi_connector.py`,
  `iris_sql_utils.py`, `llm_cache_manager.py`), `common/iris_connector.py` deleted; `core/connection.py`, `core/engine.py`,
  `core/exceptions.py`, `exceptions.py`; `pipelines/basic.py`, `crag.py`, `graphrag.py`, `hybrid_graphrag.py`,
  `colbert_iris/plaid.py`, `colbert_iris/vecindex_phase2.py`; 14 `api/` files (`main.py`, all of `models/`, `websocket/`).
- Config and gates: `pytest.ini` deleted, its content moved into `pyproject.toml` `[tool.pytest.ini_options]`;
  `scripts/gates/antipatterns.py` gains an `unguarded-dbapi-connect` rule (only `session/drivers.py` may call a driver's
  `connect`); baseline loses `inert-pytest-ini` and `unguarded-dbapi-connect connection_pool.py:109`.
- Not touched: `iris_src/`, `.github/workflows/`, `config/manager.py`, `common/utils.py`, `storage/`, `embeddings/`, `mcp/`, `services/`.
- Tests: about 45 files (fixtures, plugins, contract, integration), of which many are mechanical for the new names.

**Read diffs that matter.**

- `iris_connection.py` keeps `auto_detect_iris_port()` (the `docker ps` scan of CM-01) and `_get_connection_params_from_env()` calls it
  whenever `IRIS_PORT` is unset. `_session_connection()` calls that function even when the caller passed a port, so CM-16 stands.
  `session/identity.py` `_env_defaults()` repeats the same scan and the defaults (`localhost`, `_SYSTEM`, `SYS`, `IRIS_USER` before
  `IRIS_USERNAME`). `SessionIdentity.from_config()` reads `config_manager.get_database_config()`.
- `_hard_fix_iris_passwords` is gone, but `session/drivers.py` `NetworkDriver.recover_password()` and `_container_publishing()`
  re-implement it, including the CM-02 match: `if str(port) in line and "tcp" in line`. The session runs it once per identity
  (documented in the CHANGELOG: "Expired-password recovery is attempted once per identity").
- `common/connection_pool.py` is now a semaphore over per-thread session connections; `initialize_pool`, `get_pool`, `close_pool`,
  `_global_pool` and `queue.get(timeout=30)` no longer exist. `api/main.py` swaps `IRISConnectionPool(...)` for `_session_pool(...)`
  and replaces the nonexistent `connection_pool.dispose()` with `close_all()`. The API keeps the pool "until 0.16" (spec FR-017).
- `api/` edits beyond that: `ConnectionManager` renamed `WebSocketConnectionManager`, Pydantic `class Config` to `ConfigDict`,
  import sorting. No behaviour change in auth, rate limit, routes, services.
- `core/connection.py` `ConnectionManager` is now a thin adapter over `session_for(config_manager)`; `invalidate_connection`,
  `close_connection` and `close_all_connections` close real things.
- `core/engine.py` `embedding_dimension` no longer defaults to 1536; it reads `embeddings.dimension`, `embedding_model.dimension`, then
  `get_embedding_config()["dimension"]` (a fourth reader for the dimension, relevant to the `embedding-integrity` decision).
- `pipelines/basic.py` fixes a defect not in the ledger: `getattr(connection_manager, "connection", None)` was always `None`, so every
  non-vector retrieval mode ran without a connection.
- `pyproject.toml` pytest block: `--strict-markers --strict-config`, `filterwarnings = ["error", ...]`, `timeout = 300`,
  `timeout_method = "thread"`, and 29 registered markers. The `pytest-timeout` package is not declared by this branch; on the reviewed
  commit it is only in `[dependency-groups]` (OS-06), so with `--strict-config` the `timeout` key is an error wherever the plugin is missing.

**Findings.**

- (a) Already fixed: CM-04 (pool class deleted; the replacement blocks only when all slots are taken), CM-07 (`initialize_pool` and
  `_global_pool` deleted), CM-10 (`iris_connector.py` deleted, the nonexistent JDBC import with it), CR-08 (`_connections` dict gone, close
  methods act on the session), AP-24 (`dispose()` to `close_all()`), OS-18 (`pytest.ini` deleted; the pyproject block is now the live config,
  pinned by `tests/unit/test_pytest_config_is_live.py`; the Python 3.10 classifier half of OS-18 remains), ST-21 third item only
  (`core/exceptions.py` re-exports the one `VectorStoreCLOBError` from `exceptions.py`). Narrowed: CM-11 (the global `_cache_lock` is gone;
  the per-session `_connect_lock` is held through a connect), CM-13 (`iris_connector.py` and `get_iris_connection_old` gone, the
  two unrelated `IRISConnectionPool` classes are now one; the deprecated wrappers stay), CM-14 (only the `get_iris_connection_old` part).
- (b) Fix or deletion will conflict or must be re-aimed: CM-01, CM-02, CM-16 (the code the Connection settings decision removes has moved to
  `session/identity.py` and `session/drivers.py`, and the scan is now shared by two call paths), CM-05 (one of the two precedence sites moves
  into `SessionIdentity`), CM-03 (the `connection_pool.py:380` default site is gone, new default sites appear in `SessionIdentity`),
  OS-06 (new `timeout` key and strict config widen the failure), AP-01 AP-04 AP-05 AP-09 AP-15 AP-16 AP-17 AP-20 AP-23 (the `api/` files
  the retirement deletes are edited here, so a deletion done first produces modify/delete conflicts; see consequences).
- (c) Same file, unaffected: CM-08 (`iris_sql_utils.py` still catches everything and returns `False`), PL-11 PL-12 (`crag.py`, only the
  `LegacyConnectionManager` class rename), PL-06 (`graphrag.py`, only `_generate_answer` gains `custom_prompt`), PL-09 PL-10
  (`hybrid_graphrag.py`, only the connect call), PL-15 (`plaid.py`, `_ensure_native_conn` only), CR-16 (`core/engine.py`), OS-13 OS-22 OS-24
  (`pyproject.toml`, other hunks).

**Glossary.** `CONTEXT.md` on master already defines **Session** and **Pool** ("exists only where callers are concurrent", "avoid: connection
manager, connection wrapper, external connection, engine connection"). The glossary describes code that exists only on this branch. On master
`ConnectionManager`, `IRISConnectionManager` and the pool still are the forbidden terms.

## 086-canonical-vector-type

**Purpose.** Spec 086: declare one canonical vector element type (`FLOAT`) in every table-creating path, add a `hardcoded-vector-type`
gate detector, and stop `IRISStorage.initialize_schema` from writing its own DDL. Three commits, 2026-09-14 to 2026-09-16, forked from the master
tip, local only. 23 files. All 56 tasks ticked; the spec header still says Draft.

**Files.**

- `iris_src/src/RAG/SDK/`: `Pipeline.cls` (84 lines changed), `Schema.cls`, `Search.cls` (comment only), new `Embed.cls`.
- `iris_vector_rag/`: `storage/enterprise_storage.py`, `common/iris_sql_utils.py`, `common/db_init_complete.sql`; also `sql/schema.sql`,
  `scripts/deploy_sdk.sh`, `scripts/gates/antipatterns.py` and baseline (24 `hardcoded-vector-type`, 10 more `inline-standard-ddl` lines).
- Docs and tests: `CHANGELOG.md`, `docs/DEBT.md` (entries 3, 4, 5 rewritten), 4 spec documents,
  `tests/test_enterprise_storage.py` (edited), new `tests/integration/test_enterprise_storage_no_data_loss.py`,
  `tests/e2e/test_sdk_canonical_vector_type.py`, `tests/objectscript/.../PipelineTest.cls`, `tests/test_iris_sql_utils_connection_ownership.py`,
  `tests/unit/test_antipatterns_detectors.py`.

**Read diffs that matter.**

- `enterprise_storage.py`: only `initialize_schema` changes. The `CREATE TABLE`, `DROP TABLE` and `DELETE FROM` are removed; it calls
  `SchemaManager.ensure_table_schema(unqualified_name)` and raises if that fails, and no longer reassigns `self.table_name`. Nothing
  else in the file changes: `store_documents` still probes with `SELECT COUNT(*) ... WHERE 1=0` and calls `initialize_schema()` on any
  exception; `retrieve_documents_by_ids` still filters `WHERE id IN (...)` (`:376`); `TO_VECTOR(?, FLOAT, 384)` stays at `:254,:290,:436`;
  `JSON_EXTRACT(metadata, '$.{key}')` with an unvalidated key stays at `:447,:517`. I checked `ensure_table_schema` on master: for an
  existing document table it only caches and returns (feature 082, FR-007), so the new call cannot reach the destructive `_migrate_*` of ST-04.
- `Pipeline.cls`: the `-119` branch now calls a new `UpdateDocument` instead of the undefined `st2`; the vector dimension is `$Length(embedding, ",")`
  and the element type comes from `Search.GetVectorType` of the target column; every non-zero SQLCODE returns `{"status":"error","message":rs.%Message}`;
  `AddDocument`, `AddDocumentBatch` and `AddDocumentWithEmbed` gain a `table` argument that is concatenated into the SQL
  (`"INSERT INTO "_table_...`), unvalidated. The embedded-Python method moves to `Embed.cls` because it cannot compile on this container.
- `Schema.cls`: `Initialize(dim, schema)` and `Drop(schema)` take a schema, the DDL uses `VECTOR(FLOAT, dim)`. Unchanged: the split on `"GO"`,
  `If $$$ISERR(sc) Continue` (errors swallowed), `Set version = "0.11.4"` in `Install`.
- `iris_sql_utils.py`: `check_table_exists`, `check_index_exists` and `get_table_columns` no longer close the connection returned by
  `get_iris_dbapi_connection()` (it was the shared cached one, which killed the caller's connection; a defect not in the ledger). Same functions
  are rewritten by 083 (which renames the import to `_session_connection`).

**Findings.**

- (a) ST-01 (the drop/empty path is gone; the probe-on-any-exception remains but is now harmless), ST-20 (the second, non-canonical `CREATE TABLE` is gone;
  reads and writes still use their own column names), OS-01 (undefined `st2`, hardcoded 384 and `DOUBLE`, unchecked statuses), OS-02 (failed INSERT reported as ok).
- (b) Conflicts with planned deletions or fixes: ST-01 ST-02 ST-03 ST-20 and the enterprise half of ST-21 (all `deleted-with:enterprise_storage`; 086 edits the
  file and its test, see consequences); OS-05 (the new `table` argument is a new concatenation site in the SDK that the OS-05 fix must cover).
- (c) Same file, unaffected: ST-02 ST-03 (left exactly as reviewed), OS-03 (`Search.cls`, comment only), OS-15 and OS-16 (`Pipeline.cls`/`Schema.cls`: the
  new code keeps the bare `st`/`sc` style; `Initialize` still swallows every error, splits on "GO", hardcodes 0.11.4), CM-08 (`iris_sql_utils.py`).

## 065-iris-llm-substrate

**Purpose.** Spec 065: `iris_llm` as the LLM substrate (`SqlExecutor` protocol, `IrisLLMDSPyAdapter`, `GraphRAGToolSet`). 2026-02-21 to 2026-02-28, pushed to `origin`,
`public` and `upstream`. The merge base `4c91bdbc` (2026-02-16) is old only because master was squash-merged: master holds this work as `92090bde` (#21) and
`iris_vector_rag/executor.py`, `tools/graphrag.py` and `dspy_modules/iris_llm_lm.py` on master are byte-identical to the branch tip. The 24-file diff from the
merge base is therefore a diff against a stale base. The real remainder (`git diff 92090bde 065-iris-llm-substrate`) is black/isort formatting in 13 files and
`common/iris_globals.py`, which master deleted after the squash.

**Findings.** None in classes (a), (b) or (c) beyond what the review already saw: the ledger's PL-06 PL-09 PL-10 (`graphrag.py`, `hybrid_graphrag.py`), OE-06
(`iris_llm_lm.py:59-66`), OE-05 OE-12, OS-10 and OS-23 (`__init__.py`, `common/utils.py`) describe master's copy of this branch. The branch also carries older
`CLAUDE.md`, `CONTRIBUTING.md` and `.specify/memory/constitution.md` content that master has since superseded.

## 084-query-plan

**Purpose.** Spec 084: one normalized `QueryPlan` and one `QueryResult` for every pipeline (same key set, generation failure reported as an error instead of an
"Error generating answer" string, unknown options refused, answer built after ranking and reranking). One docs commit (spec and checklist, 313 lines), 2026-09-11,
no plan, no tasks, no code, no upstream.

**Findings.** No code overlap. Intent overlap only, to be considered when the plan is grouped: PL-01 (a failed rerank still reports `reranked=True`), PL-17 (repeated
"answer or fallback" branching), CR-04 (`retrieve()` returns the wrong type), CR-09 (the dropped `metadata_filter`), VM-13 (unknown keys copied into
`pipeline.query(**params)`, assigned to `mcp-stdio-server`). If 084 is later built, those findings would be fixed by it or collide with their own fixes; nothing
to do now. Its FR-017 and its wording on "the HTTP API" depend on `api/`, which the `api/` decision deletes.

## 067-colbert-plaid-sp

**Purpose.** Spec 067: PLAID ColBERT as an in-database stored procedure. The implementation (`b42d2704`) is already on master. The branch holds one later commit
(2026-03-28): a `--skip-sp` benchmark flag, `benchmark_results_sp.json`, and spec/research edits recording that the SP p50 (490 ms) is 1.25 times slower than Phase 2
HNSW (391 ms), SC-001 not met, superseded by 068. No upstream, no package files.

**Findings.** None in (a), (b) or (c). Context only: the spec says `pipelines/colbert_iris/sp/*.cls` is not production-ready, which bears on the open
fates of OS-11 to OS-14 (whether to fix or drop those classes).

## 001-fix-ci-security-failures

**Purpose.** Jan 2026 attempt to make the security workflow pass: hardening Dockerfiles, fixing `security.yml` triggers, then disabling CodeQL and other jobs with
`if: false`. Five commits, 2025-12-29 to 2026-01-03, pushed to `origin`. Touches `.github/workflows/security.yml`, five Dockerfiles, `Dockerfile.mcp`,
`AGENTS.md`, `.specify/model-routing.yaml`, 7 spec documents.

**Comparison with master.** `security.yml` on master has five jobs (secret-scan, infrastructure-scan, compliance-check, security-scorecard, security-policy-check);
the branch has eleven, three of them disabled, and older action majors (`checkout@v4`, `gitleaks-action@v2`, `setup-uv@v4`, `upload-artifact@v4`), an unpinned
`trivy-action@master`, and `uv sync --all-extras` (the extra set the `api/` and `mcp/` decisions change). `docker/api`, `docker/data-loader`, `docker/jupyter` and
`docker/nginx` Dockerfiles do not exist on master and would be resurrected; `docker/base/Dockerfile` is already identical to master; the `Dockerfile.mcp` change
(non-root user) is on a file the `mcp/` decision deletes. `AGENTS.md` on the branch is 347 lines shorter than master's.

**Findings.** OS-07, OS-08, OS-26 (`ci.yml`): not touched; no help, no conflict. OS-19 (`release.yml`): not touched, but the branch's `security.yml` would add a
second `trivy-action@master`. No (a) or (b).

## 060-fix-users-tdyar

**Purpose.** Not a feature branch: the name is a mangled spec directory (`specs/060-fix-users-tdyar`). Seven commits of 2026-02-16: "core v0.5.4 bug fixes"
(connection API, `_normalize_vector_data`, idt cleanup, stub fallback embedder), repo-wide ruff cleanup, v0.5.4 CHANGELOG and specs. Master holds it as `8c41c2dd`
(#20) and the trees are identical. No upstream, no reason to keep.

**Findings.** None new. It is the origin of code the ledger criticizes on master (the stub embedder of OE-01, the vector normalization neighbourhood of ST-23);
there is no separate version of it to reconcile. Its many overlaps with 083 (connection modules, pytest config, 44 contract tests) are the 2026-02 state of files 083
rewrote from the current master.

## Consequences for closed decisions

Both 083 and 086 are complete, unpushed, and would land as a "0.15" ahead of or with the remediation. Order matters as follows.

**Enterprise storage deletion (Decide the fate of enterprise storage and the devtools bridge).** The decision stands, but:

- If 086 merges first, deleting `storage/enterprise_storage.py` and `tests/test_enterprise_storage.py` is a plain delete of files 086 last edited, and the decision's
  file list is incomplete: 086 adds `tests/integration/test_enterprise_storage_no_data_loss.py`, which must go too. The decision's housekeeping is stale: DEBT entry 3
  is rewritten by 086 (six failures, not seven) and entry 4 no longer lists eleven sites (it points at 24 baselined sites in
  `scripts/gates/antipatterns-baseline.txt`, three of them `enterprise_storage.py:254,290,436`). The baseline gate fails on stale entries, so the deletion must also remove those
  three baseline lines. 086's Unreleased CHANGELOG bullet about `IRISStorage.initialize_schema` would announce a change to a class removed in the same release.
- If the deletion merges first, 086 hits a modify/delete conflict in two files; resolution is to drop 086's commit `ee88b010` (the `enterprise_storage.py` part), keep
  the SDK, SQL, gate and `iris_sql_utils.py` parts.
- ST-01, ST-02, ST-03, ST-20 keep their fate `deleted-with:enterprise_storage`. The argument "waiting a release would ship a known data-loss path" is weakened if 086
  ships in the same release, since 086 removes the destructive path; the deletion is still justified by "duplicate, no caller".

**Retire the REST API (Decide the fate of `api/`).**

- The "patch nothing" rule is already broken by 083 on 14 `api/` files (rename `ConnectionManager` to `WebSocketConnectionManager`, `ConfigDict`, pool swap, the AP-24 fix).
  If 083 merges first, the design pack and the `archive/rest-api-v1` tag should be cut from the post-083 code (new names, fewer broken findings: AP-24 fixed).
  If the retirement merges first, 083 has modify/delete conflicts in 14 `api/` files; resolve by deleting them, and `common/connection_pool.py` plus
  `tests/unit/session/test_api_pool.py` (135 lines) and spec FR-017/FR-018 lose their only consumer.
- `common/connection_pool.py` (as rewritten by 083, "removed in 0.16") exists only for `api/`; the decision's step-5 sweep list does not name it. It belongs in the sweep.
- 083 sets `filterwarnings = ["error", ...]` with a per-deprecation allow-list. The deprecation release of `api/`, and the import warning for `ontology/`, will fail any test
  that imports those packages unless each gets an `ignore:` entry (the file says "delete the entry when the API is removed").

**Connection defaults policy (Decide the connection defaults policy).** The feature `connection-settings` is written against code 083 relocates.

- "No Docker scanning" and "the password-fix helper is removed" now apply to `common/iris_connection.py` (`auto_detect_iris_port`), `session/identity.py` (`_env_defaults`) and
  `session/drivers.py` (`recover_password`, `_container_publishing`, which carries the CM-02 substring match). 083 documents and tests expired-password recovery
  (`PasswordExpired` error class, CHANGELOG line, `test_session_recovery.py`, `test_driver_internals.py`); the decision removes that behaviour. That is a direct conflict the
  maintainer has to settle: remove it in 083 before merge, or amend the decision.
- "One resolver" is best built as `SessionIdentity` (it already resolves config over environment); the decision's "env beats file" and devtester variable names apply there.
  083 orders `IRIS_USER` before `IRIS_USERNAME`; the decision wants devtester's names with `IRIS_USER` as a warned alias.
- G5's refusal hint has a natural home in `session/errors.py` (`wrap_connect_failure`, "every error names the instance and namespace"); without 083 there is no such single place.
- If `connection-settings` is implemented on master first, 083 must be redone on top of it (both rewrite `iris_connection.py`).

**Embedding integrity (Decide the fake-embedding fallback policy).** Compatible, but: 083's `engine.embedding_dimension` becomes a fourth reader of the dimension (after the
"one dimension authority" is created it must replace `_configured_dimension`); the "one warning per process" from the test embedder will become a test failure under
`filterwarnings = error` unless it is a log line or filtered.

**MCP (Decide the fate of `mcp/`).** The health check "IRIS reachable" and any connection in the new server must go through the session: 083's `unguarded-dbapi-connect`
rule fails the gate for a direct `connect` outside `session/drivers.py`.

**CI (Decide the CI repair scope, still open).** None of the branches touches `ci.yml`, `pr-integration.yml` or `release.yml`. 083's pytest config change affects what CI
runs (strict markers, warnings as errors), so the CI repair should be planned against it.

**Base branch (Decide the base branch, still open).** 083 and 086 are independent of 087 except `CHANGELOG.md`; all three forked from the master lineage.
`087-ivg-4.1` and `086` are tip-of-master based; 083 is six commits behind.

## Recommendation (input to the maintainer, not a decision)

1. **Abandon `060-fix-users-tdyar` and `065-iris-llm-substrate`.** Both are squash-merged on master (#20, #21); 065's residue is formatting and a file master removed.
   `067-colbert-plaid-sp` is one docs commit; keep only if the SC-001 record matters, otherwise abandon. `backup/083-pre-rebase-20260911` is superseded.
2. **Abandon `001-fix-ci-security-failures`.** Its content is already on master in newer form; merging regresses `security.yml`. Its spec documents may be archived.
3. **Keep `084-query-plan` parked** as a spec; do not merge it before the remediation features that touch `pipelines/*.py` are done, and amend its FR-017 for the `api/` decision.
4. **Merge `086-canonical-vector-type` first** (it is on the master tip, mostly independent, its SDK and gate work is unrelated to the remediation). Before that, consider dropping
   its `enterprise_storage.py` and `test_enterprise_storage*.py` parts so the deletion is clean and the baseline, DEBT and CHANGELOG do not need rework. It closes OS-01 and OS-02
   early, and its `hardcoded-vector-type` gate protects the `embedding-integrity` work.
5. **Merge `083-iris-session` second, before any of `connection-settings`, `retire-rest-api`, `mcp-stdio-server` and `embedding-integrity`**, after resolving the password-recovery
   conflict and the `iris_sql_utils.py` overlap with 086 (both close the shared connection; keep 083's `_session_connection` name and port 086's ownership test to it). Reason: it
   relocates the code those features change, and it closes CM-04, CM-07, CM-10, CR-08, AP-24, OS-18 on its own. Alternative if the password-recovery decision cannot be settled: do not
   merge 083 and make the `connection-settings` feature against master, accepting that 083 is then rewritten on top.
6. Run the remediation features after both, or on 087 if 087 lands first (its conflicts are `CHANGELOG.md` only).

## Ledger corrections

Proposed statuses. "Already fixed on branch X" means fixed in the branch, not on master; the finding stays open on master until the branch merges.

- **ST-01**: note "destructive path removed on 086 (`ee88b010`); fate `deleted-with:enterprise_storage` unchanged".
- **ST-02, ST-03**: no change; note that 086 edits the same file (modify/delete with the deletion).
- **ST-20**: note "`CREATE TABLE` removed on 086; moot by deletion".
- **ST-21**: third item (two `VectorStoreCLOBError` classes) already fixed on 083; enterprise item still goes with the deletion; the other duplication stays.
- **CM-01, CM-02, CM-16**: keep fate `connection-settings`; re-locate after 083: `common/iris_connection.py:64-98`, `session/identity.py:22-36`, `session/drivers.py:288-330`
  (CM-02's substring match reappears in `_container_publishing`). CM-16: still true on 083 (`_env_defaults()` runs on every call).
- **CM-03, CM-05**: keep fate; 083 removes the `connection_pool.py` default site and adds `SessionIdentity` as a site; CM-05 precedence disagreement persists between
  `SessionIdentity` and `config/manager.py`.
- **CM-04, CM-07, CM-10, CR-08**: already fixed on 083; re-verify against the branch, then set fate "fixed by 083".
- **CM-11, CM-13, CM-14**: narrowed by 083 (lock scope per session; wrapper modules partly removed; `get_iris_connection_old` removed); re-verify.
- **CM-08**: unchanged on both branches (both edit the function); keep open.
- **AP-24**: already fixed on 083 (moot under `deleted-with:api`). The other AP findings keep their fate; 083 edits 14 files of `api/`.
- **OS-01, OS-02**: already fixed on 086 (`Pipeline.cls`); re-verify against the branch; OS-01 also lost its hardcoded 384 and `DOUBLE`.
- **OS-05**: pattern widened on 086 (`table` argument concatenated into SQL in `Pipeline.cls`); include in the OS-05 fix.
- **OS-03, OS-04, OS-15, OS-16**: unchanged on 086 (`Schema.cls` still swallows errors, splits on "GO", hardcodes 0.11.4).
- **OS-06**: re-verify against 083: it adds `timeout = 300` with `--strict-config`, and the `pytest-timeout` dependency is not in the `dev` extra.
- **OS-18**: half fixed on 083 (`pytest.ini` deleted, config live); the Python 3.10 classifier half remains. `docs/DEBT.md` entry 8 and the baseline line `inert-pytest-ini` go stale on merge.
- **OS-07, OS-08, OS-19, OS-26**: no branch changes them.
- **PL, OE and CR findings in `graphrag.py`, `hybrid_graphrag.py`, `dspy_modules/iris_llm_lm.py`**: no change; 065 is already in master.
- **Candidate new findings, both fixed on the branches, for the record**: `common/iris_sql_utils.py` closed the shared cached connection in three helpers (fixed on 083 and 086,
  conflicting); `pipelines/basic.py` read a `connection_manager.connection` attribute that never existed, so non-vector retrieval ran without a connection (fixed on 083);
  `GraphRAGPipeline._generate_answer` took no `custom_prompt` although the hybrid subclass forwards one, per the 083 docstring (fixed on 083, not traced by me); two parallel exception hierarchies (ST-21's third item, more broadly, fixed on 083);
  `embedding_dimension` defaulting to 1536 (fixed on 083).
