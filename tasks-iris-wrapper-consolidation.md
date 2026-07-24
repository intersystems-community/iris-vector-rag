# Tasks: iris-embedded-python-wrapper Consolidation

Consolidate all IRIS connection logic through `iris_connection.py::get_iris_connection()`,
remove duplicate/bypass paths, fix import hazards, and add embedded-mode support.

**Source analysis**: See conversation — items 1–5 derived from package inspection.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Parallelizable (different files, no inter-task dependency)
- File paths relative to repo root

---

## Phase 1 — Tests (write before touching production code)

- [ ] T001 [P] Write unit tests for `get_iris_connection()` embedded-mode branch in `tests/unit/test_connection_api.py`: assert that when `IRISINSTALLDIR` is set and TCP is unreachable, `iris.dbapi.connect(path=IRISINSTALLDIR)` is called (mock `iris.dbapi.connect` and `socket.connect`)
- [ ] T002 [P] Write unit tests for lazy import in `iris_vector_rag/common/connection_pool.py`: assert that importing the module does NOT raise even when `iris.dbapi` is unavailable (patch `builtins.__import__` or use `importlib`)
- [ ] T003 [P] Write contract test asserting `get_iris_connector_for_embedded` is NOT exported from `iris_vector_rag/common/utils.py` (import the module, assert `not hasattr(utils, "get_iris_connector_for_embedded")`) in `tests/contract/test_iris_connection_contract.py`
- [ ] T004 [P] Write unit test for `_ensure_native_conn()` in `iris_vector_rag/pipelines/colbert_iris/plaid.py` and `vecindex_phase2.py`: assert it calls `get_iris_connection()` not `intersystems_iris.createConnection` when `intersystems_iris.IRISConnection` is unavailable, in `tests/unit/test_connection_api.py`
- [ ] T005 Write unit test for `hybrid_graphrag.py` connection path: assert `get_connection()` calls `get_iris_connection()` (mock `iris_connection.get_iris_connection`), in `tests/unit/test_connection_api.py`

---

## Phase 2 — Item 4: Fix top-level import hazard in connection_pool.py

Do this first — it breaks imports on any machine without `iris` installed.

- [ ] T006 In `iris_vector_rag/common/connection_pool.py` lines 19–26: remove the top-level `try: import iris.dbapi as iris_dbapi` block entirely; move `import iris` inside `IRISConnectionPool._create_connection()` (lazy), replacing `iris_dbapi.connect(...)` with `iris.dbapi.connect(...)`
- [ ] T007 Run `pytest tests/unit/test_connection_api.py -x` — T002 must pass

---

## Phase 3 — Item 1: Remove get_iris_connector_for_embedded() from utils.py

- [ ] T008 In `iris_vector_rag/common/utils.py`: delete lines 443–493 (comment `# ... (Embedded Python...)`, globals `_iris_connector_embedded`, `_embedding_model_embedded`, `_llm_embedded`, and the full `get_iris_connector_for_embedded()` function body). Keep `get_embedding_func_for_embedded()` and `get_llm_func_for_embedded()` — they have no equivalent in `iris_connection.py`.
- [ ] T009 Run `grep -rn "get_iris_connector_for_embedded" iris_vector_rag/ tests/` — must return zero hits (confirms no callers remain)
- [ ] T010 Run `pytest tests/contract/test_iris_connection_contract.py -x` — T003 must pass
- [ ] T011 Run `pytest tests/unit/ -x` — full unit suite must be green

---

## Phase 4 — Item 2: Replace hasattr fan-out in hybrid_graphrag.py

- [ ] T012 In `iris_vector_rag/pipelines/hybrid_graphrag.py` around lines 120–155: replace the `if hasattr(_iris_mod, "dbapi")... elif hasattr(_iris_mod, "createConnection")... else...` block with a single call: `from iris_vector_rag.common.iris_connection import get_iris_connection; iris_connection = get_iris_connection(host=str(host), port=int(port), namespace=str(namespace), username=str(username), password=str(password))`. Remove the now-unused `import iris` and `import iris as _iris_mod` lines in that block.
- [ ] T013 Run `pytest tests/unit/test_connection_api.py::test_hybrid_graphrag_uses_get_iris_connection -x` — T005 must pass

---

## Phase 5 — Item 3: Route colbert_iris through get_iris_connection()

- [ ] T014 [P] In `iris_vector_rag/pipelines/colbert_iris/plaid.py`: rewrite `_ensure_native_conn()` — remove `import intersystems_iris` and the `isinstance(conn, intersystems_iris.IRISConnection)` guard; replace the fallback `intersystems_iris.createConnection(...)` call with `from iris_vector_rag.common.iris_connection import get_iris_connection; return get_iris_connection(host=..., port=..., namespace=..., username=..., password=...)` using the same env-var lookups already present
- [ ] T015 [P] In `iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py`: same rewrite of `_ensure_native_conn()` as T014
- [ ] T016 Run `pytest tests/unit/test_connection_api.py -k "ensure_native_conn" -x` — T004 must pass
- [ ] T017 Run `grep -rn "intersystems_iris.createConnection\|intersystems_iris.IRISConnection" iris_vector_rag/` — must return zero hits

---

## Phase 6 — Item 5: Embedded-mode support in get_iris_connection()

- [ ] T018 In `iris_vector_rag/common/iris_connection.py`: add embedded-mode detection at the top of `get_iris_connection()`, before the TCP socket probe. Logic:
  1. Check `iris.runtime.get().state` — if `"embedded-kernel"` or `"embedded-local"`, call `iris.dbapi.connect(namespace=n)` (no host/port) and return immediately (skip TCP probe and cache key)
  2. If `IRISINSTALLDIR` env var is set and `iris.runtime.get().state == "unavailable"`, call `iris.runtime.configure(mode="embedded", install_dir=os.environ["IRISINSTALLDIR"])` then `iris.dbapi.connect(namespace=n)`
  3. All existing TCP path logic (socket probe, cache, retry) is unchanged and still runs when neither embedded condition applies
- [ ] T019 In `iris_vector_rag/common/iris_connection.py`: update `_get_iris_dbapi_module()` docstring and inline comment to reflect that `iris.runtime` is now used for embedded detection — no code change needed to the function itself
- [ ] T020 Run `pytest tests/unit/test_connection_api.py -k "embedded" -x` — T001 must pass
- [ ] T021 Run full unit suite: `pytest tests/unit/ -x`
- [ ] T022 Run contract suite: `pytest tests/contract/test_iris_connection_contract.py tests/contract/test_common_imports.py -x`

---

## Phase 7 — Validation & Cleanup

- [ ] T023 Run `pytest tests/unit/ tests/contract/ -x --tb=short` — full suite must be green
- [ ] T024 Run `grep -rn "intersystems_iris.createConnection" iris_vector_rag/` — must return zero hits
- [ ] T025 Run `grep -rn "import iris\.dbapi" iris_vector_rag/` — must return zero hits (only lazy `import iris` then `iris.dbapi.connect`)
- [ ] T026 [P] Run `black iris_vector_rag/common/iris_connection.py iris_vector_rag/common/connection_pool.py iris_vector_rag/common/utils.py iris_vector_rag/pipelines/hybrid_graphrag.py iris_vector_rag/pipelines/colbert_iris/plaid.py iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py`
- [ ] T027 [P] Run `isort iris_vector_rag/common/iris_connection.py iris_vector_rag/common/connection_pool.py iris_vector_rag/common/utils.py iris_vector_rag/pipelines/hybrid_graphrag.py iris_vector_rag/pipelines/colbert_iris/plaid.py iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py`

---

## Dependencies

```
Phase 1 (tests) → all phases gate on their paired tests
Phase 2 (pool import fix) → independent, do first
Phase 3 (remove duplicate) → after Phase 2
Phase 4 (hybrid_graphrag) → after Phase 3
Phase 5 (colbert) → after Phase 3, parallel with Phase 4
Phase 6 (embedded mode) → after Phase 3
Phase 7 (validation) → after all phases
```

## Parallel opportunities

- T001, T002, T003, T004 — all test writes, different files
- T014, T015 — two colbert files, independent
- T026, T027 — formatting, independent

## Total: 27 tasks across 7 phases
