# Tasks: IRIS Session

**Feature**: 083-iris-session | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

Phases follow the Phase Plan table in `plan.md`. Tests come first inside every phase,
and each user-story phase ends with an end-to-end gate that must pass before the next
phase begins.

## Phase 1: Setup

- [x] T001 Create the session package skeleton with its public exports in `iris_vector_rag/session/__init__.py` (FR-001)
- [x] T002 [P] Create the unit test package for the session in `tests/unit/session/__init__.py`
- [x] T003 [P] Add the session tuning defaults (`IRIS_HEALTH_INTERVAL`) to the configuration documentation in `docs/api/session.md` (FR-008)

## Phase 2: Foundational

Blocking prerequisites for every user story: the value objects, the error taxonomy,
the driver protocol and the registry.

- [x] T004 [P] Write unit tests for identity resolution, key equality, password redaction and validation in `tests/unit/session/test_identity.py` (FR-002)
- [x] T005 [P] Write unit tests for the error taxonomy — every error carries the identity label and subclasses `ConnectionError` — in `tests/unit/session/test_errors.py` (FR-019)
- [x] T006 [P] Write unit tests for `Rows`: values outlive the cursor, `one()` returns `None` for no rows and raises for two, `dicts()` zips columns, in `tests/unit/session/test_results.py` (FR-011)
- [x] T007 [P] Write unit tests for `select_driver` across three environments (both available, one, neither) in `tests/unit/session/test_driver_selection.py` (FR-006, FR-007)
- [x] T008 [P] Write unit tests for the registry: one session per identity key, different namespace means a different session, reset closes and clears, in `tests/unit/session/test_registry.py` (FR-002)
- [x] T009 Implement `SessionIdentity` with `from_env`, `from_config`, `key`, `label` and redacted `repr` in `iris_vector_rag/session/identity.py` (FR-002)
- [x] T010 [P] Implement the error taxonomy in `iris_vector_rag/session/errors.py` (FR-019)
- [x] T011 [P] Implement `Rows` with `one`, `dicts`, iteration and truthiness in `iris_vector_rag/session/results.py` (FR-011)
- [x] T012 Implement the `Driver` protocol, `EmbeddedDriver`, `NetworkDriver` and `select_driver` in `iris_vector_rag/session/drivers.py` (FR-006)
- [x] T013 Implement `FakeDriver` and the one-line `fake_session()` helper in `iris_vector_rag/testing/fake_driver.py` (FR-007)
- [x] T014 Implement the per-identity registry with `atexit` close and `reset_sessions_for_tests` in `iris_vector_rag/session/registry.py` (FR-002)
- [x] T015 Run the foundational unit tests and the existing `tests/unit` suite to confirm no regression

## Phase 3: User Story 1 — One way in, every protection everywhere (P1)

**Goal**: every entry point yields the same session with the same protections.
**Independent test**: build a session through each entry point against an unreachable
instance and against an expired password; the bounded failure and the recovery are
identical.

- [x] T016 [P] [US1] Write unit tests for the bounded connect, the back-off window and immediate `InBackoff` inside it, using `FakeDriver`, in `tests/unit/session/test_session_connect.py` (FR-004)
- [x] T017 [P] [US1] Write unit tests for one-shot password recovery per session and for `identity` answering without connecting in `tests/unit/session/test_session_recovery.py` (FR-005)
- [x] T018 [P] [US1] Write unit tests for `from_connection`: the session uses it, never closes it, and reports `CallerConnectionBroken` instead of replacing it, in `tests/unit/session/test_session_borrowed.py` (FR-010)
- [x] T019 [P] [US1] Write the contract test asserting the public session surface and that every deprecated entry point returns a connection from a session in `tests/contract/test_session_contract.py` (FR-001, FR-015)
- [x] T020 [US1] Implement `IRISSession` construction, `identity`, `driver_name`, `connection_for_adapter`, `close` and the context manager in `iris_vector_rag/session/session.py` (FR-001, FR-013)
- [x] T021 [US1] Implement the bounded connect, back-off window and one-shot password recovery in `iris_vector_rag/session/session.py` (FR-004, FR-005)
- [x] T022 [US1] Implement `session_for` and `IRISSession.from_connection` and export them from `iris_vector_rag/session/__init__.py` (FR-001, FR-010)
- [x] T023 [US1] Route `core.ConnectionManager.get_connection` through a session while keeping its signature and return type in `iris_vector_rag/core/connection.py` (FR-001, FR-018)
- [x] T024 [US1] Give `RAGEngine` exactly one session and pass it to the components it builds in `iris_vector_rag/core/engine.py` (FR-020)
- [x] T025 [US1] Report the real configured embedding dimension instead of the 1536 fallback in `iris_vector_rag/core/engine.py` (FR-014)
- [x] T026 [US1] Pass the session to `RetrievalEngine` instead of the attribute that does not exist in `iris_vector_rag/pipelines/basic.py` (FR-014)
- [x] T027 [US1] **Phase gate** — write and pass the end-to-end test for US1 (every entry point, unreachable instance, shared session, identity without connecting) in `tests/e2e/test_session_e2e.py` (FR-001, FR-002, FR-004)

## Phase 4: User Story 2 — Health on a cadence (P1)

**Goal**: at most one probe per interval, and a dead connection replaced without
caller retry code.
**Independent test**: count probes across 100 operations; kill the connection
server-side and confirm the next operation succeeds.

- [x] T028 [P] [US2] Write unit tests counting probes across 100 operations inside one interval and one probe after it elapses in `tests/unit/session/test_session_health.py` (FR-008)
- [x] T029 [P] [US2] Write unit tests for a single transparent reconnect per operation, `ConnectionLost` on the second failure, and `is_connected` never probing in `tests/unit/session/test_session_reconnect.py` (FR-009)
- [x] T030 [US2] Implement the health deadline and `is_connected` from recorded state in `iris_vector_rag/session/session.py` (FR-008)
- [x] T031 [US2] Implement one reconnect per operation for owned connections in `iris_vector_rag/session/session.py` (FR-009)
- [x] T032 [US2] Remove the per-call probe from the storage path so operations use the session's cadence in `iris_vector_rag/core/connection.py` (FR-008)
- [x] T033 [US2] **Phase gate** — write and pass the end-to-end test for US2 (100 operations, one probe; server-side drop recovers) in `tests/e2e/test_session_e2e.py` (FR-008, FR-009)

## Phase 5: User Story 3 — Statements safe to run and read (P2)

**Goal**: rows outlive their cursor, no cursor leaks, per-thread connections.
**Independent test**: read values after everything is closed; raise mid-statement and
confirm nothing is left open.

- [x] T034 [P] [US3] Write unit tests for `execute`, `execute_many` and `one` including cursor release on failure in `tests/unit/session/test_session_execute.py` (FR-011, FR-012)
- [x] T035 [P] [US3] Write unit tests for per-thread connections and the pid-mismatch discard in `tests/unit/session/test_session_threads.py` (FR-003)
- [x] T036 [US3] Implement `execute`, `execute_many` and `one` with copy-out results and `finally` cursor release in `iris_vector_rag/session/session.py` (FR-011, FR-012)
- [x] T037 [US3] Implement thread-local connections and the fork guard in `iris_vector_rag/session/session.py` (FR-003)
- [x] T038 [US3] **Phase gate** — write and pass the end-to-end test for US3 (values readable after close, no cursor left open, per-thread connections) in `tests/e2e/test_session_e2e.py` (FR-003, FR-011, FR-012)

## Phase 6: User Story 4 — Driver choice is a decision (P2)

**Goal**: explicit, logged driver selection, substitutable for tests.
**Independent test**: both drivers available, one, neither — inspect the reported
choice in each case.

- [x] T039 [P] [US4] Write unit tests asserting the selection log line names the driver and the reason, and that a substituted driver imports nothing real, in `tests/unit/session/test_driver_reporting.py` (FR-006, FR-007)
- [x] T040 [US4] Implement the reported selection and the `NoUsableDriver` message naming every driver tried in `iris_vector_rag/session/drivers.py` (FR-006, FR-019)
- [x] T041 [US4] Replace the import-time driver `try/except` blocks in `iris_vector_rag/common/iris_dbapi_connector.py` with the session's selection (FR-006)
- [x] T042 [US4] **Phase gate** — write and pass the end-to-end test for US4 (three environments, three outcomes, no network call for the substituted driver) in `tests/e2e/test_session_e2e.py` (FR-006, FR-007)

## Phase 7: User Story 5 — Old names keep working for one release (P3)

**Goal**: 0.15 breaks nothing and warns once per superseded name; the caller-less
re-exports go now.
**Independent test**: run the previous release's usage examples; identical results
plus one deprecation warning each.

- [x] T043 [P] [US5] Write unit tests asserting exactly one `DeprecationWarning` per superseded name and unchanged return types in `tests/unit/session/test_deprecations.py` (FR-015)
- [x] T044 [P] [US5] Write unit tests asserting the removed names are gone and the renamed classes leave no two library classes sharing a name in `tests/unit/session/test_removals.py` (FR-016)
- [x] T045 [US5] Deprecate `get_iris_connection` and delete `IRISConnectionPool` and `_PooledConnection` from `iris_vector_rag/common/iris_connection.py` (FR-015, FR-016)
- [x] T046 [P] [US5] Deprecate the manager and its function, and delete `get_iris_connection_old`, in `iris_vector_rag/common/iris_connection_manager.py` (FR-015, FR-016)
- [x] T047 [P] [US5] Deprecate `get_iris_dbapi_connection` in `iris_vector_rag/common/iris_dbapi_connector.py` (FR-015)
- [x] T048 [P] [US5] Rename `ConnectionManager` to `LegacyConnectionManager` in `iris_vector_rag/common/connection_manager.py` (FR-015)
- [x] T049 [P] [US5] Rename `ConnectionManager` to `WebSocketConnectionManager` in `iris_vector_rag/api/websocket/connection.py` and update its callers (FR-015)
- [x] T050 [US5] Delete the dead module `iris_vector_rag/common/iris_connector.py` and any re-export of it (FR-016)
- [x] T051 [US5] Rebuild `common/connection_pool.py` as a pool over sessions with no direct driver connect, deleting `get_pool`/`initialize_pool`/`close_pool` (FR-017, FR-018)
- [x] T052 [US5] Fix the HTTP API shutdown to call the pool method that exists in `iris_vector_rag/api/main.py` (FR-017)
- [x] T053 [US5] Add the direct-driver-connect detector to `scripts/gates/antipatterns.py` and record any remaining offender in `scripts/gates/antipatterns-baseline.txt` (FR-018)
- [x] T054 [US5] **Phase gate** — write and pass the end-to-end test for US5 (superseded entry points work with one warning each; the API serves concurrent requests from sessions) in `tests/e2e/test_session_e2e.py` (FR-015, FR-017)

## Phase 8: Polish

- [x] T055 [P] Document the session, its tuning variables and the migration table in `docs/api/session.md` (FR-013, FR-015)
- [X] T056 [P] Record the deprecations, the removals and the 0.16 removal note in `CHANGELOG.md` (FR-015, FR-016)
- [x] T057 Measure coverage of `iris_vector_rag/session/` and raise it to 90%+ with the tests the gaps name — 96% (`drivers.py`, `identity.py`, `results.py` at 100%)
- [x] T058 Run the full contract, unit and end-to-end suites and confirm the baselines hold with no new failures — contract 690 passed / 28 skipped / 40 deselected, unit 769 passed / 4 skipped, e2e 277 passed / 6 xfailed / 6 failed (all six on the placeholder `OPENAI_API_KEY`, none naming a connection)

## Dependencies

- Phase 2 blocks everything: the value objects, the driver protocol and the fake
  driver are what the story phases test against.
- Phase 3 (US1) blocks Phase 4 (US2) only because both edit `session.py`; their
  behaviours are independent.
- Phase 5 (US3) depends on Phase 2's `Rows` and Phase 3's connection ownership.
- Phase 6 (US4) depends on Phase 2's `select_driver`.
- Phase 7 (US5) depends on every earlier phase: it points the old names at the new
  session.

## Parallel opportunities

- T004–T008 (five foundational test files) are independent.
- T010, T011 are independent of each other and of T009 once the identity exists.
- T016–T019 are four independent test files.
- T046–T049 touch four different modules.

## MVP

Phases 1–4: one session with every protection and the health cadence. That alone
removes the hang class and halves the round trips on the storage path.
