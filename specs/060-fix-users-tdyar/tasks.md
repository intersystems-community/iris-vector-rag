---
description: "Task list for 060-fix-users-tdyar"
---

# Tasks: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Input**: Design documents from `/specs/060-fix-users-tdyar/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included (spec explicitly requires test pass rate and scenarios).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish prerequisites for running integration tests with live IRIS.

- [x] T001 Verify local environment prerequisites (Python 3.12, intersystems-irispython>=5.1.2, iris-devtester) and IRIS container availability
- [x] T002 [P] Document required env vars in `docs/development/iris_env.md`
- [x] T003 [P] Update fixture guidance in `tests/fixtures/README.md` (prefer .DAT fixtures when feasible; document exceptions)
- [x] T004 [P] Confirm iris-vector-graph>=1.6.0 installed in GraphRAG test environment (note: non-GraphRAG paths should run without it)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared test utilities used by both stories.

- [x] T005 Add/verify shared IRIS connection fixture in `tests/conftest.py` (iris-devtester, no hardcoded ports)
- [x] T006 [P] Add integration test helper for timing assertions in `tests/integration/helpers/timing.py`
- [x] T007 [P] Add/verify contract tests for connection and schema behaviors in `tests/contract/` (required before merge)

---

## Phase 3: User Story 1 - Database Connection Functionality (Priority: P1) 🎯 MVP

**Goal**: Replace `iris.connect()` usage with supported APIs and ensure connections succeed without AttributeError.

**Independent Test**: `pytest tests/integration/test_bug1_connection_fix.py tests/integration/test_iris_connection_integration.py`

### Tests for User Story 1

- [x] T008 [P] [US1] Add/adjust unit coverage for connection API selection in `tests/unit/test_connection_api.py`
- [x] T009 [P] [US1] Add/adjust integration coverage for connection success/failure in `tests/integration/test_iris_connection_integration.py`
- [x] T010 [P] [US1] Remove FHIR-AI-specific test requirement from spec/tasks/docs

### Implementation for User Story 1

- [x] T011 [US1] Replace `iris.connect()` with `iris.createConnection()`/`iris.dbapi.connect()` in `iris_vector_rag/common/iris_dbapi_connector.py`
- [x] T012 [US1] Update embedded connection helper to use supported APIs in `iris_vector_rag/common/utils.py`
- [x] T013 [US1] Update environment-based connector to use supported APIs in `iris_vector_rag/common/environment_manager.py`
- [x] T014 [US1] Update GraphRAG pipeline connection to use supported APIs in `iris_vector_rag/pipelines/hybrid_graphrag.py`
- [x] T015 [US1] Improve connection error context (include host:port/namespace) in `iris_vector_rag/common/iris_connection.py`

**Checkpoint**: Connection API fix validated by integration tests; no AttributeError.

---

## Phase 4: User Story 2 - Automatic Graph Schema Initialization (Priority: P2)

**Goal**: Ensure graph tables are created automatically and GraphRAG fails fast without iris-vector-graph.

**Independent Test**: `pytest tests/integration/test_graph_schema_integration.py`

### Tests for User Story 2

- [x] T016 [P] [US2] Add unit tests for graph table detection in `tests/unit/test_schema_detection.py`
- [x] T017 [P] [US2] Add unit tests for table init (idempotent/atomic + logging + timing assertions) in `tests/unit/test_schema_initialization.py`
- [x] T018 [P] [US2] Add integration tests for PPR validation + fail-fast when tables missing in `tests/integration/test_graph_schema_integration.py`
- [x] T019 [P] [US2] Add integration tests for missing package ImportError in `tests/integration/test_graph_schema_integration.py`
- [x] T020 [P] [US2] Add assertions for per-table error detail formatting in `tests/integration/test_graph_schema_integration.py`

### Implementation for User Story 2

- [x] T021 [US2] Enforce iris-vector-graph required for GraphRAG in `iris_vector_rag/storage/schema_manager.py` (raise ImportError with guidance)
- [x] T022 [US2] Implement automatic graph table creation in `iris_vector_rag/storage/schema_manager.py` (atomic + idempotent, include per-table error details)
- [x] T023 [US2] Add schema structure validation before PPR in `iris_vector_rag/storage/schema_manager.py`
- [x] T024 [US2] Log initialization success/failure and PPR availability in `iris_vector_rag/storage/schema_manager.py`

**Checkpoint**: Graph tables auto-created; GraphRAG fails fast without iris-vector-graph; PPR prerequisites enforced.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [x] T025 [P] Update `CHANGELOG.md` for v0.5.4 bug fixes
- [x] T026 [P] Update `README.md` with GraphRAG dependency requirement
- [x] T027 [P] Update schema manager docs in `docs/api/schema_manager.md`
- [x] T028 [P] Run v0.5.2 regression suite and capture results in `docs/testing/v0_5_2_regression.md`
- [x] T029 Run `ruff check .`, contract tests, and re-run integration tests from quickstart

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup
- **User Story 1 (P1)**: Depends on Foundational
- **User Story 2 (P2)**: Depends on Foundational (can run in parallel with US1 after Phase 2)
- **Polish (Phase 5)**: Depends on desired user stories complete

### User Story Dependencies

- **US1**: Independent after Phase 2
- **US2**: Independent after Phase 2

---

## Parallel Examples

### User Story 1

- Run in parallel: T008 (unit tests), T009 (integration test updates)
- Implementation sequence: T011 → T012/T013/T014 (parallel) → T015

### User Story 2

- Run in parallel: T016–T020 (test additions)
- Implementation sequence: T021 → T022 → T023 → T024

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1–2
2. Implement US1 (T008–T015)
3. Run US1 integration tests (quickstart Scenario 1)

### Incremental Delivery

1. US1 → validate
2. US2 → validate
3. Polish (Phase 5)
