# Feature Specification: Pure Constructors (AUD-009)

**Feature Branch**: `078-pure-constructors`
**Created**: 2026-07-19
**Status**: Draft

## Context

Construction of any pipeline currently triggers database DDL:

1. `BasicRAGPipeline()` → `RAGPipeline.__init__` (constructs `IRISVectorStore` if none passed)
2. → `IRISVectorStore.__init__` (constructs `SchemaManager`, calls `_get_connection()`)
3. → `SchemaManager.__init__` unconditionally calls `ensure_schema_metadata_table()`
4. → opens DB connection, executes `CREATE TABLE IF NOT EXISTS RAG.SchemaMetadata`, commits

A `PYTEST_CURRENT_TEST` guard in `IRISVectorStore` is a band-aid, not a fix.
Importing and constructing a pipeline object mutates a live database.

## User Scenarios & Testing

### User Story 1 — Constructing a pipeline does not touch the database (Priority: P1)

A developer imports and instantiates `BasicRAGPipeline()` in a script, test, or
REPL without an active IRIS connection. Today it fails with a connection error or
silently mutates the schema. After this fix, construction is pure.

**Why this priority**: Foundational — every test that mocks the vector store is
fighting this side effect today. Removing it simplifies all tests.

**Independent Test**: Instantiate `BasicRAGPipeline(connection_manager=cm, config_manager=cfg)` with a mock ConnectionManager that asserts `get_connection()` is never called during `__init__`. Assert the mock was not called.

**Acceptance Scenarios**:

1. **Given** no IRIS running, **When** `BasicRAGPipeline(...)` is constructed with valid-but-unconnected managers, **Then** no exception is raised and no DB calls are made.
2. **Given** a mock `ConnectionManager`, **When** any pipeline is constructed, **Then** `get_connection()` is not called on the mock.
3. **Given** `SchemaManager(connection_manager, config_manager)` is constructed, **When** `__init__` completes, **Then** no DDL has been executed.

---

### User Story 2 — Schema is initialized on first use, not on construction (Priority: P1)

The first call to `load_documents()` or `query()` triggers schema initialization
if needed. Explicit `pipeline.initialize()` is also supported for pre-warming.

**Why this priority**: Lazy init is the correct pattern; explicit init is a useful
escape hatch for startup scripts.

**Independent Test**: Construct a pipeline, assert no DB calls. Call `load_documents()`,
assert schema table was created exactly once.

**Acceptance Scenarios**:

1. **Given** a constructed pipeline with real IRIS, **When** `load_documents()` is first called, **Then** schema tables are created if they don't exist, and the documents are loaded.
2. **Given** `pipeline.initialize()` is called explicitly, **Then** schema is set up and subsequent `load_documents()` does not re-run DDL.
3. **Given** schema already exists, **When** `load_documents()` is called, **Then** no redundant DDL is executed (idempotent).

---

### User Story 3 — PYTEST_CURRENT_TEST guard is removed (Priority: P2)

The guard `if os.environ.get("PYTEST_CURRENT_TEST")` in `IRISVectorStore.__init__`
is removed. Tests pass without it because construction is now pure.

**Why this priority**: Prod code that behaves differently in tests is a trust failure.

**Independent Test**: Remove the guard, run `pytest tests/unit/` — all tests pass.

**Acceptance Scenarios**:

1. **Given** `PYTEST_CURRENT_TEST` env var is set, **When** `IRISVectorStore` is constructed, **Then** behavior is identical to when it is not set.

---

### Edge Cases

- `SchemaManager._metadata_table_ensured` flag must be per-instance, not a class variable (avoid cross-instance state).
- Lazy init must be thread-safe — use a lock or atomic flag for the first-call guard.
- `pipeline.initialize()` called twice must be idempotent (no duplicate DDL).

## Requirements

### Functional Requirements

- **FR-001**: `SchemaManager.__init__` MUST NOT call `ensure_schema_metadata_table()` or any method that opens a DB connection.
- **FR-002**: `SchemaManager` MUST have an `_initialized: bool` instance flag; `ensure_schema_metadata_table()` MUST check it and skip if already run.
- **FR-003**: `IRISVectorStore.__init__` MUST NOT call `_get_connection()` or any DB-touching method.
- **FR-004**: `RAGPipeline.__init__` MAY construct `IRISVectorStore` if none provided, but MUST NOT trigger any DB calls through that construction.
- **FR-005**: `RAGPipeline` MUST expose an `initialize()` method that triggers schema setup; `load_documents()` and `query()` MUST call `initialize()` lazily on first use.
- **FR-006**: The `PYTEST_CURRENT_TEST` guard in `IRISVectorStore.__init__` MUST be removed.
- **FR-007**: Unit tests MUST NOT require broad mocking of `SchemaManager` just to prevent DDL — construction must be safe with no mock at all.

### Key Entities

- **`_initialized` flag**: Per-instance boolean on `SchemaManager`; guards `ensure_schema_metadata_table()`.
- **`pipeline.initialize()`**: Explicit trigger for schema setup; idempotent; called lazily by `load_documents()` / `query()`.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `BasicRAGPipeline(connection_manager=mock, config_manager=mock)` construction does not call `mock.get_connection()` — verifiable with `MagicMock(spec=ConnectionManager)` and `assert_not_called()`.
- **SC-002**: All 219+ unit tests pass without the `PYTEST_CURRENT_TEST` env guard.
- **SC-003**: Live IRIS: `load_documents()` on a fresh schema creates `RAG.SchemaMetadata` exactly once (verify with SQL count before/after).
