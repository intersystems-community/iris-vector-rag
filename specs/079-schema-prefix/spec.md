# Feature Specification: Configurable Schema Prefix — Full Implementation (AUD-010)

**Feature Branch**: `079-schema-prefix`
**Created**: 2026-07-19
**Status**: Draft

## Context

Spec `074-configurable-schema-prefix` was marked merged but no code was written.
`schema_manager.py` contains 65 literal `RAG.` table references and `WHERE
TABLE_SCHEMA = 'RAG'` filter strings. `IRIS_SCHEMA_PREFIX` env var and
`schema_prefix` constructor param are not implemented anywhere in
`iris_vector_rag/`.

Class-level validation caches (`_schema_validation_cache`, `_tables_validated`)
are keyed only by `table_name:pipeline_type` — no namespace, connection, or
schema component — so two `SchemaManager` instances with different prefixes in
the same process share false cache hits.

This spec completes what 074 described.

## User Scenarios & Testing

### User Story 1 — Schema prefix configurable via env var (Priority: P1)

A developer sets `IRIS_SCHEMA_PREFIX=MYAPP` and all tables are created as
`MYAPP.SourceDocuments`, `MYAPP.DocumentChunks`, etc. No `RAG.*` tables are
created.

**Why this priority**: Enables multi-tenant deployments and CI isolation. All other
stories depend on this working.

**Independent Test**: Set `IRIS_SCHEMA_PREFIX=MYAPP_TEST`, construct `SchemaManager`,
call a method that generates DDL, assert all generated SQL uses `MYAPP_TEST.` prefix.

**Acceptance Scenarios**:

1. **Given** `IRIS_SCHEMA_PREFIX=MYAPP`, **When** `SchemaManager` is constructed, **Then** `schema_prefix` is `"MYAPP"`.
2. **Given** `IRIS_SCHEMA_PREFIX=MYAPP`, **When** any DDL is executed, **Then** all table references use `MYAPP.*` not `RAG.*`.
3. **Given** `IRIS_SCHEMA_PREFIX` not set, **When** `SchemaManager` is constructed, **Then** prefix defaults to `"RAG"` (backward compatible).

---

### User Story 2 — Schema prefix injectable via constructor (Priority: P1)

A developer can pass `schema_prefix="TEST_RAG"` to `SchemaManager(...)` directly
without relying on env vars — enabling programmatic multi-prefix use.

**Why this priority**: Env var covers the common case; constructor param covers
test isolation and library use.

**Independent Test**: `SchemaManager(connection_manager, config_manager, schema_prefix="TEST_RAG")` — assert `sm.schema_prefix == "TEST_RAG"`.

**Acceptance Scenarios**:

1. **Given** `SchemaManager(..., schema_prefix="TEST_RAG")`, **When** any table name is resolved, **Then** `"TEST_RAG"` prefix is used.
2. **Given** both env var `IRIS_SCHEMA_PREFIX=ENV_RAG` and constructor `schema_prefix="CTOR_RAG"`, **Then** constructor value wins.

---

### User Story 3 — Two prefixes coexist in the same process without cache collisions (Priority: P2)

A developer instantiates two `SchemaManager` objects with different prefixes (e.g.,
for a test harness that validates isolation). They must not share cache state.

**Why this priority**: Correctness of the caching layer. Without this, the prefix
feature is unsafe in test environments.

**Independent Test**: Create two `SchemaManager` instances with different prefixes.
Trigger cache population on both. Assert each instance's cache is keyed by its own
prefix and does not read the other's entries.

**Acceptance Scenarios**:

1. **Given** `sm1 = SchemaManager(..., schema_prefix="A")` and `sm2 = SchemaManager(..., schema_prefix="B")`, **When** both validate schemas, **Then** `sm1` does not get a cache hit from `sm2`'s entries and vice versa.

---

### User Story 4 — Schema prefix validated at construction (Priority: P2)

An invalid prefix (empty string, SQL-injection attempt, non-identifier characters)
raises a clear error at construction time, not silently generates invalid SQL.

**Why this priority**: Security and reliability.

**Independent Test**: `SchemaManager(..., schema_prefix="RAG'; DROP TABLE")` raises
`ValueError` with a clear message.

**Acceptance Scenarios**:

1. **Given** prefix contains non-alphanumeric/underscore characters, **When** `SchemaManager` is constructed, **Then** `ValueError` is raised immediately.
2. **Given** empty string prefix, **When** `SchemaManager` is constructed, **Then** `ValueError` is raised.

---

### Edge Cases

- `IRIS_SCHEMA_PREFIX` with spaces or special chars: reject early.
- Cache must be instance-level (or keyed by `id(self)` / `prefix`), not class-level sets/dicts.
- `WHERE TABLE_SCHEMA = 'RAG'` SQL filter strings must also use the prefix — not just table name prefixes.
- `ConfigurationManager` should expose `get_schema_prefix()` so the env var is read through the config authority.

## Requirements

### Functional Requirements

- **FR-001**: `SchemaManager.__init__` MUST accept `schema_prefix: str = "RAG"` param.
- **FR-002**: `ConfigurationManager` MUST read `IRIS_SCHEMA_PREFIX` env var and expose it via `get_schema_prefix()`.
- **FR-003**: `SchemaManager` MUST read `schema_prefix` from `ConfigurationManager.get_schema_prefix()` when no constructor param is passed.
- **FR-004**: All 65+ literal `"RAG."` strings in `schema_manager.py` MUST be replaced with `f"{self.schema_prefix}."` (or a `_qn(table)` helper).
- **FR-005**: All `WHERE TABLE_SCHEMA = 'RAG'` filter strings MUST use `self.schema_prefix`.
- **FR-006**: `_schema_validation_cache` and `_tables_validated` MUST be instance-level (not class-level) OR keyed by `(prefix, table_name, pipeline_type)`.
- **FR-007**: `schema_prefix` MUST be validated: only `[A-Z][A-Z0-9_]*` (case-insensitive); raises `ValueError` on invalid input.
- **FR-008**: Zero `grep -rn '"RAG\.' iris_vector_rag/storage/schema_manager.py` matches after implementation.

### Key Entities

- **`schema_prefix`**: Instance attribute on `SchemaManager`; single source of truth for all table name qualification.
- **`_qn(table_name)`**: Private helper `f"{self.schema_prefix}.{table_name}"` — all SQL generation uses this helper.
- **`get_schema_prefix()`**: `ConfigurationManager` method reading `IRIS_SCHEMA_PREFIX` env var.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `grep -c '"RAG\.' iris_vector_rag/storage/schema_manager.py` returns 0.
- **SC-002**: `SchemaManager(..., schema_prefix="MYAPP")` generates only `MYAPP.*` DDL — verifiable via unit test that captures SQL strings.
- **SC-003**: Two `SchemaManager` instances with different prefixes do not share cache state — verifiable with a unit test.
- **SC-004**: All 219+ unit tests pass.
- **SC-005**: Live IRIS: setting `IRIS_SCHEMA_PREFIX=TEST_IVR` creates `TEST_IVR.SourceDocuments`, not `RAG.SourceDocuments`.
