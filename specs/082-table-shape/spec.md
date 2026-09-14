# Feature Specification: Table Shape

**Feature Branch**: `082-table-shape`
**Created**: 2026-09-10
**Status**: Draft
**Input**: User description: "TableShape: one module owns the physical shape of RAG document tables (SourceDocuments, DocumentChunks): resolves six observed shapes once per session, canonical wins, only canonical is created, every column-naming SQL fragment comes from it."

## Background

The document tables this project reads and writes exist in the field in several
physical forms ("table shapes", see `CONTEXT.md`). Today the knowledge of which
column holds the identifier, the text, the metadata and the embedding is
rediscovered independently in eleven places, using two contradictory precedence
rules: writes prefer one text column, reads prefer another. The last two shipped
defects (0.14.0 delete by identifier, 0.14.1 metadata filter on stream columns)
were both consequences of this. This feature gives table shape one owner.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Search, write and delete behave the same on every observed shape (Priority: P1)

An operator points the library at an existing document table whose shape is one of
the six observed in production and test environments (canonical; legacy integer
identifier with filename; legacy `id`/`source`/`content`; fixture
`doc_id`/`title`/`content`; a mixed table carrying both column families after an
in-place upgrade; a table whose vectors are double precision). Vector search,
metadata filtering, insert and delete work, and return documents in one consistent
form, without the operator naming any column.

**Why this priority**: This is the defect class that shipped twice; every consumer
of the library hits it whenever their database predates the current release.

**Independent Test**: Create each of the six shapes on a fresh IRIS, load the same
documents through the library, run the same search, filter and delete operations
against each, and compare results.

**Acceptance Scenarios**:

1. **Given** a table in any of the six observed shapes, **When** a vector search
   with a metadata filter runs, **Then** the results are the same documents with
   the same identifiers, text and scores as on the canonical shape.
2. **Given** a mixed table holding both the legacy and canonical column families,
   **When** a document is written and then read back, **Then** the text read is the
   text written (the same column is used for both directions).
3. **Given** a table whose metadata column is a stream type, **When** a metadata
   filter runs, **Then** matching documents are returned rather than an empty
   result.
4. **Given** a table whose vectors are double precision, **When** a vector search
   runs, **Then** it succeeds on the first attempt with no retry.
5. **Given** a legacy table whose identifier is database-generated, **When** the
   library is asked to insert a document with its own identifier, **Then** it
   refuses with an error naming the table and the reason, rather than guessing.

---

### User Story 2 - One shape probe per table per session (Priority: P2)

A pipeline runs many searches during its lifetime. The library learns a table's
shape once per connection lifetime and reuses it; schema changes made through the
library refresh that knowledge automatically, and a caller who suspects an outside
change can ask for a refresh.

**Why this priority**: Today every search issues several catalog lookups before
the query itself; removing them is a visible latency and load win on Community
Edition where connections are scarce.

**Independent Test**: Count catalog queries during a run of N searches against one
table; observe one shape resolution followed by N searches.

**Acceptance Scenarios**:

1. **Given** a connected pipeline, **When** it runs 100 searches against one table,
   **Then** the table's shape is resolved once and each search performs exactly one
   ranked query plus one metadata fetch.
2. **Given** a resolved shape, **When** the library itself creates or alters that
   table, **Then** the next operation sees the new shape without any caller action.
3. **Given** a resolved shape and a table altered outside the library, **When** the
   caller requests a refresh, **Then** the next operation uses the new shape.

---

### User Story 3 - Only one shape is ever created (Priority: P2)

When the library must create a document table, it always creates the canonical
shape. It never emits alternative layouts, and it never mutates an existing
non-canonical table unless the operator explicitly allows destructive changes.

**Why this priority**: Two different creation layouts plus an in-place alteration
path are how mixed tables came to exist in the first place.

**Independent Test**: Create tables through every creation path the library offers
on a fresh IRIS and compare their shapes; attempt operations against a legacy table
without the destructive flag and confirm it is unchanged afterwards.

**Acceptance Scenarios**:

1. **Given** a fresh IRIS, **When** any library path creates the document or chunk
   table, **Then** the resulting shape is the canonical one, identical across paths.
2. **Given** an existing legacy table, **When** the library operates on it without
   the destructive flag, **Then** the table's columns and types are unchanged
   afterwards.
3. **Given** a request to create a table from a non-canonical description, **When**
   the request is made, **Then** it is refused with a clear error.

---

### User Story 4 - Tests describe shapes, not database chatter (Priority: P3)

A contributor writing a test for storage behaviour supplies a recorded description
of a table shape and asserts on the resulting behaviour, instead of scripting a
fake database that answers catalog queries in a particular order.

**Why this priority**: The current storage tests must reproduce five hidden catalog
round trips to get one search through, which makes them brittle and discourages
coverage.

**Independent Test**: Write the shape-dependent storage tests using recorded shape
descriptions only; no test simulates catalog query text.

**Acceptance Scenarios**:

1. **Given** a recorded description of each observed shape, **When** the storage
   tests run, **Then** they pass without any simulated catalog responses.
2. **Given** a recorded shape description captured from a real database, **When**
   the same table is resolved live, **Then** the two descriptions are identical.

---

### Edge Cases

- The table does not exist: resolution reports a missing table distinctly from an
  unrecognisable one, and the create path still works.
- Columns exist but none can serve as identifier or text: resolution fails with an
  error listing the columns found.
- The query vector's dimension differs from the table's: the operation fails before
  any query runs, naming both dimensions.
- The vector type cannot be determined from the catalog: no environment variable or
  default silently substitutes; the operation reports the gap.
- Two callers share one connection and one alters the table through the library:
  the other sees the new shape on its next operation.
- Metadata filter values that are numbers or booleans match their stored form on
  every shape, as established in 0.14.1.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: The library MUST determine a document or chunk table's shape from the
  database itself, never from configuration, environment variables or defaults.
- **FR-002**: The library MUST recognise all six observed shapes and operate
  correctly on each for search, metadata filtering, insert and delete.
- **FR-003**: When a table carries both a legacy and a canonical column for the
  same role, the canonical column MUST be used for both reading and writing.
- **FR-004**: Every read MUST present documents in one consistent form (identifier,
  text, metadata, score) regardless of the table's physical shape.
- **FR-005**: A table's shape MUST be determined at most once per connection
  lifetime per table; library-initiated schema changes MUST refresh it, and callers
  MUST be able to request a refresh explicitly.
- **FR-006**: Every creation path MUST produce the canonical shape and only the
  canonical shape.
- **FR-007**: The library MUST NOT alter a non-canonical table unless destructive
  changes are explicitly allowed by the operator.
- **FR-008**: Attempting to write a caller-supplied identifier into a table whose
  identifier is database-generated MUST fail with a descriptive error.
- **FR-009**: Metadata filters MUST work on stream-typed metadata columns as well
  as string-typed ones.
- **FR-010**: Vector searches MUST use the table's actual vector type and dimension
  and MUST reject a query vector of the wrong dimension before querying.
- **FR-011**: Missing table, unrecognisable shape, non-creatable shape and
  dimension mismatch MUST be distinct, descriptive errors.
- **FR-012**: The two document-table joins used by the graph pipelines MUST agree
  on the identifier column by using the resolved shape.
- **FR-013**: The storage layer MUST stop invoking non-existent schema operations
  during automatic table creation; a failure there MUST surface, not be swallowed.
- **FR-014**: All existing public entry points MUST keep their current signatures
  and behaviour (0.15 is additive).
- **FR-015**: Shape-dependent tests MUST be expressible with recorded shape
  descriptions, without simulating catalog query text.
- **FR-016**: A capability the table cannot provide MUST be reported rather than
  silently dropped. Writing documents whose metadata has nowhere to go MUST warn once
  per table, not once per document; a metadata filter against a table with no metadata
  column MUST raise a descriptive error naming the table, instead of returning an
  empty page that is indistinguishable from a scope that genuinely holds no rows.

### Key Entities

- **Table shape**: the physical form of one table right now: identifier, text,
  metadata and embedding columns; whether metadata is a stream; vector type and
  dimension; whether it is canonical.
- **Canonical shape**: the single shape the library creates.
- **Legacy shape**: any observed non-canonical shape; five are known.
- **Shape description (recorded)**: a captured snapshot of what the database
  reported for a table, used to test behaviour without a database.

## Assumptions

- Only the document table and the chunk table are in scope; entity, relationship
  and community tables keep their current handling until a later feature.
- The six shapes listed are the complete set observed; a seventh would be added as
  a new recorded description, not by changing precedence rules.
- Destructive alteration remains the responsibility of the schema management
  feature that follows; this feature only refuses to do it implicitly.
- The stream projection width used for metadata filtering on stream columns is
  large enough for all metadata this project writes (metadata is capped at 2,000
  characters on the canonical shape).

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: The same document set searched with the same filter returns identical
  identifiers and ordering on all six observed shapes.
- **SC-002**: A run of 100 searches against one table performs exactly one shape
  resolution; each search performs one ranked query and one metadata fetch.
- **SC-003**: Zero locations outside the shape owner read the database catalog for
  the document or chunk tables (verified by an automated check in the quality
  gates).
- **SC-004**: Every table-creation path yields a byte-identical shape on a fresh
  database.
- **SC-005**: The reader/writer column disagreement cannot recur: there is one
  precedence rule, exercised by one test per observed shape.
- **SC-006**: Existing contract, unit and end-to-end suites pass unchanged in
  behaviour; the storage unit tests no longer contain simulated catalog responses.
- **SC-007**: Release 0.15 introduces no change to any documented public call.
