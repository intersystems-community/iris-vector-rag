# Feature Specification: RAG.SDK search usable by an external ObjectScript consumer

**Feature Branch**: `085-sdk-search-filter`
**Created**: 2026-09-14
**Status**: Draft
**Input**: An ObjectScript consumer outside this repo (an agent-memory recall path) needs
`RAG.SDK.Search` to answer a scoped, synchronous top-K question over a table it attached
itself. Today it cannot: the search cannot be scoped, the rows carry no attributes beyond
id/text/score, and the vector element type is hardcoded so the search fails outright
against any table whose embedding column is `FLOAT`.

## Clarifications

### Session 2026-09-14

- Q: How should a caller scope a search — a JSON metadata filter or a SQL WHERE fragment?
  → A: A SQL WHERE fragment. The consumer already computes scoping as SQL over real
  columns of the table it attached (`agent_id`, `visibility`, `expires_at`), not as JSON
  metadata keys. A metadata-key filter would force the consumer to serialize what it
  already has as SQL and would not reach non-metadata columns.
- Q: Scope of this feature — filter only, or filter plus the two defects found while
  verifying it? → A: Filter plus both defects (metadata passthrough, resolved vector
  element type, fusion carrying text). A search test class has to be written from scratch
  either way, and the defects make the filter unusable in practice.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Scoped search (Priority: P1)

A consumer holds a scoping predicate it computed itself — for one agent, one visibility
level, unexpired rows — and needs the top K nearest rows *within that scope*. Today it can
only ask for the global top K and filter afterwards, which silently returns fewer than K
in-scope rows (or none) whenever out-of-scope rows crowd the page.

**Why this priority**: Without scoping the search cannot be used at all by a multi-tenant
consumer; every other gap is a quality problem on top of an unusable result.

**Independent Test**: Attach a table holding rows for two owners, search with a predicate
naming one owner, assert every returned row belongs to that owner and that the count is
the number of in-scope rows (not the number crowded out by the other owner).

**Acceptance Scenarios**:

1. **Given** a table with 5 rows for owner A and 5 for owner B and a caller asking for the
   top 3 with a predicate selecting owner B, **When** the search runs, **Then** exactly 3
   rows are returned and all belong to owner B.
2. **Given** the same table and a caller passing no predicate, **When** the search runs,
   **Then** the result is the same as before this feature — the predicate is optional and
   omitting it changes nothing.
3. **Given** a predicate that matches no row, **When** the search runs, **Then** the result
   is an empty list, not an error.
4. **Given** a syntactically invalid predicate, **When** the search runs, **Then** the
   result names the failure rather than returning an empty list that reads as "no matches".

---

### User Story 2 - Rows carry their attributes (Priority: P1)

The consumer maps each returned row into its own entity, whose fields (owner, type,
confidence, timestamps, expiry) live in columns or in the row's serialized metadata. Today
a row is `{id, text, score}` and nothing else, so the consumer must issue a second query
per row to rebuild what the search already read.

**Why this priority**: Same release as the filter, and driven by the same consumer. A
scoped search that returns rows the caller cannot interpret still needs N follow-up
queries.

**Independent Test**: Attach a table whose metadata column holds a known JSON object, run a
search, and assert the returned row carries that object without a second query.

**Acceptance Scenarios**:

1. **Given** an attached table with a metadata column and a row whose metadata is
   `{"owner":"a","kind":"note"}`, **When** a search returns that row, **Then** the row
   carries that metadata.
2. **Given** an attached table with no metadata column, **When** a search returns a row,
   **Then** the row omits metadata and the search still succeeds.
3. **Given** a row whose metadata is not valid JSON, **When** a search returns that row,
   **Then** the raw text is carried through and the search still succeeds.

---

### User Story 3 - Search works against a FLOAT embedding column (Priority: P1)

Any table created by this project's Python path has a `VECTOR(FLOAT, N)` embedding column.
The SDK search asks for `TO_VECTOR(?, DOUBLE, N)`. IRIS refuses to compare vectors of
different element types (SQLCODE -259, "Cannot perform vector operation on vectors of
different datatypes"), so the search returns an error object for every such table —
including the one on the project's own container today.

**Why this priority**: This is a total failure, not a degradation, and it splits the
project in two: tables created by the SDK's own schema path (`DOUBLE`) are searchable and
tables created by the Python path (`FLOAT`) are not.

**Independent Test**: Create a table with a `FLOAT` embedding column, insert a row, search
it, and assert a scored row comes back instead of an error.

**Acceptance Scenarios**:

1. **Given** a table whose embedding column is `VECTOR(FLOAT, 384)`, **When** a search
   runs, **Then** scored rows are returned.
2. **Given** a table whose embedding column is `VECTOR(DOUBLE, 384)`, **When** a search
   runs, **Then** scored rows are returned — the previously working case keeps working.
3. **Given** a table whose embedding element type cannot be determined, **When** a search
   runs, **Then** the failure names the table and column rather than surfacing a raw
   SQLCODE.

---

### User Story 4 - Hybrid results keep their text (Priority: P2)

Hybrid search fuses a vector result and a text result and returns `{id, score}` only: the
text and metadata both inputs carried are dropped on the floor. A consumer using hybrid
search must re-read every row it just ranked.

**Why this priority**: Lower than the others because the consumer's recall path uses vector
search; hybrid is the next thing it will reach for, and the fix is local to fusion.

**Independent Test**: Run a hybrid search over rows with known text and assert each fused
row carries the text that the vector or text leg already read.

**Acceptance Scenarios**:

1. **Given** rows matched by both legs, **When** either fusion strategy runs, **Then** each
   fused row carries id, score, and the text from whichever leg supplied it.
2. **Given** a row matched by only one leg, **When** fusion runs, **Then** that row still
   carries its text.

---

### Edge Cases

- A caller-supplied predicate that references a column the attached table does not have —
  the failure must name the predicate, not read as "no matches".
- A caller-supplied predicate containing a statement terminator or a second statement — it
  is a fragment, not a statement, and must not be able to extend the query.
- Search against a table whose configured columns no longer exist.
- A query vector whose length disagrees with the column's declared dimension — IRIS reports
  this distinctly from an element-type mismatch and the two must not be conflated.
- A table with no rows and a non-empty predicate — an empty list, not an error.
- The IVF fast path: it does not accept a predicate, so a predicate must not be silently
  ignored by taking that path.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A vector search MUST accept an optional caller-supplied SQL predicate and
  return only rows satisfying it, combined with the existing "embedding is present"
  condition.
- **FR-002**: Omitting the predicate MUST leave the result identical to the behaviour
  before this feature.
- **FR-003**: A predicate that cannot be executed MUST produce a named failure, distinct
  from an empty result.
- **FR-004**: The predicate MUST be treated as a fragment of the search's own WHERE clause
  and MUST NOT be able to introduce an additional statement.
- **FR-005**: When a scoping predicate is supplied, the search MUST NOT take a code path
  that cannot honour it; if the fast path cannot apply the predicate, the search MUST use
  the path that can.
- **FR-006**: A returned row MUST carry the row's metadata when the searched table has a
  metadata column, and MUST omit it when the table has none.
- **FR-007**: Metadata that is not valid JSON MUST be carried through rather than failing
  the search.
- **FR-008**: The attach and default-table configuration MUST be able to record a metadata
  column, and MUST remain valid when a table has none.
- **FR-009**: The vector element type used to build the query vector MUST be resolved from
  the searched table rather than assumed, so that a `FLOAT` column and a `DOUBLE` column
  are both searchable.
- **FR-010**: An element type that cannot be resolved MUST produce a failure naming the
  table and column.
- **FR-011**: Hybrid fusion MUST carry each fused row's text through from whichever leg
  supplied it, under both fusion strategies.
- **FR-012**: Every method changed by this feature MUST have `%UnitTest` coverage that runs
  against a live IRIS instance, including the seven search cases feature 070 declared and
  never wrote.

### Key Entities

- **Search hit**: what one returned row is — an identifier, its text, a score, and, when
  the table has one, its metadata. The consumer maps this into its own entity.
- **Attached table configuration**: the table and the columns the SDK searches — the
  identifier, text, and embedding columns today; a metadata column and a resolved vector
  element type after this feature.
- **Scoping predicate**: a caller-supplied SQL condition over the attached table's own
  columns, owned and composed by the caller, applied by the search.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A search over a table holding two owners' rows, scoped to one owner, returns
  only that owner's rows, and returns as many as were asked for whenever that many exist
  in scope.
- **SC-002**: A search against a table with a `VECTOR(FLOAT, N)` embedding column returns
  scored rows; today it returns an error for 100% of such tables.
- **SC-003**: A consumer can build its own entity from a search result with zero follow-up
  queries per row when the table stores its attributes in metadata.
- **SC-004**: A `%UnitTest` class covers every public search method, and it fails before
  each change in this feature and passes after — no method changed here ships without a
  test that demonstrably exercised it.
- **SC-005**: A caller passing no predicate, no metadata column, and a `DOUBLE` embedding
  column observes byte-identical results before and after this feature.

## Assumptions

- The caller composes its own predicate and is trusted to do so; the predicate is not
  parsed or rewritten beyond rejecting an attempt to add a statement. The SDK's existing
  configuration surface already takes caller-supplied table and column names on the same
  trust basis.
- `RAG.SDK.Schema.Initialize` continues to create `DOUBLE` embedding columns in this
  feature; making the SDK's own creation path agree with the project's canonical `FLOAT`
  shape is a schema change with its own migration question and is out of scope here. This
  feature makes search work against both, which is what unblocks the consumer.
- The consumer repository is not touched by this feature. Its rewiring is its own change in
  its own repository.
