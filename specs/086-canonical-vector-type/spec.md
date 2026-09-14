# Feature Specification: Canonical Vector Type, One DDL Owner

**Feature Branch**: `086-canonical-vector-type`
**Created**: 2026-09-14
**Status**: Draft
**Input**: Parked decisions after feature 085, items 1a and 1b: declare the canonical
vector element type everywhere the project creates a document table, detect a literal
element type before a new one ships, and stop `enterprise_storage.initialize_schema`
hand-writing (and dropping) the document table.

## Background

Feature 082 named one canonical vector element type — `table_shape.py:52`
`CANONICAL_VECTOR_TYPE = "FLOAT"` — and made the library read a table's real element
type from `%Dictionary.CompiledProperty.Parameters` at runtime. It did not change the
DDL that creates tables. Four shipped declarations still say `DOUBLE`:

| Site                                        | Live?                                            |
| ------------------------------------------- | ------------------------------------------------ |
| `iris_src/src/RAG/SDK/Schema.cls:22,23`     | yes — `Initialize()` executes `GetDefaultDDL()`  |
| `sql/schema.sql:9,18`                       | no code reads the file (feature 070 T004 is open) |
| `iris_vector_rag/common/db_init_complete.sql:31` | no code reads the file                       |

So the SDK creates a table the same library then classifies as non-canonical
(`ShapeKind.DOUBLE_VECTOR`). IRIS refuses to compare vectors of different element
types (SQLCODE -259), which is why 082 had to resolve the type at all.

The SDK's own write path is hardcoded to match its own DDL: `RAG.SDK.Pipeline`
`AddDocument` names `TO_VECTOR(?,DOUBLE,dim)` at `:9`, `:11` and `:25`. Its search path
does not — feature 085 gave `RAG.SDK.Search.GetVectorType` a runtime lookup precisely
because "Python paths create FLOAT columns, `RAG.SDK.Schema` creates DOUBLE ones, and
the SDK has to search both" (`Search.cls:47-50`). Changing the declaration without
changing the writer would leave the SDK inserting a `DOUBLE` vector into a `FLOAT`
column; changing the writer to a literal `FLOAT` would break every table the SDK has
already created. The writer has to resolve the type the same way the reader does.

`db_init_complete.sql` contradicts itself in one file: `:13` declares
`embedding VECTOR(FLOAT, 384)` and `:31` declares
`chunk_embedding VECTOR(DOUBLE, 384)`.

Nothing stops a fifth declaration. Thirty statements in shipped code name an element
type as a literal, and the quality gates have no detector for it. The nearest detector,
`inline-standard-ddl`, cannot see most DDL: it walks `iris_vector_rag/**/*.py` only, so
no `.sql` or `.cls` file is ever scanned — which also makes its own exemption for
`sql/schema.sql` unreachable — and its table-name pattern only matches a literal
standard-table name, so DDL whose table comes from a variable
(`CREATE TABLE {self.entities_table}`) is invisible.

Separately, `IRISStorage.initialize_schema`
(`iris_vector_rag/storage/enterprise_storage.py:99-205`) hand-writes its own document
table and destroys whatever is there first: `DROP TABLE` unconditionally at `:137`,
`DELETE FROM` at `:146` when the drop fails, then it re-points `self.table_name` at
whichever name worked (`:149`). It is reached lazily: `store_documents` probes the table
with `SELECT COUNT(*) ... WHERE 1=0` and calls `initialize_schema()` on any exception
(`:241-248`). A transient error against a populated table is therefore enough to empty
it.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Storing documents never destroys a populated table (Priority: P1)

An operator ingests documents into a table that already holds a corpus. Whatever the
library does about schema on the way in, every row that was there is still there
afterwards, and the documents are written to the table the operator named.

**Why this priority**: this is data loss in a path that runs on every ingest, triggered
by any error the table probe happens to raise — a lock, a licence exhaustion, a
disconnect.

**Independent Test**: populate a table, force the probe query to fail once, store one
document, and count the rows.

**Acceptance Scenarios**:

1. **Given** a populated document table, **When** the table probe in `store_documents`
   raises and schema initialisation runs, **Then** no row is deleted and no table is
   dropped.
2. **Given** a configured table name of `RAG.SourceDocuments`, **When** schema
   initialisation runs and succeeds, **Then** subsequent writes still address
   `RAG.SourceDocuments` — the configured name is never silently replaced.
3. **Given** a table that cannot be created or reached at all, **When** schema
   initialisation runs, **Then** it raises with the underlying reason rather than
   reporting success against a different table.
4. **Given** a document table that already exists in a legacy shape, **When** schema
   initialisation runs, **Then** the table's columns and types are unchanged (082
   FR-007).

---

### User Story 2 - Every creation path declares the canonical element type (Priority: P1)

An operator creates the RAG tables through any path the project ships — the Python
schema manager or `RAG.SDK.Schema.Initialize` — and the embedding columns have the same
element type, so a vector built by the library compares against them on the first
attempt.

**Why this priority**: two creation paths producing two element types is how a
`DOUBLE` corpus gets created by the SDK and then read by a `FLOAT` query, which IRIS
rejects outright.

**Independent Test**: on a fresh IRIS, create the tables through each path, read the
element type from `%Dictionary.CompiledProperty.Parameters` for each embedding column,
and compare.

**Acceptance Scenarios**:

1. **Given** a fresh namespace, **When** `RAG.SDK.Schema.Initialize(384)` runs,
   **Then** `RAG.SourceDocuments.embedding` and `RAG.DocumentChunks.chunk_embedding`
   have element type `FLOAT` and width 384.
2. **Given** tables created by the SDK path, **When** a vector search runs through the
   Python library, **Then** it succeeds on the first attempt with no SQLCODE -259 and
   no retry.
3. **Given** a table created before this change with `DOUBLE` columns, **When** the
   library reads and searches it, **Then** it still works — the declaration change
   applies to new tables only and no migration runs.
4. **Given** the two `.sql` files that declare embedding columns, **When** they are
   read, **Then** every embedding column in them declares the same element type.
5. **Given** a table created by the SDK before this change (`DOUBLE`) and one created
   after it (`FLOAT`), **When** a document with an embedding is added through the SDK
   to each, **Then** both writes succeed — the writer names the type the column has.

---

### User Story 3 - A new hardcoded element type cannot ship unnoticed (Priority: P2)

A contributor adds a SQL statement that names a vector element type as a literal. The
quality gate fails and names the file and line, before the statement reaches a release.

**Why this priority**: the thirty existing sites are a feature's worth of work to
rewrite (deferred, below). Stopping the thirty-first is a gate change.

**Independent Test**: run the gate on the tree as-is and see zero new findings; add one
literal-typed statement to a shipped module and see exactly one new finding.

**Acceptance Scenarios**:

1. **Given** the tree as it stands, **When** the antipatterns gate runs, **Then** the
   new detector reports zero new findings and every existing site is baselined.
2. **Given** a new `TO_VECTOR(?, FLOAT, 384)` or `VECTOR(DOUBLE, n)` in shipped code,
   **When** the gate runs, **Then** it fails and names that file and line.
3. **Given** prose that mentions `VECTOR(FLOAT, 384)` in a comment or docstring,
   **When** the gate runs, **Then** it is not reported — the detector fires on
   statements, not on descriptions of them.
4. **Given** a baselined site that is later removed, **When** the gate runs, **Then**
   it reports the baseline entry as stale.

---

### User Story 4 - The DDL detector can see the files that hold DDL (Priority: P3)

A contributor puts document-table DDL in a `.sql` file, a `.cls` file, or behind a
variable table name. The `inline-standard-ddl` detector reports it instead of silently
walking past it.

**Why this priority**: an exemption list naming a file the detector never opens reads
as coverage that does not exist. Today `sql/schema.sql` is exempt from a scan it was
never in.

**Independent Test**: run the detector against source it is handed, one case per blind
spot, and assert the finding.

**Acceptance Scenarios**:

1. **Given** a `.sql` or `.cls` file carrying DDL for a standard RAG table, **When**
   the detector runs, **Then** the site is reported.
2. **Given** `CREATE TABLE {some_variable}` in a non-owner module, **When** the
   detector runs, **Then** it is reported as DDL whose table it cannot identify.
3. **Given** the detector's exemption set, **When** the gate runs, **Then** every path
   in it is a path the detector actually walks.

---

### Edge Cases

- A deployment that legitimately wants a non-canonical element type: the shape
  renderer already takes the type as a parameter, so this feature changes defaults,
  not capability.
- Test fixtures that deliberately describe a `DOUBLE` table (082's sixth observed
  shape) must keep declaring `DOUBLE`; they are the record of what exists in the
  field, not a creation path.
- The detector must not fire on `.md` files, which include the constitution's own
  `TO_VECTOR(?, FLOAT, 384)` example.
- A `.cls` change cannot be verified on disk: it compiles on the server or not at all.
- `initialize_schema` delegating to the schema owner passes an unqualified table name,
  because the owner matches against unqualified standard-table names — a qualified
  `RAG.SourceDocuments` would fall through to "not recognized" and return `False`.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: Every DDL path the project ships MUST declare the canonical vector
  element type for document and chunk embedding columns.
- **FR-002**: The ObjectScript SDK's table creation and the Python schema manager's MUST
  produce the same element type for the same column.
- **FR-003**: Changing a declaration MUST NOT alter, migrate or re-create any existing
  table; tables already created with `DOUBLE` MUST keep working through the resolved
  shape.
- **FR-004**: The SDK's document write path MUST name the element type the target
  column actually has, resolved at runtime, so that it works against tables created
  before and after this change.
- **FR-005**: A single file MUST NOT declare two different element types for the
  embedding columns of the same table family.
- **FR-006**: The quality gates MUST fail when a new literal vector element type
  appears in a SQL statement in shipped code, naming the file and line.
- **FR-007**: The element-type detector MUST report statements and MUST NOT report
  prose that mentions a type in a comment or docstring.
- **FR-008**: The element-type detector MUST leave every site that exists today
  baselined, rewriting none of them.
- **FR-009**: The standard-table DDL detector MUST scan the file types that carry DDL,
  including `.sql` and `.cls`, not only `.py`.
- **FR-010**: The standard-table DDL detector MUST report DDL whose table name comes
  from a variable, because it cannot prove such a table is not a standard one.
- **FR-011**: Every path in a detector's exemption set MUST be a path that detector
  walks; an unreachable exemption MUST be removed or made reachable.
- **FR-012**: `IRISStorage.initialize_schema` MUST NOT contain document-table DDL; it
  MUST delegate table creation to the module that owns the schema.
- **FR-013**: No code path reachable from storing documents MUST drop or empty an
  existing table.
- **FR-014**: `IRISStorage.initialize_schema` MUST NOT change which table the instance
  writes to; a failure to ensure the configured table MUST raise, not fall back.
- **FR-015**: When delegating, the configured table name MUST be translated to the form
  the schema owner accepts, so that a qualified configured name still resolves.
- **FR-016**: The vector-index step of `initialize_schema` MUST keep its current
  behaviour, including its configured HNSW parameters and its tolerance of failure.
- **FR-017**: All existing public entry points MUST keep their current signatures and
  observable behaviour, except the destructive behaviour FR-013 removes.
- **FR-018**: Debt entries whose claims this feature settles MUST be updated or
  removed, and any stale line reference in them corrected.

### Deferred, deliberately

Recorded here so the boundary is explicit, not discovered later:

- **(c) The thirty literal element types in existing statements stay as they are.**
  Rewriting one means proving which table it targets and that a resolved shape is
  reachable at that point; thirty of those is a feature. Tracked as `docs/DEBT.md`
  entry 4, which this feature extends from 11 `.py` `TO_VECTOR` sites to the full
  count the new detector baselines.
- **(d) `needs_migration` still cannot see a column's physical element type**, so a
  `DOUBLE` table is never reported as needing migration to `FLOAT`. Fixing it is a
  behaviour change with a migration path attached. Tracked as `docs/DEBT.md` entry 5.

### Key Entities

- **Canonical element type**: the one vector element type the project declares when it
  creates a table; already named in the shape module.
- **Creation path**: any code that issues DDL for a document or chunk table. After this
  feature: the Python schema owner and the ObjectScript SDK.
- **Element-type finding**: one SQL statement in shipped code naming an element type as
  a literal; the detector's unit of report.

## Assumptions

- Existing `DOUBLE` tables in the field are handled: 082 resolves the element type at
  runtime and its sixth observed shape is exactly this case. No migration is in scope.
- `sql/schema.sql` and `iris_vector_rag/common/db_init_complete.sql` have no code
  consumer today. They are corrected because they are read by people and because
  feature 070's open task T004 intends `sql/schema.sql` to become the shared source,
  not because a run depends on them.
- The two `.sql` files' other declarations already say `FLOAT`; only the divergent lines
  change.
- The element-type detector's scope is shipped code — the package, the shared `sql/`
  directory and the ObjectScript sources. Tests are excluded on purpose: they create
  legacy shapes deliberately.
- `RAG.SDK.Schema` still hardcodes its DDL rather than reading `sql/schema.sql`.
  Closing that gap is feature 070's T004 and stays out of scope; this feature only
  makes the hardcoded copy agree with the canonical type.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: On a fresh IRIS, the embedding columns created by every shipped creation
  path report the same element type and width when read from
  `%Dictionary.CompiledProperty.Parameters`.
- **SC-002**: A vector search against a table created by the ObjectScript SDK path
  succeeds on the first attempt, with no SQLCODE -259 and no type retry.
- **SC-003**: The antipatterns gate reports zero new findings for the new detector on
  the unchanged tree, and exactly one new finding when a single literal-typed statement
  is added to shipped code.
- **SC-004**: `inline-standard-ddl` reports at least one `.sql` site and at least one
  variable-named site, and its exemption set contains no path the detector cannot
  reach.
- **SC-005**: `initialize_schema` contains no `CREATE TABLE`, `DROP TABLE` or
  `DELETE FROM` statement, verified by reading the method.
- **SC-006**: Storing a document into a populated table whose first probe fails leaves
  the row count unchanged and writes to the configured table.
- **SC-007**: The enterprise storage unit suite has no test asserting a table-name
  fallback, and its failure count drops by at least one from the seven recorded in
  `docs/DEBT.md` entry 3.
- **SC-008**: Every declaration of an embedding column in the two shared `.sql` files
  names one element type.
