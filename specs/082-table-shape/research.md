# Phase 0 Research: Table Shape

**Feature**: 082-table-shape | **Date**: 2026-09-10
**Method**: every decision below was probed against the live container
(`iris-vector-rag-iris`, port 51972, IRIS 2026.1) rather than inferred from the
driver docs. Probe output is quoted where it decides the design.

The spec's requirements need exactly four things from the database that the
current code either guesses at or reads from the wrong place: which columns exist
and what SQL type they are, which column is the identifier, whether a text column
is a stream, and whether an embedding column is really a `VECTOR` (and of what
element type and length). Each is one decision below.

## Decision 1: two catalog reads per table, joined on the SQL field name

**Decision**: a shape is built from exactly two queries — `INFORMATION_SCHEMA.COLUMNS`
for the SQL view of the table, and `%Dictionary.CompiledProperty` for the class view —
joined on `SqlFieldName`.

**Rationale**: neither source alone answers all four questions.
`INFORMATION_SCHEMA` gives ordinal position, nullability, length and the identity
flags, but it lies about `VECTOR` (reports `varchar`) and cannot distinguish a
stream from a long string. `%Dictionary.CompiledProperty` gives the true class type
and its parameters, but nothing about nullability or position.

`%Dictionary.CompiledProperty.Name` is the **property** name, and property names
have the underscores stripped:

```text
Name          SqlFieldName   Type
chunkindex    chunk_index    %Library.Integer
chunktext     chunk_text     %Stream.GlobalCharacter
createdat     created_at     %Library.PosixTime
documentid    document_id    %Library.String
embedding     embedding      %Library.Binary
```

So the existing lookup in `vector_store_iris.py:151-179`
(`WHERE parent = ? AND Name = 'embedding'`) only works by luck — the document
table's embedding column has no underscore in it. The same query for
`chunk_embedding` returns zero rows and falls through to the environment default.
Joining on `SqlFieldName` removes the class of bug entirely; stripping underscores
from the SQL name would also work but is lossy (`chunk_text` and `chunktext` both
map to `chunktext`).

**Alternatives considered**:

- _`%Dictionary.CompiledProperty` alone_ — rejected: no nullability, no ordinal
  position, and the class name is not derivable from the table name (Decision 2).
- _`INFORMATION_SCHEMA` alone_ — rejected: cannot see `VECTOR` or streams, which
  are two of the four questions.
- _`DESCRIBE` / cursor `description`_ — rejected: driver-level type codes are the
  same lie as `INFORMATION_SCHEMA`, and getting them costs a real query against
  the table.

## Decision 2: resolve the class name through `%Dictionary.CompiledClass`, never by string concatenation

**Decision**: map `schema.table` to a class name with
`SELECT ID FROM %Dictionary.CompiledClass WHERE SqlSchemaName = ? AND SqlTableName = ?`.

**Rationale**: the mapping is not mechanical. In the live namespace:

```text
ID              SqlSchemaName   SqlTableName
RAG.llmcache    RAG             llm_cache
```

A table whose name contains characters that are illegal in a class name gets a
mangled class name. `"RAG." + table_name` is right for eleven of the twelve tables
here and wrong for the twelfth — which is exactly the kind of near-miss that
produces a silent fallback instead of an error. The lookup is one extra row on the
same round trip budget and it is exact.

**Alternatives considered**: `%Dictionary.CompiledProperty WHERE parent = 'RAG.' || ?`
— rejected for the reason above; it degrades to "no rows found", which under
FR-001 must become an error, so it would turn a working table into a hard failure.

## Decision 3: a column is a vector iff its class type is `%Library.Vector`

**Decision**: vector-ness, element type and declared length all come from
`%Dictionary.CompiledProperty.Type` and `.Parameters`. `INFORMATION_SCHEMA.DATA_TYPE`
is never consulted for an embedding column.

**Rationale**: measured on `RAG.KGEntities.entity_embedding`, a column declared
`VECTOR(FLOAT, 384)`:

| Source                                     | Reports                                                          |
| ------------------------------------------ | ---------------------------------------------------------------- |
| `INFORMATION_SCHEMA.COLUMNS.DATA_TYPE`     | `varchar`                                                        |
| `%Dictionary.CompiledProperty.Type`        | `%Library.Vector`                                                |
| `%Dictionary.CompiledProperty.Parameters`  | `...,DATATYPE,FLOAT,...,LEN,384,...`                             |

`Parameters` is a flat comma-separated key,value sequence
(`CALCSELECTIVITY,1,CAPTION,,DATATYPE,FLOAT,EXTERNALSQLNAME,,...,LEN,384,...`);
parsing it means splitting on commas and pairing, with empty values preserved.
`DATATYPE` is the element type (`FLOAT`, `DOUBLE`, `DECIMAL`, `INTEGER`) and `LEN`
is the dimension. Both are what the shape must expose, because `TO_VECTOR()` needs
the element type and the dimension check needs the length.

This also settles the project's standing gotcha (the `varchar` report is a driver
limitation, never a schema defect) at the level of the code rather than the
documentation: the shape reads the source that tells the truth, so no call site is
ever in a position to "fix" a healthy column.

**Alternatives considered**:

- _`SELECT VECTOR_COSINE(col, col) FROM tbl` as the probe_ — this is the documented
  manual verification and it does work, but it is a query against user data, it
  returns nothing useful on an empty table (measured: zero rows on `KGEntities`,
  so no confirmation either way), and it cannot report the dimension. Rejected as
  the mechanism; it stays in the integration suite as an independent check that
  the shape's verdict matches reality.
- _Trusting a configured dimension_ — rejected outright by FR-001 and by the
  observed defect (a fixed 1536 in one path, a fixed 384 in another).

## Decision 4: a non-vector embedding column is a recorded fact, not a fallback

**Decision**: `TableShape` records the embedding column's actual type. If a column
occupying the embedding role is not `%Library.Vector`, the shape says so and any
vector search against that table raises immediately, naming the table, the column
and the type found.

**Rationale**: this is not hypothetical. The local `RAG.DocumentChunks.embedding`
is `varbinary` / `%Library.Binary`, and `RAG.SourceDocuments` has no embedding
column at all — the two shapes that the order-dependent contract test trips over.
Today those produce either a driver error deep inside a query or a silent
`FLOAT` guess. A shape that can answer "this table cannot be searched by vector,
because its embedding column is `%Library.Binary`" converts both into one clear
message at the boundary, which is what SC-005 asks for.

**Alternatives considered**: coercing (`TO_VECTOR` on a binary column) — rejected;
it would write plausible garbage. Auto-altering the column — rejected by US3 (no
implicit alter) and by the project's hard rule against altering on the basis of a
reported type.

## Decision 5: identity and primary key come from `INFORMATION_SCHEMA`

**Decision**: read `IS_IDENTITY` and `AUTO_INCREMENT` per column from
`INFORMATION_SCHEMA.COLUMNS`, and the key columns from
`INFORMATION_SCHEMA.KEY_COLUMN_USAGE`.

**Rationale**: both are available and cheap — measured on `RAG.SourceDocuments`:

```text
COLUMN_NAME   DATA_TYPE     CHAR_MAX_LEN  IS_IDENTITY  AUTO_INCREMENT  IS_NULLABLE
id            varchar       255           NO           NO              NO
content       longvarchar   2147483647    NO           NO              NO
...
KEY_COLUMN_USAGE -> id
```

The identifier role therefore does not have to be guessed from a name at all when
the table declares a key: the shape prefers the declared key column, and only
falls back to the canonical-name precedence (`doc_id`, then `id`) when a table has
no key. This matters because an identity column cannot be written to, so a writer
that picks the wrong identifier column fails at insert time rather than at
resolution time.

**Alternatives considered**: name-only precedence (what all eleven current sites
do) — kept only as the tiebreak, since it is what produced the contradiction the
feature exists to remove.

## Decision 6: a stream column is one whose class type is under `%Stream`

**Decision**: `is_stream` is true iff `Type` starts with `%Stream.`; the SQL type
(`longvarchar` with length 2147483647) is recorded alongside but is not the test.

**Rationale**: `chunk_text` is `longvarchar` in SQL and `%Stream.GlobalCharacter`
in the class. `content` on `RAG.SourceDocuments` is also `longvarchar` at
2147483647 but is a plain string property. The distinction decides two behaviours
the codebase already gets wrong in places: a stream value read through DBAPI needs
reading, and `LIKE` against a long/stream column silently matches nothing, so a
metadata or text filter must use `SUBSTRING` instead. Both are properties of the
column, so both belong on the shape rather than in a `try`/`except` at the call
site — which is what `vector_store_iris.py:1452-1500` and `:1551-1590` do today.

**Alternatives considered**: treating `CHARACTER_MAXIMUM_LENGTH = 2147483647` as
the stream test — rejected: it is true for both cases above, so it would make
every long string a stream.

## Decision 7: `IRIS_VECTOR_DATA_TYPE` stops being a fallback

**Decision**: remove the `os.environ.get("IRIS_VECTOR_DATA_TYPE") or "FLOAT"`
tail of `_get_vector_data_type`. If the catalog does not describe the embedding
column, that is an error (FR-001), not a default.

**Rationale**: the environment variable exists to paper over the underscore bug in
Decision 1 and the mapping bug in Decision 2. With both fixed there is no case
where the catalog is silent about a column that exists. Keeping the fallback keeps
the failure mode: a mistyped table name yields `FLOAT`/1536 and a query that fails
several layers away from the cause. The variable stays honoured for _creation_
defaults, which is configuration deciding what to build, not code guessing at
what exists.

**Alternatives considered**: warn-and-default — rejected; the spec's SC-005 is
about the first error message naming the real cause, and a warning is not read by
a test.

## Decision 8: one resolution per table per session, cached on the session

**Decision**: the resolver caches by `(schema, table)` on the object that owns the
connection, with an explicit `invalidate(table)` used after DDL. No module-level
or process-level cache.

**Rationale**: Community Edition runs with one connection, so round trips are the
scarce resource and a search currently pays three to five before its first real
query. A process-global cache would be wrong for a different reason: two sessions
can point at different instances or namespaces, and a shape from one is not valid
for the other. Tying the cache lifetime to the session makes the invalidation rule
trivial — the DDL that changes a shape goes through the same session.

**Alternatives considered**: TTL-based expiry — rejected as unnecessary
non-determinism in tests; a shape only changes when this process alters the table,
and that path can invalidate explicitly.

## Decision 9: the test seam is a recorded catalog, captured from real DDL

**Decision**: `CatalogReader` is a protocol with two implementations —
`IRISCatalogReader` (runs the two queries) and `RecordedCatalog` (returns rows from
a fixture). Fixtures are generated by creating each of the six shapes for real and
dumping the two result sets.

**Rationale**: this is the interface-is-the-test-surface point. Today a unit test
that wants to exercise a legacy-shaped table has to script fake cursor responses
in the right order, which encodes the number and order of the queries into the
test — so the tests break when the query count changes, which is exactly what US2
does. A recorded catalog is one function of `(schema, table)`, so the same unit
test survives the round-trip reduction. Generating the fixtures from real DDL keeps
them honest; hand-written rows would drift from what IRIS actually reports (and
would have "corrected" the `varchar` in Decision 3).

**Alternatives considered**: `unittest.mock` on the connection — rejected, that is
the status quo. An in-memory SQLite stand-in — rejected: none of the four facts
above have SQLite equivalents.

## Resolved unknowns

| Question from Technical Context               | Resolution                                      |
| --------------------------------------------- | ----------------------------------------------- |
| Can identity columns be detected?             | Yes — `IS_IDENTITY`, `AUTO_INCREMENT` (Dec. 5)  |
| Can `VECTOR` element type/length be read?     | Yes — `%Library.Vector` + `Parameters` (Dec. 3) |
| Can streams be told apart from long strings?  | Yes — `%Stream.*` class type (Dec. 6)           |
| How is a table name mapped to a class?        | `%Dictionary.CompiledClass` lookup (Dec. 2)     |
| How many round trips per shape?               | Three: class name, columns, properties (Dec. 1) |

Three round trips per table per session, once, replaces three to five per search.

## Notes for the contract

Two facts belong in `contracts/table-shape.md` because callers depend on them:

- Column names are compared case-insensitively and stored as the catalog reports
  them; SQL is emitted with the reported spelling.
- The mixed-shape read rule from `plan.md` (read the canonical text column, fall
  back per row to the legacy column when it is null) is a documented behaviour of
  the read path, not a shape precedence rule. The shape reports both columns; the
  read path decides.
