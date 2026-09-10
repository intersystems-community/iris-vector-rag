# Phase 1 Data Model: Table Shape

**Feature**: 082-table-shape | **Date**: 2026-09-10
**Depends on**: [research.md](./research.md) decisions 1-9

Every type here is a value: frozen, comparable, hashable where useful, and
constructed only from catalog rows or from a literal in a test fixture. Nothing in
this model holds a connection, and nothing in it emits SQL. That is the point of
the seam — the shape is data, so a test can hand one over and a call site cannot
tell the difference.

## Entities

### `ColumnRole` (enum)

The question a call site actually asks. Six roles, all optional on a given table
except `IDENTIFIER`:

| Role         | Meaning                                        | Canonical column |
| ------------ | ---------------------------------------------- | ---------------- |
| `IDENTIFIER` | the value a caller uses to name a document     | `doc_id`         |
| `TEXT`       | the retrievable body text                      | `text_content`   |
| `TITLE`      | short display text                             | `title`          |
| `METADATA`   | the JSON blob metadata filters run against     | `metadata`       |
| `EMBEDDING`  | the vector searched by similarity              | `embedding`      |
| `CREATED_AT` | insertion timestamp                            | `created_at`     |

Roles the codebase currently derives inline and will stop deriving: `IDENTIFIER`
(five sites), `TEXT` (four sites, two precedence rules), `EMBEDDING` (two sites),
`METADATA` (one site, type only).

### `ColumnFact`

One column as the catalog reports it. Frozen.

| Field           | Type            | Source                                            |
| --------------- | --------------- | ------------------------------------------------- |
| `name`          | `str`           | `INFORMATION_SCHEMA.COLUMNS.COLUMN_NAME`          |
| `sql_type`      | `str`           | `DATA_TYPE`, lowercased                           |
| `max_length`    | `int \| None`   | `CHARACTER_MAXIMUM_LENGTH`                        |
| `nullable`      | `bool`          | `IS_NULLABLE`                                     |
| `position`      | `int`           | `ORDINAL_POSITION`                                |
| `class_type`    | `str \| None`   | `%Dictionary.CompiledProperty.Type`               |
| `is_identity`   | `bool`          | `IS_IDENTITY` or `AUTO_INCREMENT` (either)        |
| `is_key`        | `bool`          | `INFORMATION_SCHEMA.KEY_COLUMN_USAGE`             |
| `vector`        | `VectorSpec \| None` | parsed `Parameters` when `class_type` is `%Library.Vector` |

Derived, not stored: `is_stream` (`class_type` starts with `%Stream.`),
`is_long_text` (`sql_type == "longvarchar"`), `filterable_with_like`
(`not is_stream and not is_long_text` — see FR-009; anything else needs
`SUBSTRING`).

`class_type` is `None` only when the class view has no property for the column,
which happens for `%%OID`-style system fields and must never happen for a role
column; if it does, resolution raises (FR-001).

### `VectorSpec`

| Field        | Type  | Source                              |
| ------------ | ----- | ----------------------------------- |
| `element_type` | `str` | `Parameters` key `DATATYPE` (`FLOAT`, `DOUBLE`, ...) |
| `dimension`    | `int` | `Parameters` key `LEN`              |

This is the whole of what `TO_VECTOR()` and the dimension check need, and it is the
replacement for both the fixed 1536 and the `IRIS_VECTOR_DATA_TYPE` fallback.

### `TableShape`

The aggregate. Frozen, and the only type call sites see.

| Field       | Type                        | Notes                                  |
| ----------- | --------------------------- | -------------------------------------- |
| `schema`    | `str`                       | e.g. `RAG`                             |
| `table`     | `str`                       | e.g. `SourceDocuments`                 |
| `class_name`| `str`                       | from `%Dictionary.CompiledClass` (Dec. 2) |
| `columns`   | `Mapping[str, ColumnFact]`  | keyed by lowercased name, insertion-ordered by `position` |
| `roles`     | `Mapping[ColumnRole, str]`  | role to column name; a role absent from the table is absent here |
| `kind`      | `ShapeKind`                 | classification, informational only     |

Behaviour (no SQL, no I/O):

- `column_for(role) -> ColumnFact` — raises `RoleNotPresent` naming table and role.
- `optional_column_for(role) -> ColumnFact | None`.
- `has(role) -> bool`.
- `legacy_column_for(role) -> ColumnFact | None` — the non-canonical column of the
  same family when the table carries both (the mixed shape). This is what lets the
  read path implement the per-row fallback without a precedence rule.
- `vector_spec() -> VectorSpec` — raises `NotVectorSearchable` when the embedding
  role is absent or its column is not `%Library.Vector`, naming the type found
  (Decision 4).
- `writable_identifier() -> ColumnFact` — raises `IdentifierNotWritable` when the
  identifier column `is_identity` (FR-008).
- `is_canonical` — every role present, every role bound to its canonical column
  name, embedding a `%Library.Vector`.

`roles` is resolved once, at construction, by this precedence — declared key first,
then canonical name, then legacy names in a fixed order:

| Role         | Precedence                                                     |
| ------------ | -------------------------------------------------------------- |
| `IDENTIFIER` | declared single-column key -> `doc_id` -> `id` -> `document_id` |
| `TEXT`       | `text_content` -> `content` -> `chunk_text` -> `abstract`       |
| `TITLE`      | `title` -> `source` -> `filename`                               |
| `METADATA`   | `metadata`                                                      |
| `EMBEDDING`  | `embedding` -> `chunk_embedding` -> `entity_embedding`           |
| `CREATED_AT` | `created_at`                                                    |

FR-003 is the `TEXT` row: canonical wins, unconditionally, and there is exactly one
such rule in the codebase after this feature — replacing the reader's
`content`-first rule at `vector_store_iris.py:1110-1120` and the writer's
`text_content`-first rule at `:701-715`, which currently disagree.

### `ShapeKind` (enum)

Classification of the six observed shapes, for messages and test parametrisation.
Nothing branches on it — behaviour comes from `roles` and `ColumnFact`.

| Kind             | Identifier            | Text           | Embedding             |
| ---------------- | --------------------- | -------------- | --------------------- |
| `CANONICAL`      | `doc_id` varchar      | `text_content` | `VECTOR(FLOAT, 384)`  |
| `LEGACY_SERIAL`  | integer, identity     | `content`      | `VECTOR(FLOAT, n)`    |
| `LEGACY_ID`      | `id` varchar          | `content`      | absent or non-vector  |
| `FIXTURE`        | `doc_id`              | `content`      | `VECTOR(FLOAT, 384)`  |
| `MIXED`          | `doc_id` (+ `id`)     | both families  | `VECTOR(FLOAT, 384)`  |
| `DOUBLE_VECTOR`  | `doc_id`              | `text_content` | `VECTOR(DOUBLE, 384)` |
| `UNKNOWN`        | anything else that still resolves every required role       |

`UNKNOWN` is not an error: FR-002 names six shapes, and the assumption in the spec
is that a seventh gets added as a fixture. A shape that resolves its roles works
whether or not this enum has a name for it.

## The catalog seam

### `CatalogSnapshot`

The raw material, and the unit a fixture records. Frozen.

| Field       | Type                     | Content                                      |
| ----------- | ------------------------ | -------------------------------------------- |
| `schema`    | `str`                    |                                              |
| `table`     | `str`                    |                                              |
| `class_name`| `str \| None`            | `None` when no class maps to the table       |
| `columns`   | `Sequence[Mapping]`      | `INFORMATION_SCHEMA.COLUMNS` rows            |
| `properties`| `Sequence[Mapping]`      | `%Dictionary.CompiledProperty` rows          |
| `key_columns` | `Sequence[str]`        | `KEY_COLUMN_USAGE` rows                      |

`TableShape.from_snapshot(snapshot)` is a pure function. Every unit test in this
feature calls it, and it is the reason the shape's tests need no database.

### `CatalogReader` (protocol)

```python
class CatalogReader(Protocol):
    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None: ...
```

`None` means the table does not exist — distinct from a snapshot whose `columns`
are empty, which would be a table that exists with no columns and is an error.

Two implementations:

- `IRISCatalogReader(connection)` — the three queries of Decision 1/2/5. The only
  place in the package that reads the catalog for column shape; SC-003's detector
  asserts exactly that.
- `RecordedCatalog(snapshots)` — a mapping from `(schema, table)` to
  `CatalogSnapshot`, with a loader for the JSON fixtures captured from real DDL.
  This is what replaces the scripted cursor responses US4 is about.

### `TableShapeResolver`

Owns the cache, not the shape.

```python
resolver = TableShapeResolver(reader)
shape = resolver.resolve("RAG", "SourceDocuments")   # 3 round trips, first call
shape = resolver.resolve("RAG", "SourceDocuments")   # 0 round trips
resolver.invalidate("RAG", "SourceDocuments")        # after DDL
```

- `resolve(schema, table) -> TableShape` — raises `TableNotFound` when the reader
  returns `None`.
- `invalidate(schema, table)` / `invalidate_all()` — the library's own DDL paths
  call the former (FR-005, US2 scenario 2); a caller suspecting an outside change
  calls it explicitly (US2 scenario 3).
- Cache is per-resolver, and a resolver is owned by whatever owns the connection
  (Decision 8). No module-level state, so two sessions against two instances cannot
  see each other's shapes.

## Errors

One base class so a caller can catch the family, and a distinct type per cause so
the first error message names the real reason (FR-011, SC-005). Every message
carries schema, table and — where relevant — role, column and type found.

| Error                    | Raised when                                                |
| ------------------------ | ---------------------------------------------------------- |
| `TableShapeError`        | base                                                       |
| `TableNotFound`          | no such table in the namespace                             |
| `ShapeUnresolvable`      | table exists but a required role has no candidate column   |
| `RoleNotPresent`         | a caller asks for a role this table does not have          |
| `NotVectorSearchable`    | embedding role absent, or its column is not `%Library.Vector` |
| `IdentifierNotWritable`  | caller-supplied identifier into an identity column         |
| `CatalogUnavailable`     | the catalog query itself failed (wraps the driver error)    |

`ShapeUnresolvable` is where the deleted fallbacks land. What used to be
`or "FLOAT"`, `or 1536`, or a `try`/`except` retry with legacy SQL is now one of
these, named, at the boundary.

## Relationships

```text
CatalogReader ──snapshot()──> CatalogSnapshot ──from_snapshot()──> TableShape
      ▲                                                                │
      │                                                        column_for(role)
IRISCatalogReader (connection)                                          │
RecordedCatalog (fixtures)                                              ▼
                                                                  ColumnFact
TableShapeResolver ─── caches ───> TableShape                     └─ VectorSpec
      ▲
      └── owned by IRISVectorStore / SchemaManager (one per session)
```

## What each requirement maps to

| Requirement | Where it lives in this model                                   |
| ----------- | -------------------------------------------------------------- |
| FR-001      | `IRISCatalogReader` (three queries), no defaults anywhere       |
| FR-002      | `ShapeKind` fixtures + `roles` precedence                       |
| FR-003      | `roles` precedence table, `TEXT` row                            |
| FR-004      | `column_for(role)` is the only way to name a column             |
| FR-005      | `TableShapeResolver` cache + `invalidate`                       |
| FR-006      | canonical DDL emitted from the canonical `TableShape`            |
| FR-007      | no `alter` anywhere in this model                               |
| FR-008      | `writable_identifier()` / `IdentifierNotWritable`               |
| FR-009      | `ColumnFact.is_stream`, `filterable_with_like`                  |
| FR-010      | `VectorSpec` from the catalog                                   |
| FR-011      | the error table above                                           |
| FR-012      | both graph joins call `column_for(IDENTIFIER)`                  |
| FR-013      | resolver replaces the non-existent schema operations            |
| FR-014      | nothing here is on a public signature; shape is internal        |
| FR-015      | `RecordedCatalog` + `from_snapshot` purity                      |
