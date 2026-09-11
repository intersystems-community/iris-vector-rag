# Contract: `iris_vector_rag.storage.table_shape`

**Feature**: 082-table-shape | **Date**: 2026-09-10
**Status**: contract for 0.15; additive — no public signature changes (FR-014)

This is a library module, not an HTTP surface, so the contract is the module's
public names and the promises attached to them. The contract tests in
`tests/unit/test_table_shape.py` and `tests/unit/test_shape_resolver.py` are the
executable form of this document; every numbered promise below has a test.

## Public surface

```python
# iris_vector_rag/storage/table_shape.py
class ColumnRole(Enum): IDENTIFIER, TEXT, TITLE, METADATA, EMBEDDING, CREATED_AT
class ShapeKind(Enum): CANONICAL, LEGACY_SERIAL, LEGACY_ID, FIXTURE, MIXED, DOUBLE_VECTOR, UNKNOWN

@dataclass(frozen=True)
class VectorSpec:
    element_type: str
    dimension: int

@dataclass(frozen=True)
class ColumnFact:
    name: str
    sql_type: str
    max_length: int | None
    nullable: bool
    position: int
    class_type: str | None
    is_identity: bool
    is_key: bool
    vector: VectorSpec | None
    @property
    def is_stream(self) -> bool: ...
    @property
    def is_long_text(self) -> bool: ...
    @property
    def filterable_with_like(self) -> bool: ...

@dataclass(frozen=True)
class TableShape:
    schema: str
    table: str
    class_name: str
    columns: Mapping[str, ColumnFact]
    roles: Mapping[ColumnRole, str]
    kind: ShapeKind

    @classmethod
    def from_snapshot(cls, snapshot: CatalogSnapshot) -> "TableShape": ...
    def column_for(self, role: ColumnRole) -> ColumnFact: ...
    def optional_column_for(self, role: ColumnRole) -> ColumnFact | None: ...
    def has(self, role: ColumnRole) -> bool: ...
    def legacy_column_for(self, role: ColumnRole) -> ColumnFact | None: ...
    def vector_spec(self) -> VectorSpec: ...
    def writable_identifier(self) -> ColumnFact: ...
    @property
    def is_canonical(self) -> bool: ...

CANONICAL_DOCUMENT_SHAPE: TableShape   # the shape every creation path emits
CANONICAL_CHUNK_SHAPE: TableShape

def create_table_statement(
    shape: TableShape,
    *,
    schema: str | None = None,
    table: str | None = None,
    vector: VectorSpec | None = None,
) -> str: ...                          # the CREATE TABLE a canonical shape describes

class TableShapeError(Exception): ...
class TableNotFound(TableShapeError): ...
class ShapeUnresolvable(TableShapeError): ...
class RoleNotPresent(TableShapeError): ...
class NotVectorSearchable(TableShapeError): ...
class IdentifierNotWritable(TableShapeError): ...
class CatalogUnavailable(TableShapeError): ...
class NonCanonicalCreation(TableShapeError): ...
```

```python
# iris_vector_rag/storage/catalog.py
@dataclass(frozen=True)
class CatalogSnapshot:
    schema: str
    table: str
    class_name: str | None
    columns: Sequence[Mapping[str, Any]]
    properties: Sequence[Mapping[str, Any]]
    key_columns: Sequence[str]

class CatalogReader(Protocol):
    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None: ...

class IRISCatalogReader:
    def __init__(self, connection: Any) -> None: ...
    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None: ...
    def index_names(self, table: str, cursor: Any = None) -> set[str]: ...
    # not on the protocol: a shape describes columns, and no read or write path
    # branches on an index. It lives here because this module owns the catalog SQL.

class RecordedCatalog:
    def __init__(self, snapshots: Mapping[tuple[str, str], CatalogSnapshot]) -> None: ...
    @classmethod
    def from_fixture_dir(cls, path: Path) -> "RecordedCatalog": ...
    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None: ...
```

```python
# iris_vector_rag/storage/shape_resolver.py
class TableShapeResolver:
    def __init__(self, reader: CatalogReader) -> None: ...
    def resolve(self, schema: str, table: str) -> TableShape: ...
    def invalidate(self, schema: str, table: str) -> None: ...
    def invalidate_all(self) -> None: ...
    @property
    def reads(self) -> int: ...   # catalog snapshots taken; SC-002 counts this
```

## Promises

### Resolution

1. **No defaults.** `from_snapshot` never substitutes a value the snapshot does not
   contain. A missing vector element type or dimension raises `ShapeUnresolvable`;
   it does not become `FLOAT`, 384 or 1536, and it does not consult the
   environment. (FR-001)
2. **Canonical wins.** When a table carries both a canonical and a legacy column
   for the same role, `column_for` returns the canonical one. The full precedence
   is the table in [data-model.md](./data-model.md#tableshape); it is the only such
   rule in the package. (FR-003)
3. **The declared key outranks names.** When a table declares a single-column
   primary key, that column is the `IDENTIFIER` regardless of its name. Name
   precedence applies only to tables without one. (Decision 5)
4. **Case-insensitive lookup, catalog spelling in SQL.** `columns` is keyed by
   lowercased name; `ColumnFact.name` preserves what the catalog reported, and SQL
   is emitted with that spelling.
5. **Vector-ness comes from the class view.** `ColumnFact.vector` is populated iff
   `class_type == "%Library.Vector"`. A column that `INFORMATION_SCHEMA` reports as
   `varchar` but the class view reports as `%Library.Vector` is a vector; the
   reverse is not a vector. No caller may re-derive this, and nothing in the
   package alters a column on the basis of the reported SQL type. (Decision 3)
6. **`from_snapshot` is pure.** No I/O, no clock, no environment. Given the same
   snapshot it returns an equal `TableShape`. (FR-015)

### Reads and writes

7. **One consistent document form.** Callers name roles, never columns. Every
   read presents identifier, text, metadata and score under the same keys on every
   shape. (FR-004)
8. **Mixed-table read fallback.** On a table where `legacy_column_for(TEXT)` is not
   `None`, the read path selects both columns and uses the legacy value **for that
   row** when the canonical value is null. This is a documented behaviour of the
   read path, not a shape precedence rule: `column_for(TEXT)` still returns the
   canonical column. Rows written by this release always populate the canonical
   column. (US1 scenario 2)
9. **Identity columns refuse caller identifiers.** `writable_identifier()` raises
   `IdentifierNotWritable` naming schema, table and column when the identifier
   column is an identity/auto-increment column, instead of attempting the insert.
   (FR-008)
10. **Stream and long columns are never filtered with `LIKE`.** A filter against a
    column whose `filterable_with_like` is false is expressed with `SUBSTRING`.
    `LIKE` on such a column matches nothing silently, so this is a correctness
    promise, not an optimisation. (FR-009)
11. **Vector search uses the table's own type and dimension.** `TO_VECTOR` is
    emitted with `vector_spec().element_type`, and a query embedding whose length
    differs from `vector_spec().dimension` is rejected before the query is sent,
    naming both lengths. A double-precision table succeeds on the first attempt,
    with no retry and no exception path. (FR-010, US1 scenario 4)
12. **No implicit alter.** Nothing reachable from this module issues
    `ALTER TABLE`, `DROP`, or a create-if-different. Creation paths emit
    `CANONICAL_DOCUMENT_SHAPE` / `CANONICAL_CHUNK_SHAPE` and nothing else; a
    non-canonical existing table is used as it is. `SchemaManager.migrate_table` is
    the one destructive path, and it only runs when a caller names it. (FR-006,
    FR-007) `create_table_statement` renders the `CREATE TABLE` from the shape, so a
    creation path cannot type a second spelling of a table it claims is canonical:
    only the vector element type and dimension may be supplied, and every column
    name, order and type comes from the shape. A non-canonical shape raises
    `NonCanonicalCreation` naming the table. (SC-004)

### Caching

13. **At most one snapshot per table per resolver.** N calls to `resolve` for the
    same table take one snapshot. `reads` counts snapshots and is the observable
    SC-002 asserts on. (FR-005)
14. **The library invalidates its own DDL.** Any path in the package that creates
    or alters a table calls `invalidate` for it, so the next operation sees the new
    shape with no caller action. In `IRISVectorStore` that is
    `_ensure_table_exists`, the one method every write path reaches creation
    through. A change made outside the library is the caller's to declare:
    `IRISVectorStore.refresh_shape(table=None)` invalidates and re-resolves one
    table — the store's own by default — and returns the shape every later
    operation will use. It is the only supported way to drop a cached shape.
    (US2 scenario 2, US2 scenario 3)
15. **No shared state.** The cache lives on the resolver instance. Two resolvers —
    two sessions, two instances, two namespaces — never observe each other's
    shapes. (Decision 8)

### Errors

16. **The first error names the cause.** Every raise from this module carries
    schema, table, and the role, column name or type that made it fail. No error
    from this module is raised without at least schema and table. (FR-011, SC-005)
17. **Absent table and empty table are different.** `CatalogReader.snapshot`
    returns `None` for a table that does not exist, which becomes `TableNotFound`.
    A snapshot with zero column rows is a different failure
    (`ShapeUnresolvable`).
18. **Driver failures are wrapped, not swallowed.** A failing catalog query becomes
    `CatalogUnavailable` with the driver error as `__cause__`. It is never caught
    and turned into a default shape.

### Stability

19. **Additive in 0.15.** No public entry point changes signature. `TableShape` and
    the resolver are internal collaborators of the storage layer. (FR-014)
20. **`schema_config` column overrides are deprecated.** Configuration that names
    an identifier or text column emits a `DeprecationWarning` and is ignored — the
    database decides the shape, not configuration. Removed in 0.16.

## Non-promises

- The module does not migrate data or rename columns.
- It does not create tables; it describes the shape a creation path must emit.
- It does not cache across processes, and it makes no claim about a table altered
  by another process between two `resolve` calls beyond promise 15's scope — a
  caller that suspects one calls `invalidate`.
- `ShapeKind` is informational. No behaviour branches on it, and a new observed
  shape does not require a new member to work.
