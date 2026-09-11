# TableShape

`TableShape` is what the library knows about one document or chunk table: which of
its columns is the identifier, which holds the text, which holds the metadata, which
holds the embedding, and what that embedding column actually declares.

It exists because the same table used to be described in three places at once — a
configuration file, a set of constants, and whatever the SQL string in front of you
happened to say. On a table that carried both `text_content` and `content`, the write
path picked one column and the read path picked the other, so documents loaded
through the library were invisible to search. The catalog is the only account of a
table that cannot be stale, so the catalog is the one the library reads.

## The three modules

| Module                                   | Owns                                                                |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `iris_vector_rag.storage.catalog`        | Every `INFORMATION_SCHEMA` and `%Dictionary.*` query in the package |
| `iris_vector_rag.storage.table_shape`    | What the catalog rows mean: roles, vector spec, errors              |
| `iris_vector_rag.storage.shape_resolver` | Caching, so one table costs one catalog read                        |

```python
from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import ColumnRole

shapes = TableShapeResolver(IRISCatalogReader(connection))
shape = shapes.resolve("RAG", "SourceDocuments")

shape.column_for(ColumnRole.IDENTIFIER).name   # 'doc_id'
shape.vector_spec().element_type                # 'FLOAT'
shape.vector_spec().dimension                   # 384
```

## Roles

A call site asks for a role, never a column name.

| Role         | Resolution order (first match wins)                 |
| ------------ | --------------------------------------------------- |
| `IDENTIFIER` | `doc_id`, `id`, `document_id`                       |
| `TEXT`       | `text_content`, `content`, `chunk_text`, `abstract` |
| `TITLE`      | `title`, `source`, `filename`                       |
| `METADATA`   | `metadata`                                          |
| `EMBEDDING`  | `embedding`, `chunk_embedding`, `entity_embedding`  |
| `CREATED_AT` | `created_at`                                        |

The canonical column is first in every list, so a table carrying both `text_content`
and `content` resolves to `text_content` in the reader and the writer alike. A
declared single-column primary key outranks the name order for `IDENTIFIER`: a table
whose key is `id` resolves to `id` even if it also has a `doc_id`.

Lookup is case-insensitive; `ColumnFact.name` keeps the catalog's own spelling, so
the SQL the library emits reads the way the DDL was written.

### Mixed tables

`legacy_column_for(role)` returns the _alias_ a role also matched — `content` on a
table that has both `text_content` and `content`, `None` on a table that has only
one. The read path uses it as a per-row fallback: a row whose canonical column is
empty is read from the legacy one, so data written before the table gained its
canonical column stays readable. Writes only ever go to the canonical column.

The alias list is deliberately narrower than the resolution order. `abstract` can
fill the `TEXT` role for a table that carries nothing else, but it is not an alias of
`text_content` — the canonical shape has both and they hold different things.

## API

| Member                              | Answers                                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `column_for(role)`                  | The column filling the role; raises `RoleNotPresent` if none does                            |
| `optional_column_for(role)`         | The same, or `None`                                                                          |
| `has(role)`                         | Whether any column fills the role                                                            |
| `legacy_column_for(role)`           | The alias column, for the read fallback described above                                      |
| `vector_spec()`                     | The embedding column's element type and length; raises `NotVectorSearchable`                 |
| `writable_identifier()`             | The identifier, if a caller may supply it; raises `IdentifierNotWritable` on a generated key |
| `qualified_name`, `schema`, `table` | Names for the SQL                                                                            |
| `columns`                           | Every `ColumnFact`, keyed by lower-cased name                                                |
| `kind`, `is_canonical`              | Classification, informational only                                                           |

A `ColumnFact` carries `name`, `sql_type`, `max_length`, `nullable`, `position`,
`class_type`, `is_identity`, `is_key`, `vector`, and three derived answers: `is_stream`
(the class type is a `%Stream.*`), `is_long_text` (`longvarchar`), and
`filterable_with_like` (false for both — IRIS silently matches nothing when `LIKE` is
applied to a `LONGVARCHAR`, so filters on such a column go through `SUBSTRING`).

## Errors

All of them subclass `TableShapeError` and every message is prefixed with
`schema.table`, because the table is the thing you have to go look at.

| Error                   | Raised when                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `TableNotFound`         | The catalog has no such table                                                |
| `ShapeUnresolvable`     | The table exists but cannot answer `IDENTIFIER` and `TEXT`                   |
| `RoleNotPresent`        | A required role has no column; the message names the role and the candidates |
| `NotVectorSearchable`   | No embedding column, or one whose type is not a vector                       |
| `IdentifierNotWritable` | A caller supplied an identifier for a generated key                          |
| `CatalogUnavailable`    | The catalog query itself failed; the driver error is the `__cause__`         |
| `NonCanonicalCreation`  | A creation path was asked to emit something other than the canonical shape   |

## Injecting a shape

Every component that resolves a shape takes an optional `shapes=` resolver:

```python
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver

shapes = TableShapeResolver(RecordedCatalog.from_fixture_dir("tests/fixtures/shapes"))
store = IRISVectorStore(connections, config, shapes=shapes)
```

`IRISVectorStore`, `EntityStorageAdapter`, `SetupOrchestrator` and
`HybridGraphRAGPipeline` all accept it. The recorded fixtures under
`tests/fixtures/shapes/` were captured from real DDL on a real instance by
`scripts/dev/capture_shape_fixtures.py`, and an integration test asserts each one
still matches what a live catalog reports — so a unit test using them asserts against
what IRIS says, not against what a hand-written fixture wishes it said.

## Caching and invalidation

`TableShapeResolver` reads each table once per instance, and `reads` counts how many
times it went to the catalog. `refresh_shape(table=None)` on the vector store — or
`invalidate(schema, table)` / `invalidate_all()` on the resolver — drops the cached
answer, which matters after a migration changes the table under a live process.
Nothing else invalidates: a table's shape does not change while a query runs.

## What this replaced

- Column names from `schema_config` — the `id_column`, `text_column`,
  `title_column`, `metadata_column` and `embedding_column` keys in the
  `configuration` JSON of `RAG.SchemaMetadata`. They are now ignored and raise a
  `DeprecationWarning` naming the keys found.
- `IRIS_VECTOR_DATA_TYPE` as a read-path fallback. It still applies to table
  creation; it can no longer contradict a table that exists.
- Canonical-then-legacy retry loops, which discovered the schema by letting a
  statement fail. One statement is built from the shape instead.
- A silent `FLOAT` fallback when no embedding column existed, which turned a broken
  table into an empty result set.
