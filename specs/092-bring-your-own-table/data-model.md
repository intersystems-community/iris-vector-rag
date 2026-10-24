# Data Model: Bring Your Own Table

No tables are added. The model is the in-memory description of a table the library reads.

## ColumnMapping (new, `storage/attached.py`)

| Field           | Type                        | Source key (`storage:iris:`) | Rule                                                     |
| --------------- | --------------------------- | ---------------------------- | -------------------------------------------------------- |
| `id_cols`       | tuple[str, ...]             | `id_col` (string or list)    | One or more columns. Several are joined with `#`.        |
| `text_col`      | str or None                 | `text_col`                   |                                                          |
| `embedding_col` | str or None                 | `embedding_col`              | Must be a VECTOR column for a vector search.             |
| `metadata_col`  | str or None                 | `metadata_col`               | The JSON metadata filter applies here, as today.          |
| `title_col`     | str or None                 | `title_col`                  | Surfaces as `metadata["title"]`.                          |
| `parent_col`    | str or None                 | `parent_col`                 | Surfaces as `metadata["parent_id"]`.                      |

- `declared` maps each set field to its `ColumnRole`.
- `attached` is true when any field is set.
- Constructor arguments of the same names override configuration field by field.

## FilterSpec (new)

| Field              | Type            | Source key         | Rule                                                        |
| ------------------ | --------------- | ------------------ | ----------------------------------------------------------- |
| `filter_columns`   | tuple[str, ...] | `filter_columns`   | Each must exist in the table (checked on first use).         |
| `required_filters` | tuple[str, ...] | `required_filters` | Each must be in `filter_columns`.                            |

**A search's `where`:** `Mapping[str, value | (op, value) | list]`.
- `op` is one of `=`, `<`, `<=`, `>`, `>=`.
- A list means `IN` and must not be empty.
- Values are `str`, `int`, `float`, `bool`, `date` or `datetime`.

## SearchSettings (new fields on the store)

| Field           | Values             | Default                                | Source key      |
| --------------- | ------------------ | -------------------------------------- | --------------- |
| `metric`        | `cosine`, `dot`    | `cosine` if attached, else `dot`       | `metric`        |
| `text_index`    | identifier or None | `SourceDocumentsTextIdx` when not attached | `text_index` |
| `text_language` | two-letter code    | `en` (from `feat/text-language-setting`) | `text_language` |
| `text_terms`    | `words`, `prefix`  | `words`                                | `text_terms`    |
| `embedder`      | callable or None   | None (the configured provider)         | constructor only |

## TableShape (changed, `storage/table_shape.py`)

- `roles` may now come from a declaration. A declared column outranks `ROLE_PRECEDENCE`.
- New `identifier_columns: tuple[str, ...]`, empty unless the id is composite.
- New `attached: bool`.
- New `identifier_sql: str`.
- New `ColumnRole.PARENT`, which can only be declared.

## Errors

| Error                                                   | When                                                                       |
| ------------------------------------------------------- | -------------------------------------------------------------------------- |
| `ShapeUnresolvable`                                     | A declared column is missing from the table.                               |
| `FilterError` (new, `VectorStoreConfigurationError`)    | A `where` key outside `filter_columns`, a bad op, an empty list, a bad value type, or a filter column missing from the table. |
| `RequiredFilterMissing` (new, subclass of `FilterError`) | A required filter was omitted.                                            |
| `DimensionMismatchError` (existing)                     | The query vector's length differs from the column's.                       |
| `AttachedTableReadOnly` (new, `VectorStoreDataError`)   | A write method was called on an attached table.                            |
| `TextIndexMissingError` (existing)                      | No text index was named or found.                                          |
