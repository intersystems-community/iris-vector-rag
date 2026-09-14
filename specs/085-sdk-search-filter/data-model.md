# Data Model: RAG.SDK search

Three shapes matter here. None of them is a persistent class — they are the JSON the SDK
returns and the configuration it reads.

## Search hit

One element of the array returned by `VectorSearch`, `TextSearch`, and `HybridSearch`.

| Field | Type | Present when | Notes |
| ----- | ---- | ------------ | ----- |
| `id` | string | always | The attached table's identifier column value |
| `text` | string | always | Read from the text column; a stream column is read to 2000 characters, as before |
| `score` | number | always | `VECTOR_COSINE` for vector search, `1` for text search, fused value for hybrid |
| `metadata` | object or string | only when the searched table has a metadata column and the row's value is non-empty | Parsed when the stored value is valid JSON; carried through as a string when it is not (FR-007) |

A hit never gains a field the table cannot answer for: a table with no metadata column
produces hits with no `metadata` key, which is distinguishable from `"metadata": {}`.

## Attached table configuration

Read from `^RAG.SDK.Config` by `RAG.SDK.Search.GetTableConfig()`, written by
`RAG.SDK.Bridge.SetDefaultTable()` and `RAG.SDK.Bridge.AttachTable()`.

| Key | Config subscript | Default | New in 085 |
| --- | ---------------- | ------- | ---------- |
| `table` | `default_table` | `RAG.SourceDocuments` | no |
| `id_col` | `default_id_col` | `doc_id` | no |
| `text_col` | `default_text_col` | `text_content` | no |
| `embedding_col` | `default_emb_col` | `embedding` | no |
| `metadata_col` | `default_meta_col` | `""` (absent) | **yes** |

`metadata_col` is optional in the strong sense: empty means the table has no metadata column,
and every code path that reads it must behave as though the column does not exist rather than
substituting a name. The default table's default is `metadata`, because the canonical
`RAG.SourceDocuments` has that column; an explicitly configured table gets no such
assumption.

**Element type** is deliberately *not* configuration. It is resolved from the table on each
search (`INFORMATION_SCHEMA.TABLES.CLASSNAME` → `%Dictionary.CompiledProperty.SqlFieldName`
→ `Parameters.GetAt("DATATYPE")`, see `research.md`), so a table altered underneath the SDK
cannot leave stale configuration behind. Resolution failure is an error, never a guess.

## Scoping predicate

A caller-supplied SQL condition over the attached table's own columns, passed as
`whereClause`.

| Property | Value |
| -------- | ----- |
| Ownership | The caller composes it; the SDK does not build, rewrite, or extend it |
| Combination | ANDed into the search's own clause and parenthesised: `WHERE <emb> IS NOT NULL AND (<whereClause>)` |
| Empty value | `""` — the clause is omitted entirely and behaviour is identical to before this feature |
| Rejected shapes | Contains `;`, `--`, or `/*` — a fragment must not be able to introduce a second statement or comment out the rest of the query |
| Invalid but well-shaped | Reaches `%Prepare` and returns a named error, distinct from an empty result |
| Effect on fast path | Its presence disables the IVF fast path, which cannot honour it |
