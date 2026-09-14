# Contract: RAG.SDK search API after feature 085

Every method returns a JSON string. A successful search returns a JSON **array**; a failure
returns a JSON **object** with an `error` key. A caller must therefore check the first
character, which is what the SDK's own IVF fast path already does.

## `RAG.SDK.Search.VectorSearch(queryVec, topK, whereClause)`

| Argument | Type | Default | Meaning |
| -------- | ---- | ------- | ------- |
| `queryVec` | `%String` | required | Comma-separated query vector |
| `topK` | `%Integer` | `10` | Maximum rows returned |
| `whereClause` | `%String` | `""` | **New in 085.** SQL predicate over the searched table's columns; empty means unscoped |

Returns an array of hits (see `data-model.md`), ordered by descending score.

**Behaviour**

- `whereClause = ""` — identical to the pre-085 result, including the IVF fast path.
- `whereClause` non-empty — the IVF fast path is skipped and the predicate is ANDed into the
  SQL search's clause, parenthesised.
- `whereClause` containing `;`, `--`, or `/*` — rejected before preparation:
  `{"error":"whereClause must be a single predicate: ';', '--' and '/*' are not allowed"}`.
- Predicate that fails to prepare or execute — `{"error":"<IRIS message>"}`.
- Element type unresolvable — `{"error":"cannot determine vector element type for <table>.<column>"}`.
- Query vector shorter than two elements — `[]`, as before.

## `RAG.SDK.Search.GetTableConfig()`

Returns `{"table":…,"id_col":…,"text_col":…,"embedding_col":…,"metadata_col":…}`.
`metadata_col` is `""` when the configured table has no metadata column. **New in 085**:
the `metadata_col` key.

## `RAG.SDK.Search.GetVectorType(table, column)`

**New in 085.** Returns the vector element type of `table.column` as a string (`FLOAT`,
`DOUBLE`, …), or `""` when it cannot be determined. Public so a consumer can check a table
before attaching it.

## `RAG.SDK.Search.GetVectorDim(table, column)`

**New in 085.** Returns the declared dimension of `table.column`, or `0` when it cannot be
determined. Read from the class dictionary for the same reason as the element type: `%GetData`
on a VECTOR column returns the packed vector, so counting commas in a value read back out of
the column reports `1` for every width.

## `RAG.SDK.Search.ReadText(value)`

**New in 085**, public because `RAG.SDK.Bridge.BuildBM25Index` needs it too. Returns the text
of a value that may be either a stream OID or a plain `VARCHAR`. Before 085 both `VectorSearch`
and `TextSearch` opened every text value as a stream, which threw `<LIST>` on any table whose
text column is a plain `VARCHAR`.

## `RAG.SDK.Search.TextSearch(text, topK, whereClause)`

`whereClause` is **new in 085** and behaves as above: ANDed into the existing `LIKE`
condition, disables the BM25 fast path when non-empty, same rejection rules. Hits carry
metadata on the same terms as vector search.

One asymmetry is deliberate: an **unscoped** failure to prepare or execute still returns `[]`
as it did before 085, while a **scoped** one returns `{"error":"<IRIS message>"}`. A caller
that asked for a scope has to be able to tell a predicate it got wrong from a scope that
genuinely holds no rows; a caller that asked for nothing keeps the old behaviour.

## `RAG.SDK.Search.HybridSearch(queryVec, text, topK, strategy, whereClause)`

`whereClause` is **new in 085** and is passed to both legs. Fused hits carry `id`, `score`,
and `text` — **new in 085**, previously `id` and `score` only — plus `metadata` when
available from either leg.

## `RAG.SDK.Bridge.SetDefaultTable(table, idCol, textCol, embCol, metaCol)`

`metaCol` is **new in 085**, optional, default `""`. Omitting it clears any previously
configured metadata column rather than leaving a stale one behind.

## `RAG.SDK.Bridge.AttachTable(table, idCol, textCol, embCol, label, metaCol)`

`metaCol` is **new in 085**, optional, default `""`. When supplied it is validated to exist
on the table alongside the other three columns; when absent, validation is unchanged. The
returned object gains `metadata_col` and `vector_type`.

Two further changes in 085, both bug fixes rather than new surface:

- Schema and table are split on the **last** dot, so `Cat.Sub.Docs` is looked up as table
  `Docs` in schema `Cat.Sub` instead of `Sub.Docs` in `Cat`.
- `dimension` comes from `GetVectorDim` rather than from counting commas in one row's
  embedding, which always reported `1`. It is now correct for an empty table too.

## Compatibility

Every new argument is trailing and optional, and every new response key is additive. A pre-085
caller sees these behavioural changes, all of them a failure becoming a result:

- A search against a table whose embedding column is not `DOUBLE` returns rows instead of
  `{"error":"Cannot perform vector operation on vectors of different datatypes"}`.
- A search against a table whose text column is a plain `VARCHAR` returns rows instead of
  throwing `<LIST>`.
- `AttachTable` reports the real `dimension` instead of `1`.
- A hybrid hit carries `text` (and `metadata` where the table has it) instead of `id` and
  `score` alone.
