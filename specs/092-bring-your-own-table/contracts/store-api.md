# Contract: the attached-table surface

## Configuration (`storage:iris:`; environment `RAG_STORAGE__IRIS__<KEY>`)

```yaml
storage:
  iris:
    table_name: "MDT.EvidenceChunk"
    id_col: "EvidenceId" # or a list: ["DocRef", "Seq"]
    text_col: "Text"
    embedding_col: "Embedding"
    metadata_col: null
    title_col: "Title"
    parent_col: "ResourceId"
    filter_columns: ["PatientId", "EmbeddingModel", "DocDate"]
    required_filters: ["PatientId", "EmbeddingModel"]
    metric: "cosine" # default for attached tables; "dot" for the library's own
    text_index: null # e.g. "DocChunkText"
    text_language: "en"
    text_terms: "words" # or "prefix"
```

Unset keys keep today's behaviour. Any of the six `*_col` keys makes the table attached.

## `IRISVectorStore(...)`

These keyword arguments are new. Each overrides the configuration key of the same name:

```text
id_col, text_col, embedding_col, metadata_col, title_col, parent_col,
filter_columns, required_filters, metric, text_index, text_terms,
embedder: Callable[[str], Sequence[float]] | None
```

## Search methods (new keyword `where`, default `None`)

| Method                                                                  | `where` applies |
| ----------------------------------------------------------------------- | --------------- |
| `similarity_search_by_embedding(query_embedding, top_k, filter, where)` | yes             |
| `search_by_vector(embedding, top_k, metadata_filter, where)`            | yes             |
| `search_by_text(query, top_k, metadata_filter, where)`                  | yes             |
| `search_by_text_with_score(query, top_k, metadata_filter, where)`       | yes             |
| `similarity_search_with_score(query, k, filter, where)`                 | yes             |
| `similarity_search(query, k, filter, where=...)` (LangChain form)       | yes             |
| `text_search(query, top_k, metadata_filter, where)`                     | yes             |
| `phrase_search(...)`                                                    | no: `where` is not a parameter |

## Semantics

- **`where`:**
  - `{col: v}` means `col = v`, `{col: (op, v)}` means `col op v`, and `{col: [v1, v2]}` means
    `col IN (v1, v2)`;
  - several keys are ANDed, and the result is ANDed with `filter` / `metadata_filter`;
  - an empty `where` is the same as `None`.
- **Errors:**

  | Situation                                                             | Error                         |
  | --------------------------------------------------------------------- | ----------------------------- |
  | A required filter is missing                                          | `RequiredFilterMissing`       |
  | A key outside `filter_columns`, a bad op, an empty list or a bad value type | `FilterError`           |
  | A write on an attached table                                          | `AttachedTableReadOnly`       |
  | The query length differs from the column's dimension                  | `DimensionMismatchError`      |
  | No text index on an attached table                                    | `TextIndexMissingError`, with text that says ivr will not create one on a table it does not own |

  All of these are raised before any search SQL runs.
- **Ids:** a single `id_col` gives `Document.id` as the column's value; a list joins the values
  with `#`.
- **Metadata:** `title_col` and `parent_col` values appear in `Document.metadata` as `title` and
  `parent_id`.
- **Scores:** cosine or dot, by `metric`, and `metadata["score"]` as today.

## `QueryOptions` and `pipeline.query(...)`

- `where: Optional[Mapping[str, Any]] = None`. `normalize_query_params(where=...)` passes it
  through.
- Retrieval modes `vector`, `text`, `hybrid` and `rrf` apply it to every leg.
- Modes `phrase`, `concept`, `global` and `mix` raise `ValueError` when it is set.
