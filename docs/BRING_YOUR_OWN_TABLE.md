# Bring your own table

You already have your chunks in an IRIS table: your own column names, your own embedding
model, and columns every query must filter on (a patient, a tenant, a date). Declare the
columns, and the store, the retrieval modes and the pipelines search that table as it is.

The library treats such a table as **attached** and only reads it: it never creates,
alters, indexes, inserts into or deletes from it. A write method raises
`AttachedTableReadOnly`.

**Only what you declare is used.** `id_col` and `text_col` are required. A column you don't
declare is never guessed from its name, so a column that happens to be called `metadata` or
`title` is ignored unless you name it.

## Five lines

```python
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

store = IRISVectorStore(
    table_name="MDT.EvidenceChunk", id_col="EvidenceId", text_col="Text",
    embedding_col="Embedding", filter_columns=["PatientId", "EmbeddingModel"],
    required_filters=["PatientId", "EmbeddingModel"], embedder=my_embed,
)
hits = store.search_by_text_with_score(
    "HER2 status", top_k=5,
    where={"PatientId": "P03", "EmbeddingModel": "gemini:gemini-embedding-001@1536"},
)
```

- **`embedder`** is any function from text to a vector of the column's width. The query is
  checked against the column, and a mismatch raises `DimensionMismatchError`. Without one,
  the configured embedding provider is used.
- **Required filters:** a search without `PatientId` raises `RequiredFilterMissing`. It never
  searches every patient.
- **Ranking:** by cosine similarity, the usual default for vector search; `metric="dot"`
  changes it.

## The same in configuration, for every pipeline

```yaml
storage:
  iris:
    table_name: "MDT.EvidenceChunk"
    id_col: "EvidenceId"
    text_col: "Text"
    embedding_col: "Embedding"
    title_col: "Title" # -> Document.metadata["title"]
    parent_col: "ResourceId" # -> Document.metadata["parent_id"]
    filter_columns: ["PatientId", "EmbeddingModel", "DocDate"]
    required_filters: ["PatientId", "EmbeddingModel"]
```

Every key also reads from the environment: `RAG_STORAGE__IRIS__ID_COL=EvidenceId`. A list
can be comma-separated: `RAG_STORAGE__IRIS__FILTER_COLUMNS=PatientId,EmbeddingModel`.
Constructor arguments override configuration key by key.

```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline("basic")
result = pipeline.query(
    "What is the HER2 status?", top_k=5,
    where={"PatientId": "P03", "EmbeddingModel": "gemini:gemini-embedding-001@1536"},
)
```

`where=` applies to the `vector`, `text`, `hybrid` and `rrf` retrieval modes, on every leg.
`phrase`, `concept`, `global` and `mix` refuse it.

## Filters

| `where=` value           | SQL                            |
| ------------------------ | ------------------------------ |
| `{"PatientId": "P03"}`   | `"PatientId" = ?`              |
| `{"DocDate": (">=", d)}` | `"DocDate" >= CAST(? AS DATE)` |
| `{"Kind": ["a", "b"]}`   | `"Kind" IN (?, ?)`             |

- **Operators:** `=`, `<`, `<=`, `>` and `>=`.
- **LangChain filters work as they are.** `$eq`, `$gt`, `$gte`, `$lt`, `$lte` and `$in`, in the
  tuple (`("$gte", d)`) or in the nested form
  `{"DocDate": {"$gte": a, "$lt": b}, "Kind": {"$in": ["a", "b"]}}`. `$or`, `$and`, `$ne`,
  `$nin` and `$like` are refused by name: conditions on several columns are always ANDed.
- **Allowed columns:** keys must be in `filter_columns`, and those must be columns of the table.
- **Values:** always bound as parameters. A date or time column gets a `CAST`, so
  `"2026-10-05"` compares as a date.
- **With the metadata filter:** `where=` combines with the JSON `metadata_filter` by AND.

**Use a model-identity column as a required filter.** If one table holds vectors from more
than one model, a search must read only the rows its query model wrote. The dimension check
cannot tell two 1,536-wide models apart.

## A composite key, a date rule and a Czech text index

```yaml
storage:
  iris:
    table_name: "Stafeta.DocChunk"
    id_col: ["DocRef", "Seq"] # result ids are "DocumentReference/2#3"
    text_col: "Text"
    embedding_col: "Embedding"
    parent_col: "DocRef"
    filter_columns: ["PatientRef", "AuthoredOn", "ReceivedOn"]
    required_filters: ["PatientRef"]
    text_index: "DocChunkText" # the table's own iFind index
    text_language: "cs"
    text_terms: "prefix" # vysazena -> vysaz*
```

```python
pipeline.query(
    "Kdy byla vysazena antikoagulace?", retrieval="rrf",
    where={"PatientRef": "Patient/2", "AuthoredOn": ("<", "2026-10-05")},
)
```

- **Ids:** with several `id_col` columns, a result's id is their values joined with `#`. It
  only has to be unique within the search's required filters.
- **Text search:** it reads the index named by `text_index`. Without one, a text, hybrid or
  rrf search on an attached table raises `TextIndexMissingError`; the library will not build
  an index on a table it does not own.
- **Prefix terms:** `text_terms: prefix` turns each word of four or more letters into a prefix
  wildcard (at least four letters kept, at most seven), for languages IRIS has no stemmer
  for.

## HNSW

IRIS uses an HNSW index only for the query function its `Distance` names. When a table's
index was built for a different measure than `metric`, the store logs one warning. That
covers any table, the library's own included.

Measured on IRIS 2026.1 at 5,000 rows: `SELECT TOP k ... ORDER BY` returns rows in rank order,
both scanned and through HNSW, and a test guards it.

Measured at 5,000 rows (spec 092, `plan.md`): a selective `where=` plans through the filter's
own index. That is faster than HNSW at that size, so pushing the filter into SQL is what makes
scoped search fast.

## From ObjectScript

RAG.SDK takes the same idea as a SQL predicate (spec 085), for trusted callers that already
hold SQL:

```objectscript
Set tJSON = ##class(RAG.SDK.Search).TextSearch(pText, 5, "PatientRef = 'Patient/2'")
```

From Python, `where=` is structured and bound, because Python callers often pass user input.
The graph bridge, `HybridGraphRAGPipeline.attach_existing_corpus` (spec 069), takes the same
`id_col`, `text_col` and `embedding_col` names.

## Not covered

- writing to an attached table;
- views;
- a FHIR SQL Builder projection as a source. That is the next spec; it writes the library's
  own section rows from the projection.
