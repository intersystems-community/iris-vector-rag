# Release Notes: iris-vector-rag v0.17.0

Released 2026-10-25 as `v0.17.0`.

**Bring your own table.** If your chunks are already in an IRIS table, with your own column
names, your own embedding model and columns every query must filter on, iris-vector-rag can now
search that table as it is, through the store, the retrieval modes and every pipeline. It reads
your table and never writes to it. See
[docs/BRING_YOUR_OWN_TABLE.md](docs/BRING_YOUR_OWN_TABLE.md).

```python
store = IRISVectorStore(
    table_name="MDT.EvidenceChunk", id_col="EvidenceId", text_col="Text",
    embedding_col="Embedding", filter_columns=["PatientId", "EmbeddingModel"],
    required_filters=["PatientId", "EmbeddingModel"], embedder=my_embed,
)
store.search_by_text_with_score("HER2 status", top_k=5,
    where={"PatientId": "P03", "EmbeddingModel": "gemini:gemini-embedding-001@1536"})
```

## Highlights

- **Declared columns, nothing guessed.** Name the id (one column or several), text, embedding,
  title, parent and metadata columns, in configuration or as arguments.
- **Filters on your columns.**
  - `where=` takes equals, comparisons, lists and dates, always bound as parameters.
  - LangChain's `$eq`, `$gt`, `$gte`, `$lt`, `$lte` and `$in` work as they are.
  - `required_filters` makes a search without, say, a patient an error instead of a search
    over everyone.
- **Every pipeline honours `where`:** basic, basic + rerank, CRAG, multi-query rrf, GraphRAG and
  HybridGraphRAG. Phrase search and global GraphRAG refuse it rather than ignore it.
- **Your model, your index.**
  - `embedder=` embeds queries with your model, checked against the column's width.
  - Attached tables rank by cosine by default.
  - `text_index` uses your existing iFind index, and `text_terms: prefix` handles inflected
    languages such as Czech.
  - A warning appears when an HNSW index was built for a different measure than the search
    uses.
- **The text language is a setting.** `storage:iris:text_language` drives the iFind index, stop
  words and the note-search detectors. Detectors take a caller's lexicon for other languages;
  only English ships.

## Fixed

- **multi-query rrf:** it ignored `metadata_filter`, and a filter error only reached the log.
- **CRAG:** filter errors were swallowed into its fallback.
- **HybridGraphRAG:** seed entities were read as the first character of each id.

## Upgrading

Nothing changes for tables you didn't declare: the library's own tables keep their columns, dot
product ranking and behaviour. Two things to know:

- **New HNSW warning:** if your own table's HNSW index was built for cosine while searches use
  dot product, you'll now see one warning per table saying the index isn't being used.
- **Third-party stores:** a store whose search methods don't take `where` keeps working, as
  long as no `where` is passed.

## Verification

- Unit: 2,330 passed.
- Integration on IRIS 2026.1:
  - the bring-your-own-table and pipeline suites (MedTeam's and Štafeta's table shapes):
    163 passed;
  - the existing pipeline end-to-end suites: 47 passed, 5 skipped;
  - the ivg-related suites on iris-vector-graph 4.4.1: 70 passed, 1 skipped.
- The release workflow's test steps were replayed on a fresh container before tagging.
