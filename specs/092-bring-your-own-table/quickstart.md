# Quickstart: search a table you already have

```python
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

store = IRISVectorStore(
    table_name="MDT.EvidenceChunk", id_col="EvidenceId", text_col="Text",
    embedding_col="Embedding", filter_columns=["PatientId", "EmbeddingModel"],
    required_filters=["PatientId", "EmbeddingModel"], embedder=my_gemini_embed,
)
hits = store.search_by_text_with_score(
    "HER2 status", top_k=5,
    where={"PatientId": "P03", "EmbeddingModel": "gemini:gemini-embedding-001@1536"},
)
```

What this does and does not do:

- **Read only:** the library reads `MDT.EvidenceChunk` and never writes, alters or indexes it.
- **Similarity:** ranking is cosine, the default for attached tables (set `metric="dot"` to
  change it).
- **Required filters:** leaving out `PatientId` raises `RequiredFilterMissing`, rather than
  searching every patient.

## A composite key, a date filter and a Czech text index (Štafeta)

```yaml
storage:
  iris:
    table_name: "Stafeta.DocChunk"
    id_col: ["DocRef", "Seq"]
    text_col: "Text"
    embedding_col: "Embedding"
    parent_col: "DocRef"
    filter_columns: ["PatientRef", "AuthoredOn", "ReceivedOn"]
    required_filters: ["PatientRef"]
    text_index: "DocChunkText"
    text_language: "cs"
    text_terms: "prefix"
```

```python
pipeline.query(
    "Kdy byla vysazena antikoagulace?", retrieval="rrf",
    where={"PatientRef": "Patient/2", "AuthoredOn": ("<", "2026-10-05")},
)
```

Each result's id is `DocumentReference/2#3`. Both the vector leg and the text leg (`vysaz* OR
antikoa*`) are scoped to the patient and the date.

## RAG.SDK (ObjectScript) does the same with a predicate

```objectscript
Set tJSON = ##class(RAG.SDK.Search).TextSearch(pText, 5, "PatientRef = 'Patient/2'")
```

From Python, `where=` is structured and bound. From ObjectScript, the predicate (spec 085) is
for trusted callers that already hold SQL.

## Verify

```bash
IRIS_PORT=51972 pytest tests/integration/test_byot_iris.py -v
```
