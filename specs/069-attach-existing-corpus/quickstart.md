# Quickstart: Attach Existing Corpus

## Prerequisites

- IRIS instance with existing data in a SQL table with a VECTOR column
- `iris-vector-rag >= 0.6.0` and `iris-vector-graph >= 1.27.0`
- The IVG graph schema initialized (`engine.initialize_schema()`)

## Basic Usage

```python
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

# Create pipeline (uses existing connection_manager)
pipeline = HybridGraphRAGPipeline(connection_manager=cm)

# Attach existing RAG.SourceDocuments to graph
result = pipeline.attach_existing_corpus(
    source_table="RAG.SourceDocuments",
    id_col="doc_id",
    text_col="text_content",
    embedding_col="embedding",
    graph_label="Document",
)
print(result)
# {'table': 'RAG.SourceDocuments', 'label': 'Document', 'id_col': 'doc_id',
#  'embedding_col': 'embedding', 'dimension': 768, 'row_count': 10000,
#  'has_hnsw_index': True}

# Graph queries now work over the existing table
docs = pipeline.iris_engine.query("MATCH (d:Document) RETURN d.doc_id LIMIT 5")

# Vector search uses the existing HNSW index — no re-embedding
results = pipeline.iris_engine.vector_search(
    "RAG.SourceDocuments", "embedding", query_vec, top_k=10
)
```

## Custom Table (non-RAG)

```python
result = pipeline.attach_existing_corpus(
    source_table="MyApp.ClinicalNotes",
    id_col="note_id",
    text_col="note_text",
    embedding_col="note_embedding",
    graph_label="ClinicalNote",
)
# MATCH (n:ClinicalNote) now works
```

## Re-pointing a label

```python
# Later, re-point "Document" to a different table (upsert semantics)
pipeline.attach_existing_corpus(
    source_table="MyApp.NewDocuments",
    id_col="id",
    text_col="content",
    embedding_col="emb",
    graph_label="Document",  # same label, different table
)
```
