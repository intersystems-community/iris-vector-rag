---
name: iris-rag-pipeline
description: Build production RAG pipelines with IRIS native vector search. Use when asked to implement retrieval-augmented generation, semantic search, or document Q&A on IRIS.
license: MIT
metadata:
  version: "1.0.0"
  author: InterSystems Developer Community
  compatibility: iris, python, langchain
---

## Purpose
Set up end-to-end RAG pipelines using IRIS as the vector database — no external VectorDB required.

## When to Use
- Building a Q&A system over enterprise documents stored in IRIS
- Implementing semantic search over ObjectScript class documentation or IRIS globals
- Adding AI-powered retrieval to an existing Ensemble/HealthConnect workflow

## Workflow

### Step 1 — Index Documents
```python
from iris_vector_graph import IRISVectorStore

store = IRISVectorStore.connect(
    hostname="localhost", port=1972,
    namespace="USER", username="_SYSTEM", password="SYS"
)
store.add_texts(texts=chunks, metadatas=metadata_list)
```

### Step 2 — Retrieve
```python
results = store.similarity_search(query="your question", k=5)
```

### Step 3 — Generate
Pass retrieved context to the LLM:
```python
context = "\n".join(doc.page_content for doc in results)
prompt = f"Context:\n{context}\n\nQuestion: {query}"
response = llm.invoke(prompt)
```

## IRIS Vector Index Setup
```sql
CREATE TABLE VectorStore (
    id INT IDENTITY PRIMARY KEY,
    content VARCHAR(32000),
    embedding VECTOR(FLOAT, 1536),
    metadata VARCHAR(2000)
);
CREATE INDEX hnsw_idx ON VectorStore (embedding) WITH (TYPE='HNSW', M=16, EF_CONSTRUCTION=200);
```

## Common Patterns
- **Hybrid search**: combine BM25 + vector via `iris_vector_graph.HybridSearch`
- **GraphRAG**: use `HybridGraphRAG` for entity-relationship-aware retrieval
- **RAGAS evaluation**: use `ragas` package to score faithfulness, relevancy, context recall
