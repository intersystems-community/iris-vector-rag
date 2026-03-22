---
title: "IRIS Vector Search Patterns and Gotchas"
type: kb-item
tags: [iris, vector, rag, embeddings, sql]
applies_to: [iris, healthconnect]
version: "1.0.0"
license: MIT
author: InterSystems Developer Community
---

# IRIS Vector Search Patterns and Gotchas

## The TO_VECTOR() Function
Always use `TO_VECTOR(?, FLOAT, dimensions)` to convert embedding arrays. Do NOT pass raw arrays directly.

```sql
-- WRONG
SELECT VECTOR_COSINE(embedding, ?) FROM tbl

-- CORRECT
SELECT VECTOR_COSINE(embedding, TO_VECTOR(?, FLOAT, 1536)) FROM tbl
```

## HNSW Index Tuning
- `M=16` — good default; increase to 32 for better recall at cost of memory
- `EF_CONSTRUCTION=200` — higher = better index quality, slower build
- `EF_SEARCH=50` — set at query time for recall/speed tradeoff

## Common Errors

### `SQLCODE -400: VECTOR dimensions do not match`
The stored embedding dimension doesn't match your query dimension. Check your embedding model.

### `SQLCODE -30: Table or view not found`
The VectorStore table doesn't exist in this namespace. Run the CREATE TABLE statement first.

### Slow queries without HNSW
Without an HNSW index, `VECTOR_COSINE` does a full table scan — O(n). Always create the index before production use.

## Python Connection
```python
import intersystems_iris as iris
conn = iris.connect("localhost", 1972, "USER", "_SYSTEM", "SYS")
cursor = conn.cursor()
cursor.execute("SELECT TOP 5 content, VECTOR_COSINE(embedding, TO_VECTOR(?, FLOAT, 1536)) score FROM VectorStore ORDER BY score DESC", [embedding_list])
```
