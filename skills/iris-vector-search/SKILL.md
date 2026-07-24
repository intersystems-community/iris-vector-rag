---
name: iris-vector-search
description: Implement IRIS native vector similarity search. Use when adding semantic search, embedding storage, or nearest-neighbor queries to an IRIS application.
license: MIT
source: "intersystems-community/iris-vector-rag"
metadata:
  version: "1.0.0"
  author: InterSystems Developer Community
  compatibility: iris, objectscript, python
---

## Purpose

Use IRIS SQL VECTOR type and HNSW index for fast semantic similarity search — no external VectorDB.

## When to Use

- Storing and searching embeddings in IRIS globals/tables
- Adding `VECTOR_COSINE` or `VECTOR_DOT_PRODUCT` search to existing ObjectScript classes
- Replacing an external VectorDB with native IRIS capability

## Core SQL Patterns

### Store embeddings

```sql
INSERT INTO MyEmbeddings (content, embedding)
VALUES (?, TO_VECTOR(?, FLOAT, 1536))
```

### Search by cosine similarity

```sql
SELECT TOP 10 content,
    VECTOR_COSINE(embedding, TO_VECTOR(?, FLOAT, 1536)) AS score
FROM MyEmbeddings
ORDER BY score DESC
```

### Create HNSW index

```sql
CREATE INDEX emb_hnsw ON MyEmbeddings (embedding)
WITH (TYPE='HNSW', M=16, EF_CONSTRUCTION=200, DISTANCE='COSINE')
```

## ObjectScript Access

```objectscript
Set stmt = ##class(%SQL.Statement).%New()
Do stmt.%Prepare("SELECT TOP ? content, VECTOR_COSINE(embedding, TO_VECTOR(?, FLOAT, 1536)) score FROM MyEmbeddings ORDER BY score DESC")
Set rs = stmt.%Execute(topK, embeddingVector)
While rs.%Next() { Write rs.content, " (", rs.score, ")", ! }
```

## Dimension Guide

| Model                    | Dimensions |
| ------------------------ | ---------- |
| text-embedding-3-small   | 1536       |
| text-embedding-3-large   | 3072       |
| nomic-embed-text         | 768        |
| fastembed BAAI/bge-small | 384        |
