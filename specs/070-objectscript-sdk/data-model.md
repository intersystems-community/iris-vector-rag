# Data Model: ObjectScript SDK for IVR

**Date**: 2026-04-13

## Entities

The SDK creates zero new entities. It operates on IVR's existing tables.

### RAG.SourceDocuments (existing — owned by Python IVR, shared with SDK)

| Column | Type | Notes |
|---|---|---|
| doc_id | VARCHAR(64) PK | Document identifier |
| title | VARCHAR(500) | Document title |
| text_content | LONGVARCHAR | Full text |
| metadata | LONGVARCHAR | JSON metadata |
| embedding | VECTOR(DOUBLE, N) | Pre-computed or auto-generated vector |

### RAG.DocumentChunks (existing)

| Column | Type | Notes |
|---|---|---|
| chunk_id | VARCHAR(64) PK | Chunk identifier |
| source_doc_id | VARCHAR(64) FK | Parent document |
| chunk_text | LONGVARCHAR | Chunk text |
| chunk_embedding | VECTOR(DOUBLE, N) | Chunk vector |

### RAG.Entities (existing)

| Column | Type | Notes |
|---|---|---|
| entity_id | VARCHAR(64) PK | Entity identifier |
| entity_name | VARCHAR(500) | Entity name |
| entity_type | VARCHAR(100) | Entity type label |
| source_doc_id | VARCHAR(64) FK | Source document |

### sql/schema.sql (new — shared DDL)

Single source of truth for all `CREATE TABLE` statements above. Read by both Python IVR's `SchemaManager` and ObjectScript `RAG.SDK.Schema.Initialize()`.

## Relationships

```
RAG.SourceDocuments 1──* RAG.DocumentChunks   (source_doc_id FK)
RAG.SourceDocuments 1──* RAG.Entities         (source_doc_id FK)
```

No new relationships added by the SDK.
