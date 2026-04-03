# Data Model: Attach Existing Corpus

**Date**: 2026-04-03

## Entities

### CorpusBridge (in-memory, persisted via IVG)

Represents a registered mapping between an IRIS SQL table and IVG graph nodes.

| Field | Type | Source | Notes |
|---|---|---|---|
| graph_label | str | User-provided | Graph node label (e.g., "Document") |
| source_table | str | User-provided | Fully qualified SQL table (e.g., "RAG.SourceDocuments") |
| id_col | str | User-provided | Primary key column name |
| text_col | str | User-provided | Text content column name |
| embedding_col | str | User-provided | VECTOR column name |
| dimension | int | Auto-detected | Embedding dimension from first non-NULL row |
| row_count | int | Auto-detected | Row count at attach time |
| has_hnsw_index | bool | Auto-detected | Whether HNSW index exists on vector column |

**Persistence**: The graph mapping (`graph_label` → `source_table`, `id_col`) is persisted in `Graph_KG.table_mappings` by IVG's `map_sql_table()`. The vector metadata (`embedding_col`, `dimension`) is stored in-memory in `self._attached_corpora` on the pipeline instance and re-detected on each `attach_existing_corpus()` call.

### No new SQL tables

This feature creates zero new tables. It operates entirely on:
- Existing user tables (RAG.SourceDocuments, custom tables)
- Existing IVG table (Graph_KG.table_mappings)

## State Transitions

None — this is a stateless metadata operation. The mapping is either present or absent in `Graph_KG.table_mappings`.

## Relationships

```
User SQL Table (RAG.SourceDocuments)
    ├── [map_sql_table] → Graph_KG.table_mappings (label → table bridge)
    └── [validate_vector_table] → dimension, row_count, has_hnsw (metadata)

Query flow:
    engine.query("MATCH (d:Document)") → reads Graph_KG.table_mappings → JOINs to user table
    engine.vector_search("RAG.SourceDocuments", "embedding", q) → direct SQL on user table's HNSW index
```
