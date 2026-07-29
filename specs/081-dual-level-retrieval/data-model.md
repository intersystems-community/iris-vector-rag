# Data Model: Dual-Level (Global/Mix) Retrieval — Feature 081

**Date**: 2026-07-29

---

## Entities

### RelationEmbedding (extends RAG.EntityRelationships)

Not a new table — a column addition to the existing `RAG.EntityRelationships` table.

| Field                  | Type                        | Notes                                              |
| ---------------------- | --------------------------- | -------------------------------------------------- |
| relationship_id        | VARCHAR(255) PK             | Existing                                           |
| source_entity_id       | VARCHAR(255) FK→Entities    | Existing                                           |
| target_entity_id       | VARCHAR(255) FK→Entities    | Existing                                           |
| relationship_type      | VARCHAR(255)                | Existing                                           |
| weight                 | DOUBLE DEFAULT 1.0          | Existing                                           |
| confidence             | DOUBLE DEFAULT 1.0          | Existing                                           |
| source_document        | VARCHAR(255)                | Existing                                           |
| created_timestamp      | TIMESTAMP                   | Existing                                           |
| **relation_embedding** | **VECTOR(FLOAT, 384) NULL** | **New — embedded description of the relationship** |

**Validation rules**:

- `relation_embedding` is NULL until `RelationEmbeddingStore.embed_and_store()` is called; NULLs are excluded from similarity search (`WHERE relation_embedding IS NOT NULL`)
- Dimension must be 384 — enforced by `TO_VECTOR(?, FLOAT, 384)` insert pattern
- One embedding per relationship row (relationship_id is the key for upsert)

**Index**:

```sql
CREATE INDEX idx_hnsw_rel_embedding ON RAG.EntityRelationships (relation_embedding)
  AS HNSW(M=16, efConstruction=200, Distance='COSINE');
```

---

### QueryKeywords (in-memory / metadata only — not persisted)

Produced by `KeywordExtractor` per query, stored in response `metadata`. Not written to the database.

| Field               | Type        | Notes                                                        |
| ------------------- | ----------- | ------------------------------------------------------------ |
| query               | str         | Original query string                                        |
| high_level_keywords | List[str]   | Themes/concepts extracted by LLM                             |
| low_level_keywords  | List[str]   | Specific entities/proper nouns extracted by LLM              |
| extraction_model    | str         | Model used for extraction (may differ from generation model) |
| degraded            | bool        | True if extraction returned empty arrays or fell back        |
| degradation_reason  | str \| None | Human-readable reason when degraded=True                     |

---

### DualLevelResult (in-memory — response metadata)

Carries per-source contribution data surfaced in query response `metadata`.

| Field               | Type                   | Notes                                                           |
| ------------------- | ---------------------- | --------------------------------------------------------------- |
| retrieved_documents | List[Document]         | Final fused ranked list                                         |
| low_level_docs      | List[Document]         | Documents from low-level (entity) retrieval                     |
| high_level_docs     | List[Document]         | Documents from high-level (relation/theme) retrieval            |
| naive_docs          | List[Document] \| None | Documents from vector/chunk retrieval (mix mode only)           |
| fusion_method       | str                    | `"rrf"` (default) or `"weighted_score"` (when weights supplied) |
| keywords            | QueryKeywords          | Extracted keywords for this query                               |

Each `Document` in the result carries these additions in `metadata`:

| Field            | Type  | Notes                                       |
| ---------------- | ----- | ------------------------------------------- |
| retrieval_source | str   | `"low_level"`, `"high_level"`, or `"naive"` |
| level_score      | float | Score from the contributing level           |
| fusion_score     | float | Final RRF/weighted score                    |

---

## Retrieval Mode Registry Additions

Two new entries in `iris_vector_rag/retrieval/modes.py`:

| Mode name  | Sources                                         | Prerequisites                            | Fusion                                                      |
| ---------- | ----------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| `"global"` | `["relation_embedding"]`                        | `knowledge_graph`, `relation_embeddings` | none (single source)                                        |
| `"mix"`    | `["low_level", "relation_embedding", "vector"]` | `knowledge_graph`, `relation_embeddings` | `"rrf"` (default); `"weighted_score"` when weights provided |

Prerequisite identifiers:

| ID                    | What it checks                                                                          |
| --------------------- | --------------------------------------------------------------------------------------- |
| `knowledge_graph`     | KG tables exist and have rows (existing check)                                          |
| `relation_embeddings` | `RAG.EntityRelationships` has `relation_embedding` column AND at least one non-NULL row |

---

## QueryOptions Extensions

Two new optional fields added to `iris_vector_rag/core/query_options.py`:

```python
@dataclass
class QueryOptions:
    # ... existing fields ...
    high_level_keywords: Optional[List[str]] = None   # pre-supply to skip LLM extraction
    low_level_keywords: Optional[List[str]] = None    # pre-supply to skip LLM extraction
```

When pre-supplied, `KeywordExtractor` is skipped and the provided keywords are used directly. This supports testing and advanced usage without an LLM call for keyword extraction.

---

## State Transitions

### Relation Embedding Lifecycle

```text
Document ingested → KG built (entities + relationships in RAG.EntityRelationships)
                                    ↓
                    embed_and_store() called per relationship
                                    ↓
                    relation_embedding column populated (NULL → VECTOR)
                                    ↓
                    Available for global/mix retrieval
```

**Incremental path**: New documents added → new relationships extracted → `embed_and_store()` called only for new rows → no full re-embed (FR-007).

**Degraded path**: `global`/`mix` requested but `relation_embedding` column is all-NULL → fall back to entity-level retrieval, set `degraded=True` in metadata (clarified 2026-07-29).
