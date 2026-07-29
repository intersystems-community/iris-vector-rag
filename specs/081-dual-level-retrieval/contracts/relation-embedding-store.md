# Contract: RelationEmbeddingStore

**Feature**: 081-dual-level-retrieval
**Date**: 2026-07-29
**Covers**: US4 — relation embeddings available for retrieval

---

## Interface

```python
from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

store = RelationEmbeddingStore(connection_manager, config_manager)

# Embed and store a single relationship (called during KG construction)
store.embed_and_store(
    relationship_id="rel_abc123",
    relationship_type="CAUSED_BY",
    source_entity="Basel III",
    target_entity="Capital Requirements",
    description="Basel III regulation caused stricter capital requirements for banks.",
)

# Nearest-neighbor search over relation embeddings
results = store.search(
    query_embedding=[0.1, 0.2, ...],   # List[float] len 384
    top_k=10,
)
# Returns: List[dict] with keys: relationship_id, source_entity_id,
#          target_entity_id, relationship_type, score (float)

# Check if relation embeddings are populated (for prerequisite check)
count = store.count_embedded()   # int; 0 means index is empty
```

---

## Schema Migration

`RelationEmbeddingStore.__init__()` calls `_ensure_schema()` which runs:

```sql
ALTER TABLE RAG.EntityRelationships ADD relation_embedding VECTOR(FLOAT, 384) NULL
-- (no-op if column already exists — IRIS ALTER TABLE ADD is idempotent)

CREATE INDEX idx_hnsw_rel_embedding ON RAG.EntityRelationships (relation_embedding)
  AS HNSW(M=16, efConstruction=200, Distance='COSINE')
-- (no-op if index exists)
```

No separate migration script needed. Schema is applied on first use.

---

## Incremental Ingestion (FR-007)

`embed_and_store()` uses `insert_vector(..., upsert=True)` — existing rows update their embedding; new rows insert. No full re-embed of existing corpus on incremental document addition.

---

## Embedding Text

The text embedded for each relationship is the concatenation:

```text
"{relationship_type}: {source_entity} → {target_entity}. {description}"
```

If `description` is absent, the description clause is omitted. This is the embedding input that represents the relationship's semantic content for high-level theme retrieval.

---

## NULL Handling (Degradation)

- Rows with `relation_embedding IS NULL` are excluded from `search()` via `WHERE relation_embedding IS NOT NULL`
- `count_embedded()` counts only non-NULL rows
- When `count_embedded() == 0`, `RetrievalEngine` detects the degraded state and falls back to entity-level retrieval (FR-009, clarified 2026-07-29)
