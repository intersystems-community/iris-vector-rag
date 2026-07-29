# Research: Dual-Level (Global/Mix) Retrieval — Feature 081

**Date**: 2026-07-29
**Branch**: `081-dual-level-retrieval`

---

## Decision 1: Relation Embedding Storage Schema

**Decision**: Extend `RAG.EntityRelationships` with a `relation_embedding VECTOR(FLOAT, 384) NULL` column — do not create a separate table.

**Rationale**: Mirrors the exact pattern used for `RAG.Entities` (which has `embedding VECTOR(FLOAT, 384) NULL` inline). Avoids join overhead at retrieval time. The `schema_manager.py` registry entry for `EntityRelationships` already has `"embedding_column": None` and `"supports_vector_search": False` — flip both to enable. Add an HNSW index post-column, matching the Entities pattern.

**DDL addition**:

```sql
ALTER TABLE RAG.EntityRelationships ADD relation_embedding VECTOR(FLOAT, 384) NULL;
CREATE INDEX idx_hnsw_rel_embedding ON RAG.EntityRelationships (relation_embedding)
  AS HNSW(M=16, efConstruction=200, Distance='COSINE');
```

**Files to update**:

- `iris_vector_rag/storage/schema_manager.py` line ~451: set `"embedding_column": "relation_embedding"`, `"supports_vector_search": True`
- `iris_vector_rag/common/db_init_complete.sql`: add column + index to `RAG.EntityRelationships` DDL

**Alternatives considered**: Separate `RAG.RelationEmbeddings` table — rejected; adds a join and inconsistency with the Entities pattern.

---

## Decision 2: Relation Embedding Insert Pattern

**Decision**: Use `insert_vector` from `iris_vector_rag.common.db_vector_utils` (re-exported from `iris_vector_graph.dbapi_utils`).

**Rationale**: This is the production-tested path that generates `TO_VECTOR(?, FLOAT, 384)` with a bound parameter marker. The vector is passed as a bracket-notation string `"[f1,f2,...,f384]"`. It supports upsert semantics needed for incremental ingestion (FR-007).

**Pattern**:

```python
from iris_vector_rag.common.db_vector_utils import insert_vector

insert_vector(
    cursor=cursor,
    table_name="RAG.EntityRelationships",
    vector_column="relation_embedding",
    vector_data=relation_embedding_list,   # List[float], len 384
    dimension=384,
    dtype="FLOAT",
    key_columns={"relationship_id": rel_id},
    additional_columns={...other columns...},
    upsert=True,
)
```

**Alternatives considered**: `build_safe_vector_dot_sql` inline literal — rejected; embeds vector in SQL string (security/length risk), uses DOT_PRODUCT not COSINE.

---

## Decision 3: Relation Embedding Similarity Search Pattern

**Decision**: Use `vector_similarity_search` from `iris_vector_graph.dbapi_utils` (Pattern A).

**Rationale**: Uses `VECTOR_COSINE` (correct for unit-normalized embeddings), passes the query vector as a bound parameter marker, returns dicts. The legacy `build_safe_vector_dot_sql` pattern (Pattern B, used by `IRISVectorStore`) uses DOT_PRODUCT and inline string embedding — avoid for new code.

**Pattern**:

```python
from iris_vector_graph.dbapi_utils import vector_similarity_search

results = vector_similarity_search(
    cursor=cursor,
    table_name="RAG.EntityRelationships",
    vector_column="relation_embedding",
    query_vector=query_embedding,   # List[float], len 384
    top_k=top_k,
    id_column="relationship_id",
    return_columns=["source_entity_id", "target_entity_id", "relationship_type"],
    metric="COSINE",
    dtype="FLOAT",
)
# Returns: [{"relationship_id": ..., "score": float, ...}, ...]
```

---

## Decision 4: Keyword Extraction Prompt Format

**Decision**: Adopt LightRAG's JSON-object prompt format exactly. The LLM returns a single flat JSON object with no markdown fencing:

```json
{
  "high_level_keywords": ["emerging risks", "cross-filing themes"],
  "low_level_keywords": ["Acme Corp", "LIBOR", "Basel III"]
}
```

**Rationale**: Proven in production by LightRAG. No delimiter parsing needed — separation is by JSON field name. Empty arrays on vague queries. Straightforward to parse with `json.loads()` + key extraction. Language is a runtime parameter (default `"English"`).

**Parsing**:

````python
import json, re

def parse_keywords(raw: str) -> tuple[list[str], list[str]]:
    # Strip markdown fences if model adds them despite instructions
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    data = json.loads(cleaned)
    return data.get("high_level_keywords", []), data.get("low_level_keywords", [])
````

**Alternatives considered**: Delimiter-separated plain text — rejected; less structured, harder to parse reliably across models.

---

## Decision 5: New Mode Names and Registration

**Decision**: Register two new modes in `iris_vector_rag/retrieval/modes.py`:

- `"global"` — prerequisites: `["knowledge_graph", "relation_embeddings"]`; sources: `["relation_embedding"]`; fusion: none (single source)
- `"mix"` — prerequisites: `["knowledge_graph", "relation_embeddings"]`; sources: `["low_level", "relation_embedding", "vector"]`; fusion: `"rrf"` (default, clarified 2026-07-29)

**Note on `"local"` mode**: LightRAG also has a `"local"` mode (low-level keywords → entity embeddings). This is effectively what the existing `"vector"` mode does with entity-focused queries. We do not add a separate `"local"` mode — `"vector"` covers it. The new modes are `"global"` and `"mix"` only (spec FR-001, FR-002).

**Alternatives considered**: Single `"dual_level"` mode — rejected; `"global"` and `"mix"` match LightRAG's established vocabulary and are independently useful.

---

## Decision 6: Feature 065 Components Reused vs New

### Reused unchanged

- `RetrievalMode` registry + `check_prerequisites()` — add two `_register()` calls
- `ComposableQueryMixin` — no changes; `_maybe_rerank()` works as-is
- `normalize_query_params()` — extend `_FUSION_MODES` to include `"mix"`
- `QueryOptions` — add two optional fields (see below)
- `insert_vector` + `vector_similarity_search` — reused directly

### New code required

| Component                | File                                             | Description                                                                                   |
| ------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `QueryOptions` extension | `core/query_options.py`                          | Add `high_level_keywords: Optional[List[str]]`, `low_level_keywords: Optional[List[str]]`     |
| `KeywordExtractor`       | `retrieval/keyword_extractor.py`                 | LLM call + JSON parse; accepts `query`, `llm_func`, `language`; returns `(high_kws, low_kws)` |
| Mode registration        | `retrieval/modes.py`                             | `_register("global", ...)`, `_register("mix", ...)`                                           |
| `RelationEmbeddingStore` | `storage/relation_embedding_store.py`            | `embed_and_store(relationship)`, `search(query_embedding, top_k)`, schema migration           |
| Engine dispatch          | `retrieval/engine.py`                            | `_retrieve_global(opts)`, `_retrieve_mix(opts)` branches in `retrieve()`                      |
| Schema migration         | `storage/schema_manager.py` + DDL                | ALTER TABLE + HNSW index, registry update                                                     |
| Ingestion hook           | `services/storage.py` or `pipelines/graphrag.py` | Call `RelationEmbeddingStore.embed_and_store()` when relationships are written                |

---

## Current `RAG.EntityRelationships` Schema (authoritative: schema_manager.py ~line 2432)

| Column             | Type                                 |
| ------------------ | ------------------------------------ |
| relationship_id    | VARCHAR(255) PRIMARY KEY             |
| source_entity_id   | VARCHAR(255) NOT NULL                |
| target_entity_id   | VARCHAR(255) NOT NULL                |
| relationship_type  | VARCHAR(255)                         |
| weight             | DOUBLE DEFAULT 1.0                   |
| confidence         | DOUBLE DEFAULT 1.0                   |
| source_document    | VARCHAR(255)                         |
| created_timestamp  | TIMESTAMP                            |
| relation_embedding | VECTOR(FLOAT, 384) NULL ← **to add** |

---

## Constitution Compliance

| Principle             | Status | Notes                                                                      |
| --------------------- | ------ | -------------------------------------------------------------------------- |
| P1 IRIS-First Testing | ✅     | All new components tested against live IRIS                                |
| P2 TO_VECTOR insert   | ✅     | Using `insert_vector` which generates `TO_VECTOR(?, FLOAT, 384)`           |
| P3 .DAT Fixtures      | ✅     | E2E tests ≥10 entities use .DAT fixture; unit tests use mocks              |
| P4 Test Isolation     | ✅     | Schema migration runs in test setup; cleanup tears down added column in CI |
| P5 Embedding 384d     | ✅     | `VECTOR(FLOAT, 384)` — same dimension as entity/chunk embeddings           |
| P6 Config Hygiene     | ✅     | LLM API key never logged; keyword model config via `ConfigurationManager`  |
| P7 Backend Mode       | ✅     | No new connection pool usage; reuses existing `ConnectionManager`          |
