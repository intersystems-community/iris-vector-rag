# Quickstart: Dual-Level (Global/Mix) Retrieval — Feature 081

**Date**: 2026-07-29

---

## Prerequisites

- Feature 065 (composable-retrieval) merged — `retrieval=` selector and `RetrievalEngine` must exist
- A KG-backed corpus indexed via `graphrag` pipeline (entities + relationships in IRIS)
- IRIS container running: `docker start iris-vector-rag-iris`

---

## Scenario 1: Thematic query with `retrieval="global"`

```python
from iris_rag import create_pipeline

pipeline = create_pipeline("graphrag")

# Index some documents first (builds KG)
pipeline.load_documents(documents=my_docs)

# Global mode: high-level keyword extraction → relation-embedding search
result = pipeline.query(
    "What are the systemic risks discussed across these financial filings?",
    top_k=10,
    retrieval="global",
    generate_answer=True,
)

print(result["answer"])
print("High-level keywords:", result["metadata"]["high_level_keywords"])
print("Low-level keywords:", result["metadata"]["low_level_keywords"])
print("Degraded?", result["metadata"]["degraded"])
```

**Expected**: `global` retrieves documents connected by relationship/theme embeddings that a pure vector search misses. `metadata` records extracted keywords.

---

## Scenario 2: Comprehensive retrieval with `retrieval="mix"`

```python
# Mix mode: low-level (entity) + high-level (relation) + naive (vector) fused via RRF
result = pipeline.query(
    "Explain the impact of Basel III on European bank capital structures.",
    top_k=10,
    retrieval="mix",
    generate_answer=False,
)

for doc in result["retrieved_documents"]:
    print(doc.metadata["retrieval_source"], doc.metadata["fusion_score"], doc.page_content[:80])
```

**Expected**: Results tagged with `retrieval_source` ∈ `{"low_level", "high_level", "naive"}`. RRF fusion combines all three sources.

---

## Scenario 3: Custom keyword-extraction model

```python
from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor
from iris_vector_rag.common.utils import get_llm_func

# Use a faster/cheaper model for extraction only
cheap_llm = get_llm_func(model="gpt-4o-mini")
pipeline.keyword_extractor = KeywordExtractor(llm_func=cheap_llm)

result = pipeline.query("What themes appear across the risk sections?", retrieval="global")
print("Extraction model:", result["metadata"]["extraction_model"])  # gpt-4o-mini
```

---

## Scenario 4: Handling missing relation embeddings (degradation)

```python
# If relation embeddings haven't been generated yet:
result = pipeline.query("...", retrieval="global")

if result["metadata"]["degraded"]:
    print("Degraded:", result["metadata"]["degradation_reason"])
    # "relation_embedding index empty; fell back to entity-level retrieval"
    # Results still returned (entity-level), no exception raised
```

---

## Scenario 5: Pre-supplying keywords (skip LLM extraction)

```python
result = pipeline.query(
    "Basel III capital requirements",
    retrieval="global",
    high_level_keywords=["bank regulation", "capital adequacy"],
    low_level_keywords=["Basel III", "Tier 1 capital"],
)
# KeywordExtractor LLM call is skipped; provided keywords used directly
```

---

## Verify relation embeddings are indexed

```python
from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

store = RelationEmbeddingStore(pipeline.connection_manager, pipeline.config_manager)
count = store.count_embedded()
print(f"{count} relation embeddings indexed")
# 0 → global/mix will degrade; >0 → full dual-level retrieval available
```

---

## Test commands

```bash
# Contract tests (no IRIS needed)
pytest tests/contract/test_global_mix_modes.py tests/contract/test_keyword_extractor.py -q

# Integration tests (needs IRIS)
pytest tests/integration/test_relation_embedding_store.py -v

# E2E tests (needs IRIS + LLM key)
pytest tests/e2e/test_dual_level_retrieval_e2e.py -v
```
