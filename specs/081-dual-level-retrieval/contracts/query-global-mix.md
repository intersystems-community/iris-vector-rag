# Contract: Global and Mix Retrieval Modes

**Feature**: 081-dual-level-retrieval
**Date**: 2026-07-29
**Covers**: US1, US2 — `retrieval="global"` and `retrieval="mix"` on any KG-backed pipeline

---

## Interface

All existing pipelines that expose `query()` via `ComposableQueryMixin` accept two new `retrieval=` values. No other signature change.

```python
result = pipeline.query(
    query="What are the emerging risks across these filings?",
    top_k=10,
    retrieval="global",          # or "mix"
    weights={"relation": 0.6, "vector": 0.4},  # optional; overrides RRF default for mix
    generate_answer=True,
)
```

---

## Response Contract

`result` is the existing standardized response dict, with these additions/guarantees:

```python
{
    "answer": str | None,
    "contexts": List[str],
    "retrieved_documents": List[Document],
    "sources": List[str],
    "error": None | {"type": str, "message": str, "error_class": str},
    "metadata": {
        # existing fields ...
        "retrieval_mode": "global" | "mix",
        "low_level_keywords": List[str],   # always present when global/mix ran
        "high_level_keywords": List[str],  # always present when global/mix ran
        "extraction_model": str,
        "degraded": bool,
        "degradation_reason": str | None,
        # mix-only:
        "fusion_method": "rrf" | "weighted_score",
        "low_level_count": int,
        "high_level_count": int,
        "naive_count": int | None,
    }
}
```

Each `Document` in `retrieved_documents` carries:

```python
doc.metadata["retrieval_source"]  # "low_level" | "high_level" | "naive"
doc.metadata["level_score"]       # float
doc.metadata["fusion_score"]      # float
```

---

## Prerequisite Error Contract

When `global`/`mix` prerequisites are absent (no KG, no relation_embeddings populated), raises:

```python
RetrievalPrerequisiteError(
    mode="global",           # or "mix"
    missing=["relation_embeddings"],  # list of missing prerequisite IDs
    message="retrieval='global' requires relation_embeddings: RAG.EntityRelationships "
            "has no non-NULL relation_embedding rows. Run embed_relation_embeddings() first."
)
```

`error` key in the response is only set (non-None) when `generate_answer=True` and generation fails — prerequisite errors are always raised as exceptions (consistent with Feature 065 FR-012).

---

## Degradation Contract (empty relation-embedding index)

When relation-embedding index exists but all rows are NULL (FR-009, clarified 2026-07-29):

```python
result["metadata"]["degraded"] == True
result["metadata"]["degradation_reason"] == "relation_embedding index empty; fell back to entity-level retrieval"
# No exception raised
# results contain entity-level documents (not an empty list)
```

---

## Backward Compatibility

- Omitting `retrieval=` → existing default behavior unchanged
- `retrieval="vector"`, `"text"`, `"hybrid"`, `"rrf"` → unchanged
- All existing tests pass without modification (SC-006)
