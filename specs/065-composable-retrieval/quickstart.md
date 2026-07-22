# Quickstart: Composable Query-Time Retrieval

**Feature**: 065-composable-retrieval

The point of this feature: pick your retrieval behavior **at query time**, on **any** pipeline — no need to choose a pipeline *type* up front. This mirrors MongoDB's `$rankFusion` / `$scoreFusion` / `$rerank` composability.

## Install

```bash
pip install rag-templates[hybrid-graphrag]   # includes iris-vector-graph (BM25 + fusion)
```

## Basic search (unchanged)

```python
from iris_vector_rag import create_pipeline   # NOTE: iris_vector_rag, not iris_rag

pipeline = create_pipeline("basic")
pipeline.load_documents(documents=docs)

result = pipeline.query("What is diabetes?", top_k=5)
print(result["answer"])
```

## Filtered search (now actually filters)

```python
result = pipeline.query(
    "What is diabetes?",
    top_k=5,
    metadata_filter={"source": "pubmed"},   # was silently ignored before this feature
    similarity_threshold=0.7,
)
assert all(d.metadata["source"] == "pubmed" for d in result["retrieved_documents"])
```

## Reranking — one argument, any pipeline

```python
# Before: create_pipeline("basic_rerank")   (a whole separate pipeline type)
# After:  add rerank=True to any pipeline
result = pipeline.query("What is diabetes?", top_k=5, rerank=True)

# Custom reranker
result = pipeline.query("...", rerank=my_cross_encoder_fn)
```

## Hybrid & RRF fusion — MongoDB-style

```python
# Weighted relative-score fusion (like $scoreFusion)
result = pipeline.query(
    "insulin resistance",
    retrieval="hybrid",
    weights={"vector": 0.7, "text": 0.3},
)

# Reciprocal rank fusion (like $rankFusion)
result = pipeline.query("insulin resistance", retrieval="rrf")

# Per-source scores are exposed (like scoreDetails)
for d in result["retrieved_documents"]:
    print(d.metadata.get("vector_score"), d.metadata.get("text_score"), d.metadata.get("fusion_score"))
```

## Compose them (retrieve → fuse → rerank)

```python
result = pipeline.query(
    "insulin resistance",
    retrieval="rrf",
    rerank=True,          # applied AFTER fusion, like $rerank after $rankFusion
    metadata_filter={"source": "pubmed"},
    top_k=5,
)
```

## Swap pipelines with one line (the promise now holds)

```python
for kind in ["basic", "crag", "graphrag"]:
    p = create_pipeline(kind)
    r = p.query("What is diabetes?", top_k=5, rerank=True)   # same call, every pipeline
```

## Zero-config embeddings (optional "text-in" mode)

```python
# With native IRIS EMBEDDING enabled in config, no embedding_func needed:
pipeline = create_pipeline("basic")   # embeddings generated in-engine
result = pipeline.query("What is diabetes?")
```

## Error behavior (no silent surprises)

```python
# Requesting a mode whose prerequisite is missing raises a clear, named error:
create_pipeline("basic").query("q", retrieval="rrf")
# -> PrerequisiteError: retrieval mode 'rrf' requires iris-vector-graph BM25 index (not found). Install/enable it or use retrieval='vector'.
```

## Validation checklist (maps to Success Criteria)

- [ ] SC-001: filtered query returns only matching docs
- [ ] SC-002/003/004: same call works across all pipelines; rerank/hybrid/rrf via one argument
- [ ] SC-005: reranker loads once across many queries
- [ ] SC-006: README quickstart imports run on a clean install
- [ ] SC-007: existing suite passes unchanged
