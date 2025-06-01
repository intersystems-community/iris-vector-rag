# Project Backlog / Future Enhancements

## ColBERT Optimizations & Enhancements

### Investigate `pylate` for ColBERT Re-ranking and 128-dim Embeddings

**Date Added:** 2025-06-01

**Context:**
The model card for `fjmgAI/reason-colBERT-150M-GTE-ModernColBERT` suggests using the `pylate` library for loading the model and mentions that it produces 128-dimensional token embeddings. Our current implementation using `transformers.AutoModel` results in 768-dimensional embeddings (from `last_hidden_state`).

**Potential Benefits of using `pylate`:**

1.  **128-dim Embeddings:**
    *   **Reduced Storage:** Storing 128-dim vectors instead of 768-dim vectors for `RAG.DocumentTokenEmbeddings` would significantly reduce database size.
    *   **Faster Similarity Calculations:** Cosine similarity computations during ColBERT's MaxSim stage would be faster with smaller vectors.
    *   **Alignment with Model Card:** Ensures we are using the model as intended by its author for its projected output.

2.  **`pylate.rank.rerank` Function:**
    *   The `pylate` library offers a `rank.rerank` function (see [PyLate GitHub](https://github.com/lightonai/pylate) or model card for `fjmgAI/reason-colBERT-150M-GTE-ModernColBERT`).
    *   This function allows using a ColBERT model purely for its re-ranking capabilities on a candidate set of documents retrieved by an existing first-stage retriever.
    *   This could be a way to leverage ColBERT's powerful re-ranking without needing to build and maintain a full `pylate` (e.g., Voyager HNSW) index for all token embeddings, potentially simplifying integration if we already have a satisfactory Stage 1 retriever.

**Action Items for Investigation:**

*   Add `pylate` as a project dependency.
*   Test loading `fjmgAI/reason-colBERT-150M-GTE-ModernColBERT` via `pylate.models.ColBERT` and verify if its `encode()` method produces 128-dim embeddings.
*   If successful, refactor `scripts/populate_colbert_token_embeddings_native_vector.py` to use `pylate` for model loading and encoding.
    *   This would also require updating the `RAG.DocumentTokenEmbeddings` schema to 128 dimensions.
*   Evaluate the feasibility and potential performance benefits of using `pylate.rank.rerank` as an alternative or addition to the current ColBERT pipeline's Stage 2. This would involve comparing its performance and complexity against the existing `_calculate_maxsim` logic.

**Example `pylate.rank.rerank` usage from model card:**
```python
from pylate import rank, models

# Assume 'model' is loaded via pylate.models.ColBERT
# queries_embeddings = model.encode(queries, is_query=True)
# documents_embeddings = model.encode(documents, is_query=False) # documents is a list of lists of doc texts

# reranked_documents = rank.rerank(
#     documents_ids=documents_ids, # list of lists of doc IDs
#     queries_embeddings=queries_embeddings,
#     documents_embeddings=documents_embeddings,
# )