# Project Backlog / Future Enhancements

## SQL RAG Library Initiative

### Phase 1: SQL RAG Library - BasicRAG & HyDE Proof of Concept

**Date Added:** 2025-06-01

**Context:**
The goal is to make RAG techniques accessible directly via SQL stored procedures within InterSystems IRIS, leveraging its native `EMBEDDING` data type and Embedded Python for core logic. This aims to simplify RAG integration and democratize its use for SQL-proficient developers and analysts.

**Detailed Plan:** See [docs/SQL_RAG_LIBRARY_PLAN.md](docs/SQL_RAG_LIBRARY_PLAN.md)

**Key Objectives for Phase 1:**
*   **Validate Core Architecture:**
    *   Design and implement the SQL Stored Procedure to Embedded Python interaction model.
    *   Confirm effective use of IRIS `EMBEDDING` data type for document storage and `TO_VECTOR(?)` for on-the-fly query vectorization.
*   **Implement `RAG.BasicSearch` Stored Procedure:**
    *   SQL interface for basic vector search and retrieval.
    *   Embedded Python module (`rag_py_basic.py`) for core logic.
    *   Optional LLM call for answer generation, configured via IRIS mechanisms.
*   **Implement `RAG.HyDESearch` Stored Procedure:**
    *   SQL interface for HyDE.
    *   Embedded Python module (`rag_py_hyde.py`) to:
        *   Generate hypothetical document using a configured LLM.
        *   Use hypothetical document text for vector search via `TO_VECTOR(?)`.
        *   Optional LLM call for final answer generation.
*   **Initial Configuration Management:**
    *   Develop basic IRIS SQL tables or procedures for managing LLM endpoints/keys and essential pipeline parameters (e.g., table/column names for retrieval).
*   **Helper Utilities:**
    *   Create foundational Embedded Python utility functions (e.g., in a `rag_py_utils.py`) for common tasks like fetching configurations from IRIS and making LLM calls (potentially using LiteLLM from the start for flexibility).
*   **Testing:**
    *   Unit tests for core Python logic.
    *   Basic integration tests for the SQL stored procedures.
*   **Documentation:**
    *   Update `docs/SQL_RAG_LIBRARY_PLAN.md` with learnings and refined designs from PoC.
    *   Initial user documentation for the implemented SQL procedures.

**Success Criteria for Phase 1:**
*   Successfully execute `RAG.BasicSearch` and `RAG.HyDESearch` via SQL.
*   Demonstrate retrieval of relevant documents based on query text.
*   Demonstrate (optional) answer generation using a configured LLM.
*   Configuration for LLMs and basic pipeline parameters is manageable via IRIS.
*   Core interaction patterns are established and documented.

---
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

---
## Python Integration / SDK Enhancements

### Implement "VectorStore" Interface
**Date Added:** 2025-06-03

**Context:**
The `common/vector_store.py` file was an initial stub. There's an opportunity to define and implement a more comprehensive `VectorStore` abstract base class or interface.

**Goal:**
Create a Pythonic `VectorStore` interface, similar to those found in libraries like LangChain or LlamaIndex, to abstract the interactions with the InterSystems IRIS vector database. This would provide a standardized way for Python applications to:
- Add documents/embeddings
- Delete documents/embeddings
- Search for similar documents
- Potentially manage metadata and indexes

**Benefits:**
- Improved code organization and reusability for Python-based RAG components.
- Easier integration with Python-centric RAG frameworks or for developers familiar with those patterns.
- Encapsulates IRIS-specific SQL for vector operations, making Python code cleaner.

**Action Items:**
- Define the `VectorStore` ABC or Protocol with core methods (`add`, `delete`, `search`, etc.).
- Implement an `IRISVectorStore` class that fulfills this interface, using the `intersystems-iris` DB-API and appropriate SQL (including `TO_VECTOR` and vector functions).
- Consider methods for managing HNSW indexes or other IRIS-specific vector features if appropriate at this abstraction level.
- Create unit tests for the `IRISVectorStore` implementation.