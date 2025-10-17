# Contract: Vector Search Correctness

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Contract ID**: VSC-001
**Requirements**: FR-001, FR-002, FR-003
**Test File**: `tests/contract/test_vector_search_contract.py`

## Contract Definition

### Given

```python
# Preconditions:
# 1. IRIS database running on port 11972
# 2. RAG.SourceDocuments table exists
# 3. 2,376 documents loaded with 384D embeddings
# 4. All documents have non-null embeddings
# 5. GraphRAG pipeline initialized successfully
```

### When

```python
# Action:
query = "What are the symptoms of diabetes?"
pipeline = create_pipeline("graphrag", validate_requirements=True)
result = pipeline.query(query)
```

### Then

```python
# Postconditions (all MUST be true):

# 1. Vector search returns documents (FR-001)
assert len(result.contexts) > 0, \
    "Vector search MUST return documents when embeddings exist"

# 2. Returns top-K documents (FR-002, default K=10)
assert len(result.contexts) <= 10, \
    "Vector search MUST return at most K documents (default K=10)"

# 3. All returned documents have similarity scores
assert all(hasattr(doc, 'similarity_score') for doc in result.contexts), \
    "All retrieved documents MUST have similarity scores"

# 4. Scores are sorted descending (most similar first)
scores = [doc.similarity_score for doc in result.contexts]
assert scores == sorted(scores, reverse=True), \
    "Documents MUST be sorted by similarity score DESC"

# 5. At least 1 document is relevant to query
# (Heuristic: top result score should be >0.3 for diabetes query)
assert scores[0] > 0.3, \
    "At least 1 document MUST be semantically relevant (score >0.3)"

# 6. Works with existing 384D embeddings (FR-003)
# (Verified implicitly - if retrieval works, dimensions match)
```

---

## Contract Test Implementation

### File: `tests/contract/test_vector_search_contract.py`

```python
"""
Contract tests for GraphRAG vector search correctness.

Contract: VSC-001 (specs/033-fix-graphrag-vector/contracts/vector_search_contract.md)
Requirements: FR-001, FR-002, FR-003
"""

import pytest
from iris_rag import create_pipeline


class TestVectorSearchContract:
    """Contract tests for vector search correctness (VSC-001)."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create GraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    def test_vector_search_returns_documents(self, graphrag_pipeline):
        """
        FR-001: Vector search MUST return documents when embeddings exist.

        Given: 2,376 documents with 384D embeddings in RAG.SourceDocuments
        When: Query is executed
        Then: At least 1 document is returned
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        assert len(result.contexts) > 0, \
            f"Vector search returned 0 results for query: {query}"

    def test_vector_search_respects_top_k(self, graphrag_pipeline):
        """
        FR-002: Vector search MUST retrieve top-K most similar documents.

        Given: Default top_k=10 configuration
        When: Query is executed
        Then: At most 10 documents are returned
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        # Default K=10
        assert len(result.contexts) <= 10, \
            f"Vector search returned {len(result.contexts)} > 10 documents (top_k violation)"

    def test_vector_search_results_have_scores(self, graphrag_pipeline):
        """
        All retrieved documents MUST have similarity scores.

        Given: Vector search returns documents
        When: Results are inspected
        Then: All documents have non-null similarity_score attribute
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        for i, doc in enumerate(result.contexts):
            assert hasattr(doc, 'similarity_score'), \
                f"Document {i} missing similarity_score attribute"
            assert doc.similarity_score is not None, \
                f"Document {i} has null similarity_score"
            assert isinstance(doc.similarity_score, float), \
                f"Document {i} similarity_score is not float: {type(doc.similarity_score)}"

    def test_vector_search_results_sorted_descending(self, graphrag_pipeline):
        """
        Retrieved documents MUST be sorted by similarity score DESC.

        Given: Vector search returns multiple documents
        When: Scores are compared
        Then: Each score >= next score (sorted descending)
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        scores = [doc.similarity_score for doc in result.contexts]

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i+1], \
                f"Scores not sorted DESC: score[{i}]={scores[i]:.4f} < score[{i+1}]={scores[i+1]:.4f}"

    def test_vector_search_returns_relevant_documents(self, graphrag_pipeline):
        """
        At least 1 retrieved document MUST be relevant to query.

        Given: Query "What are the symptoms of diabetes?"
        When: Vector search executes
        Then: Top result has similarity score >0.3 (relevance threshold)
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        assert len(result.contexts) > 0, "No documents retrieved"

        top_score = result.contexts[0].similarity_score
        assert top_score > 0.3, \
            f"Top result score {top_score:.4f} <= 0.3 (not relevant)"

    def test_vector_search_works_with_384d_embeddings(self, graphrag_pipeline):
        """
        FR-003: Vector search MUST work with existing 384D embeddings.

        Given: Documents have 384D all-MiniLM-L6-v2 embeddings
        When: Query is executed
        Then: Retrieval succeeds without dimension mismatch errors
        """
        query = "What are the symptoms of diabetes?"

        # Should not raise DimensionMismatchError
        result = graphrag_pipeline.query(query)

        # If retrieval works, dimensions match implicitly
        assert len(result.contexts) > 0, \
            "Vector search failed (possible dimension mismatch)"
```

---

## Test Execution

### Before Fix (Expected: FAIL)

```bash
$ .venv/bin/pytest tests/contract/test_vector_search_contract.py -v

tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_returns_documents FAILED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_respects_top_k FAILED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_results_have_scores FAILED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_results_sorted_descending FAILED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_returns_relevant_documents FAILED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_works_with_384d_embeddings FAILED

============================== FAILURES ==============================
_______ TestVectorSearchContract.test_vector_search_returns_documents _______

AssertionError: Vector search returned 0 results for query: What are the symptoms of diabetes?

============================== 6 failed in 2.34s ==============================
```

### After Fix (Expected: PASS)

```bash
$ .venv/bin/pytest tests/contract/test_vector_search_contract.py -v

tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_returns_documents PASSED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_respects_top_k PASSED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_results_have_scores PASSED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_results_sorted_descending PASSED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_returns_relevant_documents PASSED
tests/contract/test_vector_search_contract.py::TestVectorSearchContract::test_vector_search_works_with_384d_embeddings PASSED

============================== 6 passed in 3.12s ==============================
```

---

## Contract Acceptance Criteria

- ✅ All 6 contract tests PASS
- ✅ Tests execute against live IRIS database (not mocks)
- ✅ Tests cover FR-001, FR-002, FR-003 requirements
- ✅ Tests validate data integrity (scores, sorting, relevance)

---

**Contract Status**: ✅ DEFINED
**Test File**: To be created in Phase 2 (Task T001)
**Expected Result**: FAIL before fix, PASS after fix
