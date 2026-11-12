"""
Contract tests for GraphRAG vector search correctness.

Contract: VSC-001 (specs/033-fix-graphrag-vector/contracts/vector_search_contract.md)
Requirements: FR-001, FR-002, FR-003
"""

import pytest


class TestVectorSearchContract:
    """Contract tests for vector search correctness (VSC-001)."""

    def test_vector_search_returns_documents(self, hybrid_on_known_corpus):
        """
        FR-001: Vector search MUST return documents when embeddings exist.

        Given: Known corpus with 3 documents with 384D embeddings
        When: Query is executed
        Then: At least 1 document is returned
        """
        query = "Which salamander can regrow its legs?"
        result = hybrid_on_known_corpus.query(query, generate_answer=False)

        assert (
            len(result["contexts"]) > 0
        ), f"Vector search returned 0 results for query: {query}"

    def test_vector_search_results_have_scores(self, hybrid_on_known_corpus):
        """
        All retrieved documents MUST have similarity scores.

        Given: Vector search returns documents
        When: Results are inspected
        Then: All documents have non-null similarity_score in metadata
        """
        query = "Which salamander can regrow its legs?"
        result = hybrid_on_known_corpus.query(query, generate_answer=False)

        assert len(result["retrieved_documents"]) > 0, "No documents retrieved"

        for i, doc in enumerate(result["retrieved_documents"]):
            # Scores are in metadata["score"], metadata["similarity"], or metadata["similarity_score"]
            assert (
                "score" in doc.metadata
                or "similarity" in doc.metadata
                or "similarity_score" in doc.metadata
            ), f"Document {i} missing score in metadata"
            score = (
                doc.metadata.get("score")
                or doc.metadata.get("similarity")
                or doc.metadata.get("similarity_score")
            )
            assert score is not None, f"Document {i} has null score"
            assert isinstance(
                score, float
            ), f"Document {i} score is not float: {type(score)}"

    def test_vector_search_results_sorted_descending(self, hybrid_on_known_corpus):
        """
        Retrieved documents MUST be sorted by similarity score DESC.

        Given: Vector search returns multiple documents
        When: Scores are compared
        Then: Each score >= next score (sorted descending)
        """
        query = "Which salamander can regrow its legs?"
        result = hybrid_on_known_corpus.query(query, generate_answer=False)

        docs = result["retrieved_documents"]
        assert len(docs) >= 2, "Need at least 2 documents to check sorting"

        scores = []
        for doc in docs:
            score = (
                doc.metadata.get("score")
                or doc.metadata.get("similarity")
                or doc.metadata.get("similarity_score")
            )
            scores.append(score)

        for i in range(len(scores) - 1):
            assert (
                scores[i] >= scores[i + 1]
            ), f"Scores not sorted DESC: score[{i}]={scores[i]:.4f} < score[{i + 1}]={scores[i + 1]:.4f}"

    def test_vector_search_returns_relevant_documents(self, hybrid_on_known_corpus):
        """
        The top retrieved document MUST be relevant to the query.

        Given: Query "Which salamander can regrow its legs?"
        When: Vector search executes
        Then: Top result is the axolotl document (kc-axolotl)
        """
        query = "Which salamander can regrow its legs?"
        result = hybrid_on_known_corpus.query(query, generate_answer=False)

        docs = result["retrieved_documents"]
        assert len(docs) > 0, "No documents retrieved"

        # Assert the answer, not the shape: the known query must get its known document
        assert (
            docs[0].id == "kc-axolotl"
        ), f"Top result should be kc-axolotl, got {docs[0].id}"

    def test_vector_search_works_with_384d_embeddings(self, hybrid_on_known_corpus):
        """
        FR-003: Vector search MUST work with existing 384D embeddings.

        Given: Documents have 384D all-MiniLM-L6-v2 embeddings
        When: Query is executed
        Then: Retrieval succeeds without dimension mismatch errors
        """
        query = "Which salamander can regrow its legs?"

        # Should not raise DimensionMismatchError
        result = hybrid_on_known_corpus.query(query, generate_answer=False)

        # If retrieval works, dimensions match implicitly
        assert (
            len(result["contexts"]) > 0
        ), "Vector search failed (possible dimension mismatch)"
