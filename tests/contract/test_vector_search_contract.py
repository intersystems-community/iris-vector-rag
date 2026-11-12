"""
Contract tests for GraphRAG vector search correctness.

Contract: VSC-001 (specs/033-fix-graphrag-vector/contracts/vector_search_contract.md)
Requirements: FR-001, FR-002, FR-003
"""

import pytest

KNOWN_QUERY = "Which salamander can regrow its legs?"


def scored(result):
    """(id, similarity_score) for each retrieved document, in order."""
    return [
        (d.id, d.metadata.get("similarity_score"))
        for d in result["retrieved_documents"]
    ]


class TestVectorSearchContract:
    """Contract tests for vector search correctness (VSC-001), on the known corpus.

    The graph is deployed but empty, so HybridGraphRAG answers from the store's vector
    search. Scores are the store's: the fallback used to drop them and a helper then
    gave every document 1.0, which every ordering check passed.
    """

    def test_vector_search_returns_documents(self, hybrid_on_known_corpus):
        """FR-001: with embeddings stored, the known query finds its known document."""
        result = hybrid_on_known_corpus.query(KNOWN_QUERY, generate_answer=False)

        assert result["retrieved_documents"][0].id == "kc-axolotl"

    def test_vector_search_respects_top_k(self, hybrid_on_known_corpus):
        """FR-002: no more than top_k documents come back."""
        result = hybrid_on_known_corpus.query(
            KNOWN_QUERY, top_k=2, generate_answer=False
        )

        assert len(result["retrieved_documents"]) == 2

    def test_vector_search_results_have_scores(self, hybrid_on_known_corpus):
        """Every document carries the store's similarity, as a float, and they differ:
        an unrelated document does not score like the match."""
        pairs = scored(
            hybrid_on_known_corpus.query(KNOWN_QUERY, top_k=3, generate_answer=False)
        )

        assert all(isinstance(score, float) for _, score in pairs), pairs
        assert len({score for _, score in pairs}) == len(pairs), pairs

    def test_vector_search_results_sorted_descending(self, hybrid_on_known_corpus):
        """Scores fall from the first document to the last, the match first."""
        pairs = scored(
            hybrid_on_known_corpus.query(KNOWN_QUERY, top_k=3, generate_answer=False)
        )
        scores = [score for _, score in pairs]

        assert pairs[0][0] == "kc-axolotl", pairs
        assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1], pairs

    def test_vector_search_returns_relevant_documents(self, hybrid_on_known_corpus):
        """The top document is the one the query is about, for each topic."""
        for query, expected in [
            (KNOWN_QUERY, "kc-axolotl"),
            ("How does volcanic rock form from lava?", "kc-basalt"),
            ("Which medieval league controlled Baltic trade?", "kc-hanseatic"),
        ]:
            result = hybrid_on_known_corpus.query(query, generate_answer=False)
            assert result["retrieved_documents"][0].id == expected, (
                query,
                scored(result),
            )

    def test_vector_search_works_with_384d_embeddings(self, hybrid_on_known_corpus):
        """FR-003: the 384-dimension query embedding searches the 384-dimension column."""
        pipeline = hybrid_on_known_corpus

        assert len(pipeline.embedding_manager.embed_text(KNOWN_QUERY)) == 384
        assert pipeline.vector_store.vector_dimension == 384
        result = pipeline.query(KNOWN_QUERY, generate_answer=False)
        assert result["retrieved_documents"][0].id == "kc-axolotl"
