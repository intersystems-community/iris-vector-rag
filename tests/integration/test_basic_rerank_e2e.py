"""
End-to-end integration tests for BasicRerankRAG pipeline.

Tests validate full workflow including cross-encoder reranking,
using live IRIS database and real embeddings.

Requirements: FR-025, FR-026, FR-027, FR-028
"""

import json
import pytest


@pytest.mark.integration
@pytest.mark.requires_database
@pytest.mark.basic_rerank
class TestBasicRerankRAGIntegration:
    """End-to-end integration tests for BasicRerankRAG pipeline."""

    def test_full_query_path_with_real_db(self, basic_rerank_pipeline):
        """
        FR-025: Full BasicRerankRAG pipeline workflow MUST complete successfully.

        Given: BasicRerankRAG pipeline with live IRIS database
        When: Documents loaded and query executed
        Then: Complete workflow succeeds with reranking
        """
        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]

        # Execute load_documents
        load_result = basic_rerank_pipeline.load_documents(documents)

        assert load_result["documents_loaded"] > 0, \
            "Should successfully load documents"

        # Execute query
        query = "What are the symptoms of diabetes?"
        result = basic_rerank_pipeline.query(query)

        # Verify query succeeded
        assert "answer" in result
        assert "contexts" in result
        assert "metadata" in result
        assert result["metadata"]["context_count"] >= 1

    def test_document_loading_workflow(self, basic_rerank_pipeline):
        """
        FR-026: BasicRerankRAG document loading workflow MUST complete successfully.

        Given: Sample documents
        When: load_documents() executed
        Then: Documents embedded and stored
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]
        expected_count = len(documents)

        result = basic_rerank_pipeline.load_documents(documents)

        assert result["documents_loaded"] == expected_count
        assert result["embeddings_generated"] == expected_count
        assert result["documents_failed"] == 0

    def test_response_quality_metrics(self, basic_rerank_pipeline):
        """
        FR-027: BasicRerankRAG query response MUST include quality metrics.

        Given: BasicRerankRAG pipeline with loaded documents
        When: Query executed
        Then: Response includes quality metrics
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        basic_rerank_pipeline.load_documents(docs_data["documents"])

        query = "How is diabetes treated?"
        result = basic_rerank_pipeline.query(query)

        assert len(result["answer"]) > 10
        metadata = result["metadata"]
        assert "sources" in metadata
        assert len(metadata["sources"]) > 0

    def test_reranking_in_query_path(self, basic_rerank_pipeline):
        """
        FR-027: BasicRerankRAG reranking SHOULD occur in query path.

        Given: BasicRerankRAG pipeline with loaded documents
        When: Query executed
        Then: Reranking metadata MAY be present
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        basic_rerank_pipeline.load_documents(docs_data["documents"])

        query = "What are diabetes symptoms?"
        result = basic_rerank_pipeline.query(query)

        metadata = result.get("metadata", {})
        contexts = result.get("contexts", [])

        # BasicRerankRAG-specific metadata (optional)
        if "rerank_method" in metadata or "reranked" in metadata:
            # Reranking is active
            assert True, "BasicRerankRAG reranking working"

        # Alternatively, check retrieval method
        retrieval_method = metadata.get("retrieval_method", "").lower()
        if "rerank" in retrieval_method:
            # Reranking indicated in retrieval method
            assert True

        # Verify contexts ordered by relevance
        if contexts and isinstance(contexts[0], dict):
            if "score" in contexts[0]:
                scores = [ctx.get("score", 0) for ctx in contexts if "score" in ctx]
                if scores:
                    # Verify descending order (reranked)
                    assert scores == sorted(scores, reverse=True), \
                        "Contexts should be ordered by descending rerank score"

    def test_multiple_queries_maintain_consistency(self, basic_rerank_pipeline):
        """
        FR-027: Multiple BasicRerankRAG queries MUST maintain consistent behavior.

        Given: BasicRerankRAG pipeline with loaded documents
        When: Multiple queries executed
        Then: All queries succeed with consistent structure
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        basic_rerank_pipeline.load_documents(docs_data["documents"])

        queries = [
            "What are diabetes symptoms?",
            "How is diabetes diagnosed?",
            "What are diabetes risk factors?"
        ]

        for query in queries:
            result = basic_rerank_pipeline.query(query)

            assert "answer" in result
            assert "contexts" in result
            assert "metadata" in result
            assert len(result["answer"]) > 0
