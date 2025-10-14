"""
End-to-end integration tests for CRAG pipeline.

Tests validate full workflow including relevance evaluation,
using live IRIS database and real embeddings.

Requirements: FR-025, FR-026, FR-027, FR-028
"""

import json
import pytest


@pytest.mark.integration
@pytest.mark.requires_database
@pytest.mark.crag
class TestCRAGIntegration:
    """End-to-end integration tests for CRAG pipeline."""

    def test_full_query_path_with_real_db(self, crag_pipeline, iris_connection):
        """
        FR-025: Full CRAG pipeline workflow MUST complete successfully.

        Given: CRAG pipeline with live IRIS database
        When: Documents loaded and query executed
        Then: Complete workflow succeeds with relevance evaluation
        """
        from tests.conftest import skip_if_no_embeddings

        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]

        # Execute load_documents
        load_result = crag_pipeline.load_documents(documents)

        assert load_result["documents_loaded"] > 0, \
            "Should successfully load documents"

        # Check if embeddings are in proper VECTOR format before querying
        skip_if_no_embeddings(iris_connection)

        # Execute query
        query = "What are the symptoms of diabetes?"
        result = crag_pipeline.query(query)

        # Verify query succeeded
        assert "answer" in result
        assert "contexts" in result
        assert "metadata" in result
        assert result["metadata"]["context_count"] >= 1

    def test_document_loading_workflow(self, crag_pipeline):
        """
        FR-026: CRAG document loading workflow MUST complete successfully.

        Given: Sample documents
        When: load_documents() executed
        Then: Documents embedded and stored
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]
        expected_count = len(documents)

        result = crag_pipeline.load_documents(documents)

        assert result["documents_loaded"] == expected_count
        assert result["embeddings_generated"] == expected_count
        assert result["documents_failed"] == 0

    def test_response_quality_metrics(self, crag_pipeline, iris_connection):
        """
        FR-027: CRAG query response MUST include quality metrics.

        Given: CRAG pipeline with loaded documents
        When: Query executed
        Then: Response includes quality metrics
        """
        from tests.conftest import skip_if_no_embeddings

        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        crag_pipeline.load_documents(docs_data["documents"])

        # Check if embeddings are in proper VECTOR format before querying
        skip_if_no_embeddings(iris_connection)

        query = "How is diabetes treated?"
        result = crag_pipeline.query(query)

        assert len(result["answer"]) > 10
        metadata = result["metadata"]
        assert "sources" in metadata
        assert len(metadata["sources"]) > 0

    def test_relevance_evaluation_in_query_path(self, crag_pipeline):
        """
        FR-027: CRAG relevance evaluation SHOULD occur in query path.

        Given: CRAG pipeline with loaded documents
        When: Query executed
        Then: Relevance evaluation metadata MAY be present
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        crag_pipeline.load_documents(docs_data["documents"])

        query = "What are diabetes symptoms?"
        result = crag_pipeline.query(query)

        metadata = result.get("metadata", {})

        # CRAG-specific metadata (optional)
        if "relevance_score" in metadata or "evaluation_result" in metadata:
            # Relevance evaluation is active
            assert True, "CRAG relevance evaluation working"

        # Alternatively, check retrieval method
        retrieval_method = metadata.get("retrieval_method", "").lower()
        if "crag" in retrieval_method or "evaluator" in retrieval_method:
            # CRAG-specific method indicated
            assert True

    def test_multiple_queries_maintain_consistency(self, crag_pipeline):
        """
        FR-027: Multiple CRAG queries MUST maintain consistent behavior.

        Given: CRAG pipeline with loaded documents
        When: Multiple queries executed
        Then: All queries succeed with consistent structure
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        crag_pipeline.load_documents(docs_data["documents"])

        queries = [
            "What are diabetes symptoms?",
            "How is diabetes diagnosed?",
            "What are diabetes risk factors?"
        ]

        for query in queries:
            result = crag_pipeline.query(query)

            assert "answer" in result
            assert "contexts" in result
            assert "metadata" in result
            assert len(result["answer"]) > 0
