"""
End-to-end integration tests for BasicRAG pipeline.

Tests validate full workflow from document loading through query execution,
using live IRIS database and real embeddings.

Requirements: FR-025, FR-026, FR-027, FR-028
"""

import json
import pytest


@pytest.mark.integration
@pytest.mark.requires_database
@pytest.mark.basic_rag
class TestBasicRAGIntegration:
    """End-to-end integration tests for BasicRAG pipeline."""

    def test_full_query_path_with_real_db(self, basic_rag_pipeline, tmp_path):
        """
        FR-025: Full pipeline workflow MUST complete successfully.

        Given: BasicRAG pipeline with live IRIS database
        When: Documents loaded and query executed
        Then: Complete workflow succeeds (load → embed → store → retrieve → generate)
        """
        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]

        # Execute load_documents
        load_result = basic_rag_pipeline.load_documents(documents)

        # Verify loading succeeded
        assert load_result["documents_loaded"] > 0, \
            "Should successfully load documents"
        assert load_result["embeddings_generated"] > 0, \
            "Should generate embeddings"

        # Execute query
        query = "What are the symptoms of diabetes?"
        result = basic_rag_pipeline.query(query)

        # Verify query succeeded
        assert "answer" in result, "Query should return answer"
        assert "contexts" in result, "Query should return contexts"
        assert "metadata" in result, "Query should return metadata"

        # Verify retrieval occurred
        assert result["metadata"]["context_count"] >= 1, \
            "Should retrieve at least 1 context"

        # Verify answer quality
        assert len(result["answer"]) > 0, "Answer should be non-empty"

    def test_document_loading_workflow(self, basic_rag_pipeline):
        """
        FR-026: Document loading workflow MUST complete successfully.

        Given: Sample documents
        When: load_documents() executed
        Then: Documents embedded and stored in IRIS
        """
        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]
        expected_count = len(documents)

        # Execute load_documents
        result = basic_rag_pipeline.load_documents(documents)

        # Verify all documents loaded
        assert result["documents_loaded"] == expected_count, \
            f"Should load all {expected_count} documents"

        # Verify embeddings generated
        assert result["embeddings_generated"] == expected_count, \
            f"Should generate {expected_count} embeddings"

        # Verify no failures
        assert result["documents_failed"] == 0, \
            "Should have no failed documents"

    def test_response_quality_metrics(self, basic_rag_pipeline):
        """
        FR-027: Query response MUST include quality metrics.

        Given: BasicRAG pipeline with loaded documents
        When: Query executed
        Then: Response includes quality metrics (execution time, sources, etc.)
        """
        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        basic_rag_pipeline.load_documents(docs_data["documents"])

        # Execute query
        query = "How is diabetes treated?"
        result = basic_rag_pipeline.query(query)

        # Verify answer quality
        assert len(result["answer"]) > 10, \
            "Answer should be substantive (>10 chars)"

        # Verify source attribution
        metadata = result["metadata"]
        assert "sources" in metadata, "Metadata should include sources"
        assert len(metadata["sources"]) > 0, "Should have at least one source"

        # Verify execution time logged (optional)
        if "execution_time_ms" in metadata:
            assert metadata["execution_time_ms"] > 0, \
                "Execution time should be positive"

    def test_query_without_loaded_documents(self, basic_rag_pipeline):
        """
        FR-028: Query without documents SHOULD handle gracefully.

        Given: BasicRAG pipeline with no documents loaded
        When: Query executed
        Then: Graceful response (empty contexts or fallback message)
        """
        query = "What are risk factors for diabetes?"

        try:
            result = basic_rag_pipeline.query(query)

            # If query succeeds, should indicate no contexts found
            if "contexts" in result:
                # May return empty contexts or fallback answer
                assert isinstance(result["contexts"], list), \
                    "Contexts should be a list"

            # Answer may indicate no information available
            if "answer" in result:
                assert isinstance(result["answer"], str), \
                    "Answer should be a string"
        except Exception as e:
            error_msg = str(e).lower()

            # Error should be informative
            assert "document" in error_msg or "context" in error_msg or "no" in error_msg, \
                "Error should mention missing documents/contexts"

    def test_multiple_queries_maintain_consistency(self, basic_rag_pipeline):
        """
        FR-027: Multiple queries MUST maintain consistent behavior.

        Given: BasicRAG pipeline with loaded documents
        When: Multiple queries executed
        Then: All queries succeed with consistent response structure
        """
        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        basic_rag_pipeline.load_documents(docs_data["documents"])

        # Execute multiple queries
        queries = [
            "What are diabetes symptoms?",
            "How is diabetes diagnosed?",
            "What are diabetes risk factors?"
        ]

        for query in queries:
            result = basic_rag_pipeline.query(query)

            # Verify consistent structure
            assert "answer" in result, f"Query '{query}' should return answer"
            assert "contexts" in result, f"Query '{query}' should return contexts"
            assert "metadata" in result, f"Query '{query}' should return metadata"

            # Verify quality
            assert len(result["answer"]) > 0, f"Query '{query}' should have non-empty answer"
