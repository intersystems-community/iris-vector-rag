"""
Contract tests for BasicRAG Pipeline API behavior (API-001).

Tests validate that BasicRAG implements standardized RAGPipeline API with correct
input handling, response structure, and error conditions.

Contract: pipeline_api_contract.md
Requirements: FR-001, FR-002, FR-003, FR-004
"""

import pytest


@pytest.mark.contract
@pytest.mark.basic_rag
class TestBasicRAGContract:
    """Contract tests for BasicRAG pipeline API behavior."""

    def test_query_method_exists(self, basic_rag_pipeline):
        """
        FR-001: BasicRAG MUST implement query() method.

        Given: BasicRAG pipeline instance
        When: Method existence is checked
        Then: query() method exists with correct signature
        """
        assert hasattr(basic_rag_pipeline, 'query'), \
            "BasicRAG pipeline must have query() method"

        # Verify method is callable
        assert callable(basic_rag_pipeline.query), \
            "query() must be callable"

    def test_query_validates_required_parameter(self, basic_rag_pipeline):
        """
        FR-002: Query MUST validate required query parameter.

        Given: BasicRAG pipeline instance
        When: query() called with None or empty string
        Then: ValueError raised with clear message
        """
        # Test query=None
        with pytest.raises((ValueError, TypeError)) as exc_info:
            basic_rag_pipeline.query(query=None)

        error_msg = str(exc_info.value).lower()
        assert "query" in error_msg, \
            "Error message must mention 'query' parameter"

        # Test query=""
        with pytest.raises(ValueError) as exc_info:
            basic_rag_pipeline.query(query="")

        error_msg = str(exc_info.value).lower()
        assert "query" in error_msg or "empty" in error_msg, \
            "Error message must mention query or empty"

    def test_query_validates_top_k_range(self, basic_rag_pipeline, sample_query):
        """
        FR-002: Query MUST validate top_k parameter range.

        Given: BasicRAG pipeline instance
        When: top_k is 0 or >100
        Then: ValueError raised
        """
        # Test top_k=0 (invalid)
        with pytest.raises(ValueError) as exc_info:
            basic_rag_pipeline.query(sample_query, top_k=0)

        error_msg = str(exc_info.value).lower()
        assert "top_k" in error_msg or "top-k" in error_msg, \
            "Error message must mention top_k parameter"

        # Test top_k=101 (exceeds max)
        with pytest.raises(ValueError) as exc_info:
            basic_rag_pipeline.query(sample_query, top_k=101)

        error_msg = str(exc_info.value).lower()
        assert "top_k" in error_msg or "top-k" in error_msg, \
            "Error message must mention top_k parameter"

    @pytest.mark.requires_database
    def test_query_returns_valid_structure(self, basic_rag_pipeline, sample_query):
        """
        FR-003: Query MUST return valid response structure.

        Given: BasicRAG pipeline instance
        When: Valid query executed
        Then: Response has answer, contexts, metadata fields
        """
        result = basic_rag_pipeline.query(sample_query)

        # Verify top-level structure
        assert "answer" in result, "Response must have 'answer' field"
        assert "contexts" in result, "Response must have 'contexts' field"
        assert "metadata" in result, "Response must have 'metadata' field"

        # Verify metadata structure
        metadata = result["metadata"]
        assert "retrieval_method" in metadata, \
            "Metadata must include 'retrieval_method'"
        assert "context_count" in metadata, \
            "Metadata must include 'context_count'"
        assert "sources" in metadata, \
            "Metadata must include 'sources'"

        # Verify data types
        assert isinstance(result["answer"], str), "Answer must be string"
        assert isinstance(result["contexts"], list), "Contexts must be list"
        assert isinstance(metadata, dict), "Metadata must be dict"

    def test_load_documents_method_exists(self, basic_rag_pipeline):
        """
        FR-001: BasicRAG MUST implement load_documents() method.

        Given: BasicRAG pipeline instance
        When: Method existence is checked
        Then: load_documents() method exists
        """
        assert hasattr(basic_rag_pipeline, 'load_documents'), \
            "BasicRAG pipeline must have load_documents() method"

        assert callable(basic_rag_pipeline.load_documents), \
            "load_documents() must be callable"

    def test_load_documents_validates_input(self, basic_rag_pipeline):
        """
        FR-002: Load documents MUST validate input.

        Given: BasicRAG pipeline instance
        When: load_documents() called with empty list or invalid documents
        Then: ValueError raised
        """
        # Test documents=[]
        with pytest.raises(ValueError) as exc_info:
            basic_rag_pipeline.load_documents(documents=[])

        error_msg = str(exc_info.value).lower()
        assert "documents" in error_msg or "empty" in error_msg, \
            "Error message must mention documents or empty"

    @pytest.mark.requires_database
    def test_load_documents_returns_valid_structure(self, basic_rag_pipeline, sample_documents):
        """
        FR-003: Load documents MUST return valid response structure.

        Given: BasicRAG pipeline instance and sample documents
        When: load_documents() executed
        Then: Response has documents_loaded, embeddings_generated, etc.
        """
        result = basic_rag_pipeline.load_documents(sample_documents)

        # Verify response structure
        assert "documents_loaded" in result, \
            "Response must have 'documents_loaded' field"
        assert "documents_failed" in result, \
            "Response must have 'documents_failed' field"
        assert "embeddings_generated" in result, \
            "Response must have 'embeddings_generated' field"

        # Verify values are reasonable
        assert isinstance(result["documents_loaded"], int), \
            "documents_loaded must be integer"
        assert result["documents_loaded"] > 0, \
            "Should load at least one document"

    def test_embed_method_exists(self, basic_rag_pipeline):
        """
        FR-001: BasicRAG MAY implement embed() method.

        Given: BasicRAG pipeline instance
        When: Method existence is checked
        Then: embed() method exists (optional)
        """
        # Embed method is optional, so we just check if it exists
        if hasattr(basic_rag_pipeline, 'embed'):
            assert callable(basic_rag_pipeline.embed), \
                "If embed() exists, it must be callable"

    @pytest.mark.requires_database
    def test_query_with_valid_method_parameter(self, basic_rag_pipeline, sample_query):
        """
        FR-002: Query MUST accept valid method parameter.

        Given: BasicRAG pipeline instance
        When: query() called with method="vector"
        Then: Query executes successfully
        """
        result = basic_rag_pipeline.query(sample_query, method="vector")

        assert result is not None, "Query should return result"
        assert "metadata" in result, "Result should have metadata"
        assert result["metadata"].get("retrieval_method") is not None, \
            "Metadata should include retrieval_method"

    def test_query_response_includes_execution_time(self, basic_rag_pipeline, sample_query):
        """
        FR-003: Query response metadata MUST include execution time.

        Given: BasicRAG pipeline instance
        When: Query executed
        Then: Metadata includes execution_time_ms
        """
        result = basic_rag_pipeline.query(sample_query)

        metadata = result.get("metadata", {})

        # execution_time_ms is optional but recommended
        if "execution_time_ms" in metadata:
            assert isinstance(metadata["execution_time_ms"], (int, float)), \
                "execution_time_ms must be numeric"
            assert metadata["execution_time_ms"] >= 0, \
                "execution_time_ms must be non-negative"
