"""
Unit tests for BasicRAGPipeline ingestion error handling.

Tests verify that document ingestion failures are properly reported as IngestionError
with accurate counts rather than silently converting failures to success.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import IngestionError


class TestBasicRAGIngestionError:
    """Test suite for BasicRAGPipeline ingestion error handling."""

    def test_load_documents_raises_ingestion_error_on_vector_store_failure(self):
        """
        Verify that load_documents raises IngestionError when vector store raises.

        This test ensures that exceptions from the vector store are not silently
        converted to success counts, but instead raised as IngestionError with
        documents_loaded=0 and documents_failed set to the input length.
        """
        # Setup: Create a mock vector store that raises an exception
        mock_vector_store = Mock()
        mock_vector_store.add_documents.side_effect = RuntimeError(
            "Database connection failed"
        )

        # Create pipeline with mocked vector store
        pipeline = BasicRAGPipeline(vector_store=mock_vector_store)

        # Prepare test documents
        docs = [
            Document(
                page_content="Document 1 content",
                metadata={"source": "doc1.txt"},
            ),
            Document(
                page_content="Document 2 content",
                metadata={"source": "doc2.txt"},
            ),
            Document(
                page_content="Document 3 content",
                metadata={"source": "doc3.txt"},
            ),
        ]

        # Execute: Call load_documents and expect IngestionError
        with pytest.raises(IngestionError) as exc_info:
            pipeline.load_documents(documents=docs, generate_embeddings=True)

        # Assert: Check exception properties
        error = exc_info.value
        assert error.documents_loaded == 0, "documents_loaded should be 0 on failure"
        assert error.documents_failed == 3, "documents_failed should match input length"
        assert "Database connection failed" in str(
            error
        ), "Original error should be mentioned"
        assert error.original_error is not None, "original_error should be preserved"

    def test_load_documents_raises_ingestion_error_when_vector_store_missing(self):
        """
        Verify that load_documents raises IngestionError when vector_store attribute is missing.

        Note: The base class creates an IRISVectorStore by default, so we manually set
        vector_store to None on the instance to simulate a missing vector store.
        """
        # Setup: Create pipeline and then manually remove the vector store
        pipeline = BasicRAGPipeline(vector_store=None)
        pipeline.vector_store = None  # Simulate missing vector store

        # Prepare test documents
        docs = [
            Document(
                page_content="Test content",
                metadata={"source": "test.txt"},
            ),
        ]

        # Execute: Call load_documents and expect IngestionError
        with pytest.raises(IngestionError) as exc_info:
            pipeline.load_documents(documents=docs, generate_embeddings=True)

        # Assert: Check exception properties
        error = exc_info.value
        assert error.documents_loaded == 0, "documents_loaded should be 0"
        assert error.documents_failed == 1, "documents_failed should equal input length"
        assert "Vector store not available" in str(error)

    def test_load_documents_raises_ingestion_error_without_embeddings_on_failure(self):
        """
        Verify IngestionError is raised even when generate_embeddings=False.
        """
        # Setup: Create pipeline with failing vector store
        mock_vector_store = Mock()
        mock_vector_store.add_documents.side_effect = ValueError(
            "Invalid document format"
        )
        pipeline = BasicRAGPipeline(vector_store=mock_vector_store)

        # Prepare test documents
        docs = [
            Document(
                page_content="Content",
                metadata={"source": "test.txt"},
            ),
        ]

        # Execute: Call load_documents with embeddings disabled
        with pytest.raises(IngestionError) as exc_info:
            pipeline.load_documents(documents=docs, generate_embeddings=False)

        # Assert: Check that IngestionError is raised (not caught and converted to success)
        error = exc_info.value
        assert error.documents_loaded == 0, "documents_loaded should be 0 on failure"
        assert error.documents_failed == 1, "documents_failed should match input length"

    def test_load_documents_succeeds_with_valid_vector_store(self):
        """
        Verify that load_documents returns correct counts on success.

        This sanity check ensures that the fix doesn't break normal operation.
        """
        # Setup: Create mock vector store that succeeds
        mock_vector_store = Mock()
        mock_vector_store.add_documents.return_value = None  # Success
        pipeline = BasicRAGPipeline(vector_store=mock_vector_store)

        # Prepare test documents
        docs = [
            Document(
                page_content="Document 1",
                metadata={"source": "doc1.txt"},
            ),
            Document(
                page_content="Document 2",
                metadata={"source": "doc2.txt"},
            ),
        ]

        # Execute: Call load_documents
        result = pipeline.load_documents(documents=docs, generate_embeddings=True)

        # Assert: Check that it succeeds and returns correct counts
        assert result["documents_loaded"] == 2, "Should load both documents"
        assert result["documents_failed"] == 0, "No documents should fail"
        assert result["embeddings_generated"] == 2, "Should generate embeddings"

    def test_ingestion_error_preserves_original_exception(self):
        """
        Verify that IngestionError preserves the original exception chain.
        """
        # Setup
        original_error = RuntimeError("Database timeout")
        mock_vector_store = Mock()
        mock_vector_store.add_documents.side_effect = original_error
        pipeline = BasicRAGPipeline(vector_store=mock_vector_store)

        docs = [Document(page_content="test", metadata={})]

        # Execute and verify exception chain
        with pytest.raises(IngestionError) as exc_info:
            pipeline.load_documents(documents=docs)

        error = exc_info.value
        assert error.original_error is original_error
        assert error.__cause__ is original_error

    def test_load_documents_multiple_failures_accurate_count(self):
        """
        Verify that failure count accurately reflects the number of input documents.

        This ensures that when a bulk add_documents call fails, all documents
        are marked as failed (not partially counted as success).
        """
        # Setup
        mock_vector_store = Mock()
        mock_vector_store.add_documents.side_effect = Exception("Bulk insert failed")
        pipeline = BasicRAGPipeline(vector_store=mock_vector_store)

        # Create a larger set of documents to ensure counts scale correctly
        docs = [
            Document(page_content=f"Doc {i}", metadata={"source": f"doc{i}.txt"})
            for i in range(10)
        ]

        # Execute and verify
        with pytest.raises(IngestionError) as exc_info:
            pipeline.load_documents(documents=docs, generate_embeddings=True)

        error = exc_info.value
        assert error.documents_loaded == 0
        assert error.documents_failed == 10, "All documents should be marked as failed"
