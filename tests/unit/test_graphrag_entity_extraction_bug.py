"""
Unit test for GraphRAG entity extraction validation bug.

Bug Report: /Users/intersystems-community/ws/kg-ticket-resolver/RAG_TEMPLATES_BUG_ENTITY_EXTRACTION.md

The bug: GraphRAGPipeline.load_documents() incorrectly throws exception claiming
no entities were extracted, even when entities ARE successfully extracted and stored.

Root Cause: Fallback path only increments total_entities if result["stored"]=True,
but extraction can succeed while storage fails, leading to false negative.

Fix: Always count extracted entities regardless of storage status. Only fail if
zero entities are EXTRACTED (not zero entities stored).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline


@pytest.fixture
def mock_graphrag_pipeline():
    """Create a mock GraphRAGPipeline for testing without database connection."""
    with patch('iris_rag.core.base.RAGPipeline.__init__', return_value=None):
        # Create pipeline instance (bypass parent __init__)
        pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)

        # Mock required attributes
        pipeline.vector_store = Mock()
        pipeline.vector_store.add_documents = Mock()
        pipeline.entity_extraction_enabled = True
        pipeline.connection_manager = Mock()
        pipeline.config_manager = Mock()

        yield pipeline


class TestGraphRAGEntityExtractionValidationBug:
    """Test the entity extraction validation bug fix."""

    def test_should_not_throw_exception_when_entities_extracted_but_storage_fails(self, mock_graphrag_pipeline):
        """
        Verify that load_documents() does NOT throw exception when entities are
        successfully extracted but storage fails (stored=False).

        This was the bug: total_entities was only incremented if stored=True,
        so successful extraction with storage failure would incorrectly raise exception.
        """
        # Use mocked pipeline
        pipeline = mock_graphrag_pipeline

        # Mock entity extraction service to return successful extraction but failed storage
        mock_result = {
            "document_id": "test_doc_1",
            "entities_extracted": 12,  # Successfully extracted
            "relationships_extracted": 66,
            "entities_count": 12,
            "relationships_count": 66,
            "stored": False,  # But storage failed!
            "errors": ["Storage adapter not available"]
        }

        pipeline.entity_extraction_service = Mock()
        pipeline.entity_extraction_service.extract_batch_with_dspy = Mock(
            side_effect=Exception("Batch extraction not available")  # Force fallback
        )
        pipeline.entity_extraction_service.process_document = Mock(return_value=mock_result)

        # Create test document
        doc = Document(
            id="test_doc_1",
            page_content="Test ticket content with entities",
            metadata={"ticket_id": "I398850"}
        )

        # This should NOT raise exception (entities were extracted, even if not stored)
        try:
            pipeline.load_documents(documents_path="", documents=[doc], generate_embeddings=True)
            success = True
        except Exception as e:
            success = False
            error_msg = str(e)

        # Assert: Should succeed (no exception thrown)
        assert success, f"load_documents() should not throw exception when entities are extracted but storage fails. Got: {error_msg if not success else 'N/A'}"

        # Verify process_document was called (fallback path)
        assert pipeline.entity_extraction_service.process_document.called

    def test_should_throw_exception_when_zero_entities_extracted(self, mock_graphrag_pipeline):
        """
        Verify that load_documents() DOES throw exception when zero entities
        are actually extracted (systematic failure).
        """
        # Use mocked pipeline
        pipeline = mock_graphrag_pipeline

        # Mock entity extraction to return zero entities (systematic failure)
        mock_result = {
            "document_id": "test_doc_1",
            "entities_extracted": 0,  # NO entities extracted
            "relationships_extracted": 0,
            "entities_count": 0,
            "relationships_count": 0,
            "stored": False,
            "errors": []
        }

        pipeline.entity_extraction_service = Mock()
        pipeline.entity_extraction_service.extract_batch_with_dspy = Mock(
            side_effect=Exception("Batch extraction not available")
        )
        pipeline.entity_extraction_service.process_document = Mock(return_value=mock_result)

        # Create test document
        doc = Document(
            id="test_doc_1",
            page_content="",
            metadata={"ticket_id": "EMPTY"}
        )

        # This SHOULD raise exception (zero entities extracted = systematic failure)
        from iris_vector_rag.pipelines.graphrag import KnowledgeGraphNotPopulatedException

        with pytest.raises(KnowledgeGraphNotPopulatedException) as exc_info:
            pipeline.load_documents(documents_path="", documents=[doc], generate_embeddings=True)

        # Verify error message mentions extraction failure
        assert "extraction failure" in str(exc_info.value).lower()

    def test_counts_entities_from_successful_extraction_in_fallback_path(self, mock_graphrag_pipeline):
        """
        Verify that total_entities is incremented based on entities_extracted,
        not based on stored status.
        """
        pipeline = mock_graphrag_pipeline

        # Mock successful extraction with storage failure
        mock_result = {
            "document_id": "test_doc_1",
            "entities_extracted": 5,
            "relationships_extracted": 10,
            "entities_count": 5,
            "relationships_count": 10,
            "stored": False,  # Storage failed but extraction succeeded
            "errors": []
        }

        pipeline.entity_extraction_service = Mock()
        pipeline.entity_extraction_service.extract_batch_with_dspy = Mock(
            side_effect=Exception("Force fallback")
        )
        pipeline.entity_extraction_service.process_document = Mock(return_value=mock_result)

        doc = Document(id="test_doc_1", page_content="Test", metadata={})

        # Should not raise exception
        pipeline.load_documents(documents_path="", documents=[doc], generate_embeddings=True)

        # Verify the fix: process_document was called and entities were counted
        assert pipeline.entity_extraction_service.process_document.called


    def test_batch_path_falls_back_to_individual_when_no_entities_in_batch_results(self, mock_graphrag_pipeline):
        """
        Verify that when batch extraction returns no entities for a document,
        it falls back to individual processing and counts those entities.

        This is the ACTUAL bug the user reported - batch_results.get(doc.id, [])
        returns empty but individual processing extracts entities successfully.
        """
        pipeline = mock_graphrag_pipeline

        # Mock batch extraction to return empty dict (no entities for any document)
        pipeline.entity_extraction_service = Mock()
        pipeline.entity_extraction_service.extract_batch_with_dspy = Mock(return_value={})

        # Mock individual processing to return successful extraction
        mock_result = {
            "document_id": "test_doc_1",
            "entities_extracted": 15,
            "relationships_extracted": 105,
            "entities_count": 15,
            "relationships_count": 105,
            "stored": True,
            "errors": []
        }
        pipeline.entity_extraction_service.process_document = Mock(return_value=mock_result)

        doc = Document(id="test_doc_1", page_content="Test ticket", metadata={"ticket_id": "I279220"})

        # Should NOT raise exception (individual processing should work)
        try:
            pipeline.load_documents(documents_path="", documents=[doc], generate_embeddings=True)
            success = True
        except Exception as e:
            success = False
            error_msg = str(e)

        # Assert: Should succeed (entities extracted via fallback)
        assert success, f"Should not throw exception when batch returns empty but individual extraction succeeds. Got: {error_msg if not success else 'N/A'}"

        # Verify process_document was called as fallback
        assert pipeline.entity_extraction_service.process_document.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:randomly"])
