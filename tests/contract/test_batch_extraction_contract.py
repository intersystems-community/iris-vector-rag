"""
Contract tests for batch entity extraction API.

These tests validate the extract_batch() API contract before implementation.
Tests MUST fail initially, then pass after implementation (TDD).
"""

import pytest
from iris_rag.core.models import Document, BatchExtractionResult


class TestBatchExtractionContract:
    """Contract tests for EntityExtractionService.extract_batch() method."""

    def test_extract_batch_method_exists(self):
        """Validate extract_batch() method exists on EntityExtractionService."""
        from iris_rag.services.entity_extraction import EntityExtractionService

        assert hasattr(EntityExtractionService, 'extract_batch'), \
            "EntityExtractionService must have extract_batch() method"

    def test_extract_batch_signature(self):
        """Validate extract_batch() has correct signature."""
        from iris_rag.services.entity_extraction import EntityExtractionService
        import inspect

        sig = inspect.signature(EntityExtractionService.extract_batch)
        params = sig.parameters

        # Validate required parameters
        assert 'self' in params, "extract_batch must be an instance method"
        assert 'documents' in params, "extract_batch must accept 'documents' parameter"

        # Validate optional parameters with defaults
        assert 'token_budget' in params, "extract_batch must accept 'token_budget' parameter"
        assert params['token_budget'].default == 8192, \
            "token_budget default must be 8192 per FR-006"

    def test_extract_batch_returns_batch_result(self):
        """Validate extract_batch() returns BatchExtractionResult type."""
        from iris_rag.services.entity_extraction import EntityExtractionService
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import IRISConnectionManager

        # Initialize service (will need real config in implementation)
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        service = EntityExtractionService(config_manager, connection_manager)

        # Create test document
        test_doc = Document(id="test1", page_content="Test document for entity extraction.")

        # Call extract_batch
        result = service.extract_batch([test_doc])

        # Validate return type
        assert isinstance(result, BatchExtractionResult), \
            "extract_batch must return BatchExtractionResult instance"
        assert result.batch_id is not None, "BatchExtractionResult must have batch_id"
        assert result.per_document_entities is not None, \
            "BatchExtractionResult must have per_document_entities"

    def test_extract_batch_empty_documents_raises_error(self):
        """Validate extract_batch() raises ValueError on empty documents list."""
        from iris_rag.services.entity_extraction import EntityExtractionService
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import IRISConnectionManager

        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        service = EntityExtractionService(config_manager, connection_manager)

        # Empty documents list should raise ValueError
        with pytest.raises(ValueError, match="documents.*cannot be empty"):
            service.extract_batch([])

    def test_extract_batch_respects_token_budget(self):
        """Validate extract_batch() respects token_budget parameter."""
        from iris_rag.services.entity_extraction import EntityExtractionService
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import IRISConnectionManager

        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        service = EntityExtractionService(config_manager, connection_manager)

        # Create test documents
        test_docs = [
            Document(id=f"test{i}", page_content="Short test document.")
            for i in range(5)
        ]

        # Call with custom token budget
        result = service.extract_batch(test_docs, token_budget=4096)

        # Result should be valid (implementation will enforce budget)
        assert isinstance(result, BatchExtractionResult)

    def test_batch_extraction_result_has_required_fields(self):
        """Validate BatchExtractionResult has all required fields from data model."""
        from iris_rag.core.models import BatchExtractionResult
        import inspect

        # Get BatchExtractionResult attributes
        sig = inspect.signature(BatchExtractionResult.__init__)
        params = sig.parameters

        # Required fields from data-model.md
        required_fields = [
            'batch_id',
            'per_document_entities',
            'per_document_relationships',
            'processing_time',
            'success_status',
            'retry_count',
            'error_message'
        ]

        for field in required_fields:
            assert field in params, \
                f"BatchExtractionResult must have '{field}' field per data model"

    def test_batch_extraction_result_helper_methods(self):
        """Validate BatchExtractionResult has required helper methods."""
        from iris_rag.core.models import BatchExtractionResult

        # Required methods from data-model.md
        required_methods = [
            'get_all_entities',
            'get_all_relationships',
            'get_entity_count_by_document'
        ]

        for method in required_methods:
            assert hasattr(BatchExtractionResult, method), \
                f"BatchExtractionResult must have '{method}' method per data model"
