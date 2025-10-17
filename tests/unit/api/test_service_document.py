"""
Unit tests for document service.

Tests the async document upload service in isolation.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from iris_rag.api.models.upload import DocumentUploadOperation, UploadStatus
from iris_rag.api.services.document_service import DocumentService


class TestDocumentService:
    """Test document upload service."""

    @pytest.fixture
    def mock_connection_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()

        pool.get_connection.return_value.__enter__.return_value = conn
        conn.cursor.return_value = cursor

        return pool

    @pytest.fixture
    def document_service(self, mock_connection_pool):
        """Create document service instance."""
        return DocumentService(mock_connection_pool)

    def test_create_upload_operation(self, document_service, mock_connection_pool):
        """Test creating new upload operation."""
        api_key_id = "test-key-id"
        file_size_bytes = 1024000
        pipeline_type = "graphrag"
        total_documents = 10

        operation = document_service.create_upload_operation(
            api_key_id=api_key_id,
            file_size_bytes=file_size_bytes,
            pipeline_type=pipeline_type,
            total_documents=total_documents,
        )

        assert operation.operation_id is not None
        assert operation.status == UploadStatus.PENDING
        assert operation.file_size_bytes == file_size_bytes
        assert operation.pipeline_type == pipeline_type
        assert operation.total_documents == total_documents

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

    def test_get_upload_operation(self, document_service, mock_connection_pool):
        """Test retrieving upload operation by ID."""
        operation_id = "op-123"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            operation_id,
            "key-id",
            "processing",
            datetime.utcnow(),
            datetime.utcnow(),
            None,
            100,
            47,
            0,
            47.0,
            1024000,
            "graphrag",
            None,
            None,
        )

        operation = document_service.get_upload_operation(operation_id)

        assert operation.operation_id == operation_id
        assert operation.status == UploadStatus.PROCESSING
        assert operation.progress_percentage == 47.0

    def test_get_upload_operation_not_found(self, document_service, mock_connection_pool):
        """Test retrieving non-existent upload operation."""
        operation_id = "nonexistent"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = None

        operation = document_service.get_upload_operation(operation_id)

        assert operation is None

    def test_update_upload_progress(self, document_service, mock_connection_pool):
        """Test updating upload progress."""
        operation_id = "op-123"
        processed_documents = 75
        total_documents = 100

        document_service.update_upload_progress(
            operation_id=operation_id,
            processed_documents=processed_documents,
            total_documents=total_documents,
        )

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()

        # Verify progress percentage calculation
        call_args = cursor.execute.call_args[0]
        assert "75.0" in str(call_args) or "75" in str(call_args)  # 75/100 = 75%

    def test_mark_upload_completed(self, document_service, mock_connection_pool):
        """Test marking upload as completed."""
        operation_id = "op-123"

        document_service.mark_upload_completed(operation_id)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()

    def test_mark_upload_failed(self, document_service, mock_connection_pool):
        """Test marking upload as failed."""
        operation_id = "op-123"
        error_message = "File validation failed"

        document_service.mark_upload_failed(operation_id, error_message)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()

    def test_validate_document_file_valid_pdf(self, document_service):
        """Test validating valid PDF file."""
        file_content = b"%PDF-1.4\n%..."
        filename = "document.pdf"

        is_valid, error = document_service.validate_document_file(file_content, filename)

        assert is_valid is True
        assert error is None

    def test_validate_document_file_valid_txt(self, document_service):
        """Test validating valid text file."""
        file_content = b"This is a text document."
        filename = "document.txt"

        is_valid, error = document_service.validate_document_file(file_content, filename)

        assert is_valid is True
        assert error is None

    def test_validate_document_file_too_large(self, document_service):
        """Test validating file that's too large."""
        file_content = b"x" * (101 * 1024 * 1024)  # 101 MB
        filename = "large.pdf"

        is_valid, error = document_service.validate_document_file(file_content, filename)

        assert is_valid is False
        assert "size" in error.lower()

    def test_validate_document_file_invalid_format(self, document_service):
        """Test validating file with invalid format."""
        file_content = b"Some content"
        filename = "document.exe"

        is_valid, error = document_service.validate_document_file(file_content, filename)

        assert is_valid is False
        assert "format" in error.lower()

    def test_validate_document_file_invalid_encoding(self, document_service):
        """Test validating file with invalid encoding."""
        file_content = b"\xff\xfe\x00\x00"  # Invalid UTF-8
        filename = "document.txt"

        is_valid, error = document_service.validate_document_file(file_content, filename)

        assert is_valid is False
        assert "encoding" in error.lower()

    def test_process_document_async(self, document_service):
        """Test async document processing."""
        operation_id = "op-123"
        file_content = b"Test document content"
        pipeline_type = "basic"

        with patch.object(document_service, "_process_document_background") as mock_process:
            document_service.process_document_async(
                operation_id=operation_id,
                file_content=file_content,
                pipeline_type=pipeline_type,
            )

            # Should schedule background task
            mock_process.assert_called_once()

    def test_get_upload_operations_by_api_key(self, document_service, mock_connection_pool):
        """Test retrieving upload operations for API key."""
        api_key_id = "test-key-id"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            (
                "op-1",
                api_key_id,
                "completed",
                datetime.utcnow(),
                datetime.utcnow(),
                datetime.utcnow(),
                100,
                100,
                0,
                100.0,
                1024000,
                "basic",
                None,
                None,
            ),
            (
                "op-2",
                api_key_id,
                "processing",
                datetime.utcnow(),
                datetime.utcnow(),
                None,
                50,
                25,
                0,
                50.0,
                512000,
                "graphrag",
                None,
                None,
            ),
        ]

        operations = document_service.get_upload_operations_by_api_key(api_key_id)

        assert len(operations) == 2
        assert operations[0].status == UploadStatus.COMPLETED
        assert operations[1].status == UploadStatus.PROCESSING

    def test_cancel_upload_operation(self, document_service, mock_connection_pool):
        """Test canceling upload operation."""
        operation_id = "op-123"

        document_service.cancel_upload_operation(operation_id)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()

    def test_calculate_progress_percentage(self, document_service):
        """Test progress percentage calculation."""
        assert document_service.calculate_progress(50, 100) == 50.0
        assert document_service.calculate_progress(1, 3) == pytest.approx(33.33, rel=0.01)
        assert document_service.calculate_progress(100, 100) == 100.0
        assert document_service.calculate_progress(0, 100) == 0.0

    def test_estimate_processing_time(self, document_service):
        """Test processing time estimation."""
        file_size_bytes = 1024000  # 1 MB
        documents_count = 10

        estimated_seconds = document_service.estimate_processing_time(
            file_size_bytes, documents_count
        )

        assert estimated_seconds > 0
        assert isinstance(estimated_seconds, (int, float))

    def test_cleanup_old_operations(self, document_service, mock_connection_pool):
        """Test cleaning up old completed operations."""
        retention_days = 30

        document_service.cleanup_old_operations(retention_days)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()
