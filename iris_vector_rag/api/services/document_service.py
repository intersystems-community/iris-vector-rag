"""
Document Upload Service for RAG API.

Implements FR-021 to FR-024: Asynchronous document upload with validation.
Manages document upload operations with progress tracking.
"""

import logging
import asyncio
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path

from iris_vector_rag.api.models.upload import (
    DocumentUploadOperation,
    DocumentUploadRequest,
    DocumentUploadResponse,
    UploadStatus,
    PipelineType
)


logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for managing document uploads.

    Implements FR-021: Async document upload
    Implements FR-022: File size validation (100MB max)
    Implements FR-023: Progress tracking with percentages
    Implements FR-024: Validation error reporting
    """

    def __init__(self, connection_pool, pipeline_manager):
        """
        Initialize document service.

        Args:
            connection_pool: IRISConnectionPool for database operations
            pipeline_manager: PipelineManager for document indexing
        """
        self.connection_pool = connection_pool
        self.pipeline_manager = pipeline_manager

        # Track active upload operations
        self.active_operations: dict[UUID, DocumentUploadOperation] = {}

    async def initiate_upload(
        self,
        api_key_id: UUID,
        file_size_bytes: int,
        total_documents: int,
        request: DocumentUploadRequest
    ) -> DocumentUploadResponse:
        """
        Initiate document upload operation.

        Args:
            api_key_id: API key that initiated upload
            file_size_bytes: Total file size in bytes
            total_documents: Number of documents to process
            request: Upload configuration

        Returns:
            DocumentUploadResponse with operation_id

        Raises:
            ValueError: If file size exceeds 100MB limit (FR-022)

        Implements FR-021: Async document upload
        """
        # Validate file size (FR-022)
        max_size_bytes = 104857600  # 100 MB
        if file_size_bytes > max_size_bytes:
            raise ValueError(
                f"File size {file_size_bytes} bytes exceeds maximum of {max_size_bytes} bytes (100MB)"
            )

        # Generate operation ID
        operation_id = uuid4()

        # Create operation object
        operation = DocumentUploadOperation(
            operation_id=operation_id,
            api_key_id=api_key_id,
            status=UploadStatus.PENDING,
            created_at=datetime.utcnow(),
            total_documents=total_documents,
            processed_documents=0,
            failed_documents=0,
            progress_percentage=0.0,
            file_size_bytes=file_size_bytes,
            pipeline_type=request.pipeline_type
        )

        # Store in database
        self._save_operation(operation)

        # Store in active operations
        self.active_operations[operation_id] = operation

        logger.info(
            f"Initiated document upload: {operation_id} "
            f"({total_documents} docs, {file_size_bytes} bytes)"
        )

        # Start async processing (non-blocking)
        asyncio.create_task(
            self._process_upload(
                operation_id,
                request.pipeline_type,
                request.chunk_size,
                request.chunk_overlap
            )
        )

        return DocumentUploadResponse(
            operation_id=operation_id,
            status=UploadStatus.PENDING,
            message="Document upload initiated. Use operation_id to track progress."
        )

    async def _process_upload(
        self,
        operation_id: UUID,
        pipeline_type: PipelineType,
        chunk_size: int,
        chunk_overlap: int
    ):
        """
        Process document upload asynchronously.

        Args:
            operation_id: Upload operation ID
            pipeline_type: Pipeline for indexing
            chunk_size: Chunk size for splitting
            chunk_overlap: Overlap between chunks
        """
        operation = self.active_operations.get(operation_id)

        if not operation:
            logger.error(f"Operation not found: {operation_id}")
            return

        try:
            # Update status to VALIDATING
            operation.status = UploadStatus.VALIDATING
            operation.started_at = datetime.utcnow()
            self._save_operation(operation)

            logger.info(f"Validating documents for operation: {operation_id}")

            # TODO: Perform document validation
            # - Check UTF-8 encoding
            # - Validate document structure
            # - Check chunk size limits
            validation_errors = []

            if validation_errors:
                operation.status = UploadStatus.FAILED
                operation.validation_errors = validation_errors
                operation.error_message = f"Validation failed with {len(validation_errors)} errors"
                operation.completed_at = datetime.utcnow()
                self._save_operation(operation)
                return

            # Update status to PROCESSING
            operation.status = UploadStatus.PROCESSING
            self._save_operation(operation)

            logger.info(f"Processing documents for operation: {operation_id}")

            # Get pipeline for indexing
            pipeline = self.pipeline_manager.get_pipeline(pipeline_type.value)

            if not pipeline:
                raise ValueError(f"Pipeline not available: {pipeline_type.value}")

            # Process documents (simulated for now)
            # TODO: Actual document loading with pipeline.load_documents()
            for i in range(operation.total_documents):
                # Simulate processing
                await asyncio.sleep(0.1)

                # Update progress
                operation.processed_documents = i + 1
                operation.progress_percentage = (
                    operation.processed_documents / operation.total_documents * 100.0
                )

                # Save progress every 10 documents or at completion
                if i % 10 == 0 or i == operation.total_documents - 1:
                    self._save_operation(operation)

            # Mark as completed
            operation.status = UploadStatus.COMPLETED
            operation.completed_at = datetime.utcnow()
            operation.progress_percentage = 100.0
            self._save_operation(operation)

            logger.info(
                f"Completed document upload: {operation_id} "
                f"({operation.processed_documents}/{operation.total_documents} docs)"
            )

        except Exception as e:
            logger.error(f"Document upload failed for {operation_id}: {e}")

            operation.status = UploadStatus.FAILED
            operation.error_message = str(e)
            operation.completed_at = datetime.utcnow()
            self._save_operation(operation)

        finally:
            # Remove from active operations after some time
            await asyncio.sleep(300)  # Keep for 5 minutes
            self.active_operations.pop(operation_id, None)

    def get_operation_status(self, operation_id: UUID) -> Optional[DocumentUploadOperation]:
        """
        Get status of upload operation.

        Args:
            operation_id: Upload operation ID

        Returns:
            DocumentUploadOperation or None if not found

        Implements FR-023: Progress tracking
        """
        # Check active operations first
        if operation_id in self.active_operations:
            return self.active_operations[operation_id]

        # Check database
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    operation_id, api_key_id, status, created_at, started_at,
                    completed_at, total_documents, processed_documents,
                    failed_documents, progress_percentage, file_size_bytes,
                    pipeline_type, validation_errors, error_message
                FROM document_upload_operations
                WHERE operation_id = ?
            """

            cursor.execute(query, (str(operation_id),))
            row = cursor.fetchone()

            if not row:
                return None

            return DocumentUploadOperation(
                operation_id=UUID(row[0]),
                api_key_id=UUID(row[1]),
                status=UploadStatus(row[2]),
                created_at=row[3],
                started_at=row[4],
                completed_at=row[5],
                total_documents=row[6],
                processed_documents=row[7],
                failed_documents=row[8],
                progress_percentage=row[9],
                file_size_bytes=row[10],
                pipeline_type=PipelineType(row[11]),
                validation_errors=row[12].split('|') if row[12] else None,
                error_message=row[13]
            )

    def _save_operation(self, operation: DocumentUploadOperation):
        """
        Save operation to database.

        Args:
            operation: DocumentUploadOperation to save
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT COUNT(*) FROM document_upload_operations WHERE operation_id = ?",
                (str(operation.operation_id),)
            )

            exists = cursor.fetchone()[0] > 0

            if exists:
                # Update
                query = """
                    UPDATE document_upload_operations
                    SET status = ?, started_at = ?, completed_at = ?,
                        processed_documents = ?, failed_documents = ?,
                        progress_percentage = ?, validation_errors = ?,
                        error_message = ?
                    WHERE operation_id = ?
                """

                cursor.execute(query, (
                    operation.status.value,
                    operation.started_at,
                    operation.completed_at,
                    operation.processed_documents,
                    operation.failed_documents,
                    operation.progress_percentage,
                    '|'.join(operation.validation_errors) if operation.validation_errors else None,
                    operation.error_message,
                    str(operation.operation_id)
                ))

            else:
                # Insert
                query = """
                    INSERT INTO document_upload_operations (
                        operation_id, api_key_id, status, created_at, started_at,
                        completed_at, total_documents, processed_documents,
                        failed_documents, progress_percentage, file_size_bytes,
                        pipeline_type, validation_errors, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor.execute(query, (
                    str(operation.operation_id),
                    str(operation.api_key_id),
                    operation.status.value,
                    operation.created_at,
                    operation.started_at,
                    operation.completed_at,
                    operation.total_documents,
                    operation.processed_documents,
                    operation.failed_documents,
                    operation.progress_percentage,
                    operation.file_size_bytes,
                    operation.pipeline_type.value,
                    '|'.join(operation.validation_errors) if operation.validation_errors else None,
                    operation.error_message
                ))

            conn.commit()

    def list_operations(
        self,
        api_key_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[DocumentUploadOperation]:
        """
        List upload operations.

        Args:
            api_key_id: Optional filter by API key
            limit: Maximum operations to return

        Returns:
            List of DocumentUploadOperation objects
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            if api_key_id:
                query = """
                    SELECT
                        operation_id, api_key_id, status, created_at, started_at,
                        completed_at, total_documents, processed_documents,
                        failed_documents, progress_percentage, file_size_bytes,
                        pipeline_type, validation_errors, error_message
                    FROM document_upload_operations
                    WHERE api_key_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                cursor.execute(query, (str(api_key_id), limit))
            else:
                query = """
                    SELECT
                        operation_id, api_key_id, status, created_at, started_at,
                        completed_at, total_documents, processed_documents,
                        failed_documents, progress_percentage, file_size_bytes,
                        pipeline_type, validation_errors, error_message
                    FROM document_upload_operations
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))

            operations = []
            for row in cursor.fetchall():
                operations.append(DocumentUploadOperation(
                    operation_id=UUID(row[0]),
                    api_key_id=UUID(row[1]),
                    status=UploadStatus(row[2]),
                    created_at=row[3],
                    started_at=row[4],
                    completed_at=row[5],
                    total_documents=row[6],
                    processed_documents=row[7],
                    failed_documents=row[8],
                    progress_percentage=row[9],
                    file_size_bytes=row[10],
                    pipeline_type=PipelineType(row[11]),
                    validation_errors=row[12].split('|') if row[12] else None,
                    error_message=row[13]
                ))

            return operations
