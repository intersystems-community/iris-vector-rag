"""
Document API routes for RAG API.

Implements FR-021 to FR-024: Asynchronous document upload with progress tracking.
Provides POST /documents/upload and GET /documents/operations endpoints.
"""

import logging
from uuid import UUID
from typing import List

from fastapi import APIRouter, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse

from iris_vector_rag.api.models.upload import (
    DocumentUploadOperation,
    DocumentUploadRequest,
    DocumentUploadResponse,
    PipelineType
)
from iris_vector_rag.api.models.auth import ApiKey, Permission
from iris_vector_rag.api.middleware.auth import ApiKeyAuth
from iris_vector_rag.api.services.document_service import DocumentService
from iris_vector_rag.api.models.errors import (
    ErrorResponse,
    ErrorType,
    ErrorInfo,
    ErrorDetails,
    validation_error
)


logger = logging.getLogger(__name__)


def create_document_router(
    document_service: DocumentService,
    auth_service: ApiKeyAuth
) -> APIRouter:
    """
    Create document API router.

    Args:
        document_service: DocumentService for upload operations
        auth_service: Authentication service

    Returns:
        FastAPI router with document endpoints
    """
    router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

    async def require_write_permission(request: Request) -> ApiKey:
        """Dependency to check write permission."""
        return await auth_service(request, required_permission=Permission.WRITE)

    @router.post(
        "/upload",
        response_model=DocumentUploadResponse,
        responses={
            200: {"description": "Upload initiated successfully"},
            401: {"description": "Authentication required"},
            403: {"description": "Insufficient permissions (write required)"},
            413: {"description": "File size exceeds 100MB limit"},
            422: {"description": "Validation error"},
            429: {"description": "Rate limit exceeded"}
        },
        summary="Upload documents for indexing",
        description="""
        Initiate asynchronous document upload operation.

        **Requirements:**
        - Write permission required
        - Maximum file size: 100 MB
        - Supported formats: PDF, TXT, DOCX, HTML, Markdown

        **Returns:**
        - operation_id for tracking progress
        - Initial status (pending or validating)

        **Progress Tracking:**
        Use GET /documents/operations/{operation_id} to track progress.

        **Concurrent Uploads:**
        - Maximum 1-5 concurrent uploads per API key (tier-dependent)
        """
    )
    async def upload_documents(
        request: Request,
        file: UploadFile = File(..., description="Document file to upload"),
        pipeline_type: PipelineType = Form(default=PipelineType.BASIC, description="Pipeline for indexing"),
        chunk_size: int = Form(default=1000, ge=100, le=5000, description="Chunk size (characters)"),
        chunk_overlap: int = Form(default=200, ge=0, le=1000, description="Chunk overlap (characters)"),
        api_key: ApiKey = Depends(require_write_permission)
    ) -> DocumentUploadResponse:
        """
        Upload documents for indexing (FR-021).

        Args:
            request: FastAPI request
            file: Uploaded file
            pipeline_type: Pipeline to use for indexing
            chunk_size: Chunk size for document splitting
            chunk_overlap: Overlap between chunks
            api_key: Authenticated API key with write permission

        Returns:
            DocumentUploadResponse with operation_id

        Raises:
            HTTPException: On validation errors or size limits
        """
        logger.info(
            f"Document upload initiated: {file.filename} ({file.size} bytes) "
            f"- api_key={api_key.key_id}"
        )

        # Validate file size (FR-022)
        max_size_bytes = 104857600  # 100 MB

        if file.size and file.size > max_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=validation_error(
                    field="file",
                    rejected_value=f"{file.size} bytes",
                    message="File size exceeds maximum of 100 MB",
                    max_value=max_size_bytes
                ).model_dump()
            )

        # Create upload request
        upload_request = DocumentUploadRequest(
            pipeline_type=pipeline_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        try:
            # TODO: Parse file to count documents
            # For now, estimate based on file size (rough approximation)
            estimated_docs = max(1, int(file.size / 10000))  # ~10KB per doc

            # Initiate upload
            response = await document_service.initiate_upload(
                api_key_id=api_key.key_id,
                file_size_bytes=file.size or 0,
                total_documents=estimated_docs,
                request=upload_request
            )

            logger.info(f"Upload operation created: {response.operation_id}")

            return response

        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=validation_error(
                    field="file",
                    rejected_value=file.filename,
                    message=str(e)
                ).model_dump()
            )

        except Exception as e:
            logger.error(f"Upload initiation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.INTERNAL_SERVER_ERROR,
                        reason="Failed to initiate upload",
                        details=ErrorDetails(
                            message="An unexpected error occurred. Please try again."
                        )
                    )
                ).model_dump()
            )

    @router.get(
        "/operations/{operation_id}",
        response_model=DocumentUploadOperation,
        responses={
            200: {"description": "Operation status"},
            401: {"description": "Authentication required"},
            404: {"description": "Operation not found"}
        },
        summary="Get upload operation status",
        description="""
        Track progress of document upload operation.

        **Returns:**
        - Current status (pending, validating, processing, completed, failed)
        - Progress percentage (0.0-100.0)
        - Processed vs total documents
        - Validation errors (if any)
        - Completion time (if finished)

        **Status Values:**
        - pending: Operation queued
        - validating: Checking document format and encoding
        - processing: Indexing documents into pipeline
        - completed: All documents indexed successfully
        - failed: Operation failed (check error_message)
        """
    )
    async def get_operation_status(
        operation_id: UUID,
        request: Request,
        api_key: ApiKey = Depends(require_write_permission)
    ) -> DocumentUploadOperation:
        """
        Get upload operation status (FR-023).

        Args:
            operation_id: Upload operation ID
            request: FastAPI request
            api_key: Authenticated API key

        Returns:
            DocumentUploadOperation with current status

        Raises:
            HTTPException: If operation not found
        """
        logger.debug(f"Getting operation status: {operation_id}")

        operation = document_service.get_operation_status(operation_id)

        if not operation:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.BAD_REQUEST,
                        reason=f"Operation not found: {operation_id}",
                        details=ErrorDetails(
                            message=f"Upload operation '{operation_id}' does not exist"
                        )
                    )
                ).model_dump()
            )

        # Verify operation belongs to API key
        if operation.api_key_id != api_key.key_id:
            # Check if API key has admin permission
            if Permission.ADMIN not in api_key.permissions:
                raise HTTPException(
                    status_code=403,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.AUTHORIZATION_ERROR,
                            reason="Operation does not belong to this API key",
                            details=ErrorDetails(
                                message="You can only view your own upload operations"
                            )
                        )
                    ).model_dump()
                )

        return operation

    @router.get(
        "/operations",
        response_model=List[DocumentUploadOperation],
        summary="List upload operations",
        description="""
        List recent document upload operations for authenticated API key.

        **Returns:**
        - Up to 50 most recent operations
        - Sorted by creation time (newest first)
        - Only operations belonging to authenticated API key (unless admin)
        """
    )
    async def list_operations(
        request: Request,
        api_key: ApiKey = Depends(require_write_permission),
        limit: int = 50
    ) -> List[DocumentUploadOperation]:
        """
        List upload operations (FR-023).

        Args:
            request: FastAPI request
            api_key: Authenticated API key
            limit: Maximum operations to return

        Returns:
            List of DocumentUploadOperation objects
        """
        logger.debug(f"Listing operations for API key: {api_key.key_id}")

        # Admin can see all operations
        if Permission.ADMIN in api_key.permissions:
            operations = document_service.list_operations(limit=limit)
        else:
            operations = document_service.list_operations(
                api_key_id=api_key.key_id,
                limit=limit
            )

        return operations

    return router
