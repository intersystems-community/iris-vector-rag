"""
Document upload models for RAG API.

Pydantic models for Document Upload Operation entity (Entity 6 from data-model.md).
Implements async document upload with progress tracking.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from uuid import UUID
from datetime import datetime


class UploadStatus(str, Enum):
    """Document upload operation status."""

    PENDING = "pending"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineType(str, Enum):
    """Supported RAG pipeline types."""

    BASIC = "basic"
    BASIC_RERANK = "basic_rerank"
    CRAG = "crag"
    GRAPHRAG = "graphrag"
    PYLATE_COLBERT = "pylate_colbert"


class DocumentUploadOperation(BaseModel):
    """
    Document Upload Operation (Entity 6 from data-model.md).

    Represents an asynchronous document loading task with status tracking,
    progress percentage, validation results, and completion/error information.

    Implements FR-021 to FR-024: Async document upload with validation.
    """

    operation_id: UUID = Field(
        ...,
        description="Unique operation identifier",
    )

    api_key_id: UUID = Field(
        ...,
        description="API key that initiated upload",
    )

    status: UploadStatus = Field(
        ...,
        description="Current operation status",
        examples=["processing"],
    )

    created_at: datetime = Field(
        ...,
        description="When operation was created (ISO8601)",
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="When processing started (ISO8601)",
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="When operation completed/failed (ISO8601)",
    )

    total_documents: int = Field(
        ...,
        gt=0,
        description="Total documents to process",
    )

    processed_documents: int = Field(
        default=0,
        ge=0,
        description="Documents processed so far",
    )

    failed_documents: int = Field(
        default=0,
        ge=0,
        description="Documents that failed validation/indexing",
    )

    progress_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage (0.0-100.0)",
    )

    file_size_bytes: int = Field(
        ...,
        ge=1,
        le=104857600,  # 100 MB max (FR-022)
        description="Total file size in bytes (max 100MB)",
    )

    pipeline_type: PipelineType = Field(
        ...,
        description="Pipeline for indexing",
        examples=["graphrag"],
    )

    validation_errors: Optional[List[str]] = Field(
        default=None,
        max_length=100,
        description="Validation error messages (max 100)",
    )

    error_message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message if operation failed",
    )

    @field_validator("processed_documents")
    @classmethod
    def processed_cannot_exceed_total(cls, v: int, info) -> int:
        """Validate processed documents doesn't exceed total."""
        if "total_documents" in info.data and v > info.data["total_documents"]:
            raise ValueError("processed_documents cannot exceed total_documents")
        return v

    @field_validator("progress_percentage")
    @classmethod
    def progress_must_be_valid(cls, v: float) -> float:
        """Validate progress is 0-100."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("progress_percentage must be 0.0-100.0")
        return v

    @field_validator("file_size_bytes")
    @classmethod
    def file_size_within_limit(cls, v: int) -> int:
        """Validate file size is within 100MB limit (FR-022)."""
        max_size = 104857600  # 100 MB
        if v > max_size:
            raise ValueError(f"File size exceeds 100MB limit: {v} bytes")
        return v

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
                "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "status": "processing",
                "created_at": "2025-10-16T12:30:00.000Z",
                "started_at": "2025-10-16T12:30:05.000Z",
                "total_documents": 100,
                "processed_documents": 47,
                "failed_documents": 2,
                "progress_percentage": 47.0,
                "file_size_bytes": 52428800,
                "pipeline_type": "graphrag",
                "validation_errors": [
                    "Document 23: Invalid UTF-8 encoding",
                    "Document 45: Exceeds maximum chunk size",
                ],
            }
        }


class DocumentUploadRequest(BaseModel):
    """
    Request to upload documents.

    Used for POST /documents/upload endpoint.
    """

    pipeline_type: PipelineType = Field(
        default=PipelineType.BASIC,
        description="Pipeline to use for indexing",
    )

    chunk_size: Optional[int] = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Chunk size for document splitting (characters)",
    )

    chunk_overlap: Optional[int] = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks (characters)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "pipeline_type": "graphrag",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        }


class DocumentUploadResponse(BaseModel):
    """
    Response after initiating document upload.

    Returns operation_id for tracking progress.
    Implements FR-021: Async document upload.
    """

    operation_id: UUID = Field(
        ...,
        description="Operation identifier for tracking progress",
    )

    status: UploadStatus = Field(
        ...,
        description="Initial operation status (typically 'pending' or 'validating')",
    )

    message: str = Field(
        ...,
        description="Status message",
        examples=["Document upload initiated. Use operation_id to track progress."],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
                "status": "pending",
                "message": "Document upload initiated. Use operation_id to track progress.",
            }
        }
