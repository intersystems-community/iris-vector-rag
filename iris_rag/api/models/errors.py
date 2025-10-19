"""
Error response models for RAG API.

Elasticsearch-inspired structured error responses.
Implements FR-017 to FR-020: Comprehensive error handling with actionable guidance.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class ErrorType(str, Enum):
    """Error type enumeration."""

    # Authentication errors (FR-010, FR-012)
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    INVALID_API_KEY = "invalid_api_key"
    EXPIRED_API_KEY = "expired_api_key"

    # Validation errors (FR-003)
    VALIDATION_EXCEPTION = "validation_exception"
    BAD_REQUEST = "bad_request"

    # Rate limiting errors (FR-015)
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Service errors (FR-017)
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_SERVER_ERROR = "internal_server_error"

    # Processing errors
    PROCESSING_ERROR = "processing_error"
    TIMEOUT_ERROR = "timeout_error"


class ErrorDetails(BaseModel):
    """
    Error details with actionable information.

    Provides specific field-level errors, guidance, and context.
    """

    message: Optional[str] = Field(
        default=None,
        description="Detailed error message with actionable guidance",
    )

    field: Optional[str] = Field(
        default=None,
        description="Field that caused validation error",
    )

    rejected_value: Optional[Any] = Field(
        default=None,
        description="Value that was rejected",
    )

    max_length: Optional[int] = Field(
        default=None,
        description="Maximum allowed length (for string validation)",
    )

    min_value: Optional[int] = Field(
        default=None,
        description="Minimum allowed value (for numeric validation)",
    )

    max_value: Optional[int] = Field(
        default=None,
        description="Maximum allowed value (for numeric validation)",
    )

    key_id: Optional[str] = Field(
        default=None,
        description="Parsed API key ID (if available)",
    )

    required_permissions: Optional[List[str]] = Field(
        default=None,
        description="Permissions required for this operation",
    )

    current_permissions: Optional[List[str]] = Field(
        default=None,
        description="Permissions the key currently has",
    )

    limit: Optional[int] = Field(
        default=None,
        description="Rate limit maximum",
    )

    window: Optional[str] = Field(
        default=None,
        description="Rate limit window (e.g., 'requests per minute')",
    )

    retry_after_seconds: Optional[int] = Field(
        default=None,
        description="Seconds to wait before retrying",
    )

    pipeline: Optional[str] = Field(
        default=None,
        description="Pipeline name (for service errors)",
    )

    status: Optional[str] = Field(
        default=None,
        description="Pipeline status (for service errors)",
    )

    estimated_recovery_time: Optional[int] = Field(
        default=None,
        description="Estimated seconds until recovery (for 503 errors)",
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for debugging (for 500 errors)",
    )

    expired_at: Optional[str] = Field(
        default=None,
        description="When API key expired (ISO8601)",
    )

    class Config:
        """Pydantic model configuration."""

        extra = "allow"  # Allow additional fields for flexibility


class ErrorInfo(BaseModel):
    """
    Structured error information.

    Elasticsearch-inspired error format with type, reason, and details.
    """

    type: ErrorType = Field(
        ...,
        description="Specific error type",
        examples=["validation_exception"],
    )

    reason: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Invalid parameter value"],
    )

    details: Optional[ErrorDetails] = Field(
        default=None,
        description="Actionable error details",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "type": "validation_exception",
                "reason": "Invalid parameter value",
                "details": {
                    "field": "top_k",
                    "rejected_value": -5,
                    "message": "Must be positive integer between 1 and 100",
                    "min_value": 1,
                    "max_value": 100,
                },
            }
        }


class ErrorResponse(BaseModel):
    """
    Top-level error response.

    Implements FR-018: Clear error messages with actionable guidance.
    All API errors follow this structure.
    """

    error: ErrorInfo = Field(
        ...,
        description="Error information",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "examples": [
                # Authentication error (FR-010)
                {
                    "error": {
                        "type": "authentication_error",
                        "reason": "Missing Authorization header",
                        "details": {
                            "message": "API requests must include 'Authorization: ApiKey <base64(id:secret)>' header"
                        },
                    }
                },
                # Validation error (FR-003)
                {
                    "error": {
                        "type": "validation_exception",
                        "reason": "Query text exceeds maximum length",
                        "details": {
                            "field": "query",
                            "rejected_value": "[10001 character string]",
                            "message": "Query must be between 1 and 10000 characters",
                            "max_length": 10000,
                        },
                    }
                },
                # Rate limit error (FR-015)
                {
                    "error": {
                        "type": "rate_limit_exceeded",
                        "reason": "Too many requests",
                        "details": {
                            "limit": 100,
                            "window": "requests per minute",
                            "retry_after_seconds": 60,
                        },
                    }
                },
                # Service unavailable (FR-007)
                {
                    "error": {
                        "type": "service_unavailable",
                        "reason": "Pipeline is currently unavailable",
                        "details": {
                            "pipeline": "graphrag",
                            "status": "degraded",
                            "estimated_recovery_time": 120,
                            "message": "Pipeline is initializing. Please try again in 2 minutes.",
                        },
                    }
                },
                # Authorization error (FR-011)
                {
                    "error": {
                        "type": "authorization_error",
                        "reason": "Insufficient permissions for this operation",
                        "details": {
                            "required_permissions": ["write"],
                            "current_permissions": ["read"],
                        },
                    }
                },
                # Expired API key
                {
                    "error": {
                        "type": "expired_api_key",
                        "reason": "API key has expired",
                        "details": {
                            "key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                            "expired_at": "2025-01-15T08:00:00.000Z",
                        },
                    }
                },
                # Internal server error (FR-019, FR-020)
                {
                    "error": {
                        "type": "internal_server_error",
                        "reason": "Query processing failed",
                        "details": {
                            "message": "An unexpected error occurred. Please try again or contact support.",
                            "request_id": "550e8400-e29b-41d4-a716-446655440000",
                        },
                    }
                },
            ]
        }


# Convenience factory functions for common errors


def authentication_error(message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Create authentication error response."""
    return ErrorResponse(
        error=ErrorInfo(
            type=ErrorType.AUTHENTICATION_ERROR,
            reason=message,
            details=ErrorDetails(**(details or {})),
        )
    )


def validation_error(
    field: str, rejected_value: Any, message: str, **kwargs
) -> ErrorResponse:
    """Create validation error response."""
    return ErrorResponse(
        error=ErrorInfo(
            type=ErrorType.VALIDATION_EXCEPTION,
            reason=f"Invalid value for field: {field}",
            details=ErrorDetails(
                field=field,
                rejected_value=rejected_value,
                message=message,
                **kwargs,
            ),
        )
    )


def rate_limit_error(
    limit: int, window: str, retry_after_seconds: int
) -> ErrorResponse:
    """Create rate limit exceeded error response."""
    return ErrorResponse(
        error=ErrorInfo(
            type=ErrorType.RATE_LIMIT_EXCEEDED,
            reason="Too many requests",
            details=ErrorDetails(
                limit=limit,
                window=window,
                retry_after_seconds=retry_after_seconds,
                message=f"Rate limit of {limit} {window} exceeded. Retry after {retry_after_seconds} seconds.",
            ),
        )
    )


def service_unavailable_error(
    pipeline: str, status: str, estimated_recovery_time: int
) -> ErrorResponse:
    """Create service unavailable error response."""
    return ErrorResponse(
        error=ErrorInfo(
            type=ErrorType.SERVICE_UNAVAILABLE,
            reason="Pipeline is currently unavailable",
            details=ErrorDetails(
                pipeline=pipeline,
                status=status,
                estimated_recovery_time=estimated_recovery_time,
                message=f"Pipeline {pipeline} is {status}. Estimated recovery in {estimated_recovery_time} seconds.",
            ),
        )
    )


def internal_server_error(request_id: str, message: Optional[str] = None) -> ErrorResponse:
    """Create internal server error response."""
    return ErrorResponse(
        error=ErrorInfo(
            type=ErrorType.INTERNAL_SERVER_ERROR,
            reason="An unexpected error occurred",
            details=ErrorDetails(
                request_id=request_id,
                message=message
                or "An unexpected error occurred. Please try again or contact support.",
            ),
        )
    )
