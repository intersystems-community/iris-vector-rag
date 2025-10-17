"""
Request models for RAG API.

Pydantic models for API Request entity (Entity 1 from data-model.md).
Validates incoming query requests and parameters.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from uuid import UUID
from datetime import datetime


class PipelineType(str, Enum):
    """Supported RAG pipeline types."""

    BASIC = "basic"
    BASIC_RERANK = "basic_rerank"
    CRAG = "crag"
    GRAPHRAG = "graphrag"
    PYLATE_COLBERT = "pylate_colbert"


class QueryRequest(BaseModel):
    """
    Query request schema.

    Validates FR-001: Accept query text, pipeline selection, and optional parameters.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's query text (1-10000 characters)",
        examples=["What is diabetes?"],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve (1-100)",
        examples=[5],
    )

    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for document retrieval",
        examples=[{"domain": "medical", "year": 2023}],
    )

    include_sources: bool = Field(
        default=True,
        description="Whether to include source references in response",
    )

    include_metadata: bool = Field(
        default=True,
        description="Whether to include execution metadata",
    )

    pipeline: Optional[PipelineType] = Field(
        default=None,
        description="Pipeline to use (for generic /_search endpoint)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "query": "What is diabetes?",
                "top_k": 5,
                "filters": {"domain": "medical"},
                "include_sources": True,
                "include_metadata": True,
            }
        }


class HTTPMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"


class APIRequestLog(BaseModel):
    """
    API Request log entry (Entity 1 from data-model.md).

    Represents an incoming HTTP request with metadata for logging and monitoring.
    Implements FR-035: Log all API requests with standard detail.
    """

    request_id: UUID = Field(
        ...,
        description="Unique identifier for request tracing",
    )

    timestamp: datetime = Field(
        ...,
        description="When request was received (ISO8601)",
    )

    method: HTTPMethod = Field(
        ...,
        description="HTTP method",
    )

    endpoint: str = Field(
        ...,
        max_length=255,
        description="API endpoint path",
        examples=["/api/v1/graphrag/_search"],
    )

    query_text: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="User's query text (for query endpoints)",
    )

    pipeline_type: Optional[PipelineType] = Field(
        default=None,
        description="Selected RAG pipeline",
    )

    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query parameters (top_k, filters, etc.)",
    )

    api_key_id: UUID = Field(
        ...,
        description="Which API key was used",
    )

    client_ip: str = Field(
        ...,
        description="Client IP address",
        examples=["192.168.1.100"],
    )

    user_agent: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Client user agent string",
    )

    status_code: int = Field(
        ...,
        ge=200,
        le=599,
        description="HTTP response status code",
        examples=[200, 401, 429, 503],
    )

    response_time_ms: int = Field(
        ...,
        ge=0,
        description="Request execution time in milliseconds",
    )

    error_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Error type if request failed",
    )

    error_message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message if request failed",
    )

    @field_validator("endpoint")
    @classmethod
    def endpoint_must_start_with_slash(cls, v: str) -> str:
        """Validate endpoint starts with /."""
        if not v.startswith("/"):
            raise ValueError("Endpoint must start with /")
        return v

    @field_validator("status_code")
    @classmethod
    def validate_status_code(cls, v: int) -> int:
        """Validate HTTP status code is valid."""
        if v not in range(200, 600):
            raise ValueError("Status code must be 200-599")
        return v

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-16T12:34:56.789Z",
                "method": "POST",
                "endpoint": "/api/v1/graphrag/_search",
                "query_text": "What is diabetes?",
                "pipeline_type": "graphrag",
                "parameters": {"top_k": 5, "filters": {"domain": "medical"}},
                "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "client_ip": "192.168.1.100",
                "user_agent": "Python-Requests/2.31.0",
                "status_code": 200,
                "response_time_ms": 1456,
            }
        }
