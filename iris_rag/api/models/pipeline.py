"""
Pipeline models for RAG API.

Pydantic models for Pipeline Instance entity (Entity 2 from data-model.md).
Represents initialized RAG pipeline with health status and performance stats.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from uuid import UUID
from datetime import datetime


class PipelineStatus(str, Enum):
    """Pipeline health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class PipelineType(str, Enum):
    """Supported RAG pipeline types."""

    BASIC = "basic"
    BASIC_RERANK = "basic_rerank"
    CRAG = "crag"
    GRAPHRAG = "graphrag"
    PYLATE_COLBERT = "pylate_colbert"


class PipelineInstance(BaseModel):
    """
    Pipeline Instance (Entity 2 from data-model.md).

    Represents an initialized RAG pipeline with current health status,
    configuration, and performance statistics.

    Implements FR-005 to FR-008: Pipeline lifecycle management.
    """

    pipeline_id: UUID = Field(
        ...,
        description="Unique pipeline instance identifier",
    )

    pipeline_type: PipelineType = Field(
        ...,
        description="Pipeline type",
        examples=["graphrag"],
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9\-]+$",
        description="Human-readable pipeline name (alphanumeric + dash)",
        examples=["graphrag-production"],
    )

    status: PipelineStatus = Field(
        ...,
        description="Current health status",
        examples=["healthy"],
    )

    initialized_at: datetime = Field(
        ...,
        description="When pipeline was initialized (ISO8601)",
    )

    last_health_check: datetime = Field(
        ...,
        description="Last health check timestamp (ISO8601)",
    )

    total_queries: int = Field(
        default=0,
        ge=0,
        description="Total queries processed since initialization",
    )

    avg_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average query latency in milliseconds (last 100 queries)",
    )

    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors since initialization",
    )

    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Error rate (errors / total_queries)",
    )

    config: Dict[str, Any] = Field(
        ...,
        description="Pipeline configuration (model, embeddings, etc.)",
    )

    capabilities: List[str] = Field(
        ...,
        min_length=1,
        description="Supported features",
        examples=[["vector_search", "graph_traversal", "entity_extraction"]],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "pipeline_id": "a3c4e5f6-7890-4b2d-9e1f-234567890abc",
                "pipeline_type": "graphrag",
                "name": "graphrag-production",
                "status": "healthy",
                "initialized_at": "2025-10-16T10:00:00.000Z",
                "last_health_check": "2025-10-16T12:34:50.000Z",
                "total_queries": 15234,
                "avg_latency_ms": 1234.5,
                "error_count": 23,
                "error_rate": 0.0015,
                "config": {
                    "llm_model": "gpt-4",
                    "embedding_model": "text-embedding-3-small",
                    "top_k": 10,
                },
                "capabilities": [
                    "vector_search",
                    "graph_traversal",
                    "entity_extraction",
                ],
            }
        }


class PipelineListResponse(BaseModel):
    """
    Response for GET /pipelines endpoint.

    Returns list of all configured pipelines with their status.
    Implements FR-008: List available pipelines.
    """

    pipelines: List[PipelineInstance] = Field(
        ...,
        description="List of configured pipelines",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "pipelines": [
                    {
                        "pipeline_type": "basic",
                        "name": "basic-production",
                        "status": "healthy",
                        "capabilities": ["vector_search"],
                        "avg_latency_ms": 234.5,
                    },
                    {
                        "pipeline_type": "graphrag",
                        "name": "graphrag-production",
                        "status": "healthy",
                        "capabilities": [
                            "vector_search",
                            "graph_traversal",
                            "entity_extraction",
                        ],
                        "avg_latency_ms": 1234.5,
                    },
                ]
            }
        }
