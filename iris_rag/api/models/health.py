"""
Health check models for RAG API.

Pydantic models for Health Status entity (Entity 8 from data-model.md).
Implements health monitoring for all system components.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ComponentStatus(str, Enum):
    """Component health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class HealthStatus(BaseModel):
    """
    Health Status (Entity 8 from data-model.md).

    Represents current operational state of system components
    (pipelines, database, cache) with status level, last check timestamp,
    and diagnostic details.

    Implements FR-032 to FR-034: Health checks and metrics.
    """

    component_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Component identifier",
        examples=["iris_database", "redis_cache", "graphrag_pipeline"],
    )

    status: ComponentStatus = Field(
        ...,
        description="Current health status",
        examples=["healthy"],
    )

    last_checked_at: datetime = Field(
        ...,
        description="Last health check timestamp (ISO8601)",
    )

    response_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Health check response time in milliseconds",
    )

    version: Optional[str] = Field(
        default=None,
        description="Component version (semver format)",
        examples=["2025.3.0"],
    )

    dependencies: Optional[List[str]] = Field(
        default=None,
        description="Components this depends on",
        examples=[["iris_database", "redis_cache"]],
    )

    error_message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Error details if unhealthy",
    )

    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Component-specific metrics",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "component_name": "iris_database",
                "status": "healthy",
                "last_checked_at": "2025-10-16T12:34:56.789Z",
                "response_time_ms": 12,
                "version": "2025.3.0",
                "dependencies": [],
                "metrics": {
                    "connection_pool_size": 20,
                    "active_connections": 8,
                    "query_count": 15234,
                    "avg_query_time_ms": 345,
                },
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Aggregated health check response.

    Implements FR-032: Health check endpoint reporting overall system status.
    """

    status: ComponentStatus = Field(
        ...,
        description="Overall system health status",
    )

    timestamp: datetime = Field(
        ...,
        description="When health check was performed (ISO8601)",
    )

    components: Dict[str, HealthStatus] = Field(
        ...,
        description="Status of all system components",
    )

    overall_health: ComponentStatus = Field(
        ...,
        description="Aggregate health (same as status)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-16T12:34:56.789Z",
                "components": {
                    "iris_database": {
                        "status": "healthy",
                        "response_time_ms": 12,
                        "version": "2025.3.0",
                        "metrics": {
                            "connection_pool_size": 20,
                            "active_connections": 8,
                        },
                    },
                    "redis_cache": {
                        "status": "healthy",
                        "response_time_ms": 5,
                    },
                    "graphrag_pipeline": {
                        "status": "healthy",
                        "response_time_ms": 8,
                        "metrics": {
                            "total_queries": 15234,
                            "avg_latency_ms": 1234.5,
                        },
                    },
                    "basic_pipeline": {
                        "status": "healthy",
                        "response_time_ms": 6,
                    },
                },
                "overall_health": "healthy",
            }
        }


class ComponentHealthCheck(BaseModel):
    """
    Request to check health of specific component.

    Used internally for health check operations.
    """

    component_name: str = Field(
        ...,
        description="Component to check",
    )

    timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Health check timeout (1-30 seconds)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "component_name": "iris_database",
                "timeout_seconds": 5,
            }
        }
