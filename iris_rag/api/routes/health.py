"""
Health check API routes for RAG API.

Implements FR-032 to FR-034: Comprehensive health monitoring.
Provides GET /health endpoint for Kubernetes probes and monitoring.
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from iris_rag.api.models.health import HealthCheckResponse, HealthStatus, ComponentStatus
from iris_rag.api.services.pipeline_manager import PipelineManager


logger = logging.getLogger(__name__)


def create_health_router(
    pipeline_manager: PipelineManager,
    connection_pool,
    redis_client=None
) -> APIRouter:
    """
    Create health check API router.

    Args:
        pipeline_manager: PipelineManager for pipeline health checks
        connection_pool: IRISConnectionPool for database health check
        redis_client: Optional Redis client for cache health check

    Returns:
        FastAPI router with health endpoints
    """
    router = APIRouter(prefix="/api/v1", tags=["health"])

    @router.get(
        "/health",
        response_model=HealthCheckResponse,
        responses={
            200: {"description": "System is healthy"},
            503: {"description": "System is degraded or unavailable"}
        },
        summary="Health check endpoint",
        description="""
        Comprehensive health check for all system components.

        **Checks:**
        - IRIS database connectivity and response time
        - Redis cache availability (if configured)
        - All RAG pipeline statuses
        - Overall system health

        **Status Values:**
        - healthy: All components operational
        - degraded: Some components have issues but system is functional
        - unavailable: Critical components are down

        **No authentication required** - intended for Kubernetes probes and monitoring.

        **Kubernetes Usage:**
        ```yaml
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        ```
        """
    )
    async def health_check() -> HealthCheckResponse:
        """
        System health check (FR-032).

        Returns:
            HealthCheckResponse with all component statuses

        Note: Always returns 200 OK with status in body (not HTTP status).
        Kubernetes probes should check response.status field.
        """
        timestamp = datetime.utcnow()
        components = {}

        # Check IRIS database (FR-033)
        iris_start = datetime.utcnow()
        try:
            with connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()

            iris_response_time = int(
                (datetime.utcnow() - iris_start).total_seconds() * 1000
            )

            components["iris_database"] = HealthStatus(
                component_name="iris_database",
                status=ComponentStatus.HEALTHY,
                last_checked_at=datetime.utcnow(),
                response_time_ms=iris_response_time,
                version=None,  # TODO: Get IRIS version
                metrics={
                    "response_time_ms": iris_response_time
                }
            )

        except Exception as e:
            logger.error(f"IRIS database health check failed: {e}")

            components["iris_database"] = HealthStatus(
                component_name="iris_database",
                status=ComponentStatus.UNAVAILABLE,
                last_checked_at=datetime.utcnow(),
                error_message=str(e)
            )

        # Check Redis cache (FR-033)
        if redis_client:
            redis_start = datetime.utcnow()
            try:
                redis_client.ping()

                redis_response_time = int(
                    (datetime.utcnow() - redis_start).total_seconds() * 1000
                )

                components["redis_cache"] = HealthStatus(
                    component_name="redis_cache",
                    status=ComponentStatus.HEALTHY,
                    last_checked_at=datetime.utcnow(),
                    response_time_ms=redis_response_time,
                    metrics={
                        "response_time_ms": redis_response_time
                    }
                )

            except Exception as e:
                logger.error(f"Redis cache health check failed: {e}")

                components["redis_cache"] = HealthStatus(
                    component_name="redis_cache",
                    status=ComponentStatus.DEGRADED,
                    last_checked_at=datetime.utcnow(),
                    error_message=str(e)
                )

        # Check all pipelines (FR-034)
        pipeline_health_statuses = pipeline_manager.get_all_health_status()

        for pipeline_name, health_status in pipeline_health_statuses.items():
            components[f"{pipeline_name}_pipeline"] = health_status

        # Determine overall health (FR-032)
        overall_status = ComponentStatus.HEALTHY

        unavailable_count = sum(
            1 for comp in components.values()
            if comp.status == ComponentStatus.UNAVAILABLE
        )

        degraded_count = sum(
            1 for comp in components.values()
            if comp.status == ComponentStatus.DEGRADED
        )

        # If IRIS database is down, system is unavailable
        if components.get("iris_database") and \
           components["iris_database"].status == ComponentStatus.UNAVAILABLE:
            overall_status = ComponentStatus.UNAVAILABLE

        # If any critical component is unavailable, system is unavailable
        elif unavailable_count > 0:
            overall_status = ComponentStatus.UNAVAILABLE

        # If any component is degraded, system is degraded
        elif degraded_count > 0:
            overall_status = ComponentStatus.DEGRADED

        response = HealthCheckResponse(
            status=overall_status,
            timestamp=timestamp,
            components=components,
            overall_health=overall_status
        )

        # Return 503 if system is unavailable (for load balancer health checks)
        if overall_status == ComponentStatus.UNAVAILABLE:
            return JSONResponse(
                status_code=503,
                content=response.model_dump()
            )

        return response

    return router
