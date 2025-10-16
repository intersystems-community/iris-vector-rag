"""
Pipeline API routes for RAG API.

Implements FR-006 to FR-009: Pipeline discovery and health monitoring.
Provides GET /pipelines endpoints for listing and inspecting pipelines.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Path

from iris_rag.api.models.pipeline import PipelineInstance, PipelineListResponse
from iris_rag.api.services.pipeline_manager import PipelineManager
from iris_rag.api.models.errors import ErrorResponse, ErrorType, ErrorInfo, ErrorDetails


logger = logging.getLogger(__name__)


def create_pipeline_router(pipeline_manager: PipelineManager) -> APIRouter:
    """
    Create pipeline API router.

    Args:
        pipeline_manager: PipelineManager for pipeline operations

    Returns:
        FastAPI router with pipeline endpoints
    """
    router = APIRouter(prefix="/api/v1/pipelines", tags=["pipelines"])

    @router.get(
        "",
        response_model=PipelineListResponse,
        summary="List all pipelines",
        description="""
        Enumerate all available RAG pipelines with their capabilities and status.

        **Returns:**
        - List of pipelines with metadata
        - Current health status
        - Performance metrics
        - Capabilities for each pipeline

        **No authentication required** - public discovery endpoint.
        """
    )
    async def list_pipelines() -> PipelineListResponse:
        """
        List all available pipelines (FR-006).

        Returns:
            PipelineListResponse with all pipelines
        """
        logger.debug("Listing all pipelines")
        return pipeline_manager.list_pipelines()

    @router.get(
        "/{pipeline_name}",
        response_model=PipelineInstance,
        responses={
            200: {"description": "Pipeline details"},
            404: {"description": "Pipeline not found"}
        },
        summary="Get pipeline details",
        description="""
        Get detailed information about a specific pipeline.

        **Returns:**
        - Pipeline metadata
        - Health status
        - Performance metrics (avg latency, error rate)
        - Capabilities
        - Version information

        **No authentication required** - public discovery endpoint.
        """
    )
    async def get_pipeline_info(
        pipeline_name: str = Path(..., description="Pipeline name")
    ) -> PipelineInstance:
        """
        Get detailed pipeline information (FR-007).

        Args:
            pipeline_name: Pipeline identifier

        Returns:
            PipelineInstance with full metadata

        Raises:
            HTTPException: If pipeline not found
        """
        logger.debug(f"Getting pipeline info: {pipeline_name}")

        pipeline_info = pipeline_manager.get_pipeline_info(pipeline_name)

        if not pipeline_info:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.BAD_REQUEST,
                        reason=f"Pipeline not found: {pipeline_name}",
                        details=ErrorDetails(
                            message=f"Pipeline '{pipeline_name}' does not exist or is not configured"
                        )
                    )
                ).model_dump()
            )

        return pipeline_info

    return router
