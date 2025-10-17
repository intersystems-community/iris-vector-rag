"""
Pipeline Management Service for RAG API.

Implements FR-006 to FR-009: Pipeline discovery, health checks, and lifecycle management.
Manages all available RAG pipelines with status monitoring.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from iris_rag import create_pipeline
from iris_rag.core.base import RAGPipeline
from iris_rag.api.models.pipeline import (
    PipelineInstance,
    PipelineStatus,
    PipelineListResponse
)
from iris_rag.api.models.health import HealthStatus, ComponentStatus


logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages RAG pipeline lifecycle and health monitoring.

    Implements FR-006: Enumerate available pipelines
    Implements FR-007: Report pipeline availability and status
    Implements FR-008: Health checks for all pipelines
    """

    def __init__(self, config: dict):
        """
        Initialize pipeline manager.

        Args:
            config: Configuration dict with pipeline settings
        """
        self.config = config
        self.pipelines: Dict[str, RAGPipeline] = {}
        self.pipeline_metadata: Dict[str, PipelineInstance] = {}

        # Initialize configured pipelines
        self._initialize_pipelines()

    def _initialize_pipelines(self):
        """
        Initialize all configured pipelines.

        Loads pipelines from config and performs health checks.
        Implements FR-006: Pipeline discovery
        """
        enabled_pipelines = self.config.get('pipelines', {}).get('enabled', [])

        logger.info(f"Initializing {len(enabled_pipelines)} pipelines...")

        for pipeline_type in enabled_pipelines:
            try:
                # Create pipeline instance
                pipeline = create_pipeline(
                    pipeline_type=pipeline_type,
                    validate_requirements=True,
                    auto_setup=False  # Don't auto-fix in production
                )

                # Store pipeline
                self.pipelines[pipeline_type] = pipeline

                # Create metadata
                self.pipeline_metadata[pipeline_type] = PipelineInstance(
                    name=pipeline_type,
                    type=pipeline_type,
                    status=PipelineStatus.HEALTHY,
                    version="1.0.0",  # TODO: Get from pipeline
                    capabilities=self._get_pipeline_capabilities(pipeline_type),
                    description=self._get_pipeline_description(pipeline_type),
                    created_at=datetime.utcnow(),
                    last_health_check=datetime.utcnow(),
                    total_queries=0,
                    avg_latency_ms=0.0,
                    error_rate=0.0
                )

                logger.info(f"Initialized pipeline: {pipeline_type}")

            except Exception as e:
                logger.error(f"Failed to initialize pipeline {pipeline_type}: {e}")

                # Store degraded metadata
                self.pipeline_metadata[pipeline_type] = PipelineInstance(
                    name=pipeline_type,
                    type=pipeline_type,
                    status=PipelineStatus.UNAVAILABLE,
                    version="1.0.0",
                    capabilities=[],
                    description=self._get_pipeline_description(pipeline_type),
                    created_at=datetime.utcnow(),
                    last_health_check=datetime.utcnow(),
                    total_queries=0,
                    avg_latency_ms=0.0,
                    error_rate=0.0,
                    error_message=str(e)
                )

    def _get_pipeline_capabilities(self, pipeline_type: str) -> List[str]:
        """
        Get capabilities for pipeline type.

        Args:
            pipeline_type: Pipeline type identifier

        Returns:
            List of capability strings
        """
        capabilities_map = {
            "basic": ["vector_search", "semantic_retrieval"],
            "basic_rerank": ["vector_search", "semantic_retrieval", "reranking"],
            "crag": ["vector_search", "self_evaluation", "corrective_retrieval"],
            "graphrag": [
                "vector_search",
                "text_search",
                "knowledge_graph",
                "hybrid_retrieval",
                "rrf_fusion"
            ],
            "pylate_colbert": ["late_interaction", "token_level_similarity"]
        }

        return capabilities_map.get(pipeline_type, [])

    def _get_pipeline_description(self, pipeline_type: str) -> str:
        """
        Get description for pipeline type.

        Args:
            pipeline_type: Pipeline type identifier

        Returns:
            Human-readable description
        """
        descriptions = {
            "basic": "Standard vector similarity search with semantic embeddings",
            "basic_rerank": "Vector search with cross-encoder reranking for improved relevance",
            "crag": "Corrective RAG with self-evaluation and adaptive retrieval",
            "graphrag": "Hybrid search combining vector, text, and knowledge graph retrieval with RRF fusion",
            "pylate_colbert": "Late interaction retrieval using ColBERT token-level similarity"
        }

        return descriptions.get(pipeline_type, "RAG pipeline")

    def list_pipelines(self) -> PipelineListResponse:
        """
        List all available pipelines.

        Returns:
            PipelineListResponse with all pipeline metadata

        Implements FR-006: GET /api/v1/pipelines
        """
        pipelines_list = list(self.pipeline_metadata.values())

        return PipelineListResponse(
            pipelines=pipelines_list,
            total_count=len(pipelines_list)
        )

    def get_pipeline_info(self, pipeline_name: str) -> Optional[PipelineInstance]:
        """
        Get detailed information for specific pipeline.

        Args:
            pipeline_name: Pipeline identifier

        Returns:
            PipelineInstance or None if not found

        Implements FR-007: GET /api/v1/pipelines/{name}
        """
        return self.pipeline_metadata.get(pipeline_name)

    def get_pipeline(self, pipeline_name: str) -> Optional[RAGPipeline]:
        """
        Get pipeline instance for querying.

        Args:
            pipeline_name: Pipeline identifier

        Returns:
            RAGPipeline instance or None if not available
        """
        return self.pipelines.get(pipeline_name)

    def check_pipeline_health(self, pipeline_name: str) -> HealthStatus:
        """
        Check health of specific pipeline.

        Args:
            pipeline_name: Pipeline identifier

        Returns:
            HealthStatus with current state

        Implements FR-008: Pipeline health monitoring
        """
        start_time = datetime.utcnow()

        metadata = self.pipeline_metadata.get(pipeline_name)

        if not metadata:
            return HealthStatus(
                component_name=pipeline_name,
                status=ComponentStatus.UNAVAILABLE,
                last_checked_at=start_time,
                error_message="Pipeline not found"
            )

        # Check if pipeline is loaded
        pipeline = self.pipelines.get(pipeline_name)

        if not pipeline:
            return HealthStatus(
                component_name=pipeline_name,
                status=ComponentStatus.UNAVAILABLE,
                last_checked_at=start_time,
                error_message=metadata.error_message or "Pipeline not initialized"
            )

        # Perform basic health check (try to access pipeline config)
        try:
            # Simple validation - check if pipeline is callable
            if not hasattr(pipeline, 'query'):
                raise ValueError("Pipeline missing query method")

            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Update metadata
            metadata.last_health_check = end_time
            metadata.status = PipelineStatus.HEALTHY

            return HealthStatus(
                component_name=pipeline_name,
                status=ComponentStatus.HEALTHY,
                last_checked_at=end_time,
                response_time_ms=response_time_ms,
                version=metadata.version,
                metrics={
                    "total_queries": metadata.total_queries,
                    "avg_latency_ms": metadata.avg_latency_ms,
                    "error_rate": metadata.error_rate
                }
            )

        except Exception as e:
            logger.error(f"Pipeline health check failed for {pipeline_name}: {e}")

            # Update metadata
            metadata.status = PipelineStatus.DEGRADED
            metadata.error_message = str(e)

            return HealthStatus(
                component_name=pipeline_name,
                status=ComponentStatus.DEGRADED,
                last_checked_at=datetime.utcnow(),
                error_message=str(e)
            )

    def update_pipeline_metrics(
        self,
        pipeline_name: str,
        execution_time_ms: int,
        success: bool
    ):
        """
        Update pipeline performance metrics.

        Args:
            pipeline_name: Pipeline identifier
            execution_time_ms: Query execution time
            success: Whether query succeeded
        """
        metadata = self.pipeline_metadata.get(pipeline_name)

        if not metadata:
            return

        # Update query count
        metadata.total_queries += 1

        # Update average latency (moving average)
        if metadata.total_queries == 1:
            metadata.avg_latency_ms = float(execution_time_ms)
        else:
            alpha = 0.1  # Smoothing factor
            metadata.avg_latency_ms = (
                alpha * execution_time_ms +
                (1 - alpha) * metadata.avg_latency_ms
            )

        # Update error rate
        if not success:
            if metadata.total_queries == 1:
                metadata.error_rate = 1.0
            else:
                alpha = 0.1
                metadata.error_rate = alpha * 1.0 + (1 - alpha) * metadata.error_rate
        else:
            if metadata.total_queries == 1:
                metadata.error_rate = 0.0
            else:
                alpha = 0.1
                metadata.error_rate = alpha * 0.0 + (1 - alpha) * metadata.error_rate

        # Update status based on error rate
        if metadata.error_rate > 0.5:
            metadata.status = PipelineStatus.DEGRADED
        elif metadata.error_rate > 0.1:
            metadata.status = PipelineStatus.DEGRADED
        else:
            metadata.status = PipelineStatus.HEALTHY

    def get_all_health_status(self) -> Dict[str, HealthStatus]:
        """
        Get health status for all pipelines.

        Returns:
            Dictionary mapping pipeline names to health status

        Implements FR-008: Comprehensive health monitoring
        """
        health_statuses = {}

        for pipeline_name in self.pipeline_metadata.keys():
            health_statuses[pipeline_name] = self.check_pipeline_health(pipeline_name)

        return health_statuses
