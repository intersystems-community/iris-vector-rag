"""
RAG Templates Bridge Adapter - Unified Interface for kg-ticket-resolver Integration

This module provides a unified interface adapter that abstracts the RAG pipeline interface
for external consumption, enabling seamless switching between RAG techniques while
maintaining consistent response format for kg-ticket-resolver.

Architecture:
- Unified interface for all RAG pipelines (BasicRAG, CRAG, GraphRAG, BasicRAGReranking)
- Circuit breaker pattern for error handling
- Performance monitoring and metrics collection
- Incremental indexing support following LightRAG patterns
- Clean service boundaries with kg-ticket-resolver
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.base import RAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline

logger = logging.getLogger(__name__)


class RAGTechnique(Enum):
    """Supported RAG techniques."""

    BASIC = "basic"
    CRAG = "crag"
    GRAPH = "graphrag"
    RERANKING = "basic_reranking"


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RAGResponse:
    """Standardized response format for kg-ticket-resolver."""

    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    technique_used: str
    processing_time_ms: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    technique_usage: Dict[str, int] = None

    def __post_init__(self):
        if self.technique_usage is None:
            self.technique_usage = {}


class RAGTemplatesBridge:
    """
    Unified interface adapter for RAG pipeline access from kg-ticket-resolver.

    Provides:
    - Seamless switching between RAG techniques
    - Consistent response format
    - Circuit breaker pattern for resilience
    - Performance monitoring and metrics
    - Incremental indexing support
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG Templates Bridge.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_manager = ConfigurationManager(config_path)
        self.connection_manager = ConnectionManager()

        # Load bridge configuration
        self.bridge_config = self.config_manager.get("rag_integration", {})
        self.default_technique = RAGTechnique(
            self.bridge_config.get("default_technique", "basic")
        )
        self.fallback_technique = RAGTechnique(
            self.bridge_config.get("fallback_technique", "basic")
        )

        # Initialize pipeline registry
        self._pipelines: Dict[RAGTechnique, RAGPipeline] = {}
        self._circuit_breakers: Dict[RAGTechnique, Dict] = {}

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self._response_times: List[float] = []

        # Circuit breaker configuration
        self.cb_config = CircuitBreakerConfig(
            **self.bridge_config.get("circuit_breaker", {})
        )

        # Initialize available techniques
        self._initialize_pipelines()
        self._initialize_circuit_breakers()

        logger.info(
            f"RAG Templates Bridge initialized with techniques: {list(self._pipelines.keys())}"
        )

    def _initialize_pipelines(self) -> None:
        """Initialize available RAG pipelines."""
        pipeline_classes = {
            RAGTechnique.BASIC: BasicRAGPipeline,
            RAGTechnique.CRAG: CRAGPipeline,
            RAGTechnique.GRAPH: GraphRAGPipeline,
            RAGTechnique.RERANKING: BasicRAGRerankingPipeline,
        }

        for technique, pipeline_class in pipeline_classes.items():
            try:
                pipeline = pipeline_class(
                    connection_manager=self.connection_manager,
                    config_manager=self.config_manager,
                )
                self._pipelines[technique] = pipeline
                logger.info(f"Initialized {technique.value} pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize {technique.value} pipeline: {e}")

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for each technique."""
        for technique in self._pipelines.keys():
            self._circuit_breakers[technique] = {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "last_failure_time": 0,
                "success_count": 0,
            }

    def _check_circuit_breaker(self, technique: RAGTechnique) -> bool:
        """Check if circuit breaker allows execution."""
        cb = self._circuit_breakers[technique]
        current_time = time.time()

        if cb["state"] == CircuitBreakerState.OPEN:
            if current_time - cb["last_failure_time"] > self.cb_config.recovery_timeout:
                cb["state"] = CircuitBreakerState.HALF_OPEN
                cb["success_count"] = 0
                logger.info(f"Circuit breaker for {technique.value} moved to HALF_OPEN")
            else:
                return False

        elif cb["state"] == CircuitBreakerState.HALF_OPEN:
            if cb["success_count"] >= self.cb_config.half_open_max_calls:
                cb["state"] = CircuitBreakerState.CLOSED
                cb["failure_count"] = 0
                logger.info(f"Circuit breaker for {technique.value} moved to CLOSED")

        return True

    def _record_success(self, technique: RAGTechnique) -> None:
        """Record successful execution."""
        cb = self._circuit_breakers[technique]
        if cb["state"] == CircuitBreakerState.HALF_OPEN:
            cb["success_count"] += 1
        elif cb["state"] == CircuitBreakerState.CLOSED:
            cb["failure_count"] = 0

    def _record_failure(self, technique: RAGTechnique) -> None:
        """Record failed execution."""
        cb = self._circuit_breakers[technique]
        cb["failure_count"] += 1
        cb["last_failure_time"] = time.time()

        if cb["failure_count"] >= self.cb_config.failure_threshold:
            cb["state"] = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {technique.value} moved to OPEN")

    async def query(
        self,
        query_text: str,
        technique: Optional[Union[str, RAGTechnique]] = None,
        **kwargs,
    ) -> RAGResponse:
        """
        Execute RAG query with specified or default technique.

        Args:
            query_text: The query to process
            technique: RAG technique to use (defaults to configured default)
            **kwargs: Additional arguments for the pipeline

        Returns:
            RAGResponse: Standardized response object
        """
        start_time = time.time()

        # Normalize technique parameter
        if isinstance(technique, str):
            try:
                technique = RAGTechnique(technique)
            except ValueError:
                processing_time = (time.time() - start_time) * 1000
                error_response = RAGResponse(
                    answer="",
                    sources=[],
                    confidence_score=0.0,
                    technique_used="invalid",
                    processing_time_ms=processing_time,
                    metadata={"error_type": "InvalidTechnique"},
                    error=f"Invalid RAG technique: {technique}",
                )
                logger.error(f"Invalid technique specified: {technique}")
                return error_response
        elif technique is None:
            technique = self.default_technique

        # Track metrics
        self.metrics.total_queries += 1
        self.metrics.technique_usage[technique.value] = (
            self.metrics.technique_usage.get(technique.value, 0) + 1
        )

        try:
            # Check circuit breaker
            if not self._check_circuit_breaker(technique):
                logger.warning(
                    f"Circuit breaker open for {technique.value}, falling back"
                )
                technique = self.fallback_technique

            # Get pipeline
            pipeline = self._pipelines.get(technique)
            if not pipeline:
                raise ValueError(f"Pipeline not available: {technique.value}")

            # Execute query
            result = await self._execute_pipeline_query(pipeline, query_text, **kwargs)

            # Record success
            self._record_success(technique)
            self.metrics.successful_queries += 1

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self._response_times.append(processing_time)

            # Create standardized response
            response = RAGResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                confidence_score=result.get("confidence_score", 0.0),
                technique_used=technique.value,
                processing_time_ms=processing_time,
                metadata={
                    "query_id": kwargs.get("query_id"),
                    "user_context": kwargs.get("user_context"),
                    "timestamp": time.time(),
                },
            )

            logger.info(
                f"Query processed successfully with {technique.value} in {processing_time:.2f}ms"
            )
            return response

        except Exception as e:
            # Record failure
            self._record_failure(technique)
            self.metrics.failed_queries += 1

            processing_time = (time.time() - start_time) * 1000

            error_response = RAGResponse(
                answer="",
                sources=[],
                confidence_score=0.0,
                technique_used=technique.value,
                processing_time_ms=processing_time,
                metadata={"error_type": type(e).__name__},
                error=str(e),
            )

            logger.error(f"Query failed with {technique.value}: {e}")
            return error_response

    async def _execute_pipeline_query(
        self, pipeline: RAGPipeline, query_text: str, **kwargs
    ) -> Dict[str, Any]:
        """Execute query on specific pipeline with timeout."""
        timeout = self.bridge_config.get("query_timeout", 30)

        try:
            # Use asyncio.wait_for for timeout handling
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, pipeline.query, query_text
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"Query timeout after {timeout} seconds")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        # Update calculated metrics
        if self._response_times:
            self.metrics.avg_response_time_ms = sum(self._response_times) / len(
                self._response_times
            )
            self.metrics.p95_response_time_ms = sorted(self._response_times)[
                int(len(self._response_times) * 0.95)
            ]

        return asdict(self.metrics)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        health = {
            "bridge_status": "healthy",
            "pipelines": {},
            "circuit_breakers": {},
            "metrics": self.get_metrics(),
        }

        for technique, pipeline in self._pipelines.items():
            try:
                # Simple health check - attempt to get pipeline config
                _ = pipeline.config_manager
                health["pipelines"][technique.value] = "healthy"
            except Exception as e:
                health["pipelines"][technique.value] = f"unhealthy: {e}"

        for technique, cb in self._circuit_breakers.items():
            health["circuit_breakers"][technique.value] = cb["state"].value

        return health

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        technique: Optional[Union[str, RAGTechnique]] = None,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """
        Index documents using specified technique with incremental support.

        Args:
            documents: List of documents to index
            technique: RAG technique to use for indexing
            incremental: Whether to use incremental indexing

        Returns:
            Dict with indexing results
        """
        if isinstance(technique, str):
            technique = RAGTechnique(technique)
        elif technique is None:
            technique = self.default_technique

        pipeline = self._pipelines.get(technique)
        if not pipeline:
            raise ValueError(f"Pipeline not available: {technique.value}")

        start_time = time.time()

        try:
            # Convert documents to required format
            from iris_rag.core.models import Document

            doc_objects = [
                Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents
            ]

            # Use pipeline's load_documents method
            await asyncio.get_event_loop().run_in_executor(
                None, pipeline.load_documents, "", documents=doc_objects
            )

            processing_time = (time.time() - start_time) * 1000

            return {
                "status": "success",
                "documents_indexed": len(documents),
                "technique_used": technique.value,
                "processing_time_ms": processing_time,
                "incremental": incremental,
            }

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "technique_used": technique.value,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    @asynccontextmanager
    async def pipeline_context(self, technique: Union[str, RAGTechnique]):
        """Context manager for safe pipeline access."""
        if isinstance(technique, str):
            technique = RAGTechnique(technique)

        pipeline = self._pipelines.get(technique)
        if not pipeline:
            raise ValueError(f"Pipeline not available: {technique.value}")

        try:
            yield pipeline
        except Exception:
            self._record_failure(technique)
            raise
        else:
            self._record_success(technique)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all components.

        Returns:
            Dict containing health status of bridge, pipelines, and dependencies
        """
        start_time = time.time()
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "check_duration_ms": 0.0,
            "components": {
                "bridge": {"status": "healthy"},
                "pipelines": {},
                "circuit_breakers": {},
                "dependencies": {},
            },
            "metrics": await self.get_performance_metrics(),
        }

        # Check pipeline health
        for technique, pipeline in self._pipelines.items():
            try:
                # Perform lightweight health check
                if hasattr(pipeline, "health_check"):
                    await asyncio.get_event_loop().run_in_executor(
                        None, pipeline.health_check
                    )
                else:
                    # Fallback: check if pipeline can access its configuration
                    _ = pipeline.config_manager
                health_status["components"]["pipelines"][technique.value] = {
                    "status": "healthy"
                }
            except Exception as e:
                health_status["components"]["pipelines"][technique.value] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["status"] = "degraded"

        # Check circuit breaker states
        for technique, cb in self._circuit_breakers.items():
            cb_status = cb["state"].value
            health_status["components"]["circuit_breakers"][technique.value] = {
                "state": cb_status,
                "failure_count": cb["failure_count"],
            }
            if cb_status == "open":
                health_status["status"] = "degraded"

        # Check dependencies
        try:
            # Test database connectivity
            self.connection_manager.get_connection()
            health_status["components"]["dependencies"]["database"] = {
                "status": "healthy"
            }
        except Exception as e:
            health_status["components"]["dependencies"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "unhealthy"

        health_status["check_duration_ms"] = (time.time() - start_time) * 1000
        return health_status

    async def get_available_techniques(self) -> List[str]:
        """
        Get list of available RAG techniques.

        Returns:
            List of technique names that are currently available
        """
        available = []
        for technique in self._pipelines.keys():
            # Check if technique is operational (circuit breaker not open)
            cb = self._circuit_breakers.get(technique, {})
            if cb.get("state", CircuitBreakerState.CLOSED) != CircuitBreakerState.OPEN:
                available.append(technique.value)

        # Always include fallback technique if available
        if (
            self.fallback_technique.value not in available
            and self.fallback_technique in self._pipelines
        ):
            available.append(self.fallback_technique.value)

        return available

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dict containing current performance metrics and SLO compliance
        """
        current_metrics = self.get_metrics()

        # Add SLO compliance checks
        config_targets = self.bridge_config.get("performance", {})
        p95_target = config_targets.get("target_latency_p95_ms", 500)

        slo_compliance = {
            "p95_latency_compliant": current_metrics["p95_response_time_ms"]
            <= p95_target,
            "p95_target_ms": p95_target,
            "success_rate": (
                current_metrics["successful_queries"]
                / max(current_metrics["total_queries"], 1)
            )
            * 100,
        }

        return {
            **current_metrics,
            "slo_compliance": slo_compliance,
            "circuit_breaker_states": {
                technique.value: cb["state"].value
                for technique, cb in self._circuit_breakers.items()
            },
            "available_techniques": await self.get_available_techniques(),
        }
