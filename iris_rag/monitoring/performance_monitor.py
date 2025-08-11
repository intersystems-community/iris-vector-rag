"""
Performance Monitor for RAG Templates System

Provides performance monitoring and metrics collection for RAG operations.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import os

from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """A single performance metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryPerformanceData:
    """Performance data for a query operation."""

    query_text: str
    pipeline_type: str
    execution_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    documents_retrieved: int
    tokens_generated: int
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


class PerformanceMonitor:
    """
    Monitors and collects performance metrics for RAG operations.

    Tracks query performance, system resource usage, and provides
    real-time monitoring capabilities.
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the performance monitor.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.connection_manager = ConnectionManager(self.config_manager)

        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics
        self.query_performance_data = deque(maxlen=1000)  # Keep last 1k queries

        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds

        # Performance thresholds
        self.thresholds = {
            "query_time_warning_ms": 1000,
            "query_time_critical_ms": 5000,
            "retrieval_time_warning_ms": 500,
            "retrieval_time_critical_ms": 2000,
            "generation_time_warning_ms": 3000,
            "generation_time_critical_ms": 10000,
        }

        # Metrics aggregation
        self.aggregated_metrics = defaultdict(list)

    def record_metric(self, name: str, value: float, unit: str, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(name=name, value=value, unit=unit, timestamp=datetime.now(), tags=tags or {})

        self.metrics_buffer.append(metric)
        self.aggregated_metrics[name].append(value)

        # Keep only recent values for aggregation
        if len(self.aggregated_metrics[name]) > 1000:
            self.aggregated_metrics[name] = self.aggregated_metrics[name][-1000:]

    def record_query_performance(self, query_data: QueryPerformanceData):
        """Record query performance data."""
        self.query_performance_data.append(query_data)

        # Record individual metrics
        self.record_metric(
            "query_execution_time",
            query_data.execution_time_ms,
            "ms",
            {"pipeline": query_data.pipeline_type, "success": str(query_data.success)},
        )

        self.record_metric(
            "query_retrieval_time", query_data.retrieval_time_ms, "ms", {"pipeline": query_data.pipeline_type}
        )

        self.record_metric(
            "query_generation_time", query_data.generation_time_ms, "ms", {"pipeline": query_data.pipeline_type}
        )

        self.record_metric(
            "documents_retrieved", query_data.documents_retrieved, "count", {"pipeline": query_data.pipeline_type}
        )

        # Check for performance issues
        self._check_performance_thresholds(query_data)

    def _check_performance_thresholds(self, query_data: QueryPerformanceData):
        """Check if performance metrics exceed thresholds."""
        warnings = []

        if query_data.execution_time_ms > self.thresholds["query_time_critical_ms"]:
            warnings.append(f"Critical: Query execution time {query_data.execution_time_ms:.1f}ms exceeds threshold")
        elif query_data.execution_time_ms > self.thresholds["query_time_warning_ms"]:
            warnings.append(f"Warning: Query execution time {query_data.execution_time_ms:.1f}ms exceeds threshold")

        if query_data.retrieval_time_ms > self.thresholds["retrieval_time_critical_ms"]:
            warnings.append(f"Critical: Retrieval time {query_data.retrieval_time_ms:.1f}ms exceeds threshold")
        elif query_data.retrieval_time_ms > self.thresholds["retrieval_time_warning_ms"]:
            warnings.append(f"Warning: Retrieval time {query_data.retrieval_time_ms:.1f}ms exceeds threshold")

        if query_data.generation_time_ms > self.thresholds["generation_time_critical_ms"]:
            warnings.append(f"Critical: Generation time {query_data.generation_time_ms:.1f}ms exceeds threshold")
        elif query_data.generation_time_ms > self.thresholds["generation_time_warning_ms"]:
            warnings.append(f"Warning: Generation time {query_data.generation_time_ms:.1f}ms exceeds threshold")

        for warning in warnings:
            logger.warning(f"Performance threshold exceeded: {warning}")

    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        # Filter recent query data
        recent_queries = [q for q in self.query_performance_data if q.timestamp >= cutoff_time]

        if not recent_queries:
            return {
                "time_window_minutes": time_window_minutes,
                "total_queries": 0,
                "message": "No queries in time window",
            }

        # Calculate statistics
        execution_times = [q.execution_time_ms for q in recent_queries]
        retrieval_times = [q.retrieval_time_ms for q in recent_queries]
        generation_times = [q.generation_time_ms for q in recent_queries]

        successful_queries = [q for q in recent_queries if q.success]
        failed_queries = [q for q in recent_queries if not q.success]

        # Pipeline breakdown
        pipeline_stats = defaultdict(list)
        for q in recent_queries:
            pipeline_stats[q.pipeline_type].append(q.execution_time_ms)

        summary = {
            "time_window_minutes": time_window_minutes,
            "total_queries": len(recent_queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "success_rate": len(successful_queries) / len(recent_queries) * 100,
            "execution_time_stats": {
                "avg_ms": sum(execution_times) / len(execution_times),
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "p95_ms": self._percentile(execution_times, 95),
                "p99_ms": self._percentile(execution_times, 99),
            },
            "retrieval_time_stats": {
                "avg_ms": sum(retrieval_times) / len(retrieval_times),
                "min_ms": min(retrieval_times),
                "max_ms": max(retrieval_times),
            },
            "generation_time_stats": {
                "avg_ms": sum(generation_times) / len(generation_times),
                "min_ms": min(generation_times),
                "max_ms": max(generation_times),
            },
            "pipeline_performance": {
                pipeline: {"query_count": len(times), "avg_execution_time_ms": sum(times) / len(times)}
                for pipeline, times in pipeline_stats.items()
            },
        }

        return summary

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1

        return sorted_data[index]

    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_percent", cpu_percent, "percent")

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_percent", memory.percent, "percent")
            self.record_metric("system_memory_used_gb", memory.used / (1024**3), "gb")

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.record_metric("system_disk_percent", disk.percent, "percent")
            self.record_metric("system_disk_free_gb", disk.free / (1024**3), "gb")

            # Database metrics
            self._collect_database_metrics()

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_database_metrics(self):
        """Collect database performance metrics."""
        try:
            connection = self.connection_manager.get_connection("iris")

            with connection.cursor() as cursor:
                # Document counts
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                doc_count = cursor.fetchone()[0]
                self.record_metric("database_document_count", doc_count, "count")

                # Embedded document count
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                embedded_count = cursor.fetchone()[0]
                self.record_metric("database_embedded_document_count", embedded_count, "count")

                # Test query performance
                test_vector = "[" + ",".join(["0.1"] * 384) + "]"
                start_time = time.time()

                cursor.execute(
                    """
                    SELECT TOP 5 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                    FROM RAG.SourceDocuments 
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """,
                    (test_vector,),
                )

                cursor.fetchall()
                query_time_ms = (time.time() - start_time) * 1000

                self.record_metric("database_vector_query_time_ms", query_time_ms, "ms")

        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")

    def export_metrics(self, filepath: str, time_window_minutes: Optional[int] = None):
        """Export metrics to a JSON file."""
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            metrics_to_export = [
                {"name": m.name, "value": m.value, "unit": m.unit, "timestamp": m.timestamp.isoformat(), "tags": m.tags}
                for m in self.metrics_buffer
                if m.timestamp >= cutoff_time
            ]
        else:
            metrics_to_export = [
                {"name": m.name, "value": m.value, "unit": m.unit, "timestamp": m.timestamp.isoformat(), "tags": m.tags}
                for m in self.metrics_buffer
            ]

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_window_minutes": time_window_minutes,
            "metrics_count": len(metrics_to_export),
            "metrics": metrics_to_export,
            "performance_summary": self.get_performance_summary(time_window_minutes or 60),
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(metrics_to_export)} metrics to {filepath}")

    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time performance status."""
        recent_summary = self.get_performance_summary(5)  # Last 5 minutes

        # Get latest system metrics
        latest_metrics = {}
        for metric in reversed(self.metrics_buffer):
            if metric.name not in latest_metrics:
                latest_metrics[metric.name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                }

            # Stop when we have all system metrics
            if len(latest_metrics) >= 10:  # Adjust based on expected metric count
                break

        return {
            "monitoring_active": self.monitoring_active,
            "recent_performance": recent_summary,
            "latest_system_metrics": latest_metrics,
            "buffer_size": len(self.metrics_buffer),
            "query_data_size": len(self.query_performance_data),
        }
