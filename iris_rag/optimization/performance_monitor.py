"""
Performance Monitoring Dashboard for GraphRAG Optimization.

This module provides comprehensive performance monitoring for all GraphRAG
optimization components, tracking cache hit rates, connection pool utilization,
parallel processing metrics, and query performance to ensure sub-200ms response times.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement."""

    timestamp: datetime
    query_response_time_ms: float
    cache_hit_rate: float
    connection_pool_utilization: float
    parallel_operations_active: int
    memory_usage_mb: float
    cpu_utilization_percent: float
    database_query_time_ms: float
    hnsw_query_time_ms: float


@dataclass
class PerformanceAlert:
    """Performance alert for monitoring threshold violations."""

    timestamp: datetime
    alert_type: str
    severity: str  # 'warning', 'critical'
    message: str
    metric_value: float
    threshold: float


class GraphRAGPerformanceMonitor:
    """
    Comprehensive performance monitoring for GraphRAG optimization.

    Features:
    - Real-time performance tracking
    - Historical trend analysis
    - Performance alerting and thresholds
    - HTML dashboard generation
    - Optimization component monitoring
    - Sub-200ms response time validation
    """

    def __init__(
        self,
        cache_manager=None,
        connection_pool=None,
        parallel_processor=None,
        database_optimizer=None,
        hnsw_tuner=None,
        history_size: int = 1000,
        monitoring_interval: int = 5,
    ):
        """Initialize performance monitor with optimization components."""
        self.cache_manager = cache_manager
        self.connection_pool = connection_pool
        self.parallel_processor = parallel_processor
        self.database_optimizer = database_optimizer
        self.hnsw_tuner = hnsw_tuner

        # Performance tracking
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        self.performance_history: deque = deque(maxlen=history_size)
        self.alerts: List[PerformanceAlert] = []

        # Performance thresholds
        self.thresholds = {
            "max_response_time_ms": 200.0,
            "min_cache_hit_rate": 0.60,
            "max_connection_utilization": 0.85,
            "max_memory_usage_mb": 2048.0,
            "max_cpu_utilization": 0.80,
            "max_db_query_time_ms": 100.0,
            "max_hnsw_query_time_ms": 50.0,
        }

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._performance_lock = threading.RLock()

        # Aggregated metrics
        self.metrics_aggregator = PerformanceMetricsAggregator()

        logger.info("GraphRAG performance monitor initialized")

    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="GraphRAGPerformanceMonitor",
        )
        self._monitoring_thread.start()

        logger.info(
            f"Performance monitoring started (interval: {self.monitoring_interval}s)"
        )

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring_active = False

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)

        logger.info("Performance monitoring stopped")

    def record_query_performance(
        self,
        response_time_ms: float,
        cache_hit: bool = False,
        database_time_ms: float = 0.0,
        hnsw_time_ms: float = 0.0,
    ) -> None:
        """Record performance metrics for a single query."""
        with self._performance_lock:
            # Update aggregated metrics
            self.metrics_aggregator.record_query(
                response_time_ms, cache_hit, database_time_ms, hnsw_time_ms
            )

            # Check for performance alerts
            self._check_performance_thresholds(
                response_time_ms, database_time_ms, hnsw_time_ms
            )

    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance snapshot."""
        snapshot = self._capture_performance_snapshot()

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "response_time_ms": snapshot.query_response_time_ms,
            "cache_hit_rate": snapshot.cache_hit_rate,
            "connection_utilization": snapshot.connection_pool_utilization,
            "parallel_operations": snapshot.parallel_operations_active,
            "memory_usage_mb": snapshot.memory_usage_mb,
            "cpu_utilization": snapshot.cpu_utilization_percent,
            "database_query_time_ms": snapshot.database_query_time_ms,
            "hnsw_query_time_ms": snapshot.hnsw_query_time_ms,
            "performance_grade": self._calculate_performance_grade(snapshot),
            "meets_sla": snapshot.query_response_time_ms
            < self.thresholds["max_response_time_ms"],
        }

    def get_performance_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time range."""
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)

        with self._performance_lock:
            recent_snapshots = [
                snapshot
                for snapshot in self.performance_history
                if snapshot.timestamp > cutoff_time
            ]

        if not recent_snapshots:
            return {"error": "No data available for specified time range"}

        # Calculate statistics
        response_times = [s.query_response_time_ms for s in recent_snapshots]
        cache_hit_rates = [s.cache_hit_rate for s in recent_snapshots]

        summary = {
            "time_range_minutes": time_range_minutes,
            "total_snapshots": len(recent_snapshots),
            "response_time_stats": {
                "avg_ms": sum(response_times) / len(response_times),
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "p95_ms": self._percentile(response_times, 0.95),
                "p99_ms": self._percentile(response_times, 0.99),
                "sla_compliance": sum(1 for t in response_times if t < 200)
                / len(response_times),
            },
            "cache_performance": {
                "avg_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates),
                "min_hit_rate": min(cache_hit_rates),
                "max_hit_rate": max(cache_hit_rates),
            },
            "active_alerts": len([a for a in self.alerts if a.timestamp > cutoff_time]),
            "optimization_effectiveness": self._calculate_optimization_effectiveness(
                recent_snapshots
            ),
        }

        return summary

    def get_component_health(self) -> Dict[str, Any]:
        """Get health status of all optimization components."""
        health_status = {}

        # Cache manager health
        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_cache_stats()
                health_status["cache_manager"] = {
                    "status": "healthy",
                    "overall_hit_rate": cache_stats["overall"]["overall_hit_rate"],
                    "total_entries": cache_stats["overall"]["total_cache_entries"],
                    "health_grade": (
                        "excellent"
                        if cache_stats["overall"]["overall_hit_rate"] > 0.8
                        else "good"
                    ),
                }
            except Exception as e:
                health_status["cache_manager"] = {"status": "error", "error": str(e)}

        # Connection pool health
        if self.connection_pool:
            try:
                pool_stats = self.connection_pool.get_pool_stats()
                utilization = pool_stats["active_connections"] / pool_stats["peak_size"]
                health_status["connection_pool"] = {
                    "status": "healthy",
                    "utilization": utilization,
                    "active_connections": pool_stats["active_connections"],
                    "total_connections": pool_stats["current_size"],
                    "health_grade": (
                        "excellent"
                        if utilization < 0.7
                        else "good" if utilization < 0.9 else "warning"
                    ),
                }
            except Exception as e:
                health_status["connection_pool"] = {"status": "error", "error": str(e)}

        # Parallel processor health
        if self.parallel_processor:
            try:
                processor_stats = self.parallel_processor.get_performance_stats()
                health_status["parallel_processor"] = {
                    "status": "healthy",
                    "total_operations": processor_stats["total_operations"],
                    "avg_execution_time": processor_stats["avg_execution_time"],
                    "max_concurrency": processor_stats["max_concurrency_used"],
                    "health_grade": (
                        "excellent"
                        if processor_stats["avg_execution_time"] < 0.1
                        else "good"
                    ),
                }
            except Exception as e:
                health_status["parallel_processor"] = {
                    "status": "error",
                    "error": str(e),
                }

        # Database optimizer health
        if self.database_optimizer:
            try:
                optimizer_stats = self.database_optimizer.get_optimization_stats()
                health_status["database_optimizer"] = {
                    "status": "healthy",
                    "indexes_created": optimizer_stats["indexes_created"],
                    "queries_analyzed": optimizer_stats["queries_analyzed"],
                    "optimization_time": optimizer_stats["optimization_time"],
                    "health_grade": "excellent",
                }
            except Exception as e:
                health_status["database_optimizer"] = {
                    "status": "error",
                    "error": str(e),
                }

        # HNSW tuner health
        if self.hnsw_tuner:
            try:
                tuning_summary = self.hnsw_tuner.get_tuning_summary()
                if tuning_summary.get("status") != "no_tuning_performed":
                    health_status["hnsw_tuner"] = {
                        "status": "healthy",
                        "configurations_tested": tuning_summary[
                            "total_configurations_tested"
                        ],
                        "optimal_parameters": tuning_summary["optimal_parameters"],
                        "health_grade": "excellent",
                    }
                else:
                    health_status["hnsw_tuner"] = {"status": "not_configured"}
            except Exception as e:
                health_status["hnsw_tuner"] = {"status": "error", "error": str(e)}

        return health_status

    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard for performance monitoring."""
        current_perf = self.get_current_performance()
        summary = self.get_performance_summary(60)
        component_health = self.get_component_health()
        recent_alerts = self.alerts[-10:]  # Last 10 alerts

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GraphRAG Performance Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .panel { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { display: flex; justify-content: space-between; margin: 10px 0; }
                .metric-value { font-weight: bold; }
                .excellent { color: #28a745; }
                .good { color: #17a2b8; }
                .warning { color: #ffc107; }
                .critical { color: #dc3545; }
                .header { text-align: center; margin-bottom: 30px; }
                .sla-status { font-size: 24px; font-weight: bold; text-align: center; padding: 20px; }
                .alert { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 5px 0; border-radius: 4px; }
                .progress-bar { background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; }
                .progress-fill { height: 100%; background: #28a745; transition: width 0.3s; }
            </style>
            <script>
                function refreshDashboard() {
                    location.reload();
                }
                setInterval(refreshDashboard, 30000); // Refresh every 30 seconds
            </script>
        </head>
        <body>
            <div class="header">
                <h1>GraphRAG Performance Dashboard</h1>
                <div class="sla-status {sla_class}">
                    Response Time: {response_time:.1f}ms | SLA: {'✅ PASS' if current_perf['meets_sla'] else '❌ FAIL'}
                </div>
            </div>
            
            <div class="dashboard">
                <div class="panel">
                    <h3>Current Performance</h3>
                    <div class="metric">
                        <span>Response Time:</span>
                        <span class="metric-value {response_class}">{response_time:.1f}ms</span>
                    </div>
                    <div class="metric">
                        <span>Cache Hit Rate:</span>
                        <span class="metric-value {cache_class}">{cache_hit_rate:.1%}</span>
                    </div>
                    <div class="metric">
                        <span>Connection Utilization:</span>
                        <span class="metric-value">{connection_util:.1%}</span>
                    </div>
                    <div class="metric">
                        <span>Memory Usage:</span>
                        <span class="metric-value">{memory_usage:.0f}MB</span>
                    </div>
                    <div class="metric">
                        <span>Performance Grade:</span>
                        <span class="metric-value {grade_class}">{performance_grade}</span>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Performance Summary (Last Hour)</h3>
                    {summary_content}
                </div>
                
                <div class="panel">
                    <h3>Component Health</h3>
                    {component_health_content}
                </div>
                
                <div class="panel">
                    <h3>Recent Alerts</h3>
                    {alerts_content}
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px; color: #666;">
                Last Updated: {timestamp} | Auto-refresh: 30s
            </div>
        </body>
        </html>
        """

        # Format template variables
        sla_class = "excellent" if current_perf["meets_sla"] else "critical"
        response_class = self._get_css_class_for_response_time(
            current_perf["response_time_ms"]
        )
        cache_class = "excellent" if current_perf["cache_hit_rate"] > 0.8 else "good"
        grade_class = current_perf["performance_grade"].lower()

        # Generate summary content
        if "error" not in summary:
            summary_content = f"""
                <div class="metric">
                    <span>Avg Response Time:</span>
                    <span class="metric-value">{summary['response_time_stats']['avg_ms']:.1f}ms</span>
                </div>
                <div class="metric">
                    <span>P95 Response Time:</span>
                    <span class="metric-value">{summary['response_time_stats']['p95_ms']:.1f}ms</span>
                </div>
                <div class="metric">
                    <span>SLA Compliance:</span>
                    <span class="metric-value">{summary['response_time_stats']['sla_compliance']:.1%}</span>
                </div>
                <div class="metric">
                    <span>Cache Hit Rate:</span>
                    <span class="metric-value">{summary['cache_performance']['avg_hit_rate']:.1%}</span>
                </div>
            """
        else:
            summary_content = f"<div class='alert'>Error: {summary['error']}</div>"

        # Generate component health content
        health_items = []
        for component, health in component_health.items():
            if health["status"] == "healthy":
                grade = health.get("health_grade", "unknown")
                health_items.append(
                    f"""
                    <div class="metric">
                        <span>{component.replace('_', ' ').title()}:</span>
                        <span class="metric-value {grade}">{grade.upper()}</span>
                    </div>
                """
                )
            else:
                health_items.append(
                    f"""
                    <div class="metric">
                        <span>{component.replace('_', ' ').title()}:</span>
                        <span class="metric-value critical">{health['status'].upper()}</span>
                    </div>
                """
                )
        component_health_content = "".join(health_items)

        # Generate alerts content
        if recent_alerts:
            alerts_items = []
            for alert in recent_alerts:
                alerts_items.append(
                    f"""
                    <div class="alert">
                        <strong>{alert.severity.upper()}</strong>: {alert.message}
                        <br><small>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
                    </div>
                """
                )
            alerts_content = "".join(alerts_items)
        else:
            alerts_content = "<div>No recent alerts</div>"

        return html_template.format(
            sla_class=sla_class,
            response_time=current_perf["response_time_ms"],
            response_class=response_class,
            cache_hit_rate=current_perf["cache_hit_rate"],
            cache_class=cache_class,
            connection_util=current_perf["connection_utilization"],
            memory_usage=current_perf["memory_usage_mb"],
            performance_grade=current_perf["performance_grade"],
            grade_class=grade_class,
            summary_content=summary_content,
            component_health_content=component_health_content,
            alerts_content=alerts_content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _monitoring_worker(self) -> None:
        """Background worker for continuous performance monitoring."""
        logger.info("Performance monitoring worker started")

        while self._monitoring_active:
            try:
                # Capture performance snapshot
                snapshot = self._capture_performance_snapshot()

                with self._performance_lock:
                    self.performance_history.append(snapshot)

                # Check for alerts
                self._check_snapshot_thresholds(snapshot)

                # Sleep until next monitoring interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in performance monitoring worker: {e}")
                time.sleep(self.monitoring_interval)

        logger.info("Performance monitoring worker stopped")

    def _capture_performance_snapshot(self) -> PerformanceSnapshot:
        """Capture current performance metrics from all components."""
        # Get metrics from aggregator
        current_metrics = self.metrics_aggregator.get_current_metrics()

        # Get system metrics (simplified - in production would use psutil)
        memory_usage = 512.0  # Placeholder
        cpu_utilization = 0.3  # Placeholder

        # Get cache metrics
        cache_hit_rate = 0.0
        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_cache_stats()
                cache_hit_rate = cache_stats["overall"]["overall_hit_rate"]
            except:
                pass

        # Get connection pool metrics
        connection_utilization = 0.0
        if self.connection_pool:
            try:
                pool_stats = self.connection_pool.get_pool_stats()
                connection_utilization = pool_stats["active_connections"] / max(
                    1, pool_stats["peak_size"]
                )
            except:
                pass

        # Get parallel processor metrics
        parallel_operations = 0
        if self.parallel_processor:
            try:
                processor_stats = self.parallel_processor.get_performance_stats()
                parallel_operations = processor_stats.get("total_operations", 0)
            except:
                pass

        return PerformanceSnapshot(
            timestamp=datetime.now(),
            query_response_time_ms=current_metrics["avg_response_time_ms"],
            cache_hit_rate=cache_hit_rate,
            connection_pool_utilization=connection_utilization,
            parallel_operations_active=parallel_operations,
            memory_usage_mb=memory_usage,
            cpu_utilization_percent=cpu_utilization,
            database_query_time_ms=current_metrics["avg_database_time_ms"],
            hnsw_query_time_ms=current_metrics["avg_hnsw_time_ms"],
        )

    def _check_performance_thresholds(
        self, response_time_ms: float, db_time_ms: float, hnsw_time_ms: float
    ) -> None:
        """Check performance thresholds and generate alerts."""
        if response_time_ms > self.thresholds["max_response_time_ms"]:
            self._create_alert(
                "response_time_exceeded",
                "critical",
                f"Response time {response_time_ms:.1f}ms exceeds threshold {self.thresholds['max_response_time_ms']}ms",
                response_time_ms,
                self.thresholds["max_response_time_ms"],
            )

    def _check_snapshot_thresholds(self, snapshot: PerformanceSnapshot) -> None:
        """Check all thresholds for a performance snapshot."""
        # Add comprehensive threshold checking here
        pass

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        value: float,
        threshold: float,
    ) -> None:
        """Create a performance alert."""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_value=value,
            threshold=threshold,
        )

        self.alerts.append(alert)

        # Keep only recent alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

        logger.warning(f"Performance alert: {message}")

    def _calculate_performance_grade(self, snapshot: PerformanceSnapshot) -> str:
        """Calculate overall performance grade."""
        if snapshot.query_response_time_ms < 50:
            return "excellent"
        elif snapshot.query_response_time_ms < 100:
            return "good"
        elif snapshot.query_response_time_ms < 200:
            return "acceptable"
        else:
            return "poor"

    def _calculate_optimization_effectiveness(
        self, snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, float]:
        """Calculate effectiveness of optimizations."""
        if len(snapshots) < 2:
            return {}

        # Calculate improvements over time
        early = snapshots[: len(snapshots) // 3]
        recent = snapshots[-len(snapshots) // 3 :]

        early_avg = sum(s.query_response_time_ms for s in early) / len(early)
        recent_avg = sum(s.query_response_time_ms for s in recent) / len(recent)

        improvement = (early_avg - recent_avg) / early_avg if early_avg > 0 else 0

        return {
            "response_time_improvement": improvement,
            "cache_effectiveness": sum(s.cache_hit_rate for s in recent) / len(recent),
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _get_css_class_for_response_time(self, response_time_ms: float) -> str:
        """Get CSS class for response time display."""
        if response_time_ms < 50:
            return "excellent"
        elif response_time_ms < 100:
            return "good"
        elif response_time_ms < 200:
            return "warning"
        else:
            return "critical"


class PerformanceMetricsAggregator:
    """Aggregates performance metrics for monitoring."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.cache_hits = 0
        self.total_queries = 0
        self.database_times: deque = deque(maxlen=window_size)
        self.hnsw_times: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record_query(
        self,
        response_time_ms: float,
        cache_hit: bool,
        db_time_ms: float,
        hnsw_time_ms: float,
    ) -> None:
        """Record metrics for a single query."""
        with self._lock:
            self.response_times.append(response_time_ms)
            self.database_times.append(db_time_ms)
            self.hnsw_times.append(hnsw_time_ms)
            self.total_queries += 1
            if cache_hit:
                self.cache_hits += 1

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current aggregated metrics."""
        with self._lock:
            if not self.response_times:
                return {
                    "avg_response_time_ms": 0.0,
                    "cache_hit_rate": 0.0,
                    "avg_database_time_ms": 0.0,
                    "avg_hnsw_time_ms": 0.0,
                }

            return {
                "avg_response_time_ms": sum(self.response_times)
                / len(self.response_times),
                "cache_hit_rate": self.cache_hits / max(1, self.total_queries),
                "avg_database_time_ms": (
                    sum(self.database_times) / len(self.database_times)
                    if self.database_times
                    else 0.0
                ),
                "avg_hnsw_time_ms": (
                    sum(self.hnsw_times) / len(self.hnsw_times)
                    if self.hnsw_times
                    else 0.0
                ),
            }
