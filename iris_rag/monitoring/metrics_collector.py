"""
Metrics Collector for RAG Templates System

Centralized metrics collection and aggregation.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os

from common.llm_cache_manager import get_global_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class MetricsCollector:
    """
    Centralized metrics collection and aggregation system.

    Collects metrics from various system components and provides
    aggregation, storage, and export capabilities.
    """

    def __init__(self, max_points: int = 100000):
        """
        Initialize the metrics collector.

        Args:
            max_points: Maximum number of metric points to keep in memory
        """
        self.max_points = max_points
        self.metrics = deque(maxlen=max_points)
        self.aggregated_metrics = defaultdict(list)
        self.collectors = {}  # Registered metric collectors
        self.collection_thread = None
        self.collection_active = False
        self.collection_interval = 60  # seconds

        # Metric aggregation windows
        self.aggregation_windows = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
        }

    def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Add a metric point."""
        metric = MetricPoint(name=name, value=value, timestamp=datetime.now(), tags=tags or {}, unit=unit)

        self.metrics.append(metric)

        # Update aggregated metrics
        metric_key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.aggregated_metrics[metric_key].append((metric.timestamp, value))

        # Keep only recent values for aggregation
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.aggregated_metrics[metric_key] = [
            (ts, val) for ts, val in self.aggregated_metrics[metric_key] if ts >= cutoff_time
        ]

    def register_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register a metric collector function."""
        self.collectors[name] = collector_func
        logger.info(f"Registered metric collector: {name}")

    def start_collection(self):
        """Start automatic metric collection."""
        if self.collection_active:
            logger.warning("Metric collection already active")
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        logger.info("Metric collection started")

    def stop_collection(self):
        """Stop automatic metric collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)

        logger.info("Metric collection stopped")

    def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metric collection loop: {e}")
                time.sleep(self.collection_interval)

    def _collect_all_metrics(self):
        """Collect metrics from all registered collectors."""
        for collector_name, collector_func in self.collectors.items():
            try:
                metrics = collector_func()
                for metric_name, value in metrics.items():
                    self.add_metric(name=metric_name, value=value, tags={"collector": collector_name})
            except Exception as e:
                logger.error(f"Error collecting metrics from {collector_name}: {e}")

    def get_metrics(
        self,
        name_pattern: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        time_window: Optional[timedelta] = None,
    ) -> List[MetricPoint]:
        """Get metrics matching the specified criteria."""
        filtered_metrics = list(self.metrics)

        # Filter by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]

        # Filter by name pattern
        if name_pattern:
            filtered_metrics = [m for m in filtered_metrics if name_pattern in m.name]

        # Filter by tags
        if tags:
            filtered_metrics = [m for m in filtered_metrics if all(m.tags.get(k) == v for k, v in tags.items())]

        return filtered_metrics

    def get_aggregated_metrics(self, name: str, window: str = "1h", aggregation: str = "avg") -> Dict[str, float]:
        """Get aggregated metrics for a specific metric name."""
        if window not in self.aggregation_windows:
            raise ValueError(f"Invalid window: {window}. Available: {list(self.aggregation_windows.keys())}")

        time_window = self.aggregation_windows[window]
        cutoff_time = datetime.now() - time_window

        # Find matching metrics
        matching_metrics = []
        for metric_key, values in self.aggregated_metrics.items():
            if metric_key.startswith(f"{name}:"):
                recent_values = [val for ts, val in values if ts >= cutoff_time]
                if recent_values:
                    matching_metrics.extend(recent_values)

        if not matching_metrics:
            return {}

        # Calculate aggregations
        result = {}

        if aggregation in ["avg", "all"]:
            result["avg"] = sum(matching_metrics) / len(matching_metrics)

        if aggregation in ["min", "all"]:
            result["min"] = min(matching_metrics)

        if aggregation in ["max", "all"]:
            result["max"] = max(matching_metrics)

        if aggregation in ["sum", "all"]:
            result["sum"] = sum(matching_metrics)

        if aggregation in ["count", "all"]:
            result["count"] = len(matching_metrics)

        if aggregation in ["p95", "all"]:
            sorted_values = sorted(matching_metrics)
            p95_index = int(len(sorted_values) * 0.95)
            result["p95"] = sorted_values[min(p95_index, len(sorted_values) - 1)]

        if aggregation in ["p99", "all"]:
            sorted_values = sorted(matching_metrics)
            p99_index = int(len(sorted_values) * 0.99)
            result["p99"] = sorted_values[min(p99_index, len(sorted_values) - 1)]

        return result

    def get_metric_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        if time_window is None:
            time_window = timedelta(hours=1)

        metrics = self.get_metrics(time_window=time_window)

        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.name].append(metric.value)

        summary = {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_metrics": len(metrics),
            "unique_metric_names": len(metric_groups),
            "metric_statistics": {},
        }

        for name, values in metric_groups.items():
            if values:
                summary["metric_statistics"][name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else None,
                }

        return summary

    def export_metrics(self, filepath: str, time_window: Optional[timedelta] = None, format: str = "json") -> None:
        """Export metrics to a file."""
        metrics = self.get_metrics(time_window=time_window)

        if format == "json":
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_window_hours": time_window.total_seconds() / 3600 if time_window else None,
                "metrics_count": len(metrics),
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "tags": m.tags,
                        "unit": m.unit,
                    }
                    for m in metrics
                ],
            }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format == "csv":
            import csv

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "name", "value", "unit", "tags"])

                for metric in metrics:
                    writer.writerow(
                        [metric.timestamp.isoformat(), metric.name, metric.value, metric.unit, json.dumps(metric.tags)]
                    )

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(metrics)} metrics to {filepath}")

    def clear_old_metrics(self, max_age: timedelta = timedelta(days=7)):
        """Clear metrics older than the specified age."""
        cutoff_time = datetime.now() - max_age

        # Filter metrics
        original_count = len(self.metrics)
        self.metrics = deque((m for m in self.metrics if m.timestamp >= cutoff_time), maxlen=self.max_points)

        # Filter aggregated metrics
        for metric_key in list(self.aggregated_metrics.keys()):
            self.aggregated_metrics[metric_key] = [
                (ts, val) for ts, val in self.aggregated_metrics[metric_key] if ts >= cutoff_time
            ]

            # Remove empty entries
            if not self.aggregated_metrics[metric_key]:
                del self.aggregated_metrics[metric_key]

        cleared_count = original_count - len(self.metrics)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old metrics")

    def get_status(self) -> Dict[str, Any]:
        """Get collector status information."""
        return {
            "collection_active": self.collection_active,
            "collection_interval_seconds": self.collection_interval,
            "total_metrics": len(self.metrics),
            "registered_collectors": list(self.collectors.keys()),
            "unique_metric_names": len(set(m.name for m in self.metrics)),
            "oldest_metric": min(m.timestamp for m in self.metrics).isoformat() if self.metrics else None,
            "newest_metric": max(m.timestamp for m in self.metrics).isoformat() if self.metrics else None,
            "memory_usage_mb": len(self.metrics) * 0.001,  # Rough estimate
        }

    def collect_cache_metrics(self) -> Dict[str, float]:
        """Collect LLM cache performance metrics."""
        try:
            cache_manager = get_global_cache_manager()

            if not cache_manager:
                return {"llm_cache_enabled": 0.0, "llm_cache_configured": 0.0}

            cache_stats = cache_manager.get_cache_stats()
            cache_metrics = cache_stats.get("metrics", {})

            metrics = {
                "llm_cache_enabled": 1.0 if cache_stats.get("enabled", False) else 0.0,
                "llm_cache_configured": 1.0 if cache_stats.get("configured", False) else 0.0,
                "llm_cache_hit_rate": cache_metrics.get("hit_rate", 0.0),
                "llm_cache_total_requests": float(cache_metrics.get("total_requests", 0)),
                "llm_cache_hits": float(cache_metrics.get("hits", 0)),
                "llm_cache_misses": float(cache_metrics.get("misses", 0)),
                "llm_cache_avg_response_time_cached_ms": cache_metrics.get("avg_response_time_cached", 0.0),
                "llm_cache_avg_response_time_uncached_ms": cache_metrics.get("avg_response_time_uncached", 0.0),
            }

            # Calculate speedup ratio
            cached_time = cache_metrics.get("avg_response_time_cached", 0.0)
            uncached_time = cache_metrics.get("avg_response_time_uncached", 0.0)
            if cached_time > 0:
                metrics["llm_cache_speedup_ratio"] = uncached_time / cached_time
            else:
                metrics["llm_cache_speedup_ratio"] = 0.0

            # Add backend-specific metrics if available
            backend_stats = cache_stats.get("backend_stats", {})
            for key, value in backend_stats.items():
                if isinstance(value, (int, float)):
                    metrics[f"llm_cache_backend_{key}"] = float(value)

            return metrics

        except Exception as e:
            logger.warning(f"Failed to collect cache metrics: {e}")
            return {"llm_cache_enabled": 0.0, "llm_cache_configured": 0.0, "llm_cache_error": 1.0}
