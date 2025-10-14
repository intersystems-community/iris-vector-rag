"""
Comprehensive test suite for GraphRAGPerformanceMonitor.

Tests cover all aspects of performance monitoring including:
- Performance data collection and storage
- Alert generation and thresholds
- Dashboard generation
- Component health monitoring
- SLA compliance tracking
- Real-time monitoring lifecycle
"""

import json
import threading
import time
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

from iris_rag.optimization.performance_monitor import (
    GraphRAGPerformanceMonitor,
    PerformanceAlert,
    PerformanceSnapshot,
)


class TestPerformanceSnapshot(unittest.TestCase):
    """Test cases for PerformanceSnapshot dataclass."""

    def test_performance_snapshot_creation(self):
        """Test creating a performance snapshot with all fields."""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            query_response_time_ms=150.5,
            cache_hit_rate=0.85,
            connection_pool_utilization=0.7,
            parallel_operations_active=3,
            memory_usage_mb=512.0,
            cpu_utilization_percent=45.2,
            database_query_time_ms=75.3,
            hnsw_query_time_ms=25.1,
        )

        self.assertEqual(snapshot.timestamp, timestamp)
        self.assertEqual(snapshot.query_response_time_ms, 150.5)
        self.assertEqual(snapshot.cache_hit_rate, 0.85)
        self.assertEqual(snapshot.connection_pool_utilization, 0.7)
        self.assertEqual(snapshot.parallel_operations_active, 3)
        self.assertEqual(snapshot.memory_usage_mb, 512.0)
        self.assertEqual(snapshot.cpu_utilization_percent, 45.2)
        self.assertEqual(snapshot.database_query_time_ms, 75.3)
        self.assertEqual(snapshot.hnsw_query_time_ms, 25.1)

    def test_performance_snapshot_serialization(self):
        """Test that performance snapshots can be serialized."""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            query_response_time_ms=150.5,
            cache_hit_rate=0.85,
            connection_pool_utilization=0.7,
            parallel_operations_active=3,
            memory_usage_mb=512.0,
            cpu_utilization_percent=45.2,
            database_query_time_ms=75.3,
            hnsw_query_time_ms=25.1,
        )

        # Test that asdict works (used in monitor)
        from dataclasses import asdict

        snapshot_dict = asdict(snapshot)
        self.assertIsInstance(snapshot_dict, dict)
        self.assertEqual(snapshot_dict["query_response_time_ms"], 150.5)


class TestPerformanceAlert(unittest.TestCase):
    """Test cases for PerformanceAlert dataclass."""

    def test_performance_alert_creation(self):
        """Test creating performance alerts."""
        timestamp = datetime.now()
        alert = PerformanceAlert(
            timestamp=timestamp,
            alert_type="response_time",
            severity="critical",
            message="Query response time exceeded 200ms threshold",
            metric_value=250.5,
            threshold=200.0,
        )

        self.assertEqual(alert.timestamp, timestamp)
        self.assertEqual(alert.alert_type, "response_time")
        self.assertEqual(alert.severity, "critical")
        self.assertEqual(alert.message, "Query response time exceeded 200ms threshold")
        self.assertEqual(alert.metric_value, 250.5)
        self.assertEqual(alert.threshold, 200.0)


class TestGraphRAGPerformanceMonitor(unittest.TestCase):
    """Test cases for the main GraphRAGPerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = GraphRAGPerformanceMonitor(
            history_size=100, monitoring_interval=1
        )
        # Override thresholds for testing
        self.monitor.thresholds = {
            "max_response_time_ms": 200.0,
            "min_cache_hit_rate": 0.7,
            "max_memory_usage_mb": 1000.0,
            "max_connection_utilization": 0.85,
            "max_cpu_utilization": 0.80,
            "max_db_query_time_ms": 100.0,
            "max_hnsw_query_time_ms": 50.0,
        }

    def tearDown(self):
        """Clean up after tests."""
        if (
            hasattr(self.monitor, "_monitoring_active")
            and self.monitor._monitoring_active
        ):
            self.monitor.stop_monitoring()

    def test_monitor_initialization(self):
        """Test monitor initialization with default and custom parameters."""
        # Test default initialization
        default_monitor = GraphRAGPerformanceMonitor()
        self.assertEqual(default_monitor.history_size, 1000)
        self.assertIsInstance(default_monitor.thresholds, dict)

        # Test custom initialization
        custom_monitor = GraphRAGPerformanceMonitor(
            history_size=50, monitoring_interval=10
        )
        self.assertEqual(custom_monitor.history_size, 50)
        self.assertEqual(custom_monitor.monitoring_interval, 10)

    def test_record_query_performance(self):
        """Test recording query performance metrics."""
        # Record performance data using the actual API
        self.monitor.record_query_performance(
            response_time_ms=150.5,
            cache_hit=True,
            database_time_ms=75.3,
            hnsw_time_ms=25.1,
        )

        # Verify data was recorded
        current_perf = self.monitor.get_current_performance()
        self.assertEqual(current_perf["response_time_ms"], 150.5)
        self.assertEqual(current_perf["cache_hit_rate"], 0.85)
        self.assertTrue(current_perf["meets_sla"])  # Should meet 200ms SLA

    def test_alert_generation_response_time(self):
        """Test alert generation for response time threshold violations."""
        # Record performance that exceeds response time threshold
        self.monitor.record_query_performance(
            query_response_time_ms=250.0,  # Exceeds 200ms threshold
            cache_hit_rate=0.85,
            connection_pool_utilization=0.7,
            parallel_operations_active=2,
            memory_usage_mb=400.0,
            cpu_utilization_percent=30.0,
            database_query_time_ms=150.0,
            hnsw_query_time_ms=50.0,
        )

        # Check that alert was generated
        current_perf = self.monitor.get_current_performance()
        self.assertFalse(current_perf["meets_sla"])

        # Check for alerts in performance summary
        summary = self.monitor.get_performance_summary(time_range_minutes=1)
        self.assertGreater(len(summary["recent_alerts"]), 0)

        alert = summary["recent_alerts"][0]
        self.assertEqual(alert["alert_type"], "response_time")
        self.assertEqual(alert["severity"], "critical")

    def test_alert_generation_cache_hit_rate(self):
        """Test alert generation for low cache hit rate."""
        # Record performance with low cache hit rate
        self.monitor.record_query_performance(
            query_response_time_ms=150.0,
            cache_hit_rate=0.6,  # Below 0.7 threshold
            connection_pool_utilization=0.7,
            parallel_operations_active=2,
            memory_usage_mb=400.0,
            cpu_utilization_percent=30.0,
            database_query_time_ms=100.0,
            hnsw_query_time_ms=25.0,
        )

        # Check for cache hit rate alert
        summary = self.monitor.get_performance_summary(time_range_minutes=1)
        cache_alerts = [
            a for a in summary["recent_alerts"] if a["alert_type"] == "cache_hit_rate"
        ]
        self.assertGreater(len(cache_alerts), 0)

    def test_performance_summary_calculation(self):
        """Test calculation of performance summary statistics."""
        # Record multiple performance measurements
        measurements = [
            {"query_response_time_ms": 100.0, "cache_hit_rate": 0.9},
            {"query_response_time_ms": 150.0, "cache_hit_rate": 0.8},
            {"query_response_time_ms": 200.0, "cache_hit_rate": 0.7},
        ]

        for measurement in measurements:
            self.monitor.record_query_performance(
                connection_pool_utilization=0.5,
                parallel_operations_active=1,
                memory_usage_mb=300.0,
                cpu_utilization_percent=25.0,
                database_query_time_ms=75.0,
                hnsw_query_time_ms=20.0,
                **measurement,
            )

        summary = self.monitor.get_performance_summary(time_range_minutes=60)

        # Verify summary calculations
        self.assertEqual(summary["total_queries"], 3)
        self.assertEqual(summary["avg_response_time_ms"], 150.0)
        self.assertEqual(summary["max_response_time_ms"], 200.0)
        self.assertEqual(summary["min_response_time_ms"], 100.0)
        self.assertAlmostEqual(summary["avg_cache_hit_rate"], 0.8, places=2)

    def test_component_health_assessment(self):
        """Test component health status assessment."""
        # Record good performance
        self.monitor.record_query_performance(
            query_response_time_ms=120.0,
            cache_hit_rate=0.9,
            connection_pool_utilization=0.6,
            parallel_operations_active=2,
            memory_usage_mb=400.0,
            cpu_utilization_percent=35.0,
            database_query_time_ms=80.0,
            hnsw_query_time_ms=20.0,
        )

        health = self.monitor.get_component_health()

        # Verify health assessment structure
        self.assertIn("overall_status", health)
        self.assertIn("query_performance", health)
        self.assertIn("cache_system", health)
        self.assertIn("database_layer", health)
        self.assertIn("resource_utilization", health)

        # Verify good performance results in healthy status
        self.assertEqual(health["query_performance"]["status"], "healthy")
        self.assertEqual(health["cache_system"]["status"], "healthy")

    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        # Monitor should not be active initially
        self.assertFalse(
            hasattr(self.monitor, "_monitoring_active")
            and self.monitor._monitoring_active
        )

        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring_active)

        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring_active)

    def test_max_snapshots_limit(self):
        """Test that snapshot storage respects maximum limit."""
        # Create monitor with small limit
        limited_monitor = GraphRAGPerformanceMonitor(history_size=3)

        # Record more snapshots than the limit
        for i in range(5):
            limited_monitor.record_query_performance(
                query_response_time_ms=100.0 + i,
                cache_hit_rate=0.8,
                connection_pool_utilization=0.5,
                parallel_operations_active=1,
                memory_usage_mb=300.0,
                cpu_utilization_percent=25.0,
                database_query_time_ms=75.0,
                hnsw_query_time_ms=20.0,
            )

        # Verify only history_size entries are kept
        self.assertEqual(len(limited_monitor.performance_history), 3)

        # Verify most recent snapshots are kept
        recent_times = [
            s.query_response_time_ms for s in limited_monitor.performance_history
        ]
        self.assertEqual(recent_times, [102.0, 103.0, 104.0])  # Last 3 recorded

    def test_html_dashboard_generation(self):
        """Test HTML dashboard generation."""
        # Record some performance data
        self.monitor.record_query_performance(
            query_response_time_ms=150.0,
            cache_hit_rate=0.85,
            connection_pool_utilization=0.7,
            parallel_operations_active=3,
            memory_usage_mb=512.0,
            cpu_utilization_percent=45.0,
            database_query_time_ms=100.0,
            hnsw_query_time_ms=30.0,
        )

        # Generate dashboard
        html_dashboard = self.monitor.generate_html_dashboard()

        # Verify HTML structure
        self.assertIsInstance(html_dashboard, str)
        self.assertIn("<html>", html_dashboard)
        self.assertIn("GraphRAG Performance Dashboard", html_dashboard)
        self.assertIn("Response Time", html_dashboard)
        self.assertIn("Cache Hit Rate", html_dashboard)
        self.assertIn("150.0", html_dashboard)  # Should show our recorded response time
        self.assertIn("85%", html_dashboard)  # Should show cache hit rate as percentage

    @patch("time.time")
    def test_performance_grade_calculation(self):
        """Test performance grade calculation logic."""
        # Mock time for consistent testing
        mock_time = 1000.0
        time.time.return_value = mock_time

        # Test excellent performance (sub-150ms, high cache hit rate)
        self.monitor.record_query_performance(
            query_response_time_ms=120.0,
            cache_hit_rate=0.95,
            connection_pool_utilization=0.5,
            parallel_operations_active=1,
            memory_usage_mb=300.0,
            cpu_utilization_percent=25.0,
            database_query_time_ms=80.0,
            hnsw_query_time_ms=20.0,
        )

        current_perf = self.monitor.get_current_performance()
        self.assertEqual(current_perf["performance_grade"], "A")

        # Test poor performance (over 200ms, low cache hit rate)
        self.monitor.record_query_performance(
            query_response_time_ms=250.0,
            cache_hit_rate=0.5,
            connection_pool_utilization=0.8,
            parallel_operations_active=5,
            memory_usage_mb=800.0,
            cpu_utilization_percent=75.0,
            database_query_time_ms=180.0,
            hnsw_query_time_ms=50.0,
        )

        current_perf = self.monitor.get_current_performance()
        self.assertIn(current_perf["performance_grade"], ["C", "D", "F"])

    def test_thread_safety(self):
        """Test that performance recording is thread-safe."""
        import threading
        import time as time_module

        # Function to record performance in a thread
        def record_performance(thread_id):
            for i in range(10):
                self.monitor.record_query_performance(
                    query_response_time_ms=100.0 + thread_id + i,
                    cache_hit_rate=0.8,
                    connection_pool_utilization=0.5,
                    parallel_operations_active=1,
                    memory_usage_mb=300.0,
                    cpu_utilization_percent=25.0,
                    database_query_time_ms=75.0,
                    hnsw_query_time_ms=20.0,
                )
                time_module.sleep(0.001)  # Small delay to allow interleaving

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=record_performance, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all recordings were captured
        self.assertEqual(
            len(self.monitor.performance_history), 30
        )  # 3 threads * 10 recordings each

        # Verify no data corruption
        summary = self.monitor.get_performance_summary(time_range_minutes=1)
        self.assertEqual(summary["total_queries"], 30)

    def test_empty_state_handling(self):
        """Test behavior when no performance data has been recorded."""
        empty_monitor = GraphRAGPerformanceMonitor()

        # Test current performance with no data
        current_perf = empty_monitor.get_current_performance()
        self.assertEqual(current_perf["response_time_ms"], 0.0)
        self.assertEqual(current_perf["cache_hit_rate"], 0.0)
        self.assertEqual(current_perf["performance_grade"], "N/A")

        # Test summary with no data
        summary = empty_monitor.get_performance_summary()
        self.assertEqual(summary["total_queries"], 0)
        self.assertEqual(summary["avg_response_time_ms"], 0.0)

        # Test component health with no data
        health = empty_monitor.get_component_health()
        self.assertEqual(health["overall_status"], "no_data")


class TestPerformanceMonitorIntegration(unittest.TestCase):
    """Integration tests for performance monitor with realistic scenarios."""

    def test_realistic_workload_monitoring(self):
        """Test monitoring a realistic workload pattern."""
        monitor = GraphRAGPerformanceMonitor(history_size=100)
        monitor.thresholds.update(
            {"max_response_time_ms": 200.0, "min_cache_hit_rate": 0.75}
        )

        # Simulate a realistic workload: fast queries with occasional slow ones
        workload_pattern = [
            # Fast queries (cache hits)
            (120, 0.9),
            (135, 0.85),
            (110, 0.95),
            (145, 0.88),
            # Slower query (cache miss)
            (280, 0.6),
            # Recovery to normal
            (125, 0.9),
            (140, 0.87),
            (115, 0.92),
        ]

        for response_time, cache_rate in workload_pattern:
            monitor.record_query_performance(
                query_response_time_ms=response_time,
                cache_hit_rate=cache_rate,
                connection_pool_utilization=0.6,
                parallel_operations_active=2,
                memory_usage_mb=450.0,
                cpu_utilization_percent=40.0,
                database_query_time_ms=response_time * 0.6,
                hnsw_query_time_ms=response_time * 0.2,
            )

        # Analyze results
        summary = monitor.get_performance_summary(time_range_minutes=60)
        health = monitor.get_component_health()

        # Verify workload was captured correctly
        self.assertEqual(summary["total_queries"], 8)
        self.assertEqual(summary["max_response_time_ms"], 280.0)
        self.assertEqual(summary["min_response_time_ms"], 110.0)

        # Verify alerts were generated for the slow query
        response_time_alerts = [
            a for a in summary["recent_alerts"] if a["alert_type"] == "response_time"
        ]
        self.assertGreater(len(response_time_alerts), 0)

        # Verify overall health reflects mixed performance
        self.assertIn(health["overall_status"], ["degraded", "healthy"])


if __name__ == "__main__":
    unittest.main()
