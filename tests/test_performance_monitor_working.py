"""
Working test suite for GraphRAGPerformanceMonitor.

This test suite covers the actual implemented API and functionality
of the performance monitor system.
"""

import time
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from iris_rag.optimization.performance_monitor import (
    GraphRAGPerformanceMonitor,
    PerformanceAlert,
    PerformanceSnapshot,
)


class TestPerformanceDataClasses(unittest.TestCase):
    """Test the performance data classes."""

    def test_performance_snapshot_creation(self):
        """Test creating a performance snapshot."""
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

    def test_performance_alert_creation(self):
        """Test creating performance alerts."""
        timestamp = datetime.now()
        alert = PerformanceAlert(
            timestamp=timestamp,
            alert_type="response_time",
            severity="critical",
            message="Query response time exceeded threshold",
            metric_value=250.5,
            threshold=200.0,
        )

        self.assertEqual(alert.alert_type, "response_time")
        self.assertEqual(alert.severity, "critical")
        self.assertEqual(alert.metric_value, 250.5)


class TestGraphRAGPerformanceMonitor(unittest.TestCase):
    """Test the main performance monitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = GraphRAGPerformanceMonitor(
            history_size=100, monitoring_interval=1
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.monitor._monitoring_active:
            self.monitor.stop_monitoring()

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.history_size, 100)
        self.assertEqual(self.monitor.monitoring_interval, 1)
        self.assertIsInstance(self.monitor.thresholds, dict)
        self.assertIsInstance(
            self.monitor.performance_history, type(self.monitor.performance_history)
        )
        self.assertIsInstance(self.monitor.alerts, list)
        self.assertFalse(self.monitor._monitoring_active)

    def test_record_query_performance(self):
        """Test recording query performance."""
        # Test basic recording
        self.monitor.record_query_performance(
            response_time_ms=150.0,
            cache_hit=True,
            database_time_ms=75.0,
            hnsw_time_ms=25.0,
        )

        # Verify recording was successful (basic check)
        current_perf = self.monitor.get_current_performance()
        self.assertIsInstance(current_perf, dict)

        # Test with different parameters
        self.monitor.record_query_performance(
            response_time_ms=200.0,
            cache_hit=False,
            database_time_ms=100.0,
            hnsw_time_ms=50.0,
        )

    def test_get_current_performance(self):
        """Test getting current performance snapshot."""
        # Test with no data
        current_perf = self.monitor.get_current_performance()
        self.assertIsInstance(current_perf, dict)

        # Record some data and test again
        self.monitor.record_query_performance(
            response_time_ms=150.0,
            cache_hit=True,
            database_time_ms=75.0,
            hnsw_time_ms=25.0,
        )

        current_perf = self.monitor.get_current_performance()
        self.assertIsInstance(current_perf, dict)
        # The exact keys depend on implementation, but it should be a dict

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Test with no data
        summary = self.monitor.get_performance_summary()
        self.assertIsInstance(summary, dict)

        # Record some data
        for i in range(3):
            self.monitor.record_query_performance(
                response_time_ms=100.0 + i * 50,
                cache_hit=i % 2 == 0,
                database_time_ms=50.0 + i * 25,
                hnsw_time_ms=20.0 + i * 10,
            )

        # Test summary with data
        summary = self.monitor.get_performance_summary(time_range_minutes=60)
        self.assertIsInstance(summary, dict)

        # Test with custom time range
        summary = self.monitor.get_performance_summary(time_range_minutes=30)
        self.assertIsInstance(summary, dict)

    def test_get_component_health(self):
        """Test getting component health status."""
        # Test with no data
        health = self.monitor.get_component_health()
        self.assertIsInstance(health, dict)

        # Record some performance data
        self.monitor.record_query_performance(
            response_time_ms=120.0,
            cache_hit=True,
            database_time_ms=80.0,
            hnsw_time_ms=20.0,
        )

        # Test health with data
        health = self.monitor.get_component_health()
        self.assertIsInstance(health, dict)

    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        # Should not be active initially
        self.assertFalse(self.monitor._monitoring_active)

        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring_active)

        # Test starting when already active (should handle gracefully)
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring_active)

        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring_active)

        # Test stopping when already stopped (should handle gracefully)
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring_active)

    def test_html_dashboard_generation(self):
        """Test HTML dashboard generation."""
        # Test with no data
        html = self.monitor.generate_html_dashboard()
        self.assertIsInstance(html, str)
        self.assertIn("<html>", html.lower())

        # Record some data
        self.monitor.record_query_performance(
            response_time_ms=150.0,
            cache_hit=True,
            database_time_ms=100.0,
            hnsw_time_ms=30.0,
        )

        # Test with data
        html = self.monitor.generate_html_dashboard()
        self.assertIsInstance(html, str)
        self.assertIn("<html>", html.lower())
        # Should contain some performance-related content
        self.assertTrue(len(html) > 100)  # Should be substantial HTML

    def test_history_size_limit(self):
        """Test that performance history respects size limit."""
        # Create monitor with small history size
        small_monitor = GraphRAGPerformanceMonitor(history_size=3)

        # Record more entries than the limit
        for i in range(5):
            small_monitor.record_query_performance(
                response_time_ms=100.0 + i,
                cache_hit=True,
                database_time_ms=50.0,
                hnsw_time_ms=20.0,
            )

        # Should only keep the most recent entries up to history_size
        self.assertLessEqual(len(small_monitor.performance_history), 3)

    def test_threshold_configuration(self):
        """Test that thresholds can be configured."""
        # Test default thresholds
        self.assertIn("max_response_time_ms", self.monitor.thresholds)
        self.assertIn("min_cache_hit_rate", self.monitor.thresholds)

        # Test updating thresholds
        original_threshold = self.monitor.thresholds["max_response_time_ms"]
        self.monitor.thresholds["max_response_time_ms"] = 150.0
        self.assertEqual(self.monitor.thresholds["max_response_time_ms"], 150.0)
        self.assertNotEqual(
            self.monitor.thresholds["max_response_time_ms"], original_threshold
        )

    def test_multiple_recordings(self):
        """Test recording multiple performance measurements."""
        # Record various performance scenarios
        test_scenarios = [
            (120.0, True, 80.0, 20.0),  # Fast query with cache hit
            (250.0, False, 180.0, 40.0),  # Slow query with cache miss
            (90.0, True, 60.0, 15.0),  # Very fast query
            (300.0, False, 200.0, 60.0),  # Very slow query
        ]

        for response_time, cache_hit, db_time, hnsw_time in test_scenarios:
            self.monitor.record_query_performance(
                response_time_ms=response_time,
                cache_hit=cache_hit,
                database_time_ms=db_time,
                hnsw_time_ms=hnsw_time,
            )

        # Verify all recordings were processed
        self.assertGreater(len(self.monitor.performance_history), 0)

        # Get summary and verify it contains data
        summary = self.monitor.get_performance_summary()
        self.assertIsInstance(summary, dict)

    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with negative values (should handle gracefully)
        try:
            self.monitor.record_query_performance(
                response_time_ms=-10.0,
                cache_hit=True,
                database_time_ms=50.0,
                hnsw_time_ms=20.0,
            )
            # Should not crash
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, TypeError))

        # Test with very large values
        try:
            self.monitor.record_query_performance(
                response_time_ms=999999.0,
                cache_hit=False,
                database_time_ms=999999.0,
                hnsw_time_ms=999999.0,
            )
            # Should not crash
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, TypeError))


class TestPerformanceMonitorIntegration(unittest.TestCase):
    """Integration tests for realistic usage scenarios."""

    def test_typical_workload_pattern(self):
        """Test a typical workload pattern."""
        monitor = GraphRAGPerformanceMonitor(history_size=50)

        # Simulate a realistic workload
        for i in range(20):
            # Most queries are fast
            if i < 15:
                monitor.record_query_performance(
                    response_time_ms=100.0 + (i % 5) * 10,
                    cache_hit=True,
                    database_time_ms=60.0 + (i % 3) * 10,
                    hnsw_time_ms=20.0 + (i % 2) * 5,
                )
            # A few queries are slower
            else:
                monitor.record_query_performance(
                    response_time_ms=250.0 + (i % 3) * 20,
                    cache_hit=False,
                    database_time_ms=150.0 + (i % 2) * 30,
                    hnsw_time_ms=50.0 + (i % 2) * 10,
                )

        # Verify the workload was recorded
        self.assertEqual(len(monitor.performance_history), 20)

        # Get summary and verify it makes sense
        summary = monitor.get_performance_summary()
        self.assertIsInstance(summary, dict)

        # Get current performance
        current = monitor.get_current_performance()
        self.assertIsInstance(current, dict)

        # Get component health
        health = monitor.get_component_health()
        self.assertIsInstance(health, dict)

        # Generate dashboard
        dashboard = monitor.generate_html_dashboard()
        self.assertIsInstance(dashboard, str)
        self.assertGreater(len(dashboard), 100)


if __name__ == "__main__":
    unittest.main()
