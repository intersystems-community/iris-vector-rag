"""
Tests for LLM Cache Monitoring Integration

Tests the integration of LLM cache monitoring with the health monitor,
metrics collector, and dashboard components.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from iris_rag.monitoring.health_monitor import HealthMonitor
from iris_rag.monitoring.metrics_collector import MetricsCollector
from common.llm_cache_manager import LangchainCacheManager, CacheMetrics
from common.llm_cache_config import CacheConfig


class TestLLMCacheHealthMonitoring:
    """Test LLM cache health monitoring functionality."""
    
    def test_cache_health_check_disabled(self):
        """Test health check when cache is disabled."""
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=None):
            health_monitor = HealthMonitor()
            result = health_monitor.check_llm_cache_performance()
            
            assert result.component == 'llm_cache_performance'
            assert result.status == 'warning'
            assert 'not configured or disabled' in result.message
            assert result.metrics['cache_enabled'] is False
            assert result.metrics['cache_configured'] is False
    
    def test_cache_health_check_healthy(self):
        """Test health check when cache is healthy."""
        # Mock cache manager with good performance
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'memory',
            'configured': True,
            'metrics': {
                'hits': 80,
                'misses': 20,
                'total_requests': 100,
                'hit_rate': 0.8,
                'avg_response_time_cached': 50.0,
                'avg_response_time_uncached': 200.0
            },
            'backend_stats': {
                'cache_size': 1000,
                'memory_usage_mb': 10.5
            }
        }
        
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=mock_cache_manager):
            health_monitor = HealthMonitor()
            result = health_monitor.check_llm_cache_performance()
            
            assert result.component == 'llm_cache_performance'
            assert result.status == 'healthy'
            assert 'healthy' in result.message
            assert result.metrics['cache_enabled'] is True
            assert result.metrics['hit_rate'] == 0.8
            assert result.metrics['total_requests'] == 100
            assert result.metrics['cache_speedup_ratio'] == 4.0  # 200/50
    
    def test_cache_health_check_low_hit_rate_warning(self):
        """Test health check with low hit rate (warning)."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'memory',
            'configured': True,
            'metrics': {
                'hits': 25,
                'misses': 75,
                'total_requests': 100,
                'hit_rate': 0.25,
                'avg_response_time_cached': 50.0,
                'avg_response_time_uncached': 200.0
            }
        }
        
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=mock_cache_manager):
            health_monitor = HealthMonitor()
            result = health_monitor.check_llm_cache_performance()
            
            assert result.component == 'llm_cache_performance'
            assert result.status == 'warning'
            assert 'hit rate low' in result.message
            assert '25.0%' in result.message
    
    def test_cache_health_check_critical_hit_rate(self):
        """Test health check with critically low hit rate."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'memory',
            'configured': True,
            'metrics': {
                'hits': 5,
                'misses': 95,
                'total_requests': 100,
                'hit_rate': 0.05,
                'avg_response_time_cached': 50.0,
                'avg_response_time_uncached': 200.0
            }
        }
        
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=mock_cache_manager):
            health_monitor = HealthMonitor()
            result = health_monitor.check_llm_cache_performance()
            
            assert result.component == 'llm_cache_performance'
            assert result.status == 'critical'
            assert 'critically low' in result.message
    
    def test_cache_health_check_no_requests(self):
        """Test health check when no requests have been made."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'memory',
            'configured': True,
            'metrics': {
                'hits': 0,
                'misses': 0,
                'total_requests': 0,
                'hit_rate': 0.0,
                'avg_response_time_cached': 0.0,
                'avg_response_time_uncached': 0.0
            }
        }
        
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=mock_cache_manager):
            health_monitor = HealthMonitor()
            result = health_monitor.check_llm_cache_performance()
            
            assert result.component == 'llm_cache_performance'
            assert result.status == 'warning'
            assert 'No cache requests recorded' in result.message
    
    def test_cache_health_check_exception_handling(self):
        """Test health check exception handling."""
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', side_effect=Exception("Test error")):
            health_monitor = HealthMonitor()
            result = health_monitor.check_llm_cache_performance()
            
            assert result.component == 'llm_cache_performance'
            assert result.status == 'critical'
            assert 'Test error' in result.message
            assert result.metrics == {}
    
    def test_comprehensive_health_check_includes_cache(self):
        """Test that comprehensive health check includes cache monitoring."""
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=None):
            health_monitor = HealthMonitor()
            
            # Mock other health checks to avoid dependencies
            with patch.object(health_monitor, 'check_system_resources') as mock_sys, \
                 patch.object(health_monitor, 'check_database_connectivity') as mock_db, \
                 patch.object(health_monitor, 'check_docker_containers') as mock_docker, \
                 patch.object(health_monitor, 'check_vector_performance') as mock_vector:
                
                # Configure mocks to return healthy results
                for mock_check in [mock_sys, mock_db, mock_docker, mock_vector]:
                    mock_result = Mock()
                    mock_result.status = 'healthy'
                    mock_check.return_value = mock_result
                
                results = health_monitor.run_comprehensive_health_check()
                
                assert 'llm_cache_performance' in results
                assert results['llm_cache_performance'].status == 'warning'


class TestLLMCacheMetricsCollection:
    """Test LLM cache metrics collection functionality."""
    
    def test_collect_cache_metrics_disabled(self):
        """Test metrics collection when cache is disabled."""
        with patch('iris_rag.monitoring.metrics_collector.get_global_cache_manager', return_value=None):
            collector = MetricsCollector()
            metrics = collector.collect_cache_metrics()
            
            assert metrics['llm_cache_enabled'] == 0.0
            assert metrics['llm_cache_configured'] == 0.0
    
    def test_collect_cache_metrics_enabled(self):
        """Test metrics collection when cache is enabled."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'iris',
            'configured': True,
            'metrics': {
                'hits': 150,
                'misses': 50,
                'total_requests': 200,
                'hit_rate': 0.75,
                'avg_response_time_cached': 45.0,
                'avg_response_time_uncached': 180.0
            },
            'backend_stats': {
                'cache_entries': 500,
                'memory_usage_mb': 25.5,
                'disk_usage_mb': 100.0
            }
        }
        
        with patch('iris_rag.monitoring.metrics_collector.get_global_cache_manager', return_value=mock_cache_manager):
            collector = MetricsCollector()
            metrics = collector.collect_cache_metrics()
            
            assert metrics['llm_cache_enabled'] == 1.0
            assert metrics['llm_cache_configured'] == 1.0
            assert metrics['llm_cache_hit_rate'] == 0.75
            assert metrics['llm_cache_total_requests'] == 200.0
            assert metrics['llm_cache_hits'] == 150.0
            assert metrics['llm_cache_misses'] == 50.0
            assert metrics['llm_cache_avg_response_time_cached_ms'] == 45.0
            assert metrics['llm_cache_avg_response_time_uncached_ms'] == 180.0
            assert metrics['llm_cache_speedup_ratio'] == 4.0  # 180/45
            
            # Backend metrics
            assert metrics['llm_cache_backend_cache_entries'] == 500.0
            assert metrics['llm_cache_backend_memory_usage_mb'] == 25.5
            assert metrics['llm_cache_backend_disk_usage_mb'] == 100.0
    
    def test_collect_cache_metrics_zero_cached_time(self):
        """Test metrics collection when cached time is zero."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'memory',
            'configured': True,
            'metrics': {
                'hits': 0,
                'misses': 10,
                'total_requests': 10,
                'hit_rate': 0.0,
                'avg_response_time_cached': 0.0,
                'avg_response_time_uncached': 200.0
            }
        }
        
        with patch('iris_rag.monitoring.metrics_collector.get_global_cache_manager', return_value=mock_cache_manager):
            collector = MetricsCollector()
            metrics = collector.collect_cache_metrics()
            
            assert metrics['llm_cache_speedup_ratio'] == 0.0
    
    def test_collect_cache_metrics_exception_handling(self):
        """Test metrics collection exception handling."""
        with patch('iris_rag.monitoring.metrics_collector.get_global_cache_manager', side_effect=Exception("Test error")):
            collector = MetricsCollector()
            metrics = collector.collect_cache_metrics()
            
            assert metrics['llm_cache_enabled'] == 0.0
            assert metrics['llm_cache_configured'] == 0.0
            assert metrics['llm_cache_error'] == 1.0
    
    def test_metrics_collector_registration(self):
        """Test that cache metrics can be registered with the collector."""
        collector = MetricsCollector()
        
        # Register cache metrics collector
        collector.register_collector('cache_metrics', collector.collect_cache_metrics)
        
        assert 'cache_metrics' in collector.collectors
        assert callable(collector.collectors['cache_metrics'])
    
    def test_automatic_metrics_collection(self):
        """Test automatic collection of cache metrics."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            'enabled': True,
            'backend': 'memory',
            'configured': True,
            'metrics': {
                'hits': 10,
                'misses': 5,
                'total_requests': 15,
                'hit_rate': 0.67,
                'avg_response_time_cached': 30.0,
                'avg_response_time_uncached': 150.0
            }
        }
        
        with patch('iris_rag.monitoring.metrics_collector.get_global_cache_manager', return_value=mock_cache_manager):
            collector = MetricsCollector()
            collector.register_collector('cache_metrics', collector.collect_cache_metrics)
            
            # Manually trigger collection (simulating automatic collection)
            collector._collect_all_metrics()
            
            # Check that metrics were added
            cache_metrics = collector.get_metrics(name_pattern='llm_cache')
            assert len(cache_metrics) > 0
            
            # Check specific metrics
            hit_rate_metrics = [m for m in cache_metrics if m.name == 'llm_cache_hit_rate']
            assert len(hit_rate_metrics) == 1
            assert hit_rate_metrics[0].value == 0.67


class TestLLMCacheMonitoringIntegration:
    """Test integration of cache monitoring with dashboard and other components."""
    
    def test_dashboard_cache_metrics_display(self):
        """Test that dashboard can display cache metrics."""
        # This would be an integration test that would require
        # the full dashboard setup, which is complex to mock
        # For now, we test that the method exists and can be called
        from scripts.monitoring_dashboard import MonitoringDashboard
        
        # Mock the dependencies
        with patch('scripts.monitoring_dashboard.HealthMonitor'), \
             patch('scripts.monitoring_dashboard.PerformanceMonitor'), \
             patch('scripts.monitoring_dashboard.MetricsCollector') as mock_collector_class:
            
            mock_collector = Mock()
            mock_collector.collect_cache_metrics.return_value = {
                'llm_cache_enabled': 1.0,
                'llm_cache_configured': 1.0,
                'llm_cache_hit_rate': 0.8,
                'llm_cache_total_requests': 100.0
            }
            mock_collector_class.return_value = mock_collector
            
            dashboard = MonitoringDashboard()
            
            # Test that the method exists and can be called
            assert hasattr(dashboard, '_display_cache_metrics')
            assert callable(dashboard._display_cache_metrics)
    
    def test_end_to_end_cache_monitoring_flow(self):
        """Test the complete flow from cache usage to monitoring display."""
        # Create a real cache manager with metrics
        config = CacheConfig(
            enabled=True,
            backend='memory',
            include_model_name=True
        )
        
        cache_manager = LangchainCacheManager(config)
        
        # Simulate some cache activity
        cache_manager.metrics.record_hit(50.0)
        cache_manager.metrics.record_hit(45.0)
        cache_manager.metrics.record_miss(200.0)
        cache_manager.metrics.record_miss(180.0)
        
        # Mock the global cache manager to return our test instance
        with patch('iris_rag.monitoring.health_monitor.get_global_cache_manager', return_value=cache_manager), \
             patch('iris_rag.monitoring.metrics_collector.get_global_cache_manager', return_value=cache_manager):
            
            # Test health monitoring
            health_monitor = HealthMonitor()
            health_result = health_monitor.check_llm_cache_performance()
            
            assert health_result.status == 'healthy'  # 50% hit rate is healthy (above 30% threshold)
            assert health_result.metrics['hit_rate'] == 0.5
            assert health_result.metrics['total_requests'] == 4
            
            # Test metrics collection
            metrics_collector = MetricsCollector()
            cache_metrics = metrics_collector.collect_cache_metrics()
            
            assert cache_metrics['llm_cache_enabled'] == 1.0
            assert cache_metrics['llm_cache_hit_rate'] == 0.5
            assert cache_metrics['llm_cache_total_requests'] == 4.0
            assert cache_metrics['llm_cache_speedup_ratio'] > 3.0  # Should be around 4x


if __name__ == '__main__':
    pytest.main([__file__])