"""
LLM Cache Monitoring Demonstration Script

This script demonstrates the LLM cache monitoring capabilities by:
1. Setting up a cache with some simulated activity
2. Running health checks
3. Collecting metrics
4. Displaying dashboard-style output
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iris_rag.monitoring.health_monitor import HealthMonitor
from iris_rag.monitoring.metrics_collector import MetricsCollector
from common.llm_cache_manager import LangchainCacheManager
from common.llm_cache_config import CacheConfig


def simulate_cache_activity(cache_manager: LangchainCacheManager, num_requests: int = 50):
    """Simulate cache activity with hits and misses."""
    print(f"🔄 Simulating {num_requests} cache requests...")
    
    # Simulate cache hits (faster responses)
    for i in range(int(num_requests * 0.7)):  # 70% hit rate
        response_time = 30 + (i % 20)  # 30-50ms
        cache_manager.metrics.record_hit(response_time)
    
    # Simulate cache misses (slower responses)
    for i in range(int(num_requests * 0.3)):  # 30% miss rate
        response_time = 150 + (i % 100)  # 150-250ms
        cache_manager.metrics.record_miss(response_time)
    
    print(f"✅ Simulated {cache_manager.metrics.total_requests} total requests")
    print(f"   Hit rate: {cache_manager.metrics.hit_rate:.1%}")
    print(f"   Avg cached time: {cache_manager.metrics.avg_response_time_cached:.1f}ms")
    print(f"   Avg uncached time: {cache_manager.metrics.avg_response_time_uncached:.1f}ms")


def demonstrate_health_monitoring(cache_manager: LangchainCacheManager):
    """Demonstrate health monitoring capabilities."""
    print("\n🏥 HEALTH MONITORING DEMONSTRATION")
    print("=" * 50)
    
    # Mock the global cache manager for health monitoring
    import iris_rag.monitoring.health_monitor as health_module
    original_get_cache = health_module.get_global_cache_manager
    health_module.get_global_cache_manager = lambda: cache_manager
    
    try:
        health_monitor = HealthMonitor()
        
        # Run cache-specific health check
        print("Running LLM cache health check...")
        cache_health = health_monitor.check_llm_cache_performance()
        
        # Display results
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '❌'
        }
        
        print(f"\nCache Health Status: {status_emoji.get(cache_health.status, '❓')} {cache_health.status.upper()}")
        print(f"Message: {cache_health.message}")
        print(f"Check Duration: {cache_health.duration_ms:.1f}ms")
        
        print("\nKey Metrics:")
        for key, value in cache_health.metrics.items():
            if isinstance(value, float):
                if 'rate' in key or 'percent' in key:
                    print(f"  • {key}: {value:.1%}")
                elif 'time' in key:
                    print(f"  • {key}: {value:.1f}ms")
                elif 'ratio' in key:
                    print(f"  • {key}: {value:.1f}x")
                else:
                    print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")
        
        # Run comprehensive health check
        print("\n" + "-" * 30)
        print("Running comprehensive health check (cache included)...")
        
        # Mock other health checks to avoid dependencies
        def mock_health_check():
            from iris_rag.monitoring.health_monitor import HealthCheckResult
            return HealthCheckResult(
                component='mock',
                status='healthy',
                message='Mock check',
                metrics={},
                timestamp=datetime.now(),
                duration_ms=10.0
            )
        
        health_monitor.check_system_resources = mock_health_check
        health_monitor.check_database_connectivity = mock_health_check
        health_monitor.check_docker_containers = mock_health_check
        health_monitor.check_vector_performance = mock_health_check
        
        all_health = health_monitor.run_comprehensive_health_check()
        overall_status = health_monitor.get_overall_health_status(all_health)
        
        print(f"\nOverall System Health: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        print("Component Status:")
        for component, result in all_health.items():
            emoji = status_emoji.get(result.status, '❓')
            print(f"  {emoji} {component.replace('_', ' ').title()}: {result.status}")
    
    finally:
        # Restore original function
        health_module.get_global_cache_manager = original_get_cache


def demonstrate_metrics_collection(cache_manager: LangchainCacheManager):
    """Demonstrate metrics collection capabilities."""
    print("\n📊 METRICS COLLECTION DEMONSTRATION")
    print("=" * 50)
    
    # Mock the global cache manager for metrics collection
    import iris_rag.monitoring.metrics_collector as metrics_module
    original_get_cache = metrics_module.get_global_cache_manager
    metrics_module.get_global_cache_manager = lambda: cache_manager
    
    try:
        metrics_collector = MetricsCollector()
        
        print("Collecting cache metrics...")
        cache_metrics = metrics_collector.collect_cache_metrics()
        
        print("\nCollected Metrics:")
        for key, value in cache_metrics.items():
            if isinstance(value, float):
                if 'rate' in key:
                    print(f"  • {key}: {value:.1%}")
                elif 'time' in key and 'ms' in key:
                    print(f"  • {key}: {value:.1f}ms")
                elif 'ratio' in key:
                    print(f"  • {key}: {value:.1f}x")
                elif value == 0.0 or value == 1.0:
                    print(f"  • {key}: {'Yes' if value == 1.0 else 'No'}")
                else:
                    print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")
        
        # Register and test automatic collection
        print("\n" + "-" * 30)
        print("Testing automatic metrics collection...")
        
        metrics_collector.register_collector('cache_metrics', metrics_collector.collect_cache_metrics)
        
        # Manually trigger collection (simulating automatic collection)
        metrics_collector._collect_all_metrics()
        
        # Check collected metrics
        recent_metrics = metrics_collector.get_metrics(name_pattern='llm_cache')
        print(f"Collected {len(recent_metrics)} cache-related metrics")
        
        if recent_metrics:
            print("Sample metrics:")
            for metric in recent_metrics[:5]:  # Show first 5
                print(f"  • {metric.name}: {metric.value} (at {metric.timestamp.strftime('%H:%M:%S')})")
    
    finally:
        # Restore original function
        metrics_module.get_global_cache_manager = original_get_cache


def demonstrate_dashboard_display(cache_manager: LangchainCacheManager):
    """Demonstrate dashboard-style display."""
    print("\n🖥️  DASHBOARD DISPLAY DEMONSTRATION")
    print("=" * 50)
    
    # Mock the global cache manager for dashboard
    import iris_rag.monitoring.metrics_collector as metrics_module
    original_get_cache = metrics_module.get_global_cache_manager
    metrics_module.get_global_cache_manager = lambda: cache_manager
    
    try:
        from scripts.utilities.monitoring_dashboard import MonitoringDashboard
        
        # Create dashboard instance (with mocked dependencies)
        class MockHealthMonitor:
            def __init__(self, *args, **kwargs):
                pass
        
        class MockPerformanceMonitor:
            def __init__(self, *args, **kwargs):
                pass
            def start_monitoring(self):
                pass
        
        # Mock the imports to avoid full initialization
        import scripts.utilities.monitoring_dashboard as dashboard_module
        dashboard_module.HealthMonitor = MockHealthMonitor
        dashboard_module.PerformanceMonitor = MockPerformanceMonitor
        
        dashboard = MonitoringDashboard()
        
        print("Displaying cache metrics (dashboard style):")
        print("-" * 40)
        
        # Call the cache metrics display method
        dashboard._display_cache_metrics()
    
    except Exception as e:
        print(f"Dashboard demo encountered an issue: {e}")
        print("This is expected in a demo environment without full system setup.")
    
    finally:
        # Restore original function
        metrics_module.get_global_cache_manager = original_get_cache


def main():
    """Main demonstration function."""
    print("🚀 LLM CACHE MONITORING DEMONSTRATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create cache configuration
    print("\n1️⃣  Setting up cache configuration...")
    config = CacheConfig(
        enabled=True,
        backend='memory',
        include_model_name=True,
        include_temperature=True
    )
    
    # Create cache manager
    print("2️⃣  Creating cache manager...")
    cache_manager = LangchainCacheManager(config)
    
    # Simulate cache activity
    print("\n3️⃣  Simulating cache activity...")
    simulate_cache_activity(cache_manager, num_requests=100)
    
    # Demonstrate health monitoring
    demonstrate_health_monitoring(cache_manager)
    
    # Demonstrate metrics collection
    demonstrate_metrics_collection(cache_manager)
    
    # Demonstrate dashboard display
    demonstrate_dashboard_display(cache_manager)
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("\nTo see the full monitoring dashboard in action:")
    print("  python scripts/monitoring_dashboard.py")
    print("\nTo run the monitoring tests:")
    print("  python -m pytest tests/test_llm_cache_monitoring.py -v")


if __name__ == "__main__":
    main()