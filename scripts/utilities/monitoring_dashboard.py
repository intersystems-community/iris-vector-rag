"""
Real-time Monitoring Dashboard for RAG Templates System

Provides a real-time dashboard for monitoring system health and performance.
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iris_rag.monitoring.health_monitor import HealthMonitor
from iris_rag.monitoring.performance_monitor import PerformanceMonitor
from iris_rag.monitoring.metrics_collector import MetricsCollector
from iris_rag.config.manager import ConfigurationManager

class MonitoringDashboard:
    """
    Real-time monitoring dashboard for the RAG system.
    """
    
    def __init__(self, config_path: str = None, refresh_interval: int = 30):
        """
        Initialize the monitoring dashboard.
        
        Args:
            config_path: Path to configuration file
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.config_manager = ConfigurationManager(config_path)
        self.health_monitor = HealthMonitor(self.config_manager)
        self.performance_monitor = PerformanceMonitor(self.config_manager)
        self.metrics_collector = MetricsCollector()
        self.refresh_interval = refresh_interval
        self.running = False
    
    def start_dashboard(self):
        """Start the real-time dashboard."""
        print("🚀 Starting RAG System Monitoring Dashboard...")
        print(f"Refresh interval: {self.refresh_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        # Start monitoring components
        self.performance_monitor.start_monitoring()
        
        # Register cache metrics collector
        self.metrics_collector.register_collector('cache_metrics', self.metrics_collector.collect_cache_metrics)
        
        self.metrics_collector.start_collection()
        
        self.running = True
        
        try:
            while self.running:
                self._display_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped by user")
        finally:
            self._cleanup()
    
    def _display_dashboard(self):
        """Display the current dashboard."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Header
        print("="*80)
        print("🏥 RAG SYSTEM MONITORING DASHBOARD")
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        try:
            # System Health
            self._display_health_status()
            
            # Performance Metrics
            self._display_performance_metrics()
            
            # System Resources
            self._display_system_resources()
            
            # LLM Cache Metrics
            self._display_cache_metrics()
            
            # Recent Activity
            self._display_recent_activity()
            
        except Exception as e:
            print(f"❌ Error updating dashboard: {e}")
        
        print("="*80)
        print(f"Next refresh in {self.refresh_interval} seconds... (Press Ctrl+C to stop)")
    
    def _display_health_status(self):
        """Display system health status."""
        print("\n🏥 SYSTEM HEALTH")
        print("-" * 40)
        
        try:
            health_results = self.health_monitor.run_comprehensive_health_check()
            overall_status = self.health_monitor.get_overall_health_status(health_results)
            
            # Overall status
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌'
            }
            
            print(f"Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
            print()
            
            # Component status
            for component, result in health_results.items():
                emoji = status_emoji.get(result.status, '❓')
                duration = f"({result.duration_ms:.1f}ms)"
                print(f"  {emoji} {component.replace('_', ' ').title()}: {result.status.upper()} {duration}")
                
                # Show key metrics
                if result.metrics:
                    for key, value in list(result.metrics.items())[:2]:  # Show first 2 metrics
                        if isinstance(value, (int, float)):
                            if 'percent' in key:
                                print(f"    └─ {key}: {value:.1f}%")
                            elif 'count' in key:
                                print(f"    └─ {key}: {value:,}")
                            else:
                                print(f"    └─ {key}: {value}")
            
        except Exception as e:
            print(f"❌ Failed to get health status: {e}")
    
    def _display_performance_metrics(self):
        """Display performance metrics."""
        print("\n📊 PERFORMANCE METRICS (Last 5 minutes)")
        print("-" * 40)
        
        try:
            summary = self.performance_monitor.get_performance_summary(5)
            
            if summary.get('total_queries', 0) > 0:
                print(f"Total Queries: {summary['total_queries']}")
                print(f"Success Rate: {summary['success_rate']:.1f}%")
                print(f"Failed Queries: {summary['failed_queries']}")
                
                exec_stats = summary.get('execution_time_stats', {})
                if exec_stats:
                    print(f"Avg Execution Time: {exec_stats.get('avg_ms', 0):.1f}ms")
                    print(f"P95 Execution Time: {exec_stats.get('p95_ms', 0):.1f}ms")
                    print(f"Max Execution Time: {exec_stats.get('max_ms', 0):.1f}ms")
                
                # Pipeline breakdown
                pipeline_perf = summary.get('pipeline_performance', {})
                if pipeline_perf:
                    print("\nPipeline Performance:")
                    for pipeline, stats in pipeline_perf.items():
                        print(f"  • {pipeline}: {stats['query_count']} queries, "
                              f"{stats['avg_execution_time_ms']:.1f}ms avg")
            else:
                print("No queries in the last 5 minutes")
            
        except Exception as e:
            print(f"❌ Failed to get performance metrics: {e}")
    
    def _display_system_resources(self):
        """Display system resource usage."""
        print("\n💻 SYSTEM RESOURCES")
        print("-" * 40)
        
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_emoji = "🔴" if cpu_percent > 90 else "🟡" if cpu_percent > 70 else "🟢"
            print(f"{cpu_emoji} CPU Usage: {cpu_percent:.1f}%")
            
            # Memory
            memory = psutil.virtual_memory()
            memory_emoji = "🔴" if memory.percent > 90 else "🟡" if memory.percent > 70 else "🟢"
            print(f"{memory_emoji} Memory Usage: {memory.percent:.1f}% "
                  f"({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_emoji = "🔴" if disk.percent > 90 else "🟡" if disk.percent > 80 else "🟢"
            print(f"{disk_emoji} Disk Usage: {disk.percent:.1f}% "
                  f"({disk.free / (1024**3):.1f}GB free)")
            
            # Docker container (if available)
            try:
                import docker
                client = docker.from_env()
                containers = client.containers.list()
                iris_container = None
                
                for container in containers:
                    if 'iris' in container.name.lower():
                        iris_container = container
                        break
                
                if iris_container:
                    container_emoji = "🟢" if iris_container.status == 'running' else "🔴"
                    print(f"{container_emoji} IRIS Container: {iris_container.status}")
                else:
                    print("🟡 IRIS Container: Not found")
                    
            except Exception:
                print("❓ Docker: Not available")
            
        except Exception as e:
            print(f"❌ Failed to get system resources: {e}")
    
    def _display_cache_metrics(self):
        """Display LLM cache performance metrics."""
        print("\n🧠 LLM CACHE PERFORMANCE")
        print("-" * 40)
        
        try:
            # Collect cache metrics
            cache_metrics = self.metrics_collector.collect_cache_metrics()
            
            if cache_metrics.get('llm_cache_enabled', 0) == 0:
                print("🔴 LLM Cache: Disabled or not configured")
                return
            
            # Cache status
            configured = cache_metrics.get('llm_cache_configured', 0) == 1
            status_emoji = "🟢" if configured else "🟡"
            print(f"{status_emoji} Cache Status: {'Configured' if configured else 'Not Configured'}")
            
            # Hit rate with color coding
            hit_rate = cache_metrics.get('llm_cache_hit_rate', 0.0)
            total_requests = int(cache_metrics.get('llm_cache_total_requests', 0))
            
            if total_requests > 0:
                hit_rate_emoji = "🟢" if hit_rate >= 0.5 else "🟡" if hit_rate >= 0.3 else "🔴"
                print(f"{hit_rate_emoji} Hit Rate: {hit_rate:.1%} ({total_requests:,} total requests)")
                
                hits = int(cache_metrics.get('llm_cache_hits', 0))
                misses = int(cache_metrics.get('llm_cache_misses', 0))
                print(f"  └─ Hits: {hits:,}, Misses: {misses:,}")
                
                # Response time comparison
                cached_time = cache_metrics.get('llm_cache_avg_response_time_cached_ms', 0.0)
                uncached_time = cache_metrics.get('llm_cache_avg_response_time_uncached_ms', 0.0)
                speedup = cache_metrics.get('llm_cache_speedup_ratio', 0.0)
                
                if cached_time > 0 and uncached_time > 0:
                    speedup_emoji = "🟢" if speedup >= 3 else "🟡" if speedup >= 2 else "🔴"
                    print(f"{speedup_emoji} Performance Speedup: {speedup:.1f}x")
                    print(f"  └─ Cached: {cached_time:.1f}ms, Uncached: {uncached_time:.1f}ms")
                
            else:
                print("🟡 No cache requests recorded yet")
            
            # Backend-specific metrics
            backend_metrics = {k: v for k, v in cache_metrics.items() if k.startswith('llm_cache_backend_')}
            if backend_metrics:
                print("\nBackend Metrics:")
                for key, value in backend_metrics.items():
                    metric_name = key.replace('llm_cache_backend_', '').replace('_', ' ').title()
                    if isinstance(value, float):
                        print(f"  • {metric_name}: {value:.2f}")
                    else:
                        print(f"  • {metric_name}: {value}")
            
        except Exception as e:
            print(f"❌ Failed to get cache metrics: {e}")
    
    def _display_recent_activity(self):
        """Display recent system activity."""
        print("\n📈 RECENT ACTIVITY")
        print("-" * 40)
        
        try:
            # Get recent metrics
            metrics_summary = self.metrics_collector.get_metric_summary(timedelta(minutes=5))
            
            print(f"Metrics Collected (5min): {metrics_summary.get('total_metrics', 0)}")
            print(f"Unique Metric Types: {metrics_summary.get('unique_metric_names', 0)}")
            
            # Show some key metrics
            metric_stats = metrics_summary.get('metric_statistics', {})
            
            # Database metrics
            if 'database_document_count' in metric_stats:
                doc_count = metric_stats['database_document_count'].get('latest', 0)
                print(f"Documents in Database: {doc_count:,}")
            
            if 'database_embedded_document_count' in metric_stats:
                embedded_count = metric_stats['database_embedded_document_count'].get('latest', 0)
                print(f"Embedded Documents: {embedded_count:,}")
            
            if 'database_vector_query_time_ms' in metric_stats:
                query_time = metric_stats['database_vector_query_time_ms'].get('latest', 0)
                query_emoji = "🔴" if query_time > 1000 else "🟡" if query_time > 500 else "🟢"
                print(f"{query_emoji} Vector Query Time: {query_time:.1f}ms")
            
            # Performance monitoring status
            perf_status = self.performance_monitor.get_real_time_status()
            print(f"Performance Monitoring: {'🟢 Active' if perf_status['monitoring_active'] else '🔴 Inactive'}")
            print(f"Query Buffer Size: {perf_status.get('query_data_size', 0)}")
            
        except Exception as e:
            print(f"❌ Failed to get recent activity: {e}")
    
    def _cleanup(self):
        """Cleanup monitoring components."""
        try:
            self.performance_monitor.stop_monitoring()
            self.metrics_collector.stop_collection()
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
    
    def export_current_status(self, filepath: str = None):
        """Export current system status to a file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"reports/dashboard_status_{timestamp}.json"
        
        try:
            # Collect all current data
            health_results = self.health_monitor.run_comprehensive_health_check()
            performance_summary = self.performance_monitor.get_performance_summary(60)
            metrics_summary = self.metrics_collector.get_metric_summary(timedelta(hours=1))
            
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'overall_health': self.health_monitor.get_overall_health_status(health_results),
                'health_details': {
                    name: {
                        'status': result.status,
                        'message': result.message,
                        'metrics': result.metrics,
                        'duration_ms': result.duration_ms
                    }
                    for name, result in health_results.items()
                },
                'performance_summary': performance_summary,
                'metrics_summary': metrics_summary,
                'system_info': self._get_system_info()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            print(f"✅ Status exported to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to export status: {e}")
            return None
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            import psutil
            import platform
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
            }
        except Exception:
            return {}

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Monitoring Dashboard")
    parser.add_argument(
        '--refresh-interval',
        type=int,
        default=30,
        help='Dashboard refresh interval in seconds (default: 30)'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--export-status',
        action='store_true',
        help='Export current status and exit'
    )
    parser.add_argument(
        '--export-file',
        help='File path for status export'
    )
    
    args = parser.parse_args()
    
    try:
        dashboard = MonitoringDashboard(args.config, args.refresh_interval)
        
        if args.export_status:
            filepath = dashboard.export_current_status(args.export_file)
            if filepath:
                print(f"Status exported to: {filepath}")
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            dashboard.start_dashboard()
            
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()