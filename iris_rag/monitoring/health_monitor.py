"""
Health Monitor for RAG Templates System

Provides comprehensive health checking capabilities for all system components.
"""

import logging
import time
import psutil
import docker
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from common.llm_cache_manager import get_global_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    duration_ms: float

class HealthMonitor:
    """
    Comprehensive health monitoring for the RAG system.
    
    Monitors system resources, database connectivity, container status,
    and application-specific health indicators.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the health monitor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.connection_manager = ConnectionManager(self.config_manager)
        self.docker_client = None
        
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client initialization failed: {e}")
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        start_time = time.time()
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine status
            status = 'healthy'
            issues = []
            
            if memory_percent > 90:
                status = 'critical'
                issues.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > 80:
                status = 'warning'
                issues.append(f"Memory usage high: {memory_percent:.1f}%")
            
            if cpu_percent > 90:
                status = 'critical'
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = 'warning'
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if disk_percent > 95:
                status = 'critical'
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 85:
                status = 'warning'
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            
            metrics = {
                'memory_percent': memory_percent,
                'memory_gb_used': memory.used / (1024**3),
                'memory_gb_total': memory.total / (1024**3),
                'cpu_percent': cpu_percent,
                'disk_percent': disk_percent,
                'disk_gb_free': disk.free / (1024**3),
                'disk_gb_total': disk.total / (1024**3)
            }
            
        except Exception as e:
            status = 'critical'
            message = f"Failed to check system resources: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            component='system_resources',
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(),
            duration_ms=duration_ms
        )
    
    def check_database_connectivity(self) -> HealthCheckResult:
        """Check database connectivity and basic operations."""
        start_time = time.time()
        
        try:
            # Test connection
            connection = self.connection_manager.get_connection('iris')
            
            with connection.cursor() as cursor:
                # Basic connectivity test
                cursor.execute("SELECT 1 AS test")
                result = cursor.fetchone()
                
                if not result or result[0] != 1:
                    raise Exception("Basic connectivity test failed")
                
                # Check schema
                cursor.execute(
                    "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'RAG'"
                )
                table_count = cursor.fetchone()[0]
                
                # Test vector operations
                cursor.execute("SELECT TO_VECTOR('[0.1, 0.2, 0.3]') AS test_vector")
                vector_result = cursor.fetchone()
                
                if not vector_result:
                    raise Exception("Vector operations test failed")
                
                # Check document count
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                doc_count = cursor.fetchone()[0]
                
                # Check embedding count
                cursor.execute(
                    "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL"
                )
                embedded_count = cursor.fetchone()[0]
            
            status = 'healthy'
            message = "Database connectivity and operations healthy"
            
            metrics = {
                'table_count': table_count,
                'document_count': doc_count,
                'embedded_document_count': embedded_count,
                'embedding_completion_percent': (embedded_count / doc_count * 100) if doc_count > 0 else 0
            }
            
        except Exception as e:
            status = 'critical'
            message = f"Database connectivity failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            component='database_connectivity',
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(),
            duration_ms=duration_ms
        )
    
    def check_docker_containers(self) -> HealthCheckResult:
        """Check Docker container status."""
        start_time = time.time()
        
        if not self.docker_client:
            return HealthCheckResult(
                component='docker_containers',
                status='warning',
                message="Docker client not available",
                metrics={},
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            containers = self.docker_client.containers.list(all=True)
            iris_container = None
            
            for container in containers:
                if 'iris' in container.name.lower():
                    iris_container = container
                    break
            
            if not iris_container:
                status = 'critical'
                message = "IRIS container not found"
                metrics = {}
            elif iris_container.status != 'running':
                status = 'critical'
                message = f"IRIS container not running (status: {iris_container.status})"
                metrics = {'container_status': iris_container.status}
            else:
                # Get container stats
                stats = iris_container.stats(stream=False)
                memory_usage = stats['memory_stats']['usage'] / (1024**3)
                memory_limit = stats['memory_stats']['limit'] / (1024**3)
                memory_percent = (memory_usage / memory_limit) * 100
                
                status = 'healthy'
                if memory_percent > 90:
                    status = 'warning'
                    message = f"IRIS container memory usage high: {memory_percent:.1f}%"
                else:
                    message = "IRIS container healthy"
                
                metrics = {
                    'container_status': iris_container.status,
                    'memory_usage_gb': memory_usage,
                    'memory_limit_gb': memory_limit,
                    'memory_percent': memory_percent
                }
            
        except Exception as e:
            status = 'critical'
            message = f"Failed to check Docker containers: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            component='docker_containers',
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(),
            duration_ms=duration_ms
        )
    
    def check_vector_performance(self) -> HealthCheckResult:
        """Check vector query performance."""
        start_time = time.time()
        
        try:
            connection = self.connection_manager.get_connection('iris')
            
            with connection.cursor() as cursor:
                # Check if we have enough data for testing
                cursor.execute(
                    "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL"
                )
                embedded_count = cursor.fetchone()[0]
                
                if embedded_count < 10:
                    status = 'warning'
                    message = f"Insufficient embedded documents for performance testing: {embedded_count}"
                    metrics = {'embedded_document_count': embedded_count}
                else:
                    # Test vector query performance
                    test_vector = "[" + ",".join(["0.1"] * 384) + "]"
                    
                    query_start = time.time()
                    cursor.execute("""
                        SELECT TOP 10 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                        FROM RAG.SourceDocuments 
                        WHERE embedding IS NOT NULL
                        ORDER BY similarity DESC
                    """, (test_vector,))
                    
                    results = cursor.fetchall()
                    query_time_ms = (time.time() - query_start) * 1000
                    
                    # Determine status based on performance
                    if query_time_ms < 100:
                        status = 'healthy'
                        message = f"Vector query performance excellent: {query_time_ms:.1f}ms"
                    elif query_time_ms < 500:
                        status = 'healthy'
                        message = f"Vector query performance good: {query_time_ms:.1f}ms"
                    elif query_time_ms < 1000:
                        status = 'warning'
                        message = f"Vector query performance slow: {query_time_ms:.1f}ms"
                    else:
                        status = 'critical'
                        message = f"Vector query performance critical: {query_time_ms:.1f}ms"
                    
                    metrics = {
                        'embedded_document_count': embedded_count,
                        'query_time_ms': query_time_ms,
                        'results_returned': len(results)
                    }
            
        except Exception as e:
            status = 'critical'
            message = f"Vector performance check failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            component='vector_performance',
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(),
            duration_ms=duration_ms
        )
    
    def check_llm_cache_performance(self) -> HealthCheckResult:
        """Check LLM cache performance and health."""
        start_time = time.time()
        
        try:
            cache_manager = get_global_cache_manager()
            
            if not cache_manager:
                status = 'warning'
                message = "LLM cache not configured or disabled"
                metrics = {
                    'cache_enabled': False,
                    'cache_configured': False
                }
            else:
                # Get cache statistics
                cache_stats = cache_manager.get_cache_stats()
                
                # Extract metrics
                cache_metrics = cache_stats.get('metrics', {})
                hit_rate = cache_metrics.get('hit_rate', 0.0)
                total_requests = cache_metrics.get('total_requests', 0)
                avg_cached_time = cache_metrics.get('avg_response_time_cached', 0.0)
                avg_uncached_time = cache_metrics.get('avg_response_time_uncached', 0.0)
                
                # Determine status based on performance thresholds
                status = 'healthy'
                issues = []
                
                # Check hit rate (warning if < 30%, critical if < 10%)
                if total_requests > 10:  # Only check if we have meaningful data
                    if hit_rate < 0.1:
                        status = 'critical'
                        issues.append(f"Cache hit rate critically low: {hit_rate:.1%}")
                    elif hit_rate < 0.3:
                        status = 'warning'
                        issues.append(f"Cache hit rate low: {hit_rate:.1%}")
                
                # Check response time performance (cached should be much faster)
                if avg_cached_time > 0 and avg_uncached_time > 0:
                    speedup_ratio = avg_uncached_time / avg_cached_time
                    if speedup_ratio < 2:  # Cache should provide at least 2x speedup
                        if status != 'critical':
                            status = 'warning'
                        issues.append(f"Cache speedup low: {speedup_ratio:.1f}x")
                
                # Check if cache is actually being used
                if total_requests == 0:
                    status = 'warning'
                    issues.append("No cache requests recorded")
                
                message = "LLM cache healthy" if not issues else "; ".join(issues)
                
                metrics = {
                    'cache_enabled': cache_stats.get('enabled', False),
                    'cache_backend': cache_stats.get('backend', 'unknown'),
                    'cache_configured': cache_stats.get('configured', False),
                    'hit_rate': hit_rate,
                    'total_requests': total_requests,
                    'cache_hits': cache_metrics.get('hits', 0),
                    'cache_misses': cache_metrics.get('misses', 0),
                    'avg_response_time_cached_ms': avg_cached_time,
                    'avg_response_time_uncached_ms': avg_uncached_time,
                    'cache_speedup_ratio': avg_uncached_time / avg_cached_time if avg_cached_time > 0 else 0
                }
                
                # Add backend-specific stats if available
                backend_stats = cache_stats.get('backend_stats', {})
                if backend_stats:
                    metrics.update({f'backend_{k}': v for k, v in backend_stats.items()})
            
        except Exception as e:
            status = 'critical'
            message = f"LLM cache health check failed: {e}"
            metrics = {}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            component='llm_cache_performance',
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(),
            duration_ms=duration_ms
        )
    
    def run_comprehensive_health_check(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        logger.info("Starting comprehensive health check...")
        
        checks = {
            'system_resources': self.check_system_resources,
            'database_connectivity': self.check_database_connectivity,
            'docker_containers': self.check_docker_containers,
            'vector_performance': self.check_vector_performance,
            'llm_cache_performance': self.check_llm_cache_performance
        }
        
        results = {}
        
        for check_name, check_func in checks.items():
            logger.info(f"Running {check_name} check...")
            try:
                results[check_name] = check_func()
            except Exception as e:
                logger.error(f"Health check {check_name} failed with exception: {e}")
                results[check_name] = HealthCheckResult(
                    component=check_name,
                    status='critical',
                    message=f"Check failed with exception: {e}",
                    metrics={},
                    timestamp=datetime.now(),
                    duration_ms=0
                )
        
        # Log summary
        healthy_count = sum(1 for r in results.values() if r.status == 'healthy')
        warning_count = sum(1 for r in results.values() if r.status == 'warning')
        critical_count = sum(1 for r in results.values() if r.status == 'critical')
        
        logger.info(f"Health check complete: {healthy_count} healthy, {warning_count} warnings, {critical_count} critical")
        
        return results
    
    def get_overall_health_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Determine overall system health status."""
        if any(r.status == 'critical' for r in results.values()):
            return 'critical'
        elif any(r.status == 'warning' for r in results.values()):
            return 'warning'
        else:
            return 'healthy'