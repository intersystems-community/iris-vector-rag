"""
Monitoring Dashboard for RAG System

This module provides a dashboard interface for monitoring system health,
performance metrics, and cache statistics.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Dashboard for monitoring RAG system health and performance."""
    
    def __init__(self, config_manager=None):
        """Initialize the monitoring dashboard.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logger
        
    def display_cache_metrics(self, metrics: Dict[str, Any]) -> str:
        """Display cache metrics in a formatted way.
        
        Args:
            metrics: Dictionary containing cache metrics
            
        Returns:
            Formatted string representation of metrics
        """
        if not metrics:
            return "No cache metrics available"
            
        output = []
        output.append("=== Cache Metrics ===")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                output.append(f"{key}: {value:.2f}")
            else:
                output.append(f"{key}: {value}")
                
        return "\n".join(output)
        
    def _display_cache_metrics(self, metrics: Dict[str, Any]) -> str:
        """Private method for displaying cache metrics (used by tests).
        
        Args:
            metrics: Dictionary containing cache metrics
            
        Returns:
            Formatted string representation of metrics
        """
        return self.display_cache_metrics(metrics)
        
    def display_health_status(self, health_report: Dict[str, Any]) -> str:
        """Display system health status.
        
        Args:
            health_report: Dictionary containing health check results
            
        Returns:
            Formatted string representation of health status
        """
        if not health_report:
            return "No health data available"
            
        output = []
        output.append("=== System Health ===")
        
        overall_status = health_report.get('overall_status', 'unknown')
        output.append(f"Overall Status: {overall_status}")
        
        checks = health_report.get('checks', {})
        for check_name, result in checks.items():
            status = result.get('status', 'unknown')
            output.append(f"{check_name}: {status}")
            
        return "\n".join(output)
        
    def generate_report(self, include_cache: bool = True, include_health: bool = True) -> str:
        """Generate a comprehensive monitoring report.
        
        Args:
            include_cache: Whether to include cache metrics
            include_health: Whether to include health status
            
        Returns:
            Formatted monitoring report
        """
        output = []
        output.append(f"=== Monitoring Report - {datetime.now().isoformat()} ===")
        
        if include_cache:
            # This would normally fetch real cache metrics
            cache_metrics = {
                "hit_rate": 0.85,
                "total_requests": 1000,
                "cache_hits": 850,
                "cache_misses": 150
            }
            output.append(self.display_cache_metrics(cache_metrics))
            
        if include_health:
            # This would normally fetch real health status
            health_report = {
                "overall_status": "healthy",
                "checks": {
                    "database": {"status": "healthy"},
                    "cache": {"status": "healthy"},
                    "memory": {"status": "warning"}
                }
            }
            output.append(self.display_health_status(health_report))
            
        return "\n\n".join(output)