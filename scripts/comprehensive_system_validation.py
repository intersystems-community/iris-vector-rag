"""
Comprehensive System Validation Script

This script provides a complete validation of the RAG templates system,
including health checks, performance monitoring, and system validation.
"""

import sys
import os
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iris_rag.monitoring.health_monitor import HealthMonitor
from iris_rag.monitoring.performance_monitor import PerformanceMonitor
from iris_rag.monitoring.system_validator import SystemValidator
from iris_rag.monitoring.metrics_collector import MetricsCollector
from iris_rag.config.manager import ConfigurationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/system_validation.log')
    ]
)

logger = logging.getLogger(__name__)

class ComprehensiveValidator:
    """
    Comprehensive system validator that orchestrates all monitoring components.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the comprehensive validator."""
        self.config_manager = ConfigurationManager(config_path)
        self.health_monitor = HealthMonitor(self.config_manager)
        self.performance_monitor = PerformanceMonitor(self.config_manager)
        self.system_validator = SystemValidator(self.config_manager)
        self.metrics_collector = MetricsCollector()
        
        # Setup metrics collectors
        self._setup_metrics_collectors()
    
    def _setup_metrics_collectors(self):
        """Setup automatic metrics collection."""
        def collect_health_metrics():
            """Collect health metrics."""
            try:
                health_results = self.health_monitor.run_comprehensive_health_check()
                metrics = {}
                
                for component, result in health_results.items():
                    # Convert status to numeric
                    status_value = {'healthy': 1, 'warning': 0.5, 'critical': 0}.get(result.status, 0)
                    metrics[f'health_{component}_status'] = status_value
                    metrics[f'health_{component}_duration_ms'] = result.duration_ms
                    
                    # Add specific metrics from health check details
                    for metric_name, metric_value in result.metrics.items():
                        if isinstance(metric_value, (int, float)):
                            metrics[f'health_{component}_{metric_name}'] = metric_value
                
                return metrics
            except Exception as e:
                logger.error(f"Error collecting health metrics: {e}")
                return {}
        
        def collect_performance_metrics():
            """Collect performance metrics."""
            try:
                summary = self.performance_monitor.get_performance_summary(5)  # Last 5 minutes
                metrics = {}
                
                if summary.get('total_queries', 0) > 0:
                    exec_stats = summary.get('execution_time_stats', {})
                    metrics.update({
                        'performance_total_queries': summary.get('total_queries', 0),
                        'performance_success_rate': summary.get('success_rate', 0),
                        'performance_avg_execution_time_ms': exec_stats.get('avg_ms', 0),
                        'performance_p95_execution_time_ms': exec_stats.get('p95_ms', 0),
                        'performance_p99_execution_time_ms': exec_stats.get('p99_ms', 0)
                    })
                
                return metrics
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
                return {}
        
        # Register collectors
        self.metrics_collector.register_collector('health', collect_health_metrics)
        self.metrics_collector.register_collector('performance', collect_performance_metrics)
    
    def run_quick_validation(self) -> dict:
        """Run a quick validation check."""
        logger.info("🚀 Starting quick system validation...")
        
        results = {
            'validation_type': 'quick',
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        try:
            # Health check
            logger.info("Running health checks...")
            health_results = self.health_monitor.run_comprehensive_health_check()
            overall_health = self.health_monitor.get_overall_health_status(health_results)
            
            results['results']['health_check'] = {
                'overall_status': overall_health,
                'component_results': {
                    name: {
                        'status': result.status,
                        'message': result.message,
                        'duration_ms': result.duration_ms
                    }
                    for name, result in health_results.items()
                }
            }
            
            # Basic validation
            logger.info("Running basic system validation...")
            validation_results = self.system_validator.run_comprehensive_validation()
            
            results['results']['system_validation'] = {
                name: {
                    'success': result.success,
                    'message': result.message,
                    'duration_ms': result.duration_ms
                }
                for name, result in validation_results.items()
            }
            
            # Overall status
            health_ok = overall_health in ['healthy', 'warning']
            validation_ok = all(r.success for r in validation_results.values())
            
            results['overall_status'] = 'PASS' if (health_ok and validation_ok) else 'FAIL'
            results['summary'] = {
                'health_status': overall_health,
                'validations_passed': sum(1 for r in validation_results.values() if r.success),
                'validations_total': len(validation_results),
                'recommendations': self._generate_quick_recommendations(health_results, validation_results)
            }
            
        except Exception as e:
            logger.error(f"Quick validation failed: {e}")
            results['overall_status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    def run_comprehensive_validation(self, duration_minutes: int = 10) -> dict:
        """Run a comprehensive validation with performance monitoring."""
        logger.info(f"🔍 Starting comprehensive system validation (duration: {duration_minutes} minutes)...")
        
        results = {
            'validation_type': 'comprehensive',
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration_minutes,
            'results': {}
        }
        
        try:
            # Start performance monitoring
            logger.info("Starting performance monitoring...")
            self.performance_monitor.start_monitoring()
            self.metrics_collector.start_collection()
            
            # Run initial health check
            logger.info("Running initial health checks...")
            initial_health = self.health_monitor.run_comprehensive_health_check()
            
            # Run system validation
            logger.info("Running comprehensive system validation...")
            validation_results = self.system_validator.run_comprehensive_validation()
            
            # Monitor for specified duration
            logger.info(f"Monitoring system performance for {duration_minutes} minutes...")
            import time
            time.sleep(duration_minutes * 60)
            
            # Run final health check
            logger.info("Running final health checks...")
            final_health = self.health_monitor.run_comprehensive_health_check()
            
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            self.metrics_collector.stop_collection()
            
            # Collect results
            performance_summary = self.performance_monitor.get_performance_summary(duration_minutes)
            metrics_summary = self.metrics_collector.get_metric_summary(timedelta(minutes=duration_minutes))
            
            results['results'] = {
                'initial_health_check': {
                    name: {
                        'status': result.status,
                        'message': result.message,
                        'metrics': result.metrics
                    }
                    for name, result in initial_health.items()
                },
                'final_health_check': {
                    name: {
                        'status': result.status,
                        'message': result.message,
                        'metrics': result.metrics
                    }
                    for name, result in final_health.items()
                },
                'system_validation': self.system_validator.generate_validation_report(validation_results),
                'performance_monitoring': performance_summary,
                'metrics_summary': metrics_summary
            }
            
            # Determine overall status
            initial_health_ok = self.health_monitor.get_overall_health_status(initial_health) != 'critical'
            final_health_ok = self.health_monitor.get_overall_health_status(final_health) != 'critical'
            validation_ok = all(r.success for r in validation_results.values())
            performance_ok = performance_summary.get('success_rate', 0) > 80 if performance_summary.get('total_queries', 0) > 0 else True
            
            results['overall_status'] = 'PASS' if all([initial_health_ok, final_health_ok, validation_ok, performance_ok]) else 'FAIL'
            
            results['summary'] = {
                'initial_health_status': self.health_monitor.get_overall_health_status(initial_health),
                'final_health_status': self.health_monitor.get_overall_health_status(final_health),
                'validations_passed': sum(1 for r in validation_results.values() if r.success),
                'validations_total': len(validation_results),
                'performance_queries': performance_summary.get('total_queries', 0),
                'performance_success_rate': performance_summary.get('success_rate', 0),
                'metrics_collected': metrics_summary.get('total_metrics', 0),
                'recommendations': self._generate_comprehensive_recommendations(
                    initial_health, final_health, validation_results, performance_summary
                )
            }
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            results['overall_status'] = 'ERROR'
            results['error'] = str(e)
        finally:
            # Ensure monitoring is stopped
            try:
                self.performance_monitor.stop_monitoring()
                self.metrics_collector.stop_collection()
            except:
                pass
        
        return results
    
    def _generate_quick_recommendations(self, health_results, validation_results) -> list:
        """Generate recommendations for quick validation."""
        recommendations = []
        
        # Health recommendations
        for name, result in health_results.items():
            if result.status == 'critical':
                recommendations.append(f"CRITICAL: Fix {name} issues immediately")
            elif result.status == 'warning':
                recommendations.append(f"WARNING: Address {name} issues when possible")
        
        # Validation recommendations
        for name, result in validation_results.items():
            if not result.success:
                recommendations.append(f"Fix {name} validation failures")
        
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(self, initial_health, final_health, validation_results, performance_summary) -> list:
        """Generate recommendations for comprehensive validation."""
        recommendations = []
        
        # Health trend analysis
        initial_status = self.health_monitor.get_overall_health_status(initial_health)
        final_status = self.health_monitor.get_overall_health_status(final_health)
        
        if final_status == 'critical' and initial_status != 'critical':
            recommendations.append("URGENT: System health degraded during monitoring - investigate immediately")
        elif final_status == 'warning' and initial_status == 'healthy':
            recommendations.append("System health declined during monitoring - monitor closely")
        elif final_status == 'healthy' and initial_status != 'healthy':
            recommendations.append("System health improved during monitoring - good trend")
        
        # Performance recommendations
        if performance_summary.get('total_queries', 0) > 0:
            success_rate = performance_summary.get('success_rate', 0)
            avg_time = performance_summary.get('execution_time_stats', {}).get('avg_ms', 0)
            
            if success_rate < 90:
                recommendations.append(f"Low query success rate ({success_rate:.1f}%) - investigate failures")
            
            if avg_time > 2000:
                recommendations.append(f"High average query time ({avg_time:.1f}ms) - optimize performance")
        
        # Validation recommendations
        failed_validations = [name for name, result in validation_results.items() if not result.success]
        if failed_validations:
            recommendations.append(f"Fix validation failures: {', '.join(failed_validations)}")
        
        if not recommendations:
            recommendations.append("System performing well - maintain current monitoring")
        
        return recommendations
    
    def export_results(self, results: dict, output_dir: str = "reports/validation"):
        """Export validation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_type = results.get('validation_type', 'unknown')
        
        # Export main results
        results_file = f"{output_dir}/validation_{validation_type}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results exported to {results_file}")
        
        # Export performance metrics if available
        if hasattr(self, 'performance_monitor'):
            try:
                metrics_file = f"{output_dir}/performance_metrics_{timestamp}.json"
                self.performance_monitor.export_metrics(metrics_file, timedelta(hours=1))
            except Exception as e:
                logger.warning(f"Failed to export performance metrics: {e}")
        
        # Export collected metrics if available
        if hasattr(self, 'metrics_collector'):
            try:
                collector_metrics_file = f"{output_dir}/collected_metrics_{timestamp}.json"
                self.metrics_collector.export_metrics(collector_metrics_file, timedelta(hours=1))
            except Exception as e:
                logger.warning(f"Failed to export collected metrics: {e}")
        
        return results_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive RAG System Validation")
    parser.add_argument(
        '--type', 
        choices=['quick', 'comprehensive'], 
        default='quick',
        help='Type of validation to run'
    )
    parser.add_argument(
        '--duration', 
        type=int, 
        default=10,
        help='Duration in minutes for comprehensive validation (default: 10)'
    )
    parser.add_argument(
        '--output-dir', 
        default='reports/validation',
        help='Output directory for results (default: reports/validation)'
    )
    parser.add_argument(
        '--config', 
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure log directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        validator = ComprehensiveValidator(args.config)
        
        if args.type == 'quick':
            results = validator.run_quick_validation()
        else:
            results = validator.run_comprehensive_validation(args.duration)
        
        # Export results
        results_file = validator.export_results(results, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print(f"🏥 RAG SYSTEM VALIDATION COMPLETE")
        print("="*60)
        print(f"Validation Type: {results['validation_type'].upper()}")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Timestamp: {results['timestamp']}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"\nSummary:")
            for key, value in summary.items():
                if key != 'recommendations':
                    print(f"  {key}: {value}")
            
            print(f"\nRecommendations:")
            for rec in summary.get('recommendations', []):
                print(f"  • {rec}")
        
        print(f"\nDetailed results saved to: {results_file}")
        print("="*60)
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_status'] == 'PASS' else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()