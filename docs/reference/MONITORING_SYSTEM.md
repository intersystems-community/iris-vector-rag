# RAG Templates Monitoring System

This document describes the comprehensive monitoring system for the RAG Templates project, including health monitoring, performance tracking, system validation, and metrics collection.

## Overview

The monitoring system provides:

- **Health Monitoring**: Real-time health checks for system components
- **Performance Monitoring**: Query performance tracking and metrics collection
- **System Validation**: Comprehensive validation of data integrity and functionality
- **Metrics Collection**: Automated metrics gathering and export
- **LLM Cache Monitoring**: Performance tracking for LLM caching system

## Architecture

### Core Components

#### 1. Health Monitor ([`iris_rag.monitoring.health_monitor`](../../iris_rag/monitoring/health_monitor.py))

Monitors the health of system components:

- **System Resources**: CPU, memory, disk usage
- **Database Connectivity**: Connection status and basic operations
- **Docker Containers**: Container status and resource usage
- **Vector Performance**: Vector query performance and HNSW indexes
- **LLM Cache Performance**: Cache hit rates and response times

```python
from iris_rag.monitoring.health_monitor import HealthMonitor

monitor = HealthMonitor()
results = monitor.run_comprehensive_health_check()
overall_status = monitor.get_overall_health_status(results)
```

#### 2. Performance Monitor ([`iris_rag.monitoring.performance_monitor`](../../iris_rag/monitoring/performance_monitor.py))

Tracks query performance and system metrics:

- **Query Performance**: Execution time, success rates, pipeline breakdown
- **System Metrics**: Real-time resource monitoring
- **Performance Thresholds**: Configurable alerting thresholds
- **Metrics Export**: JSON export capabilities

```python
from iris_rag.monitoring.performance_monitor import PerformanceMonitor, QueryPerformanceData

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Record query performance
query_data = QueryPerformanceData(
    query_text="test query",
    pipeline_type="basic_rag",
    execution_time_ms=150.0,
    retrieval_time_ms=50.0,
    generation_time_ms=100.0,
    documents_retrieved=5,
    tokens_generated=100,
    timestamp=datetime.now(),
    success=True
)
monitor.record_query_performance(query_data)
```

#### 3. System Validator ([`iris_rag.monitoring.system_validator`](../../iris_rag/monitoring/system_validator.py))

Validates system integrity and functionality:

- **Data Integrity**: Checks for duplicates, orphaned data, consistency
- **Pipeline Functionality**: Tests RAG pipeline execution
- **Vector Operations**: Validates vector operations and HNSW performance
- **System Configuration**: Verifies dependencies and configuration

```python
from iris_rag.monitoring.system_validator import SystemValidator

validator = SystemValidator()
results = validator.run_comprehensive_validation()
report = validator.generate_validation_report(results)
```

#### 4. Metrics Collector ([`iris_rag.monitoring.metrics_collector`](../../iris_rag/monitoring/metrics_collector.py))

Centralized metrics collection and aggregation:

- **Metric Collection**: Automated collection from registered sources
- **Aggregation**: Time-window based metric aggregation
- **Export**: Multiple export formats (JSON, CSV)
- **Real-time Access**: Live metric querying
- **LLM Cache Metrics**: Specialized cache performance tracking

```python
from iris_rag.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()
collector.start_collection()

# Add custom metrics
collector.add_metric("custom_metric", 42.0, tags={"source": "test"})

# Get aggregated metrics
summary = collector.get_metric_summary(timedelta(hours=1))
```

## Usage

### Quick System Validation

Run a quick validation to check system health:

```bash
python scripts/utilities/comprehensive_system_validation.py --type quick
```

### Comprehensive Validation

Run a comprehensive validation with performance monitoring:

```bash
python scripts/utilities/comprehensive_system_validation.py --type comprehensive --duration 10
```

### Programmatic Usage

```python
from iris_rag.monitoring import HealthMonitor, PerformanceMonitor, SystemValidator
from iris_rag.config.manager import ConfigurationManager

# Initialize components
config_manager = ConfigurationManager()
health_monitor = HealthMonitor(config_manager)
performance_monitor = PerformanceMonitor(config_manager)
validator = SystemValidator(config_manager)

# Run health check
health_results = health_monitor.run_comprehensive_health_check()
print(f"Overall health: {health_monitor.get_overall_health_status(health_results)}")

# Start performance monitoring
performance_monitor.start_monitoring()

# Run validation
validation_results = validator.run_comprehensive_validation()
validation_report = validator.generate_validation_report(validation_results)

# Stop monitoring
performance_monitor.stop_monitoring()
```

## Configuration

The monitoring system is configured via [`config/monitoring.json`](../../config/monitoring.json):

### Key Configuration Sections

#### Performance Thresholds
```json
{
  "performance_thresholds": {
    "vector_query_max_ms": 100,
    "ingestion_rate_min_docs_per_sec": 10,
    "memory_usage_max_percent": 85,
    "disk_usage_max_percent": 90,
    "query_success_rate_min_percent": 95,
    "response_time_p95_max_ms": 500,
    "response_time_p99_max_ms": 1000
  }
}
```

#### Health Check Schedule
```json
{
  "health_check_schedule": {
    "interval_minutes": 15,
    "full_check_interval_hours": 6,
    "quick_check_interval_minutes": 5,
    "enable_continuous_monitoring": true
  }
}
```

#### Alert Settings
```json
{
  "alert_settings": {
    "enable_alerts": true,
    "alert_log_file": "logs/alerts.log",
    "critical_threshold_breaches": 3,
    "alert_cooldown_minutes": 15,
    "notification_channels": {
      "email": {
        "enabled": false,
        "recipients": []
      },
      "webhook": {
        "enabled": false,
        "url": ""
      }
    }
  }
}
```

#### Metrics Collection
```json
{
  "metrics_collection": {
    "collection_interval_seconds": 60,
    "buffer_size": 10000,
    "export_interval_hours": 24,
    "export_format": "json",
    "export_directory": "reports/metrics"
  }
}
```

## Validation Tests

The system includes comprehensive validation tests:

### Data Integrity Validation
- Checks for duplicate documents
- Validates embedding consistency
- Identifies orphaned chunks
- Verifies content completeness
- Checks embedding dimension consistency

### Pipeline Functionality Validation
- Tests RAG pipeline execution with sample queries
- Validates response structure and content
- Checks retrieval and generation components
- Measures performance metrics
- Verifies required result keys

### Vector Operations Validation
- Tests basic vector operations (TO_VECTOR, VECTOR_COSINE)
- Validates HNSW index performance
- Checks vector similarity calculations
- Measures query performance
- Verifies index existence and configuration

### System Configuration Validation
- Verifies required Python dependencies
- Checks configuration file validity
- Validates log directories
- Tests overall system health
- Confirms package versions

## Metrics and Monitoring

### Collected Metrics

#### System Metrics
- CPU usage percentage
- Memory usage (percentage and absolute)
- Disk usage (percentage and free space)
- Container status and resource usage

#### Database Metrics
- Document count
- Embedded document count
- Vector query performance
- Connection status and health

#### Performance Metrics
- Query execution time (avg, p95, p99)
- Success rate
- Pipeline-specific performance
- Retrieval and generation times

#### Health Metrics
- Component health status
- Health check duration
- Issue counts and types

#### LLM Cache Metrics
- Cache hit rate and miss rate
- Average response times (cached vs uncached)
- Cache speedup ratio
- Backend-specific statistics
- Total requests and cache utilization

### Metric Export

Metrics can be exported in multiple formats:

```python
# Export to JSON
collector.export_metrics("metrics.json", format="json")

# Export to CSV
collector.export_metrics("metrics.csv", format="csv")

# Export with time window
collector.export_metrics("recent_metrics.json", time_window=timedelta(hours=1))
```

## Health Check Components

### System Resources Check
- **Memory**: Warns at 80%, critical at 90%
- **CPU**: Warns at 80%, critical at 90%
- **Disk**: Warns at 85%, critical at 95%

### Database Connectivity Check
- Basic connectivity test
- Schema validation (RAG tables)
- Vector operations test
- Document and embedding counts

### Docker Containers Check
- IRIS container status and health
- Container resource usage
- Memory utilization monitoring

### Vector Performance Check
- Query performance measurement
- HNSW index validation
- Embedding availability check
- Performance threshold validation

### LLM Cache Performance Check
- Cache configuration validation
- Hit rate analysis
- Response time comparison
- Backend health monitoring

## Testing

Run the monitoring system tests:

```bash
# Run all monitoring tests
pytest tests/test_monitoring/

# Run specific test modules
pytest tests/test_monitoring/test_health_monitor.py
pytest tests/test_monitoring/test_performance_monitor.py
pytest tests/test_monitoring/test_system_validator.py
pytest tests/test_monitoring/test_metrics_collector.py
```

### Test Coverage

The test suite covers:
- Health check functionality for all components
- Performance monitoring and metrics collection
- System validation across all categories
- Metrics collection and aggregation
- Error handling and edge cases
- Configuration validation

## Troubleshooting

### Common Issues

#### Health Check Failures
1. **Database Connectivity**: Check IRIS container status and connection parameters
2. **System Resources**: Monitor CPU, memory, and disk usage
3. **Docker Issues**: Verify Docker daemon is running and containers are healthy
4. **Vector Operations**: Ensure HNSW indexes are properly created

#### Performance Issues
1. **Slow Vector Queries**: Check HNSW index status and document count
2. **High Resource Usage**: Monitor system resources and optimize queries
3. **Low Success Rate**: Check pipeline configuration and error logs
4. **Cache Performance**: Verify LLM cache configuration and hit rates

#### Validation Failures
1. **Data Integrity**: Run data cleanup and re-embedding processes
2. **Pipeline Functionality**: Verify pipeline dependencies and configuration
3. **Vector Operations**: Check vector data quality and index configuration
4. **System Configuration**: Install missing dependencies and fix configuration

### Log Files

Monitor these log files for issues:
- `logs/system.log`: General system logs
- `logs/performance/performance.log`: Performance monitoring logs
- `logs/health_checks/health.log`: Health check logs
- `logs/validation/validation.log`: Validation logs
- `logs/alerts.log`: Alert notifications

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger('iris_rag.monitoring').setLevel(logging.DEBUG)
```

## Integration

### With Existing Scripts

The monitoring system integrates with existing validation scripts:
- Extends existing health checks
- Provides metrics for performance scripts
- Validates system integrity
- Monitors long-running processes

### With CI/CD

Include monitoring in CI/CD pipelines:

```bash
# Quick validation in CI
python scripts/utilities/comprehensive_system_validation.py --type quick

# Export status for reporting
python scripts/utilities/comprehensive_system_validation.py --export-status
```

### Custom Metrics

Add custom metrics to the system:

```python
from iris_rag.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()

# Register custom collector
def collect_custom_metrics():
    return {
        "custom_metric_1": get_custom_value_1(),
        "custom_metric_2": get_custom_value_2()
    }

collector.register_collector("custom", collect_custom_metrics)
```

## Performance Thresholds

### Default Thresholds
- **Vector Query Time**: < 100ms (warning), < 500ms (critical)
- **Memory Usage**: < 85% (warning), < 90% (critical)
- **Disk Usage**: < 85% (warning), < 95% (critical)
- **Query Success Rate**: > 95%
- **Response Time P95**: < 500ms
- **Response Time P99**: < 1000ms

### Configurable Thresholds
All thresholds can be customized in [`config/monitoring.json`](../../config/monitoring.json) to match your system requirements and performance expectations.

## Best Practices

1. **Regular Monitoring**: Run health checks every 15 minutes
2. **Performance Baselines**: Establish performance baselines for comparison
3. **Alert Thresholds**: Set appropriate alert thresholds based on system capacity
4. **Log Retention**: Configure appropriate log retention policies (default: 30 days)
5. **Metric Export**: Regularly export metrics for historical analysis
6. **Validation Schedule**: Run comprehensive validation daily or after major changes
7. **Cache Monitoring**: Monitor LLM cache performance for optimization opportunities

## Future Enhancements

Planned improvements:
- Email/webhook alert notifications
- Historical trend analysis
- Predictive monitoring
- Custom dashboard widgets
- Integration with external monitoring systems
- Automated remediation actions
- Enhanced cache analytics
- Real-time dashboard interface