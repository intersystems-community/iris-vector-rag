# RAG Templates Monitoring System

This document describes the comprehensive monitoring system for the RAG Templates project, including health monitoring, performance tracking, system validation, and real-time dashboards.

## Overview

The monitoring system provides:

- **Health Monitoring**: Real-time health checks for system components
- **Performance Monitoring**: Query performance tracking and metrics collection
- **System Validation**: Comprehensive validation of data integrity and functionality
- **Real-time Dashboard**: Live monitoring interface
- **Metrics Collection**: Automated metrics gathering and export

## Architecture

### Core Components

#### 1. Health Monitor (`rag_templates.monitoring.health_monitor`)

Monitors the health of system components:

- **System Resources**: CPU, memory, disk usage
- **Database Connectivity**: Connection status and basic operations
- **Docker Containers**: Container status and resource usage
- **Vector Performance**: Vector query performance and HNSW indexes

```python
from rag_templates.monitoring.health_monitor import HealthMonitor

monitor = HealthMonitor()
results = monitor.run_comprehensive_health_check()
overall_status = monitor.get_overall_health_status(results)
```

#### 2. Performance Monitor (`rag_templates.monitoring.performance_monitor`)

Tracks query performance and system metrics:

- **Query Performance**: Execution time, success rates, pipeline breakdown
- **System Metrics**: Real-time resource monitoring
- **Performance Thresholds**: Configurable alerting thresholds
- **Metrics Export**: JSON/CSV export capabilities

```python
from rag_templates.monitoring.performance_monitor import PerformanceMonitor

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

#### 3. System Validator (`rag_templates.monitoring.system_validator`)

Validates system integrity and functionality:

- **Data Integrity**: Checks for duplicates, orphaned data, consistency
- **Pipeline Functionality**: Tests RAG pipeline execution
- **Vector Operations**: Validates vector operations and HNSW performance
- **System Configuration**: Verifies dependencies and configuration

```python
from rag_templates.monitoring.system_validator import SystemValidator

validator = SystemValidator()
results = validator.run_comprehensive_validation()
report = validator.generate_validation_report(results)
```

#### 4. Metrics Collector (`rag_templates.monitoring.metrics_collector`)

Centralized metrics collection and aggregation:

- **Metric Collection**: Automated collection from registered sources
- **Aggregation**: Time-window based metric aggregation
- **Export**: Multiple export formats (JSON, CSV)
- **Real-time Access**: Live metric querying

```python
from rag_templates.monitoring.metrics_collector import MetricsCollector

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
python scripts/comprehensive_system_validation.py --type quick
```

### Comprehensive Validation

Run a comprehensive validation with performance monitoring:

```bash
python scripts/comprehensive_system_validation.py --type comprehensive --duration 10
```

### Real-time Dashboard

Start the monitoring dashboard:

```bash
python scripts/monitoring_dashboard.py
```

Options:
- `--refresh-interval 30`: Set refresh interval in seconds
- `--export-status`: Export current status and exit
- `--config path/to/config.json`: Use custom configuration

### Programmatic Usage

```python
from rag_templates.monitoring import HealthMonitor, PerformanceMonitor, SystemValidator

# Initialize components
health_monitor = HealthMonitor()
performance_monitor = PerformanceMonitor()
validator = SystemValidator()

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

The monitoring system is configured via [`config/monitoring.json`](../config/monitoring.json):

### Key Configuration Sections

#### Performance Thresholds
```json
{
  "performance_thresholds": {
    "vector_query_max_ms": 100,
    "memory_usage_max_percent": 85,
    "disk_usage_max_percent": 90,
    "query_success_rate_min_percent": 95
  }
}
```

#### Health Check Schedule
```json
{
  "health_check_schedule": {
    "interval_minutes": 15,
    "full_check_interval_hours": 6,
    "enable_continuous_monitoring": true
  }
}
```

#### Alert Settings
```json
{
  "alert_settings": {
    "enable_alerts": true,
    "critical_threshold_breaches": 3,
    "alert_cooldown_minutes": 15
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

### Pipeline Functionality Validation
- Tests RAG pipeline execution
- Validates response structure
- Checks retrieval and generation components
- Measures performance metrics

### Vector Operations Validation
- Tests basic vector operations
- Validates HNSW index performance
- Checks vector similarity calculations
- Measures query performance

### System Configuration Validation
- Verifies required dependencies
- Checks configuration files
- Validates log directories
- Tests overall system health

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
- Connection status

#### Performance Metrics
- Query execution time (avg, p95, p99)
- Success rate
- Pipeline-specific performance
- Retrieval and generation times

#### Health Metrics
- Component health status
- Health check duration
- Issue counts and types

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

## Dashboard Features

The real-time dashboard provides:

### System Health Overview
- Overall health status with color coding
- Component-level health details
- Key metrics display
- Duration tracking

### Performance Metrics
- Query statistics (last 5 minutes)
- Success rate tracking
- Execution time breakdown
- Pipeline performance comparison

### System Resources
- Real-time CPU, memory, disk usage
- Container status monitoring
- Resource usage trends
- Alert indicators

### Recent Activity
- Metrics collection status
- Database statistics
- Query buffer information
- Performance monitoring status

## Testing

Run the monitoring system tests:

```bash
# Run all monitoring tests
pytest tests/test_monitoring/

# Run specific test modules
pytest tests/test_monitoring/test_health_monitor.py
pytest tests/test_monitoring/test_system_validator.py
```

### Test Coverage

The test suite covers:
- Health check functionality
- Performance monitoring
- System validation
- Metrics collection
- Error handling
- Configuration validation

## Troubleshooting

### Common Issues

#### Health Check Failures
1. **Database Connectivity**: Check IRIS container status and connection parameters
2. **System Resources**: Monitor CPU, memory, and disk usage
3. **Docker Issues**: Verify Docker daemon is running and containers are healthy

#### Performance Issues
1. **Slow Vector Queries**: Check HNSW index status and document count
2. **High Resource Usage**: Monitor system resources and optimize queries
3. **Low Success Rate**: Check pipeline configuration and error logs

#### Validation Failures
1. **Data Integrity**: Run data cleanup and re-embedding processes
2. **Pipeline Functionality**: Verify pipeline dependencies and configuration
3. **Vector Operations**: Check vector data quality and index configuration

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
logging.getLogger('rag_templates.monitoring').setLevel(logging.DEBUG)
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
python scripts/comprehensive_system_validation.py --type quick

# Export status for reporting
python scripts/monitoring_dashboard.py --export-status --export-file ci_status.json
```

### Custom Metrics

Add custom metrics to the system:

```python
from rag_templates.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()

# Register custom collector
def collect_custom_metrics():
    return {
        "custom_metric_1": get_custom_value_1(),
        "custom_metric_2": get_custom_value_2()
    }

collector.register_collector("custom", collect_custom_metrics)
```

## Best Practices

1. **Regular Monitoring**: Run health checks regularly (every 15 minutes)
2. **Performance Baselines**: Establish performance baselines for comparison
3. **Alert Thresholds**: Set appropriate alert thresholds based on system capacity
4. **Log Retention**: Configure appropriate log retention policies
5. **Metric Export**: Regularly export metrics for historical analysis
6. **Validation Schedule**: Run comprehensive validation daily or after major changes
7. **Dashboard Monitoring**: Use the dashboard for real-time system oversight

## Future Enhancements

Planned improvements:
- Email/webhook alert notifications
- Historical trend analysis
- Predictive monitoring
- Custom dashboard widgets
- Integration with external monitoring systems
- Automated remediation actions