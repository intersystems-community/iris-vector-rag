# TDD+RAGAS Integration Guide

This document provides comprehensive guidance on the Test-Driven Development (TDD) integration with RAGAS (Retrieval Augmented Generation Assessment) framework for performance benchmarking and quality assessment of RAG pipelines.

## Overview

The TDD+RAGAS integration combines:
- **Test-Driven Development principles** for systematic validation
- **RAGAS quality metrics** for industry-standard assessment
- **Performance benchmarking** for scalability analysis
- **Automated reporting** for comprehensive analysis

## Key Components

### Test File
- **Location**: [`tests/test_tdd_performance_with_ragas.py`](../tests/test_tdd_performance_with_ragas.py)
- **Purpose**: Comprehensive TDD-based performance and quality testing with RAGAS metrics

### Reporting Script
- **Location**: [`scripts/generate_tdd_ragas_performance_report.py`](../scripts/generate_tdd_ragas_performance_report.py)
- **Purpose**: Generates detailed Markdown reports from test results

### Pytest Markers

The integration introduces four new pytest markers for targeted test execution:

| Marker | Purpose | Usage |
|--------|---------|-------|
| `performance_ragas` | Performance benchmarking with RAGAS quality metrics | `pytest -m performance_ragas` |
| `scalability_ragas` | Scalability testing across document corpus sizes | `pytest -m scalability_ragas` |
| `tdd_ragas` | General TDD+RAGAS integration tests | `pytest -m tdd_ragas` |
| `ragas_integration` | All RAGAS integration aspects | `pytest -m ragas_integration` |

## Make Targets

### Core Testing Targets

#### `make test-performance-ragas-tdd`
**Purpose**: Run TDD performance benchmark tests with RAGAS quality metrics

**What it does**:
- Validates pipeline performance meets minimum thresholds
- Measures RAGAS quality metrics (answer relevancy, context precision, faithfulness, context recall)
- Ensures success rates above defined minimums
- Tests individual query results for proper RAGAS scores

**When to use**: Regular performance validation during development

#### `make test-scalability-ragas-tdd`
**Purpose**: Run TDD scalability tests with RAGAS across different document corpus sizes

**What it does**:
- Tests performance and quality scaling across document counts (100, 500, 1000+ docs)
- Validates that quality metrics remain acceptable at scale
- Analyzes response time degradation patterns
- Ensures success rates don't drop below thresholds

**When to use**: Before production deployment or when testing scalability limits

#### `make test-tdd-comprehensive-ragas`
**Purpose**: Run all TDD RAGAS integration tests (performance & scalability)

**What it does**:
- Executes complete TDD+RAGAS test suite
- Combines performance and scalability validation
- Provides comprehensive system validation

**When to use**: Full system validation, CI/CD pipelines, release testing

### Enhanced Testing Targets

#### `make test-1000-enhanced`
**Purpose**: Run TDD RAGAS tests with 1000+ documents for comprehensive validation

**What it does**:
- Sets `TEST_DOCUMENT_COUNT=1000` environment variable
- Ensures large-scale testing with substantial document corpus
- Validates performance and quality at production scale

**When to use**: Production readiness testing, comprehensive validation

#### `make test-tdd-ragas-quick`
**Purpose**: Run a quick version of TDD RAGAS performance tests for development

**What it does**:
- Sets `TDD_RAGAS_QUICK_MODE=true` for limited test scope
- Provides rapid feedback during development
- Focuses on core performance metrics

**When to use**: Development cycles, rapid iteration, debugging

#### `make ragas-with-tdd`
**Purpose**: Run comprehensive TDD RAGAS tests and generate detailed report

**What it does**:
1. Executes `test-tdd-comprehensive-ragas`
2. Locates latest test results JSON file
3. Generates comprehensive Markdown report
4. Saves report to `reports/tdd_ragas_reports/`

**When to use**: Complete validation with documentation, stakeholder reporting

## RAGAS Quality Metrics

The integration validates the following RAGAS metrics against defined thresholds:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Answer Relevancy | ≥ 0.7 | How relevant the generated answer is to the query |
| Context Precision | ≥ 0.6 | Precision of retrieved context for answering the query |
| Context Recall | ≥ 0.7 | Recall of retrieved context for answering the query |
| Faithfulness | ≥ 0.8 | How faithful the answer is to the retrieved context |

### Threshold Configuration

Thresholds are defined in the `RAGASThresholds` class in [`tests/test_tdd_performance_with_ragas.py`](../tests/test_tdd_performance_with_ragas.py):

```python
class RAGASThresholds:
    MIN_ANSWER_RELEVANCY = 0.7
    MIN_CONTEXT_PRECISION = 0.6
    MIN_FAITHFULNESS = 0.8
    MIN_CONTEXT_RECALL = 0.7
    MIN_SUCCESS_RATE = 0.8
```

## Performance Metrics

Beyond RAGAS quality metrics, the framework tracks:

- **Response Time**: Average and standard deviation
- **Success Rate**: Percentage of successful pipeline executions
- **Documents Retrieved**: Average number of documents per query
- **Answer Length**: Average length of generated answers
- **Efficiency Score**: Success rate divided by response time

## Report Generation

### Automatic Report Generation

The `make ragas-with-tdd` target automatically generates reports after test execution.

### Manual Report Generation

```bash
# Generate report from specific results file
python scripts/generate_tdd_ragas_performance_report.py path/to/results.json

# Specify custom output directory
python scripts/generate_tdd_ragas_performance_report.py results.json --output-dir custom/reports/

# Custom report name
python scripts/generate_tdd_ragas_performance_report.py results.json --report-name custom_report
```

### Report Contents

Generated reports include:

1. **Executive Summary**: High-level findings and recommendations
2. **Performance Analysis**: Response times, success rates, efficiency metrics
3. **RAGAS Quality Metrics**: Quality scores and threshold compliance
4. **Scalability Analysis**: Trends and bottleneck identification
5. **Recommendations**: Actionable insights for optimization
6. **Detailed Data**: Raw analysis data in expandable sections

## Environment Variables

### Test Configuration

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `TEST_DOCUMENT_COUNT` | Set document count for testing | varies | `1000` |
| `TDD_RAGAS_QUICK_MODE` | Enable quick testing mode | `false` | `true` |
| `PYTEST_FAST_MODE` | Limit queries for faster testing | `false` | `true` |

### Usage Examples

```bash
# Run with specific document count
TEST_DOCUMENT_COUNT=500 make test-scalability-ragas-tdd

# Quick development testing
TDD_RAGAS_QUICK_MODE=true make test-tdd-ragas-quick

# Fast mode for development
PYTEST_FAST_MODE=true pytest tests/test_tdd_performance_with_ragas.py -m performance_ragas
```

## Integration with Existing Framework

### Fixture Dependencies

The TDD+RAGAS tests leverage existing pytest fixtures:

- `iris_connection_auto`: Database connection management
- `iris_with_pmc_data`: PMC data loading and validation
- `evaluation_dataset`: Test queries and expected answers

### Configuration Integration

Tests use the existing `ComprehensiveRAGASEvaluationFramework` for:
- Pipeline initialization
- RAGAS metric calculation
- Result aggregation and analysis

## Best Practices

### Development Workflow

1. **Start with Quick Tests**: Use `make test-tdd-ragas-quick` during development
2. **Regular Performance Checks**: Run `make test-performance-ragas-tdd` for feature validation
3. **Scalability Validation**: Use `make test-scalability-ragas-tdd` before major releases
4. **Comprehensive Validation**: Run `make ragas-with-tdd` for complete assessment

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run TDD+RAGAS Performance Tests
  run: make test-performance-ragas-tdd

- name: Run TDD+RAGAS Scalability Tests
  run: make test-scalability-ragas-tdd

- name: Generate Performance Report
  run: make ragas-with-tdd
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

### Threshold Tuning

1. **Baseline Establishment**: Run tests with current system to establish baselines
2. **Gradual Improvement**: Incrementally raise thresholds as system improves
3. **Context-Specific Adjustment**: Adjust thresholds based on use case requirements
4. **Regular Review**: Periodically review and update thresholds

## Troubleshooting

### Common Issues

#### No Test Results Found
```bash
# Error: No TDD RAGAS JSON result file found
# Solution: Ensure tests run successfully first
make test-tdd-comprehensive-ragas
```

#### Threshold Failures
```bash
# Error: RAGAS metrics below threshold
# Solution: Check pipeline configuration and data quality
# Review individual test results for specific failures
```

#### Scalability Test Failures
```bash
# Error: Response time degradation too severe
# Solution: Optimize pipeline performance or adjust degradation limits
# Check database indexing and query optimization
```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
# Run with verbose output
pytest tests/test_tdd_performance_with_ragas.py -v -s

# Enable debug logging
PYTEST_LOG_LEVEL=DEBUG pytest tests/test_tdd_performance_with_ragas.py -m performance_ragas
```

## Future Enhancements

### Planned Features

1. **Multi-Scale Testing**: Support for testing across more document scales
2. **Custom Threshold Configuration**: External configuration file for thresholds
3. **Comparative Analysis**: Compare results across different test runs
4. **Performance Regression Detection**: Automated detection of performance degradation
5. **Integration with Monitoring**: Real-time performance monitoring integration

### Extension Points

The framework is designed for extensibility:

- **Custom Metrics**: Add new performance or quality metrics
- **Additional Pipelines**: Support for new RAG pipeline types
- **Enhanced Reporting**: Custom report formats and visualizations
- **Integration APIs**: REST APIs for external system integration

## Related Documentation

- [Testing Guide](TESTING.md) - General testing strategies and setup
- [Performance Guide](PERFORMANCE_GUIDE.md) - Performance optimization recommendations
- [Benchmark Execution Plan](BENCHMARK_EXECUTION_PLAN.md) - Comprehensive benchmarking strategy
- [API Reference](API_REFERENCE.md) - Complete API documentation

---

*This guide is part of the RAG Templates documentation suite. For questions or contributions, please refer to the project's contribution guidelines.*