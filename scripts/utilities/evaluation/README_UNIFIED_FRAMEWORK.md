# Unified RAGAS-based Evaluation Framework

This document describes the comprehensive refactored RAGAS-based evaluation framework that consolidates all scattered testing code and provides a unified approach to RAG system evaluation.

## Overview

The Unified RAGAS Evaluation Framework addresses the major inconsistencies found in the previous evaluation system:

1. **Fixed Import Issues**: All imports now use correct paths from `core_pipelines/` instead of deprecated `src.deprecated.*` and `src.experimental.*` paths
2. **Unified Framework**: Single comprehensive framework in `eval/unified_ragas_evaluation_framework.py`
3. **Consolidated E2E Tests**: Comprehensive test suite in `tests/test_unified_e2e_rag_evaluation.py`
4. **Standardized Pipeline Initialization**: Consistent parameter patterns across all pipelines
5. **Configuration Management**: Proper configuration system with validation and environment support

## Architecture

### Core Components

#### 1. Unified RAGAS Evaluation Framework (`eval/unified_ragas_evaluation_framework.py`)

The main framework class that provides:

- **Consistent Pipeline Imports**: Safe imports with fallback handling for missing pipelines
- **RAGAS Integration**: Full integration with RAGAS metrics for retrieval and generation quality
- **Parameter Optimization**: Configurable parameters for chunking methods and RAG configurations
- **DBAPI vs JDBC Support**: Configurable connection types
- **Statistical Analysis**: Statistical significance testing with SciPy
- **Comprehensive Metrics**: Deep evaluation beyond syntactic success

#### 2. Configuration Management (`eval/config_manager.py`)

Centralized configuration system with:

- **Environment Variable Support**: Load configuration from environment variables
- **File-based Configuration**: Support for JSON and YAML configuration files
- **Validation**: Comprehensive configuration validation
- **Hierarchical Configuration**: Separate configs for database, embedding, LLM, chunking, retrieval, evaluation, and output

#### 3. Comprehensive E2E Tests (`tests/test_unified_e2e_rag_evaluation.py`)

Following TDD principles with:

- **Unit Tests**: Test individual components and data structures
- **Integration Tests**: Test pipeline execution and framework integration
- **Performance Tests**: Test metrics calculation and performance characteristics
- **Error Handling Tests**: Test graceful error handling and recovery

## Usage

### Basic Usage

```python
from eval.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework

# Initialize with default configuration
framework = UnifiedRAGASEvaluationFramework()

# Run comprehensive evaluation
results = framework.run_comprehensive_evaluation()

# Generate report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = framework.generate_report(results, timestamp)
print(report)
```

### Configuration-based Usage

```python
from eval.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework
from eval.config_manager import ConfigManager

# Load configuration from file
config_manager = ConfigManager()
config = config_manager.load_config("eval/config/dev_config.json")

# Initialize framework with configuration
framework = UnifiedRAGASEvaluationFramework(config)

# Run evaluation
results = framework.run_comprehensive_evaluation()
```

### Environment-based Configuration

```bash
# Set environment variables
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_USERNAME=demo
export IRIS_PASSWORD=demo
export OPENAI_API_KEY=your_api_key
export ENABLE_RAGAS=true
export NUM_ITERATIONS=3

# Run evaluation
python -m eval.unified_ragas_evaluation_framework
```

## Configuration

### Configuration Files

Three sample configuration files are provided:

1. **`eval/config/default_config.json`**: Full-featured configuration with all pipelines enabled
2. **`eval/config/dev_config.json`**: Development configuration with faster settings and limited pipelines
3. **Custom configurations**: Create your own by copying and modifying the defaults

### Configuration Sections

#### Database Configuration
```json
{
  "database": {
    "host": "localhost",
    "port": 1972,
    "namespace": "USER",
    "username": "demo",
    "password": "demo",
    "connection_type": "dbapi",
    "schema": "RAG",
    "timeout": 30
  }
}
```

#### Embedding Configuration
```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "batch_size": 32,
    "max_length": 512,
    "normalize_embeddings": true,
    "cache_dir": null
  }
}
```

#### LLM Configuration
```json
{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-3.5-turbo",
    "api_key": null,
    "base_url": null,
    "temperature": 0.0,
    "max_tokens": 1000,
    "timeout": 30
  }
}
```

#### Pipeline Configuration
```json
{
  "pipelines": {
    "BasicRAG": {
      "enabled": true,
      "timeout": 60,
      "retry_attempts": 3,
      "custom_params": {}
    },
    "HyDE": {
      "enabled": true,
      "timeout": 90,
      "retry_attempts": 3,
      "custom_params": {}
    }
  }
}
```

## Features

### RAGAS Integration

The framework provides comprehensive RAGAS evaluation with:

- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Context Precision**: Measures the precision of retrieved context
- **Context Recall**: Measures the recall of retrieved context
- **Faithfulness**: Measures how faithful the answer is to the context
- **Answer Similarity**: Measures similarity to ground truth answers
- **Answer Correctness**: Measures factual correctness of answers

### Statistical Analysis

When SciPy is available, the framework provides:

- **Pairwise Comparisons**: Statistical comparison between techniques
- **Significance Testing**: T-tests and Mann-Whitney U tests
- **Performance Analysis**: Statistical analysis of response times and quality metrics

### Visualization

Comprehensive visualizations including:

- **Performance Comparison Charts**: Bar charts comparing response times, documents retrieved, similarity scores
- **RAGAS Metrics Comparison**: Grouped bar charts for quality metrics
- **Spider Charts**: Radar charts showing overall technique comparison
- **Export Formats**: PNG, PDF, and other formats

### Error Handling

Robust error handling with:

- **Graceful Degradation**: Continue evaluation even if some pipelines fail
- **Detailed Logging**: Comprehensive logging with configurable levels
- **Retry Logic**: Configurable retry attempts for failed operations
- **Timeout Handling**: Configurable timeouts for long-running operations

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/test_unified_e2e_rag_evaluation.py -v

# Run specific test categories
pytest tests/test_unified_e2e_rag_evaluation.py::TestUnifiedE2ERAGEvaluation -v
pytest tests/test_unified_e2e_rag_evaluation.py::TestRAGASIntegration -v
pytest tests/test_unified_e2e_rag_evaluation.py::TestStatisticalAnalysis -v

# Run integration tests only
pytest tests/test_unified_e2e_rag_evaluation.py -m integration -v

# Run performance tests only
pytest tests/test_unified_e2e_rag_evaluation.py -m performance -v
```

### Test Categories

1. **Unit Tests**: Test individual components and data structures
2. **Integration Tests**: Test pipeline execution and framework integration
3. **Performance Tests**: Test metrics calculation and performance characteristics
4. **RAGAS Tests**: Test RAGAS integration when available
5. **Statistical Tests**: Test statistical analysis when SciPy is available

## Migration Guide

### From Old Evaluation Files

If you were using the old scattered evaluation files:

1. **Replace imports**: Update imports to use the new unified framework
2. **Update configuration**: Convert your parameters to the new configuration format
3. **Update test files**: Use the new comprehensive test suite
4. **Update scripts**: Use the new framework's main execution method

### Example Migration

**Old approach:**
```python
from eval.comprehensive_rag_benchmark_with_ragas import ComprehensiveRAGBenchmark
benchmark = ComprehensiveRAGBenchmark()
results = benchmark.run_comprehensive_benchmark()
```

**New approach:**
```python
from eval.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework
framework = UnifiedRAGASEvaluationFramework()
results = framework.run_comprehensive_evaluation()
```

## Performance Considerations

### Development vs Production

- **Development**: Use `dev_config.json` for faster iteration with limited pipelines and iterations
- **Production**: Use `default_config.json` or custom configuration for comprehensive evaluation

### Optimization Tips

1. **Disable RAGAS**: Set `enable_ragas: false` for faster evaluation during development
2. **Reduce Iterations**: Set `num_iterations: 1` for quick testing
3. **Limit Pipelines**: Disable unused pipelines in configuration
4. **Parallel Execution**: Enable `parallel_execution: true` for faster evaluation with multiple workers

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all pipeline dependencies are installed
2. **Connection Errors**: Verify database connection parameters
3. **RAGAS Errors**: Ensure OpenAI API key is set for RAGAS evaluation
4. **Memory Issues**: Reduce batch sizes and number of iterations

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the evaluation framework:

1. **Follow TDD**: Write tests first, then implement features
2. **Update Configuration**: Add new parameters to the configuration system
3. **Update Documentation**: Keep this README updated with new features
4. **Test Thoroughly**: Run the full test suite before submitting changes

## Dependencies

### Required
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`

### Optional
- `ragas` and `datasets`: For RAGAS evaluation
- `scipy`: For statistical analysis
- `langchain-openai` and `langchain-community`: For RAGAS LLM integration
- `pyyaml`: For YAML configuration support

Install all dependencies:
```bash
pip install numpy pandas matplotlib seaborn plotly ragas datasets scipy langchain-openai langchain-community pyyaml