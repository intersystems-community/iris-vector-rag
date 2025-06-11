# Evaluation Framework Migration Guide

This guide helps you migrate from the old scattered evaluation files to the new unified RAGAS evaluation framework.

## Overview

The unified evaluation framework consolidates multiple scattered evaluation files into a single, comprehensive system that provides:

- Consistent pipeline imports and initialization
- Unified configuration management
- Comprehensive RAGAS integration
- Statistical analysis capabilities
- Better error handling and logging

## Migration Mapping

### Old Files → New Framework

| Old File | New Component | Migration Notes |
|----------|---------------|-----------------|
| `eval/comprehensive_rag_benchmark_with_ragas.py` | `eval/unified_ragas_evaluation_framework.py` | Main framework class |
| `eval/enterprise_rag_benchmark.py` | `eval/unified_ragas_evaluation_framework.py` | Integrated into main framework |
| `eval/comprehensive_scaling_orchestrator.py` | `eval/unified_ragas_evaluation_framework.py` | Scaling features integrated |
| `eval/compare_jdbc_vs_odbc.py` | Built into framework | Connection type comparison |
| Various test scripts | `tests/test_unified_e2e_rag_evaluation.py` | Consolidated test suite |

## Step-by-Step Migration

### 1. Update Imports

**Old approach:**
```python
# Scattered imports from various locations
from eval.comprehensive_rag_benchmark_with_ragas import ComprehensiveRAGBenchmark
from eval.enterprise_rag_benchmark import EnterpriseRAGBenchmark
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline  # Deprecated path
```

**New approach:**
```python
# Single unified import
from eval.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework
from eval.config_manager import ConfigManager
```

### 2. Update Pipeline Initialization

**Old approach:**
```python
# Inconsistent parameter names across pipelines
basic_rag = BasicRAGPipeline(connection=conn, embed_func=embedder)
hyde = HyDEPipeline(iris_connection=conn, embedding_func=embedder)
colbert = ColBERTPipeline(conn=conn, embedder=embedder)
```

**New approach:**
```python
# Consistent parameter patterns
framework = UnifiedRAGASEvaluationFramework(config)
# Framework handles all pipeline initialization internally with consistent parameters
```

### 3. Update Configuration

**Old approach:**
```python
# Hard-coded configuration scattered across files
IRIS_HOST = "localhost"
IRIS_PORT = 1972
IRIS_USERNAME = "demo"
IRIS_PASSWORD = "demo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

**New approach:**
```python
# Configuration-driven with validation
config = {
    "database": {
        "host": "localhost",
        "port": 1972,
        "username": "demo",
        "password": "demo",
        "connection_type": "dbapi"
    },
    "llm": {
        "provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    "evaluation": {
        "enable_ragas": True,
        "num_iterations": 3
    }
}
```

### 4. Update Evaluation Execution

**Old approach:**
```python
# Multiple separate scripts with different interfaces
benchmark = ComprehensiveRAGBenchmark()
results = benchmark.run_comprehensive_benchmark()

# Or
orchestrator = ComprehensiveScalingOrchestrator()
orchestrator.run_scaling_evaluation()
```

**New approach:**
```python
# Single unified interface
framework = UnifiedRAGASEvaluationFramework(config)
results = framework.run_comprehensive_evaluation()

# Generate report
report = framework.generate_report(results, timestamp)
```

### 5. Update Test Scripts

**Old approach:**
```python
# Scattered test files with different patterns
pytest tests/test_comprehensive_rag_benchmark.py
pytest tests/test_enterprise_rag.py
pytest tests/test_scaling_evaluation.py
```

**New approach:**
```python
# Single comprehensive test suite
pytest tests/test_unified_e2e_rag_evaluation.py -v

# Run specific test categories
pytest tests/test_unified_e2e_rag_evaluation.py::TestRAGASIntegration -v
```

## Configuration Migration

### Environment Variables

The new framework supports environment variables with a consistent naming pattern:

```bash
# Database configuration
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_NAMESPACE=USER
export IRIS_USERNAME=demo
export IRIS_PASSWORD=demo
export IRIS_CONNECTION_TYPE=dbapi

# LLM configuration
export OPENAI_API_KEY=your_key
export LLM_PROVIDER=openai
export LLM_MODEL_NAME=gpt-3.5-turbo

# Evaluation configuration
export ENABLE_RAGAS=true
export NUM_ITERATIONS=3
export PARALLEL_EXECUTION=false
```

### Configuration Files

Create a configuration file based on the provided templates:

```bash
# Copy and modify the default configuration
cp eval/config/default_config.json eval/config/my_config.json

# Edit the configuration
vim eval/config/my_config.json

# Use the configuration
python scripts/run_unified_evaluation.py --config eval/config/my_config.json
```

## Feature Comparison

### RAGAS Integration

**Old:** Inconsistent RAGAS integration across different evaluation scripts
**New:** Comprehensive RAGAS integration with all metrics:
- Answer Relevancy
- Context Precision
- Context Recall
- Faithfulness
- Answer Similarity
- Answer Correctness

### Statistical Analysis

**Old:** Limited or no statistical analysis
**New:** Built-in statistical significance testing:
- Pairwise comparisons between techniques
- T-tests and Mann-Whitney U tests
- Performance percentile analysis

### Error Handling

**Old:** Basic error handling, often failing entire evaluation
**New:** Robust error handling:
- Graceful degradation
- Detailed error logging
- Retry logic with configurable attempts
- Timeout handling

### Visualization

**Old:** Basic or no visualization
**New:** Comprehensive visualization suite:
- Performance comparison charts
- RAGAS metrics comparison
- Spider/radar charts
- Statistical significance indicators

## Common Migration Issues

### 1. Import Errors

**Problem:** `ImportError: cannot import name 'BasicRAGPipeline' from 'src.deprecated.basic_rag'`

**Solution:** Update imports to use `core_pipelines`:
```python
from core_pipelines.basic_rag_pipeline import BasicRAGPipeline
```

### 2. Parameter Mismatch

**Problem:** `TypeError: __init__() got an unexpected keyword argument 'connection'`

**Solution:** Use consistent parameter names:
```python
# Old: connection, iris_connection, conn
# New: iris_connector (consistent across all pipelines)
```

### 3. Missing Configuration

**Problem:** `KeyError: 'openai_api_key'`

**Solution:** Use the configuration manager:
```python
config_manager = ConfigManager()
config = config_manager.load_config("path/to/config.json")
# Or use environment variables
config = config_manager.get_config()
```

### 4. RAGAS Dependency

**Problem:** `ImportError: No module named 'ragas'`

**Solution:** Install RAGAS dependencies:
```bash
pip install ragas datasets langchain-openai langchain-community
```

## Best Practices

1. **Use Configuration Files**: Store your configuration in version-controlled JSON/YAML files
2. **Environment Variables for Secrets**: Use environment variables for sensitive data like API keys
3. **Test Migration**: Run the comprehensive test suite after migration
4. **Incremental Migration**: Migrate one evaluation script at a time
5. **Validate Results**: Compare results between old and new frameworks initially

## Example Migration Script

Here's a complete example of migrating an old evaluation script:

**Old Script:**
```python
# old_evaluation.py
import os
from eval.comprehensive_rag_benchmark_with_ragas import ComprehensiveRAGBenchmark

# Hard-coded configuration
os.environ["OPENAI_API_KEY"] = "your_key"

# Run benchmark
benchmark = ComprehensiveRAGBenchmark()
results = benchmark.run_comprehensive_benchmark()
print(results)
```

**New Script:**
```python
# new_evaluation.py
from eval.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework
from eval.config_manager import ConfigManager
from datetime import datetime

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("eval/config/production_config.json")

# Initialize framework
framework = UnifiedRAGASEvaluationFramework(config)

# Run evaluation
results = framework.run_comprehensive_evaluation()

# Generate and save report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = framework.generate_report(results, timestamp)

# Save results
framework.save_results(results, f"reports/evaluation_{timestamp}")
print(f"Evaluation complete. Results saved to reports/evaluation_{timestamp}")
```

## Support and Troubleshooting

If you encounter issues during migration:

1. Check the comprehensive test suite for examples
2. Review the framework documentation in [`eval/README_UNIFIED_FRAMEWORK.md`](../eval/README_UNIFIED_FRAMEWORK.md)
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Validate your configuration using the ConfigManager's validation methods

The unified framework is designed to be backward-compatible where possible, but the benefits of migration include better maintainability, consistent behavior, and access to new features like comprehensive RAGAS evaluation and statistical analysis.