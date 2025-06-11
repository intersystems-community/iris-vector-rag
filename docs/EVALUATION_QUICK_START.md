# RAG Evaluation Framework Quick Start Guide

Get started with the unified RAGAS evaluation framework in 5 minutes.

## Prerequisites

1. **Python Environment**
   ```bash
   # Ensure Python 3.8+ is installed
   python --version
   ```

2. **IRIS Database Running**
   ```bash
   # Check if IRIS container is running
   docker ps | grep iris
   
   # If not, start it
   docker-compose -f docker-compose.iris-only.yml up -d
   ```

3. **Dependencies Installed**
   ```bash
   # Install required packages
   pip install numpy pandas matplotlib seaborn plotly
   
   # For RAGAS evaluation (recommended)
   pip install ragas datasets langchain-openai langchain-community
   ```

4. **OpenAI API Key** (for RAGAS evaluation)
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start Examples

### 1. Basic Evaluation (2 minutes)

Run a quick evaluation with minimal configuration:

```bash
# Run with development configuration (fast, limited pipelines)
python scripts/run_unified_evaluation.py --config eval/config/dev_config.json
```

### 2. Single Pipeline Test (1 minute)

Test a specific RAG pipeline:

```bash
# Test only BasicRAG pipeline
python scripts/run_unified_evaluation.py --pipelines BasicRAG --num-iterations 1
```

### 3. Full Evaluation with RAGAS (5 minutes)

Run comprehensive evaluation with semantic metrics:

```bash
# Ensure OpenAI API key is set
export OPENAI_API_KEY="your-api-key"

# Run full evaluation
python scripts/run_unified_evaluation.py \
  --config eval/config/default_config.json \
  --output-format all \
  --output-dir reports/
```

### 4. Custom Configuration (3 minutes)

Create and use a custom configuration:

```bash
# Copy template
cp eval/config/dev_config.json eval/config/my_config.json

# Edit configuration (adjust pipelines, parameters, etc.)
vim eval/config/my_config.json

# Run with custom config
python scripts/run_unified_evaluation.py --config eval/config/my_config.json
```

## Understanding the Output

### Console Output

```
Starting Unified RAGAS Evaluation Framework...
Loading configuration from: eval/config/dev_config.json
Initializing pipelines...
  ✓ BasicRAG initialized
  ✓ HyDE initialized
  ✗ ColBERT failed (missing dependencies)

Running evaluation...
  Query 1/5: "What are the symptoms of COVID-19?"
    BasicRAG: 0.234s, 5 docs retrieved
    HyDE: 0.456s, 5 docs retrieved
  
Calculating RAGAS metrics...
  Answer Relevancy: BasicRAG=0.82, HyDE=0.89
  Faithfulness: BasicRAG=0.91, HyDE=0.94
  
Generating report...
Report saved to: reports/evaluation_20250605_120000/
```

### Report Structure

```
reports/evaluation_20250605_120000/
├── summary.md              # Human-readable summary
├── results.json           # Raw results data
├── metrics_comparison.png # Visual comparison
├── statistical_analysis.txt # Statistical significance tests
└── failed_queries.json    # Debug information
```

## Common Use Cases

### Development Testing

Quick iteration during development:

```python
# Python script for development testing
from eval.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework

# Minimal configuration for speed
config = {
    "pipelines": {
        "BasicRAG": {"enabled": True}
    },
    "evaluation": {
        "num_iterations": 1,
        "enable_ragas": False  # Skip RAGAS for speed
    }
}

framework = UnifiedRAGASEvaluationFramework(config)
results = framework.run_comprehensive_evaluation()
print(f"BasicRAG average response time: {results['BasicRAG']['avg_response_time']:.3f}s")
```

### A/B Testing

Compare two pipeline configurations:

```bash
# Run A/B comparison
python scripts/run_unified_evaluation.py \
  --pipelines BasicRAG HyDE \
  --num-iterations 5 \
  --enable-statistical-analysis
```

### Performance Benchmarking

Measure performance characteristics:

```bash
# Run performance benchmark
python scripts/run_unified_evaluation.py \
  --config eval/config/benchmark_config.json \
  --enable-profiling \
  --output-format json
```

### Production Validation

Validate before deployment:

```bash
# Full validation suite
python scripts/run_unified_evaluation.py \
  --config eval/config/production_config.json \
  --num-iterations 10 \
  --fail-on-regression
```

## Configuration Quick Reference

### Minimal Configuration

```json
{
  "pipelines": {
    "BasicRAG": {
      "enabled": true
    }
  }
}
```

### Development Configuration

```json
{
  "pipelines": {
    "BasicRAG": {"enabled": true},
    "HyDE": {"enabled": true}
  },
  "evaluation": {
    "num_iterations": 1,
    "enable_ragas": false,
    "queries": ["What is machine learning?", "How do vaccines work?"]
  }
}
```

### Production Configuration

```json
{
  "pipelines": {
    "BasicRAG": {"enabled": true, "timeout": 60},
    "HyDE": {"enabled": true, "timeout": 90},
    "ColBERT": {"enabled": true, "timeout": 120},
    "CRAG": {"enabled": true, "timeout": 90},
    "NodeRAG": {"enabled": true, "timeout": 90}
  },
  "evaluation": {
    "num_iterations": 5,
    "enable_ragas": true,
    "parallel_execution": true,
    "max_workers": 4
  },
  "ragas": {
    "metrics": ["answer_relevancy", "faithfulness", "context_precision"]
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'ragas'**
   ```bash
   pip install ragas datasets
   ```

2. **OpenAI API Key Missing**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # Or add to .env file
   ```

3. **Database Connection Failed**
   ```bash
   # Check IRIS is running
   docker ps
   # Check connection settings in config
   ```

4. **Pipeline Initialization Failed**
   ```bash
   # Check pipeline dependencies
   pip install -r requirements.txt
   ```

### Debug Mode

Enable detailed logging:

```bash
# Run with debug logging
python scripts/run_unified_evaluation.py --debug

# Or set in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

1. **Read the Full Documentation**
   - [`eval/README_UNIFIED_FRAMEWORK.md`](../eval/README_UNIFIED_FRAMEWORK.md) - Complete framework documentation
   - [`docs/EVALUATION_BEST_PRACTICES.md`](EVALUATION_BEST_PRACTICES.md) - Best practices guide

2. **Explore Configuration Options**
   - Review [`eval/config/default_config.json`](../eval/config/default_config.json)
   - Create custom configurations for your use case

3. **Run Comprehensive Tests**
   ```bash
   pytest tests/test_unified_e2e_rag_evaluation.py -v
   ```

4. **Integrate with CI/CD**
   - Add evaluation to your build pipeline
   - Set up automated regression testing

## Quick Command Reference

```bash
# Basic evaluation
python scripts/run_unified_evaluation.py

# With specific config
python scripts/run_unified_evaluation.py --config path/to/config.json

# Test specific pipelines
python scripts/run_unified_evaluation.py --pipelines BasicRAG HyDE

# Full evaluation with all outputs
python scripts/run_unified_evaluation.py --output-format all --output-dir reports/

# Quick test (1 iteration, no RAGAS)
python scripts/run_unified_evaluation.py --num-iterations 1 --disable-ragas

# Production validation
python scripts/run_unified_evaluation.py --config production --fail-on-regression

# Debug mode
python scripts/run_unified_evaluation.py --debug --verbose
```

## Getting Help

- **Framework Documentation**: [`eval/README_UNIFIED_FRAMEWORK.md`](../eval/README_UNIFIED_FRAMEWORK.md)
- **Migration Guide**: [`docs/EVALUATION_FRAMEWORK_MIGRATION.md`](EVALUATION_FRAMEWORK_MIGRATION.md)
- **Test Examples**: [`tests/test_unified_e2e_rag_evaluation.py`](../tests/test_unified_e2e_rag_evaluation.py)
- **Configuration Templates**: [`eval/config/`](../eval/config/)

Start with the development configuration and gradually move to more comprehensive evaluations as you become familiar with the framework.