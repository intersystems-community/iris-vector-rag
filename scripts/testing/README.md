# Example Testing Framework

A comprehensive testing framework for validating RAG-Templates example scripts and demonstrations with **live IRIS database integration**, performance monitoring, and CI/CD integration.

## ⚠️ Constitutional Requirement

**All tests MUST execute against a live IRIS database instance** per Section III of the project constitution. Mock mode violates the constitution and should only be used for development debugging.

## Quick Start

```bash
# Run all examples with live IRIS database (constitutional requirement)
make test-examples

# Run specific category with live IRIS
make test-examples-basic

# Run with pattern matching with live IRIS
make test-examples-pattern PATTERN=basic

# Run CI validation mode with live IRIS
make test-examples-ci

# Development debugging only (constitutional violation)
make test-examples-mock
```

## Framework Components

### Core Testing Infrastructure

- **ExampleTestRunner**: Executes examples in isolated environments with performance monitoring
- **ValidationSuite**: Validates outputs against expected formats and quality standards
- **MockProviders**: Deterministic mock implementations for LLM and data providers
- **PerformanceMonitor**: Real-time resource usage tracking during execution

### Mock Provider System (Development Only)

The framework includes mock providers for development debugging, but these **violate the constitution** and should not be used for validation:

```python
# ⚠️ CONSTITUTIONAL VIOLATION - Only for development debugging
mock_llm = MockLLMProvider(mode="realistic")
response = mock_llm.generate_response("What is diabetes?")

# ⚠️ Constitutional requirement: Use live IRIS database instead
mock_data = MockDataProvider()  # Should not be used for validation
```

**Note**: Mock providers remain in the codebase for development debugging purposes only.

### Validation and Quality Assessment

Each example is validated across multiple dimensions:

- **Output Format**: Ensures required fields (answer, sources, metadata) are present
- **Content Quality**: Validates answer length, relevance, and coherence
- **Performance**: Monitors execution time, memory usage, and resource consumption
- **Error Handling**: Categorizes and reports different types of failures

## Usage Patterns

### Local Development

```bash
# Constitutional requirement: Use live IRIS database for validation
python scripts/testing/run_example_tests.py --pattern "basic_rag" --verbose --mode real

# Standard testing with live IRIS (default behavior)
python scripts/testing/run_example_tests.py --mode real --timeout 600

# Validate specific category with live IRIS
python scripts/testing/run_example_tests.py --category advanced --continue-on-failure --mode real

# Development debugging only (constitutional violation)
python scripts/testing/run_example_tests.py --mode mock --pattern "basic_rag"
```

### CI/CD Integration

The framework integrates with GitHub Actions, GitLab CI, and other CI systems:

```bash
# CI script with live IRIS database (constitutional requirement)
scripts/ci/run-example-tests.sh --mode real --continue-on-failure --upload-artifacts

# Environment-specific testing with live IRIS
EXAMPLE_TEST_MODE=real scripts/ci/run-example-tests.sh --verbose
```

### Configuration Management

Examples can be configured individually in `config.yaml`:

```yaml
examples:
  "basic/try_basic_rag_pipeline.py":
    timeout: 180
    expected_outputs: ["answer", "sources"]
    test_queries:
      - "What is diabetes?"
      - "How does insulin work?"
    performance_bounds:
      max_execution_time: 120
      max_memory_mb: 512
```

## Advanced Features

### Performance Benchmarking

```bash
# Generate performance reports
python scripts/testing/run_example_tests.py --mode mock --generate-reports

# Compare against baselines
python scripts/testing/run_example_tests.py --compare-baseline --upload-artifacts
```

### Multi-Environment Testing

```bash
# Test across different Python versions
for version in 3.8 3.9 3.10 3.11; do
  python$version scripts/testing/run_example_tests.py --category basic
done

# Test with different LLM providers
MOCK_LLM_MODE=deterministic python scripts/testing/run_example_tests.py
```

### Integration with Existing Testing

The framework complements existing pytest-based tests:

```bash
# Run all test types together
make test-all && make test-examples
pytest tests/ && python scripts/testing/run_example_tests.py
```

## Example Categories

### Basic RAG Examples
- `try_basic_rag_pipeline.py`: Standard vector similarity search
- Simple query-response validation
- Performance baseline establishment

### Advanced RAG Examples
- `try_crag_pipeline.py`: Corrective RAG with relevance evaluation
- `try_hybrid_graphrag_pipeline.py`: Graph-enhanced retrieval
- Complex multi-step validation

### Demonstration Scripts
- `demo_graph_visualization.py`: Interactive graph visualization
- `demo_ontology_support.py`: Entity extraction and mapping
- Output file and artifact validation

### Reranking Examples
- `try_basic_rerank.py`: Result reranking and scoring
- Quality and relevance metric validation

## Configuration Options

### Execution Modes

- **Mock Mode**: Fast testing with simulated responses (default)
- **Real Mode**: Full integration testing with actual APIs

### Validation Levels

- **Basic**: Syntax and format validation
- **Standard**: Content quality and performance checks
- **Comprehensive**: Deep semantic validation and benchmarking

### Report Generation

- **Markdown**: Human-readable development reports
- **JSON**: Machine-readable CI/CD integration
- **HTML**: Interactive reports with visualizations

## CI/CD Integration

### GitHub Actions

The framework integrates with the existing `.github/workflows/ci.yml`:

```yaml
- name: Run example tests
  run: |
    poetry run scripts/ci/run-example-tests.sh --mode mock --verbose

- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: example-test-results
    path: test-results/examples/
```

### Failure Handling

```bash
# Continue testing after failures for comprehensive reporting
scripts/ci/run-example-tests.sh --continue-on-failure

# Fail fast for quick feedback
scripts/ci/run-example-tests.sh --fail-fast
```

### Artifact Management

Test results are automatically uploaded as CI artifacts:

- `example_test_report_TIMESTAMP.md`: Human-readable summary
- `example_test_report_TIMESTAMP.json`: Machine-readable data
- `ci_summary.txt`: Concise CI status summary

## Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Core settings
EXAMPLE_TEST_MODE=mock
EXAMPLE_TEST_TIMEOUT=300

# LLM provider settings (for real mode)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Performance limits
MAX_MEMORY_MB=1024
MAX_CPU_PERCENT=80

# CI integration
CI_FAIL_ON_EXAMPLE_FAILURE=true
CI_UPLOAD_ARTIFACTS=false
```

## Troubleshooting

### Common Issues

**Import Errors**: Ensure virtual environment is activated and dependencies installed
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Timeout Errors**: Increase timeout for slow examples
```bash
python scripts/testing/run_example_tests.py --timeout 600
```

**Memory Errors**: Run examples individually or increase limits
```bash
python scripts/testing/run_example_tests.py --pattern "specific_example"
```

### Debug Mode

Enable detailed logging and error tracking:
```bash
DEBUG_MODE=true python scripts/testing/run_example_tests.py --verbose
```

### Mock Provider Issues

Verify mock providers are working correctly:
```python
from scripts.testing.mock_providers import MockLLMProvider
provider = MockLLMProvider(mode="deterministic")
response = provider.generate_response("test query")
print(response)  # Should return deterministic mock response
```

## Contributing

### Adding New Examples

1. Create example script in appropriate directory (`scripts/basic/`, `scripts/crag/`, etc.)
2. Add configuration to `scripts/testing/config.yaml`
3. Test with framework: `python scripts/testing/run_example_tests.py --pattern "new_example"`
4. Update validation rules if needed

### Extending Validation

Add custom validators in `scripts/testing/validation_suite.py`:

```python
def validate_custom_output(self, output: str) -> ValidationResult:
    # Custom validation logic
    return ValidationResult(score=0.9, issues=[], metrics={})
```

### Mock Provider Enhancement

Add new response templates in `scripts/testing/mock_providers.py`:

```python
RESPONSE_TEMPLATES = {
    "new_topic": "Comprehensive response for new topic...",
    # Add realistic responses for better testing
}
```

## Performance and Scaling

### Resource Usage

- Mock mode: ~50MB memory, <30s execution per example
- Real mode: ~200MB memory, 60-300s execution per example
- Parallel execution: 2-4x speedup with adequate resources

### Optimization Tips

- Use mock mode for development and CI
- Reserve real mode for integration validation
- Filter by category/pattern for focused testing
- Enable parallel execution for large test suites

## Integration Examples

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: example-tests
      name: Example Tests
      entry: scripts/ci/run-example-tests.sh --mode mock --category basic
      language: script
      pass_filenames: false
```

### Development Workflow

```bash
# Before committing changes
make test-examples-basic  # Quick validation
git commit -m "Updated basic RAG pipeline"

# Before releasing
make test-examples        # Full validation
make test-examples-real   # Integration testing
```

This framework ensures that RAG-Templates examples remain functional, well-documented, and provide consistent user experiences across different environments and use cases.