# Existing Tests Guide

This guide categorizes all existing tests in the RAG templates project to help you understand which tests are real end-to-end tests versus mock-based tests, and provides clear command sequences for different validation scenarios.

## 🎯 Quick Reference

### Post-Installation Verification
```bash
# Basic functionality check
make test-unit

# Database connectivity
make test-dbapi

# Package validation
make validate-iris-rag
```

### Real End-to-End Validation
```bash
# Comprehensive E2E with 1000+ documents
make test-1000

# All RAG techniques with real data
pytest tests/test_comprehensive_e2e_iris_rag_1000_docs.py -v

# Individual E2E tests
pytest tests/test_e2e_rag_pipelines.py -v
```

### Performance Testing
```bash
# RAGAs evaluation with real data
make test-ragas-1000-enhanced

# Benchmark all techniques
make eval-all-ragas-1000

# TDD performance tests
make test-performance-ragas-tdd
```

### Retrieval Path Testing (NEW)
```bash
# Test all explicit retrieval paths
make test-retrieval-paths

# Test specific pipeline paths
pytest tests/test_hybrid_ifind_retrieval_paths.py -v
pytest tests/test_graphrag_retrieval_paths.py -v
pytest tests/test_fallback_behavior_validation.py -v
```
### 🔧 Comprehensive System Test Workup

The **Comprehensive System Test Workup** is a centralized test orchestration system that provides a unified way to execute, manage, and report on the entire test suite across all categories. This system is designed to give you a complete picture of system health and functionality.

#### Quick Start
```bash
# Run comprehensive system test workup (standard)
make test-system-workup

# Run with verbose output for detailed debugging
make test-system-workup-verbose
```

#### Direct Script Usage
```bash
# Basic usage with default settings
python scripts/run_comprehensive_system_tests.py

# Show all available command-line options
python scripts/run_comprehensive_system_tests.py --help

# Run specific test categories only
python scripts/run_comprehensive_system_tests.py --categories core_pytest validation

# Run specific test targets
python scripts/run_comprehensive_system_tests.py --targets test-unit test-integration validate-iris-rag

# Enable parallel execution for compatible tests
python scripts/run_comprehensive_system_tests.py --parallel

# Skip setup targets (useful for development)
python scripts/run_comprehensive_system_tests.py --skip-setup

# Custom output directory
python scripts/run_comprehensive_system_tests.py --output-dir custom/reports/path
```

#### Key Features

**🎯 Comprehensive Coverage**: The system orchestrates tests across multiple categories:
- **Core Pytest**: Unit, integration, and E2E pytest-based tests
- **Comprehensive E2E**: Large-scale tests with 1000+ documents
- **RAGAS Evaluation**: Quality metrics and performance evaluation
- **TDD RAGAS**: Test-driven development with quality metrics
- **Validation**: System validation and pipeline verification
- **Test Mode Framework**: Mock control and mode-specific testing
- **Data Healing**: Self-healing data validation and repair

**📊 Intelligent Orchestration**:
- Dependency resolution and execution ordering
- Parallel execution for compatible tests
- Setup target management with failure handling
- Category-based filtering and target selection

**📈 Comprehensive Reporting**:
- **JSON Reports**: Machine-readable detailed results with timestamps, durations, and full output
- **Markdown Summaries**: Human-readable executive summaries with failure analysis
- **Execution Logs**: Detailed logging for debugging and audit trails

#### Output and Reports

**Default Output Location**: [`outputs/system_workup_reports/`](../outputs/system_workup_reports/)

**Generated Files**:
- `run_YYYYMMDD_HHMMSS_report.json` - Complete test results in JSON format
- `run_YYYYMMDD_HHMMSS_summary.md` - Executive summary in Markdown format  
- `run_YYYYMMDD_HHMMSS.log` - Detailed execution log

**Report Contents**:
- Environment information (Python version, platform, conda environment)
- Execution summary with success/failure counts by status
- Detailed results table with durations and return codes
- Failure analysis with stderr/stdout excerpts for debugging
- Dependency resolution and execution order documentation

#### Advanced Usage

**List Available Targets**:
```bash
# Show all defined test targets and their descriptions
python scripts/run_comprehensive_system_tests.py --list-targets

# Show available test categories
python scripts/run_comprehensive_system_tests.py --list-categories
```

**Category-Based Execution**:
```bash
# Run only core pytest tests
python scripts/run_comprehensive_system_tests.py --categories core_pytest

# Run validation and setup tests
python scripts/run_comprehensive_system_tests.py --categories validation setup

# Run RAGAS evaluations only
python scripts/run_comprehensive_system_tests.py --categories ragas_evaluation ragas_lightweight
```

**Performance Optimization**:
```bash
# Enable parallel execution with custom worker count
python scripts/run_comprehensive_system_tests.py --parallel --parallel-workers 8

# Set custom timeout for long-running tests
python scripts/run_comprehensive_system_tests.py --timeout 7200  # 2 hours
```

#### Prerequisites

**Environment Setup**:
- Conda environment `iris_vector` must be active or available
- All dependencies installed via `make install`
- IRIS database connection configured and accessible

**Data Requirements**:
- For comprehensive tests: 1000+ PMC documents loaded
- For validation tests: Basic test data and schema setup
- For RAGAS tests: Real document corpus with embeddings

#### Integration with Existing Workflows

The system test workup integrates seamlessly with existing testing workflows:

**Post-Installation Validation**:
```bash
make install
make test-system-workup  # Comprehensive validation
```

**Development Workflow**:
```bash
# Quick validation during development
python scripts/run_comprehensive_system_tests.py --categories core_pytest --skip-setup

# Full validation before commits
make test-system-workup-verbose
```

**CI/CD Integration**:
```bash
# Automated testing with structured output
python scripts/run_comprehensive_system_tests.py --output-dir ci_reports/ --categories core_pytest validation
```

For detailed information about individual test categories and their scope, see the [Testing System Analysis](../testing_system_analysis.md) document.

## 📊 Test Categories

### ✅ Real End-to-End Tests (No Mocks - Use for Final Validation)

These tests use real databases, real data, and real models. They provide the most reliable validation of system functionality.

#### Core E2E Tests
- **[`test_comprehensive_e2e_iris_rag_1000_docs.py`](../tests/test_comprehensive_e2e_iris_rag_1000_docs.py)** - Comprehensive validation of all 7 RAG techniques with 1000+ PMC documents
- **[`test_e2e_iris_rag_full_pipeline.py`](../tests/test_e2e_iris_rag_full_pipeline.py)** - Full pipeline testing with real IRIS database
- **[`test_e2e_rag_pipelines.py`](../tests/test_e2e_rag_pipelines.py)** - Individual RAG technique validation

#### Technique-Specific E2E Tests
- **[`test_colbert_e2e.py`](../tests/test_colbert_e2e.py)** - ColBERT RAG end-to-end validation
- **[`test_crag_e2e.py`](../tests/test_crag_e2e.py)** - CRAG (Corrective RAG) end-to-end validation
- **[`test_graphrag_e2e.py`](../tests/test_graphrag_e2e.py)** - GraphRAG end-to-end validation
- **[`test_hyde_e2e.py`](../tests/test_hyde_e2e.py)** - HyDE RAG end-to-end validation
- **[`test_hybrid_ifind_e2e.py`](../tests/test_hybrid_ifind_e2e.py)** - Hybrid iFind RAG end-to-end validation
- **[`test_noderag_e2e.py`](../tests/test_noderag_e2e.py)** - NodeRAG end-to-end validation

#### Data and Infrastructure E2E Tests
- **[`test_real_data_integration.py`](../tests/test_real_data_integration.py)** - Real PMC data integration testing
- **[`test_pmc_processor.py`](../tests/test_pmc_processor.py)** - PMC document processing with real files

**Markers:** `@pytest.mark.requires_real_data`, `@pytest.mark.requires_1000_docs`, `@pytest.mark.e2e`

**Commands:**
```bash
# Run all E2E tests
pytest -m "e2e or requires_real_data" -v

# Run with 1000+ documents
make test-1000

# Individual technique testing
pytest tests/test_colbert_e2e.py -v
```

### ⚠️ Mixed Tests (Some Real, Some Mocks)

These tests combine real components with mocked dependencies. Useful for integration testing but not for final validation.

#### Integration Tests
- **[`test_context_reduction.py`](../tests/test_context_reduction.py)** - Context reduction with real IRIS connection but mocked models
- **[`test_iris_connector.py`](../tests/test_iris_connector.py)** - Database connectivity with fallback to mocks
- **[`test_llm_caching.py`](../tests/test_llm_caching.py)** - LLM caching with real IRIS but mocked LLM
- **[`test_reconciliation_daemon.py`](../tests/test_reconciliation_daemon.py)** - System reconciliation with mixed real/mock components

#### Evaluation Framework Tests
- **[`test_unified_e2e_rag_evaluation.py`](../tests/test_unified_e2e_rag_evaluation.py)** - Evaluation framework with real pipelines but controlled data
- **[`test_ragas_context_debug_harness.py`](../tests/test_ragas_context_debug_harness.py)** - RAGAs debugging with mixed components

**Markers:** `@pytest.mark.integration`

**Commands:**
```bash
# Run integration tests
pytest -m integration -v

# Run specific integration test
pytest tests/test_context_reduction.py::test_context_reduction_end_to_end -v
```

### 🎯 Explicit Retrieval Path Tests (NEW - Essential for Pipeline Validation)

These tests explicitly validate different retrieval paths and fallback behaviors in pipelines. They ensure that fallback mechanisms work correctly and are not buried in integration tests.

#### Hybrid IFind Retrieval Paths
- **[`test_hybrid_ifind_retrieval_paths.py`](../tests/test_hybrid_ifind_retrieval_paths.py)** - Explicitly tests:
  - IFind working path (when indexes are functional)
  - IFind fallback to LIKE search (when IFind fails)
  - Vector-only results (when text search returns nothing)
  - Result fusion (combining scores from both systems)
  - Empty results handling
  - Score normalization

#### GraphRAG Retrieval Paths  
- **[`test_graphrag_retrieval_paths.py`](../tests/test_graphrag_retrieval_paths.py)** - Explicitly tests:
  - Graph-only retrieval (entity-based traversal)
  - Vector-only retrieval (no entities extracted)
  - Combined graph + vector retrieval
  - Entity extraction failure handling
  - Graph traversal at different depths (0, 1, 2)
  - Entity confidence threshold filtering

#### Fallback Behavior Validation
- **[`test_fallback_behavior_validation.py`](../tests/test_fallback_behavior_validation.py)** - Tests all pipelines for:
  - Index creation failures (IFind, etc.)
  - Component failures (entity extraction, chunking, hypothesis generation)
  - Embedding service failures
  - Database connection failures
  - Partial results handling (return what's available)

**Markers:** `@pytest.mark.retrieval_paths`

**Commands:**
```bash
# Run all retrieval path tests
make test-retrieval-paths

# Run specific pipeline path tests
pytest tests/test_hybrid_ifind_retrieval_paths.py -v
pytest tests/test_graphrag_retrieval_paths.py -v

# Run specific test case
pytest tests/test_hybrid_ifind_retrieval_paths.py::TestHybridIFindRetrievalPaths::test_ifind_fallback_to_like_search -v
```

### ❌ Mock-Heavy Tests (Skip for Final Validation)

These tests primarily use mocks and are designed for unit testing and development. They're fast but don't validate real system behavior.

#### Unit Tests
- **[`test_bench_runner.py`](../tests/test_bench_runner.py)** - Benchmark runner with mocked dependencies
- **[`test_simple_api_phase1.py`](../tests/test_simple_api_phase1.py)** - Simple API with mocked pipelines
- **[`test_pipelines/test_refactored_pipelines.py`](../tests/test_pipelines/test_refactored_pipelines.py)** - Pipeline testing with mocked storage and models

#### Mock-Based Component Tests
- **[`test_monitoring/test_health_monitor.py`](../tests/test_monitoring/test_health_monitor.py)** - Health monitoring with mocked system resources
- **[`test_monitoring/test_system_validator.py`](../tests/test_monitoring/test_system_validator.py)** - System validation with mocked components
- **[`test_validation/`](../tests/test_validation/)** - Validation framework tests with extensive mocking

#### Development and Debug Tests
- **[`debug_basic_rag_ragas_retrieval.py`](../tests/debug_basic_rag_ragas_retrieval.py)** - Debug harness with mocked components
- **[`test_ipm_integration.py`](../tests/test_ipm_integration.py)** - IPM integration with mocked subprocess calls

**Markers:** `@pytest.mark.unit`

**Commands:**
```bash
# Run unit tests only
pytest -m unit -v

# Run all mock-based tests
pytest tests/test_pipelines/ tests/test_monitoring/ tests/test_validation/ -v
```

## 🔍 Identifying Test Types

### Patterns for Real E2E Tests

Look for these patterns to identify real end-to-end tests:

```python
# Real database connections
@pytest.mark.requires_real_db
@pytest.mark.requires_real_data
@pytest.mark.e2e

# Real data fixtures
def test_with_real_data(iris_connection, use_real_data):
    if not use_real_data:
        pytest.skip("Real data required")

# Environment variable checks
required_env_vars = ["IRIS_HOST", "IRIS_PORT", "IRIS_NAMESPACE"]
for var in required_env_vars:
    if var not in os.environ:
        pytest.skip(f"Environment variable {var} not set")

# Real model loading
embedding_model = get_embedding_model(mock=False)
llm_func = get_llm_func(mock=False)
```

### Patterns for Mock-Heavy Tests

Look for these patterns to identify mock-heavy tests:

```python
# Extensive mocking
from unittest.mock import Mock, patch, MagicMock

@patch('module.function')
def test_with_mocks(mock_function):

# Mock fixtures
@pytest.fixture
def mock_iris_connector():
    return MagicMock()

# Mock assertions
mock_function.assert_called_once()
assert isinstance(result, MockClass)
```

### Patterns for Mixed Tests

Look for these patterns to identify mixed tests:

```python
# Integration markers
@pytest.mark.integration

# Conditional real/mock usage
if real_iris_available():
    connection = get_real_connection()
else:
    connection = get_mock_connection()

# Real database with mocked models
def test_integration(iris_connection, mock_embedding_func):
```

## 🚀 Command Sequences

### Post-Installation Verification

Run these commands after installing the package to verify basic functionality:

```bash
# 1. Verify package installation
make validate-iris-rag

# 2. Test database connectivity
make test-dbapi

# 3. Run unit tests
make test-unit

# 4. Check data availability
make check-data

# 5. Validate pipeline configurations
make validate-all-pipelines
```

### Real End-to-End Validation

For comprehensive validation with real data and components:

```bash
# 1. Ensure 1000+ documents are loaded
make load-1000

# 2. Run comprehensive E2E test
make test-1000

# 3. Run individual technique E2E tests
pytest tests/test_*_e2e.py -v

# 4. Run RAGAs evaluation
make test-ragas-1000-enhanced

# 5. Performance benchmarking
make eval-all-ragas-1000
```

### Performance Testing

For performance analysis and benchmarking:

```bash
# 1. TDD performance tests with RAGAs
make test-performance-ragas-tdd

# 2. Scalability testing
make test-scalability-ragas-tdd

# 3. Comprehensive benchmark
make ragas-full

# 4. Individual pipeline debugging
make debug-ragas-basic
make debug-ragas-colbert
make debug-ragas-hyde
```

### Development Testing

For development and debugging:

```bash
# 1. Fast unit tests
pytest tests/test_pipelines/ -v

# 2. Integration tests
pytest -m integration -v

# 3. Mock-based component tests
pytest tests/test_monitoring/ tests/test_validation/ -v

# 4. Debug specific issues
pytest tests/debug_* -v
```

## 🎛️ Test Mode Configuration

The project supports different test modes controlled by the [`test_modes.py`](../tests/test_modes.py) system:

### Test Modes

- **UNIT**: Fast tests with mocks (development)
- **INTEGRATION**: Mixed real/mock tests
- **E2E**: Full end-to-end tests with real components (final validation)

### Setting Test Mode

```bash
# Set via environment variable
export RAG_TEST_MODE=e2e
pytest tests/

# Auto-detection based on available resources
# - If database available: defaults to integration
# - If no database: defaults to unit
```

### Mode-Specific Behavior

```python
# Tests are automatically skipped based on mode
@pytest.mark.unit      # Only runs in unit mode
@pytest.mark.e2e       # Only runs in e2e mode
@pytest.mark.integration  # Runs in integration mode

# Fixtures respect mode settings
@pytest.fixture
def ensure_no_mocks():
    """Ensures no mocks are used in E2E mode"""
    if not MockController.are_mocks_disabled():
        pytest.skip("Test requires mocks to be disabled")
```

## 📋 Test Selection Guidelines

### For Final Validation
- Use only **✅ Real E2E Tests**
- Run with `make test-1000` or `pytest -m "e2e or requires_real_data"`
- Ensure 1000+ documents are loaded
- Verify all environment variables are set

### For Development
- Use **❌ Mock-Heavy Tests** for fast iteration
- Run with `pytest -m unit` or `make test-unit`
- No external dependencies required

### For Integration Testing
- Use **⚠️ Mixed Tests** for component integration
- Run with `pytest -m integration`
- Requires database but allows mocked models

### For Performance Analysis
- Use **✅ Real E2E Tests** with performance markers
- Run with `make test-performance-ragas-tdd`
- Includes timing and resource usage metrics

### For Retrieval Path Validation (Critical)
- Use **🎯 Explicit Retrieval Path Tests**
- Run with `make test-retrieval-paths`
- Essential for validating fallback behaviors
- Ensures robustness when components fail

## 🔧 Troubleshooting

### Common Issues

1. **Tests Skip Due to Missing Environment Variables**
   ```bash
   # Set required variables
   export IRIS_HOST=localhost
   export IRIS_PORT=1972
   export IRIS_NAMESPACE=USER
   export IRIS_USERNAME=demo
   export IRIS_PASSWORD=demo
   ```

2. **Insufficient Test Data**
   ```bash
   # Load more documents
   make load-1000
   make check-data
   ```

3. **Mock Conflicts in E2E Mode**
   ```bash
   # Ensure E2E mode is set
   export RAG_TEST_MODE=e2e
   pytest tests/test_comprehensive_e2e_iris_rag_1000_docs.py -v
   ```

4. **Database Connection Issues**
   ```bash
   # Test connectivity
   make test-dbapi
   
   # Check Docker container
   make docker-logs
   ```

### Test Debugging

```bash
# Run with verbose output
pytest tests/test_name.py -v -s

# Run specific test method
pytest tests/test_name.py::test_method_name -v

# Run with debugging
pytest tests/test_name.py --pdb

# Show test markers
pytest --markers
```

## 📚 Related Documentation

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Examples](EXAMPLES.md)** - Usage examples and patterns
- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrading and migration information

---

**Note**: This guide reflects the current test structure. As the project evolves, test categorizations may change. Always verify test behavior by examining the actual test code and markers.
