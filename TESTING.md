# RAG Templates Testing Guide

This document provides instructions for testing the RAG templates with different datasets and scale levels, from small test sets to large-scale (92,000+ documents) testing.

## Testing Requirements

According to our .clinerules file, all tests must:

1. Use pytest for testing (not shell scripts)
2. Verify RAG techniques work with real data
3. Use real PMC documents (not synthetic data)
4. Run with at least 1000 documents
5. Test the complete pipeline from ingestion to answer generation
6. Include assertions on actual results (not just logging)

## Testing Levels

### 1. Basic Component Tests

Run basic component tests to verify individual components:

```bash
# Run all basic component tests
pytest tests/test_basic_rag.py tests/test_colbert.py tests/test_noderag.py tests/test_graphrag.py tests/test_hyde.py tests/test_crag.py
```

### 2. 1000+ Document Tests

These tests verify functionality with at least 1000 documents (required by .clinerules):

```bash
# Run all tests with 1000+ documents
make test-1000

# Run a specific technique with 1000+ documents
PYTEST_CONFTEST_PATH=tests/conftest_real_pmc.py pytest tests/test_minimal_real_pmc.py::test_basic_rag_simple -v
```

### 3. Large-Scale Tests (92,000+ Documents)

For large-scale testing as specified in .clinerules:

```bash
# Run the full 92K document test suite (may take hours)
./run_92k_docs_tests.sh

# Run large-scale tests with performance metrics
python run_large_scale_tests.py --documents=92000 --measure-performance
```

## Performance Metrics

To collect performance metrics during large-scale tests:

```bash
# Run with performance measurement enabled
python run_large_scale_tests.py --documents=92000 --measure-performance

# Generate performance report
python -m eval.bench_runner --scale=large --report-format=json
```

Performance metrics are saved to test_results/ with timestamps.

## Test Structure

- **test_minimal_real_pmc.py**: Core tests with real PMC data and 1000+ docs
- **test_full_pipeline_integration.py**: Tests that verify end-to-end functionality with realistic data
- **test_technique_mocked_retrieval.py**: Tests each technique with mocked retrieval to validate multi-document processing
- **test_all_with_1000_docs.py**: Tests all techniques with 1000+ documents
- **tests/test_[technique]_[scale].py**: Individual technique tests at different scales

## Vector Similarity Function

The VECTOR_COSINE function, which is built into IRIS SQL, is used for vector similarity calculations. The TO_VECTOR helper function in common/vector_similarity.sql is installed during database initialization to facilitate proper vector similarity testing in both small and large-scale scenarios.

## Results Validation

All tests perform assertions on:
1. Document retrieval functionality
2. Proper document processing by each technique 
3. Answer generation from retrieved documents

## Coverage Testing

To see test coverage statistics:

```bash
# Run tests with coverage
python -m pytest --cov=. tests/

# Generate coverage report
python -m pytest --cov=. --cov-report=html tests/
```

This will show which parts of the codebase have test coverage and which need additional testing.

## Test Environment

Tests can run in different environments:
- Local mock environment (default)
- Docker test container (uses testcontainers)
- Real IRIS instance (requires configuration)

To specify the environment:

```bash
# Use testcontainer
TEST_IRIS=true pytest tests/test_minimal_real_pmc.py

# Use real IRIS instance
IRIS_HOST=localhost IRIS_PORT=1972 pytest tests/test_minimal_real_pmc.py
