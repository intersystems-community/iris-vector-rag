# Real Data Testing with Testcontainers

This document explains how to run tests using real PMC medical data with isolated IRIS database containers.

## Overview

These testing utilities provide a no-boilerplate way to test all RAG techniques against real data while maintaining isolation and reproducibility. Instead of mocking databases or requiring a dedicated IRIS instance, this approach uses testcontainers to:

1. **Spin up ephemeral IRIS containers** for each test session
2. **Automatically populate** them with real PMC data
3. **Run tests** against the real database
4. **Clean up** when done

This approach allows for true end-to-end testing with real data while keeping tests isolated, reproducible, and suitable for CI/CD pipelines.

## Installation

To use this testing framework, you need to install the testcontainers-iris package and its dependencies:

```bash
pip install -r testcontainer-requirements.txt
```

You also need Docker installed and running on your system.

## Running Tests

There are two ways to run the real data tests:

### 1. Using the run_real_data_tests.py script

The easiest way is to use the provided script:

```bash
# Run all tests with testcontainers
./run_real_data_tests.py --verbose

# Test only GraphRAG with real embeddings (default)
./run_real_data_tests.py --techniques graphrag

# Use mock embeddings for faster testing (CI environments)
./run_real_data_tests.py --techniques graphrag --use-mock-embeddings

# Use fewer PMC documents for faster tests
./run_real_data_tests.py --pmc-limit 10
```

### 2. Directly with pytest

You can also run the tests directly with pytest by setting the appropriate environment variables:

```bash
# Set up environment
export TEST_IRIS=true
export TEST_PMC_LIMIT=20
export USE_MOCK_EMBEDDINGS=true

# Run specific test
pytest tests/test_graphrag_with_testcontainer.py -v
```

## Test Organization

The real data tests are designed to follow these principles:

1. **Test-Driven Development (TDD)**: Tests are written before implementation to guide development
2. **Real Data**: Tests use actual PMC XML files instead of synthetic data
3. **Isolation**: Each test session gets its own clean database instance
4. **Standardization**: Common fixtures and utilities ensure all RAG techniques are tested consistently

## Key Components

This framework consists of several components:

### 1. Enhanced IRIS Connector

The `common/iris_connector.py` module provides:
- Support for testcontainers, real connections, and mocks
- Automatic cleanup of testcontainers
- Environment-aware connection selection

### 2. Session-Scoped Testcontainer Fixtures

The `tests/conftest.py` file includes:
- `iris_testcontainer`: Creates an ephemeral IRIS container
- `iris_testcontainer_connection`: Provides a connection to the container
- `iris_with_pmc_data`: Preloads PMC data into the container

### 3. Test Utilities

The `tests/utils.py` module offers:
- `build_knowledge_graph()`: Builds a KG from real PMC documents
- `run_standardized_queries()`: Runs the same queries against any RAG technique
- `compare_rag_techniques()`: Compares multiple techniques with the same data

### 4. Technique-Specific Test Files

Each RAG technique has its own testcontainer test file:
- `tests/test_graphrag_with_testcontainer.py`
- `tests/test_colbert_with_testcontainer.py` (to be implemented)
- etc.

## Context Reduction Verification

These tests specifically verify that context reduction techniques are working properly by:

1. Measuring the total corpus size
2. Measuring the retrieved context size 
3. Calculating the reduction factor
4. Verifying that the reduction is significant

## Example Test

Here's an example of a test that verifies GraphRAG's context reduction:

```python
@pytest.mark.force_testcontainer
def test_graphrag_context_reduction(iris_with_pmc_data, real_embedding_model):
    # Create pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_pmc_data,
        embedding_func=lambda text: real_embedding_model.encode(text),
        llm_func=lambda prompt: f"Response to: {prompt[:100]}..."
    )
    
    # Run query
    query = "What is the relationship between diabetes and insulin?"
    result = pipeline.run(query)
    
    # Count total document size vs. retrieved context size
    # [calculation code...]
    
    # Verify reduction
    assert reduction_factor > 1.0, "GraphRAG should reduce context"
```

## Extending for New RAG Techniques

To add tests for a new RAG technique:

1. Create a new test file: `tests/test_yourrag_with_testcontainer.py`
2. Use the existing fixtures and utilities
3. Add the technique to the `run_real_data_tests.py` script

## Troubleshooting

Common issues:

1. **Docker not running**: Make sure Docker is installed and running
2. **Port conflicts**: If tests fail with port binding errors, stop other containers
3. **Missing dependencies**: Install required packages with `pip install -r testcontainer-requirements.txt`
