# Real Data Testing with Testcontainers (Supplementary Guide)

**Note:** The primary and most up-to-date guide for all testing is [`docs/TESTING.md`](docs/TESTING.md:1). This document provides supplementary details on using Testcontainers for running tests with real PMC medical data in isolated IRIS database containers. This may be an alternative or specialized testing approach.

## Current Testing Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, testing with newly loaded real PMC data that requires vector embeddings is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, preventing the successful loading of document embeddings. While Testcontainers can spin up IRIS instances and load text data, full validation with new real embeddings is impacted.

For more details on this blocker, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1).

## Overview

This Testcontainer-based approach aims to provide a way to test RAG techniques against real data (primarily text content, or mock/pre-loaded embeddings due to the blocker) with isolation and reproducibility:

1. **Spin up ephemeral IRIS containers** for each test session.
2. **Automatically populate** them with real PMC data (text content; embedding loading is currently BLOCKED).
3. **Run tests** against this database.
4. **Clean up** when done.

This approach can be useful for CI/CD pipelines or specific isolated test scenarios.

## Prerequisites & Installation

- Docker installed and running.
- Python 3.11+ environment set up with `uv` as per [`README.md`](README.md:1). Ensure your virtual environment is active.
- To use this Testcontainer framework, install `testcontainers-iris`:
  ```bash
  # Ensure .venv is active
  uv pip install "testcontainers-iris>=1.2.0" "testcontainers>=3.7.0"
  ```
  Project dependencies should already be installed via `uv pip install -r requirements.txt` or `uv pip install .` as per [`README.md`](README.md:1).

## Running Tests

There are two main ways to run these Testcontainer-based real data tests:

### 1. Using the `run_real_data_tests.py` script (from `scripts_to_review/`)

A script, potentially `scripts_to_review/run_real_data_tests.py` (Note: path and status to be confirmed post Phase 0 review), may provide an easy way to run these tests:

```bash
# Example: Run all tests with testcontainers
python scripts_to_review/run_real_data_tests.py --verbose

# Example: Test only GraphRAG (embeddings might be mocked or pre-loaded due to blocker)
python scripts_to_review/run_real_data_tests.py --techniques graphrag

# Example: Use mock embeddings for faster testing
python scripts_to_review/run_real_data_tests.py --techniques graphrag --use-mock-embeddings

# Example: Use fewer PMC documents for faster tests
python scripts_to_review/run_real_data_tests.py --pmc-limit 10
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
- `iris_with_pmc_data`: Preloads PMC data (text content) into the container. (Loading of new embeddings is BLOCKED).

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
3. Add the technique to the relevant test runner script (e.g., `scripts_to_review/run_real_data_tests.py` if using that).

## Troubleshooting

Common issues:

1. **Docker not running**: Make sure Docker is installed and running
2. **Port conflicts**: If tests fail with port binding errors, stop other containers
3. **Missing dependencies**: Ensure `testcontainers-iris` and `testcontainers` are installed in your `uv` environment (see Installation section).
