# Context Reduction Testing with Real Data (Testcontainer Approach)

**Note:** For an overview of context reduction strategies, see [`docs/CONTEXT_REDUCTION_STRATEGY.md`](docs/CONTEXT_REDUCTION_STRATEGY.md:1). For general project testing, refer to [`docs/TESTING.md`](docs/TESTING.md:1). This document details a specific approach for testing context reduction strategies using Testcontainers for managing IRIS instances.

## Current Testing Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, testing context reduction strategies with newly loaded real PMC data that requires vector embeddings is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, preventing the successful loading of document embeddings. While text data can be loaded, and context reduction can be tested on text or with mock/pre-existing embeddings, full validation with new real embeddings is impacted.

For more details on this blocker, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1).

## Overview

This document describes a framework for testing context reduction strategies using Testcontainers to spin up ephemeral IRIS database instances. This allows for:

1. Isolated IRIS database containers for testing.
2. Loading real PMC medical articles (text content) as test data.
3. Measuring and validating context reduction effectiveness (potentially with mock or pre-loaded embeddings due to the blocker).

## Prerequisites

- Docker installed and running.
- Python 3.11+ environment set up with `uv` as per [`README.md`](README.md:1). Ensure your virtual environment is active.
- PMC XML files in the data directory (e.g., `data/pmc_oas_downloaded/`).

## Installation

Install `testcontainers-iris` if you intend to use this Testcontainer-based approach:

```bash
# Ensure .venv is active
uv pip install "testcontainers-iris>=1.2.0" "testcontainers>=3.7.0"
```
Project dependencies should already be installed via `uv pip install -r requirements.txt` or `uv pip install .` as per [`README.md`](README.md:1).

## Running Tests

Tests using this Testcontainer approach are typically invoked via `pytest`, targeting specific test files or markers. (The `poetry run <script_alias>` commands below are illustrative of how `pyproject.toml` might define shortcuts; direct `pytest` commands are generally preferred for clarity if aliases are not universally known).

```bash
# Example: Run tests specifically designed for context reduction with Testcontainers
# (Assuming tests/test_context_reduction.py uses @pytest.mark.force_testcontainer or similar)
pytest tests/test_context_reduction.py -m force_testcontainer

# Example: Run GraphRAG tests that might be set up to use Testcontainers
pytest tests/test_graphrag.py -m force_testcontainer # Adjust marker as needed

# General command to run tests marked for Testcontainers
pytest -m force_testcontainer tests/
```

## Test Architecture

The test framework consists of these key components:

### 1. Enhanced IRIS Connector

The `common/iris_connector.py` module now includes:
- Support for real connections, mock connections, and testcontainers
- Automatic cleanup of containers when tests finish
- Connection prioritization (real > testcontainer > mock)

### 2. Test Fixtures

The `tests/conftest.py` file provides:
- `iris_testcontainer` - Creates a containerized IRIS instance
- `iris_testcontainer_connection` - Provides a connection to the container
- `iris_with_pmc_data` - Pre-loads PMC documents into the container
- `real_embedding_model` - Provides real or mock embeddings

### 3. Knowledge Graph Builder

The `tests/utils.py` module contains:
- Functions to process PMC documents
- Knowledge graph construction utilities
- Entity extraction and relationship generation
- Context reduction measurement tools

### 4. Test Runner

The `scripts_to_review/run_real_data_tests.py` script (Note: path and status to be confirmed post Phase 0 review) might offer:
- Selection of specific RAG techniques to test with Testcontainers.
- Control over document sample size.
- Options for real or mock embeddings.

## Measuring Context Reduction

The framework provides robust measurements of context reduction:

1. **Document Count Reduction**: Measuring how many documents are retrieved vs. total corpus
2. **Token Count Reduction**: Calculating the reduction in total tokens
3. **Entity-Based Reduction**: Evaluating how entities focus the retrieval
4. **Graph Traversal Efficiency**: Measuring the path efficiency in graph-based RAG

## Writing Custom Tests

You can write your own tests following this pattern:

```python
@pytest.mark.force_testcontainer
def test_your_reduction_strategy(iris_with_pmc_data, real_embedding_model):
    # Create your RAG pipeline
    pipeline = YourRAGPipeline(
        iris_connector=iris_with_pmc_data,
        embedding_func=lambda text: real_embedding_model.encode(text)
    )
    
    # Run a query
    query = "What is the relationship between diabetes and insulin?"
    result = pipeline.run(query)
    
    # Measure context reduction
    total_corpus_size = measure_corpus_size(iris_with_pmc_data)
    retrieved_context_size = measure_context_size(result)
    reduction_factor = calculate_reduction_factor(total_corpus_size, retrieved_context_size)
    
    # Verify significant reduction
    assert reduction_factor > 1.5, "Expected at least 1.5x context reduction"
```

## Troubleshooting

If you encounter issues:

1. **Docker Not Running**: Ensure Docker is installed and running
2. **Port Conflicts**: Check for port conflicts with existing containers
3. **IRIS Driver Issues**: Verify you have the correct IRIS driver version
4. **Connection Errors**: Set `TEST_IRIS=true` in your environment

## Tips for Effective Testing

- Use small PMC samples during development (`TEST_PMC_LIMIT=10`)
- Use real embeddings for accurate context testing (Note: loading new real embeddings is currently BLOCKED).
- Register custom `pytest` markers for your testing needs.
- Use session-scoped fixtures to avoid repeated container creation when using Testcontainers.
