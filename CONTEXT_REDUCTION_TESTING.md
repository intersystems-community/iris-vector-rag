# Context Reduction Testing with Real Data

This document explains how to test context reduction strategies using real PMC data with Poetry and Testcontainers.

## Overview

The RAG templates project now includes a comprehensive framework for testing context reduction strategies with real data. By using testcontainers, you can:

1. Spin up ephemeral IRIS database containers for testing
2. Load real PMC medical articles as test data
3. Measure and validate context reduction effectiveness
4. Compare different RAG techniques against the same queries

## Prerequisites

- Docker installed and running
- Poetry for dependency management
- PMC XML files in the data directory

## Installation

Install the required dependencies using Poetry:

```bash
# Add testcontainers-iris dependency
poetry add testcontainers-iris@>=1.2.0 testcontainers@>=3.7.0

# Install all dependencies
poetry install
```

## Running Tests with Poetry

The project includes several Poetry scripts to easily run the tests:

```bash
# Test GraphRAG with testcontainer
poetry run test-graphrag-testcontainer

# Test context reduction specifically
poetry run test-context-reduction

# Run all testcontainer tests
poetry run test-with-testcontainer
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

The `run_real_data_tests.py` script offers:
- Selection of specific RAG techniques to test
- Control over document sample size
- Options for real or mock embeddings

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
- Use real embeddings for accurate context testing
- Register custom pytest markers for your testing needs
- Use session-scoped fixtures to avoid repeated container creation
