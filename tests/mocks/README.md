# Standardized Testing Mocks

This module provides standardized mock implementations for RAG templates testing. These mocks help reduce code duplication and provide consistent, reliable test behavior across different RAG implementations.

## Overview

The module is organized into specialized mock components:

- `db.py`: Database connectivity mocks (IRIS connector, cursor)
- `models.py`: Machine learning model mocks (embeddings, LLM, ColBERT)

## How to Use

### In Test Files

Import and use mocks directly from conftest.py fixtures:

```python
# Your test file

def test_some_feature(mock_iris_connector, mock_embedding_func):
    # The fixtures are pre-configured with the standardized mocks
    result = your_function_under_test(
        iris_connector=mock_iris_connector,
        embedding_func=mock_embedding_func
    )
    assert result is not None
```

### Direct Mock Usage

You can also import the mock implementations directly:

```python
from tests.mocks.db import MockIRISConnector
from tests.mocks.models import mock_embedding_func

# Create your own instance
connector = MockIRISConnector()

# Use the function directly
embeddings = mock_embedding_func("Test text", dimensions=10)
```

## Mock Capabilities

### MockIRISConnector / MockIRISCursor

- Tracks SQL queries and parameters
- Stores and retrieves documents
- Manages token embeddings
- Supports knowledge graph operations
- Implements context manager protocol

### mock_embedding_func

- Generates deterministic embeddings
- Supports configurable dimensions
- Handles both single strings and lists of strings

### mock_llm_func

- Generates deterministic responses
- Supports predefined responses for specific prompts
- Configurable response length

### mock_colbert_doc_encoder / mock_colbert_query_encoder

- Generates token-level embeddings
- Configurable token count and dimensions
- Suitable for testing ColBERT-based retrieval

## Extending

To add new mock implementations:

1. Add your mock to the appropriate file or create a new one
2. Update `__init__.py` to export your new mock
3. Consider adding a pytest fixture in conftest.py

## Benefits

- Reduces test boilerplate
- Provides consistent behavior across tests
- Simulates real components effectively
- Makes tests more maintainable and readable
