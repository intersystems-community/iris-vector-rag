# GraphRAG Test Fixtures

This directory contains test fixtures for GraphRAG functionality testing.

## Structure

- `medical_docs.json` - Medical domain test documents with entities
- `technical_docs.json` - Technical/AI domain test documents
- `fixture_service.py` - Fixture loading and management service
- `validator_service.py` - Test data validation service

## Usage

```python
from tests.fixtures.graphrag.fixture_service import load_fixtures

fixtures = load_fixtures('medical')
for doc in fixtures:
    # Use in tests
    pass
```

## Fixture Requirements

All test documents must meet these criteria:

1. **Content Length**: Minimum 100 characters
2. **Entity Count**: Minimum 2 entities per document
3. **Entity Presence**: All expected entities must appear in content
4. **Relationship Consistency**: Source and target entities must exist

See `validator_service.py` for validation details.
