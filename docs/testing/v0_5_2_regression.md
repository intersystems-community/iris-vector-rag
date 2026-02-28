# v0.5.2 Regression Suite Results

This document captures regression checks for backward compatibility with v0.5.2 behaviors.

## Scope

- Connection API compatibility
- GraphRAG schema initialization and validation

## Commands Executed

```bash
pytest tests/unit/test_connection_api.py \
       tests/integration/test_bug1_connection_fix.py \
       tests/integration/test_iris_connection_integration.py

pytest tests/unit/test_schema_detection.py \
       tests/unit/test_schema_initialization.py \
       tests/contract/test_graph_schema_validation.py \
       tests/contract/test_graph_schema_initialization.py \
       tests/integration/test_graph_schema_integration.py
```

## Results

- Connection suite: **17 passed** (warnings only: testcontainers + deprecated ConnectionManager)
- Graph schema suite: **27 passed** (warnings only: testcontainers + deprecated ConnectionPool)

## Notes

- FHIR-AI regression suite is no longer a requirement for this repo.
