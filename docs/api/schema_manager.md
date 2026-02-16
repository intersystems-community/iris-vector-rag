# SchemaManager (GraphRAG)

This document describes GraphRAG-specific schema behaviors managed by `SchemaManager`.

## Dependency Requirement

- **GraphRAG pipelines require `iris-vector-graph`**.
- Missing package → `ImportError` with install guidance.
- Non-GraphRAG pipelines remain functional without the package.

## Core Methods

### `ensure_iris_vector_graph_tables(pipeline_type="graphrag")`

- Ensures tables exist:
  - `rdf_labels`
  - `rdf_props`
  - `rdf_edges`
  - `kg_NodeEmbeddings_optimized`
- Records per-table errors and returns `InitializationResult`.
- Logs initialization success or failure.

### `validate_graph_prerequisites(pipeline_type="graphrag")`

- Validates package presence, table existence, and schema structure.
- Logs PPR availability.
- Returns `ValidationResult` unless GraphRAG package missing (raises `ImportError`).

## Logging

- Success logs include initialization completion and PPR availability.
- Failure logs include per-table error details and schema mismatch summary.
