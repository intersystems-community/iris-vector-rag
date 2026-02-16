# Changelog

## v0.5.4

- Fix IRIS connection API usage (replace unsupported `iris.connect()` with supported APIs).
- Require `iris-vector-graph` for GraphRAG pipelines; fail fast with clear ImportError when missing.
- Auto-initialize GraphRAG schema tables with validation, logging, and per-table error reporting.
