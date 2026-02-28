# Quickstart: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature**: 060-fix-users-tdyar
**Date**: 2026-02-09

---

## Prerequisites

- Python 3.12
- intersystems-irispython>=5.1.2
- iris-vector-graph>=1.6.0 (required for GraphRAG pipelines)
- Live IRIS instance via iris-devtester (no hardcoded ports)

---

## Scenario 1: Connection API Fix (Bug 1)

**Goal**: Ensure `iris.createConnection()` or `iris.dbapi.connect()` is used; no `iris.connect()` calls remain.

**Run integration tests**:

```bash
pytest tests/integration/test_bug1_connection_fix.py tests/integration/test_iris_connection_integration.py
```

**Expected**:
- All tests pass
- No AttributeError for `iris.connect()`
- Connection errors include host:port/namespace context

---

## Scenario 2: Required Dependency Enforcement (Bug 2)

**Goal**: GraphRAG pipelines fail fast with clear ImportError when iris-vector-graph missing.

**Run integration tests**:

```bash
pytest tests/integration/test_graph_schema_integration.py -k "missing_package"
```

**Expected**:
- ImportError raised with guidance to install iris-vector-graph
- Non-GraphRAG pipelines continue to work

---

## Scenario 3: Automatic Graph Schema Initialization

**Goal**: Tables are created automatically, idempotently, and within performance limits.

**Run integration tests**:

```bash
pytest tests/integration/test_graph_schema_integration.py -k "init"
```

**Expected**:
- Tables created: rdf_labels, rdf_props, rdf_edges, kg_NodeEmbeddings_optimized
- Initialization completes in <5s
- Schema validation completes in <1s
- Re-running does not error (idempotent)

---

## Suggested Full Integration Run

```bash
pytest tests/integration/
```
