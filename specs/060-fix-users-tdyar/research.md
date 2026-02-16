# Research Phase: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature**: 060-fix-users-tdyar
**Date**: 2026-02-09
**Status**: ✅ Complete

---

## Decision 1: IRIS Connection API

**Decision**: Replace any use of `iris.connect()` with supported APIs: `iris.createConnection()` (native) or `iris.dbapi.connect()` (DB-API).

**Rationale**: `iris.connect()` does not exist in intersystems-irispython; using supported APIs eliminates AttributeError and restores connectivity.

**Alternatives considered**:
- Keep `iris.connect()` and add compatibility shim → rejected (non-existent API, brittle).
- Use only `iris_devtester.IRISContainer().dbapi_connection()` everywhere → rejected (test-only helper, not appropriate for runtime library).

---

## Decision 2: GraphRAG Dependency Policy

**Decision**: `iris-vector-graph` is required for HybridGraphRAG pipelines; non-GraphRAG pipelines remain functional without it.

**Rationale**: GraphRAG features depend on iris-vector-graph tables and APIs. Failing fast with clear ImportError avoids silent degradation and mismatched user expectations.

**Alternatives considered**:
- Graceful fallback to uniform scoring when missing → rejected (silent degradation, contradicts requirement).
- Make iris-vector-graph required for entire package → rejected (unnecessary for non-GraphRAG use cases).

---

## Decision 3: Graph Schema Initialization Location

**Decision**: Integrate graph table initialization and validation into `SchemaManager` during pipeline initialization.

**Rationale**: SchemaManager already owns table creation patterns; centralizing graph schema operations ensures consistent behavior and logging.

**Alternatives considered**:
- Initialize tables inside pipeline classes directly → rejected (duplication, inconsistent behavior).
- Separate CLI/manual setup → rejected (violates automatic initialization requirement).
