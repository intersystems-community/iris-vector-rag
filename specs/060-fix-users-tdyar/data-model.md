# Data Model: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature**: 060-fix-users-tdyar
**Date**: 2026-02-09

---

## Bug 1: Connection API Fix

### ConnectionEstablisher

**Purpose**: Establish IRIS connections using supported APIs.

**Key fields**:
- `connection_kwargs`: host, port, namespace, username, password (env defaults allowed)
- `api_used`: `createConnection` or `dbapi.connect`

**Validation rules**:
- Must not call `iris.connect()` (non-existent API)
- Errors must include host:port/namespace context

### ConnectionResult

**Fields**:
- `success: bool`
- `api_used: str`
- `error_message: Optional[str]`

---

## Bug 2: GraphRAG Schema Initialization

### GraphTableDetector

**Purpose**: Detect iris-vector-graph availability.

**Fields**:
- `is_available: bool`
- `error: Optional[str]`

**Validation rules**:
- If GraphRAG pipeline requested and package missing → raise `ImportError`

### GraphTableInitializer

**Purpose**: Ensure required graph tables exist.

**Tables**:
- `rdf_labels`
- `rdf_props`
- `rdf_edges`
- `kg_NodeEmbeddings_optimized`

**Validation rules**:
- Creation must be **atomic** (all or none)
- Must be **idempotent** (safe to re-run)
- Log success/failure outcomes

### PrerequisiteValidator

**Purpose**: Validate schema and prerequisites before PPR usage.

**Validation rules**:
- Validate table existence and **schema structure**
- Log PPR availability status
- Fail fast if PPR invoked without required tables

### InitializationResult

**Fields**:
- `created: bool`
- `tables_present: list[str]`
- `errors: list[str]`

### ValidationResult

**Fields**:
- `is_valid: bool`
- `missing_tables: list[str]`
- `schema_errors: list[str]`
- `message: Optional[str]`
