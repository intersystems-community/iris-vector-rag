# Data Model: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature**: 060-fix-users-tdyar
**Date**: 2025-01-13
**Updated**: 2025-01-14 (Added Bug 1 components)
**Status**: Design Phase

---

## Bug 1 Components: Connection API Fix

### 1. ConnectionEstablisher

**Purpose**: Establish IRIS database connection using correct intersystems-irispython API.

**Attributes**:
- `host`: str - IRIS database host
- `port`: int - IRIS database port
- `namespace`: str - IRIS namespace
- `user`: str - Database user
- `password`: str - Database password
- `connection_timeout`: int - Timeout in seconds (default: 30)

**Methods**:
- `create_connection(host: str, port: int, namespace: str, user: str, password: str) → ConnectionResult`: Create connection using correct API
- `validate_connection(conn: Connection) → bool`: Verify connection is active

**Relationships**:
- Used by: ConnectionManager (iris_dbapi_connector.py)
- Produces: ConnectionResult
- Replaces: Non-existent `iris.connect()` call at line 210

**State Transitions**:
1. **Not Started** → **Connecting** (attempting connection)
2. **Connecting** → **Connected** (successful connection)
3. **Connecting** → **Failed** (connection error)
4. **Connecting** → **Retry** (transient error, retry with exponential backoff)

**Validation Rules**:
- Must use `iris.createConnection()` API (NOT `iris.connect()`)
- Must validate all connection parameters are non-empty
- Must handle SSL configuration if provided
- Must raise ConnectionError with clear message on failure (not AttributeError)

---

### 2. ConnectionResult

**Purpose**: Track result of connection establishment attempt.

**Fields**:
- `success`: bool - Whether connection was established
- `connection`: Optional[Connection] - Connection object if successful
- `error_message`: str - Error details if failed
- `api_used`: str - API method used (e.g., "iris.createConnection")
- `retry_count`: int - Number of retry attempts made
- `elapsed_seconds`: float - Time taken to establish connection

**Example**:
```python
# Success case
ConnectionResult(
    success=True,
    connection=<iris.Connection object>,
    error_message="",
    api_used="iris.createConnection",
    retry_count=0,
    elapsed_seconds=0.45
)

# Failure case (Bug 1 - AttributeError)
ConnectionResult(
    success=False,
    connection=None,
    error_message="AttributeError: module 'iris' has no attribute 'connect'. Use iris.createConnection() instead.",
    api_used="iris.connect",  # Wrong API
    retry_count=3,
    elapsed_seconds=2.1
)

# Fixed success case
ConnectionResult(
    success=True,
    connection=<iris.Connection object>,
    error_message="",
    api_used="iris.createConnection",  # Correct API
    retry_count=0,
    elapsed_seconds=0.52
)
```

**Validation Rules**:
- If success=True, connection must be non-None
- If success=False, error_message must be non-empty
- elapsed_seconds must be >= 0
- retry_count must be >= 0

---

### 3. ConnectionValidator

**Purpose**: Validate connection configuration before attempting connection.

**Attributes**:
- `required_params`: List[str] - Required connection parameters ["host", "port", "namespace", "user", "password"]

**Methods**:
- `validate_config(config: Dict[str, Any]) → Tuple[bool, List[str]]`: Validate connection parameters
- `validate_api_available() → Tuple[bool, str]`: Check if correct IRIS API is available

**Relationships**:
- Used by: ConnectionEstablisher (before connection attempt)
- Prevents: AttributeError from invalid API calls

**Validation Rules**:
- Must verify all required parameters are present
- Must verify iris.createConnection exists (not iris.connect)
- Must provide specific list of missing/invalid parameters

---

## Bug 2 Components: Schema Initialization

## Core Components

### 1. GraphTableDetector

**Purpose**: Detect whether iris-vector-graph package is available in the Python environment.

**Attributes**:
- None (stateless utility)

**Methods**:
- `detect() → bool`: Returns True if iris-vector-graph is importable

**Relationships**:
- Used by: GraphTableInitializer, PrerequisiteValidator

**State Transitions**:
- N/A (stateless detection)

**Validation Rules**:
- Must not import iris-vector-graph (use importlib.util.find_spec)
- Must return boolean (no exceptions)

---

### 2. GraphTableInitializer

**Purpose**: Create required iris-vector-graph tables when package is detected.

**Attributes**:
- `schema_manager`: Reference to SchemaManager instance
- `pipeline_type`: Pipeline type for table creation context (e.g., "graphrag")
- `required_tables`: List of table names ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]

**Methods**:
- `initialize(pipeline_type: str) → InitializationResult`: Create all required tables
- `initialize_table(table_name: str, pipeline_type: str) → bool`: Create single table

**Relationships**:
- Uses: GraphTableDetector (to check if package installed)
- Uses: SchemaManager.ensure_table_schema() (for actual table creation)
- Produces: InitializationResult

**State Transitions**:
1. **Not Started** → **Detecting Package** (check if iris-vector-graph installed)
2. **Detecting Package** → **Skipped** (if not installed)
3. **Detecting Package** → **Creating Tables** (if installed)
4. **Creating Tables** → **Complete** (all tables created)
5. **Creating Tables** → **Partial Failure** (some tables failed)

**Validation Rules**:
- Must check package detection before creating tables
- Must create tables in dependency order (nodes before edges)
- Must track success/failure for each table
- Must log clear messages for each step

---

### 3. PrerequisiteValidator

**Purpose**: Validate that all iris-vector-graph prerequisites are met before PPR operations.

**Attributes**:
- `schema_manager`: Reference to SchemaManager instance
- `required_tables`: List of table names to validate

**Methods**:
- `validate() → ValidationResult`: Check all prerequisites
- `table_exists(table_name: str) → bool`: Check if specific table exists

**Relationships**:
- Uses: GraphTableDetector (to check if package installed)
- Uses: SchemaManager.table_exists() (to check table presence)
- Produces: ValidationResult

**State Transitions**:
1. **Not Started** → **Checking Package** (verify iris-vector-graph installed)
2. **Checking Package** → **Invalid** (if not installed)
3. **Checking Package** → **Checking Tables** (if installed)
4. **Checking Tables** → **Valid** (all tables present)
5. **Checking Tables** → **Invalid** (missing tables)

**Validation Rules**:
- Must check package presence first
- Must validate all 4 required tables
- Must provide specific list of missing components
- Must distinguish "package not installed" from "tables missing"

---

## Data Structures

### InitializationResult

**Purpose**: Track results of graph table initialization.

**Fields**:
- `package_detected`: bool - Whether iris-vector-graph was found
- `tables_attempted`: List[str] - Tables that initialization was attempted for
- `tables_created`: Dict[str, bool] - Mapping of table names to creation success
- `total_time_seconds`: float - Time taken for initialization
- `error_messages`: Dict[str, str] - Error messages for failed tables

**Example**:
```python
InitializationResult(
    package_detected=True,
    tables_attempted=["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"],
    tables_created={
        "rdf_labels": True,
        "rdf_props": True,
        "rdf_edges": False,  # Failed
        "kg_NodeEmbeddings_optimized": True
    },
    total_time_seconds=3.2,
    error_messages={
        "rdf_edges": "Permission denied: CREATE TABLE"
    }
)
```

**Validation Rules**:
- If package_detected=False, tables_created must be empty dict
- len(tables_created) must equal len(tables_attempted)
- error_messages keys must be subset of tables_attempted
- total_time_seconds must be >= 0

---

### ValidationResult

**Purpose**: Track results of prerequisite validation.

**Fields**:
- `is_valid`: bool - Whether all prerequisites are met
- `package_installed`: bool - Whether iris-vector-graph is installed
- `missing_tables`: List[str] - Tables that don't exist
- `error_message`: str - Human-readable error message if not valid

**Example**:
```python
# Success case
ValidationResult(
    is_valid=True,
    package_installed=True,
    missing_tables=[],
    error_message=""
)

# Package not installed
ValidationResult(
    is_valid=False,
    package_installed=False,
    missing_tables=[],
    error_message="iris-vector-graph package not installed"
)

# Tables missing
ValidationResult(
    is_valid=False,
    package_installed=True,
    missing_tables=["rdf_edges", "kg_NodeEmbeddings_optimized"],
    error_message="Missing required tables: rdf_edges, kg_NodeEmbeddings_optimized"
)
```

**Validation Rules**:
- If package_installed=False, is_valid must be False
- If is_valid=True, missing_tables must be empty
- If is_valid=False, error_message must be non-empty
- error_message should list specific missing components

---

## Database Schema (Existing)

### Table: rdf_labels

**Purpose**: Entity type/label mapping for graph nodes.

**Columns**:
- `s`: VARCHAR(256) NOT NULL - Subject (entity ID)
- `label`: VARCHAR(128) NOT NULL - Entity type/label

**Indexes**:
- `idx_labels_label_s`: INDEX(label, s) - Lookup entities by type
- `idx_labels_s_label`: INDEX(s, label) - Lookup labels for entity

**Created By**: iris-vector-graph package
**Used By**: PPR neighborhood expansion, entity type filtering

---

### Table: rdf_props

**Purpose**: Entity properties/attributes storage.

**Columns**:
- `s`: VARCHAR(256) NOT NULL - Subject (entity ID)
- `key`: VARCHAR(128) NOT NULL - Property key
- `val`: VARCHAR(4000) - Property value

**Indexes**:
- `idx_props_s_key`: INDEX(s, key) - Lookup properties for entity
- `idx_props_key_val`: INDEX(key, val) - Search by property value

**Created By**: iris-vector-graph package
**Used By**: Entity attribute retrieval, property-based filtering

---

### Table: rdf_edges

**Purpose**: Graph relationships/edges between entities.

**Columns**:
- `edge_id`: BIGINT IDENTITY PRIMARY KEY - Unique edge ID
- `s`: VARCHAR(256) NOT NULL - Source entity ID
- `p`: VARCHAR(128) NOT NULL - Predicate (relationship type)
- `o_id`: VARCHAR(256) NOT NULL - Target entity ID (object)
- `qualifiers`: VARCHAR(4000) - JSON metadata (confidence, etc.)

**Indexes**:
- `idx_edges_s_p`: INDEX(s, p) - Outgoing edges from entity
- `idx_edges_p_oid`: INDEX(p, o_id) - Incoming edges to entity
- `idx_edges_s`: INDEX(s) - All edges from entity

**Created By**: iris-vector-graph package
**Used By**: PPR graph traversal, relationship queries

---

### Table: kg_NodeEmbeddings_optimized

**Purpose**: HNSW-optimized vector embeddings for graph nodes.

**Columns**:
- `id`: VARCHAR(256) PRIMARY KEY - Entity ID
- `emb`: VECTOR(FLOAT, dimension) NOT NULL - Vector embedding

**Indexes**:
- HNSW index on `emb` column - High-performance vector similarity search

**Created By**: iris-vector-graph package
**Used By**: PPR vector-based ranking, semantic similarity search

---

## Component Interactions

### Initialization Flow

```
Pipeline Setup
     ↓
SchemaManager.ensure_iris_vector_graph_tables()
     ↓
GraphTableDetector.detect()
     ↓ (if True)
GraphTableInitializer.initialize()
     ↓
For each table in ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]:
     SchemaManager.ensure_table_schema(table)
     ↓
InitializationResult (returned to caller)
```

### Validation Flow

```
Before PPR Operation
     ↓
PrerequisiteValidator.validate()
     ↓
GraphTableDetector.detect()
     ↓ (if True)
For each table:
     SchemaManager.table_exists(table)
     ↓
ValidationResult (returned to caller)
     ↓ (if not valid)
Raise RuntimeError with specific missing components
```

---

## Error Handling Patterns

### Error Type 1: Package Not Installed

**Condition**: iris-vector-graph not found by GraphTableDetector
**Response**: Log DEBUG message, return empty InitializationResult, continue gracefully
**Rationale**: This is expected state for pipelines not using graph features

### Error Type 2: Table Creation Failed

**Condition**: SchemaManager.ensure_table_schema() returns False
**Response**: Log ERROR with specific table name and underlying database error, continue to next table, return InitializationResult with failures
**Rationale**: Partial failure should be tracked but not block other tables

### Error Type 3: Prerequisites Missing (PPR Call Time)

**Condition**: PrerequisiteValidator finds missing tables
**Response**: Raise RuntimeError with specific missing components list, do not attempt PPR
**Rationale**: Silent fallback eliminated per Constitution VI

### Error Type 4: Database Permission Denied

**Condition**: CREATE TABLE fails due to insufficient permissions
**Response**: Log ERROR with permission details, suggest remediation (grant CREATE TABLE), return False
**Rationale**: Clear actionable error message per specification FR-005

---

## Performance Characteristics

### Initialization Performance

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Package detection | <10ms | importlib.util.find_spec() is fast |
| rdf_labels creation | ~500ms | Simple table with indexes |
| rdf_props creation | ~500ms | Simple table with indexes |
| rdf_edges creation | ~800ms | More complex with IDENTITY |
| kg_NodeEmbeddings_optimized | ~2000ms | VECTOR column + HNSW index |
| **Total** | **~3.8 seconds** | Within <5s requirement |

### Validation Performance

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Package detection | <10ms | Cached by importlib |
| Table existence checks (4 tables) | <100ms | Fast metadata queries |
| **Total** | **<200ms** | Well within <1s requirement |

---

## Design Status

✅ **Phase 1 Design Complete**

- [x] Core components defined (GraphTableDetector, GraphTableInitializer, PrerequisiteValidator)
- [x] Data structures specified (InitializationResult, ValidationResult)
- [x] Database schema documented (existing tables)
- [x] Component interactions mapped
- [x] Error handling patterns defined
- [x] Performance characteristics validated

**Ready for Contract Generation**
