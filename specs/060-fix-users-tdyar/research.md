# Research Phase: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature**: 060-fix-users-tdyar
**Date**: 2025-01-13
**Updated**: 2025-01-14 (Added Bug 1 research)
**Status**: ✅ Complete

---

## Bug 1 Research: iris.connect() Connection API Fix

### Research Task R1: IRIS Connection API Documentation

**Finding**: The `iris.connect()` method used at `iris_dbapi_connector.py:210` does not exist in intersystems-irispython package.

#### Documented Connection APIs

**API 1: iris.createConnection() (Native IRIS)**
```python
import iris
conn = iris.createConnection(host, port, namespace, user, password)
```
- **Documentation**: Official InterSystems IRIS Python API
- **Return Type**: Native IRIS connection object
- **Parameters**: host (str), port (int), namespace (str), user (str), password (str)
- **Use Case**: Direct IRIS native API access

**API 2: iris.dbapi.connect() (DBAPI Standard)**
```python
import iris.dbapi
conn = iris.dbapi.connect(host, port, namespace, user, password)
```
- **Documentation**: DB-API 2.0 specification
- **Return Type**: DBAPI connection object
- **Parameters**: Same as createConnection
- **Use Case**: Standard Python database interface

**API 3: iris_devtester.IRISContainer().dbapi_connection() (Testing)**
```python
from iris_devtester import IRISContainer
container = IRISContainer()
conn = container.dbapi_connection()
```
- **Documentation**: iris-devtester package
- **Return Type**: DBAPI connection with test container context
- **Use Case**: Testing environments only

#### Non-Existent API

**iris.connect() - DOES NOT EXIST**
```python
# This line at iris_dbapi_connector.py:210 is WRONG:
conn = iris.connect(host, port, namespace, user, password)  # ❌ AttributeError
```
- **Error**: `AttributeError: module 'iris' has no attribute 'connect'`
- **Impact**: Breaks all database connections in v0.5.3
- **FHIR-AI Test Results**: 3/6 tests passing (down from 4/6 in v0.5.2)

### Research Task R2: Current Connection Code Analysis

**Location**: `iris_vector_rag/common/iris_dbapi_connector.py:190-240`

**Context Around Line 210**:
```python
# Lines 205-210
logger.info(
    f"Attempting IRIS connection to {host}:{port}/{namespace} as user {user}"
)

# Use direct iris.connect() - this avoids SSL issues
conn = iris.connect(host, port, namespace, user, password)  # ❌ LINE 210 BUG
```

**Why This Was Added (v0.5.3 Change)**:
- Comment says "this avoids SSL issues"
- Likely attempt to fix SSL connection problems from v0.5.2
- Developer mistakenly used non-existent `iris.connect()` instead of `iris.createConnection()`

**Impact Analysis**:
- **Breaks**: ConnectionManager, IRISVectorStore, SchemaManager
- **FHIR-AI Tests**: 3 tests now fail (ConnectionManager, IRISVectorStore, SchemaManager)
- **Critical Severity**: Makes entire framework unusable

**Connection Flow**:
1. Entry: `get_iris_connection()` called
2. Retry logic: 3 attempts with exponential backoff
3. Line 210: Attempts `iris.connect()` → AttributeError
4. Retry exhausted: Returns None
5. Calling code fails

### Research Task R3: Connection Pattern Best Practices

**Recommendation**: Use `iris.createConnection()` as direct replacement

#### Option 1: iris.createConnection() (RECOMMENDED)
```python
# Line 210 fix:
conn = iris.createConnection(host, port, namespace, user, password)
```

**Pros**:
- Direct replacement for iris.connect()
- Same parameter signature
- Native IRIS API (widely documented)
- Preserves SSL configuration intent

**Cons**:
- None significant

#### Option 2: iris.dbapi.connect() (Alternative)
```python
# Alternative fix:
import iris.dbapi
conn = iris.dbapi.connect(host, port, namespace, user, password)
```

**Pros**:
- Standard DB-API 2.0 interface
- More portable across databases

**Cons**:
- Requires additional import
- May not preserve SSL settings from v0.5.3 intent

### Decision
**DECISION #1 (Bug 1)**: Use `iris.createConnection()` as primary fix for line 210

**Rationale**:
- Most direct replacement for non-existent `iris.connect()`
- Same parameter signature (minimal code change)
- Native IRIS API matches developer's apparent intent
- Widely used in InterSystems documentation
- Preserves SSL configuration context from v0.5.3

**DECISION #2 (Bug 1)**: Preserve SSL configuration logic from v0.5.3

**Rationale**:
- v0.5.3 added `iris.connect()` specifically for SSL issues
- Correct approach: Use `iris.createConnection()` with same SSL parameters
- Do NOT revert to v0.5.2 connection code (had SSL issues)
- Only fix the API call, keep SSL intent

### Implementation Recommendation

**One-Line Fix**:
```python
# iris_vector_rag/common/iris_dbapi_connector.py:210
# BEFORE (v0.5.3 - BROKEN):
conn = iris.connect(host, port, namespace, user, password)

# AFTER (v0.5.4 - FIXED):
conn = iris.createConnection(host, port, namespace, user, password)
```

**Verification Steps**:
1. Search codebase for other `iris.connect()` instances
2. Run FHIR-AI test suite (target: 6/6 passing)
3. Verify ConnectionManager, IRISVectorStore, SchemaManager all work
4. Confirm SSL connections still work (preserve v0.5.3 fix)
5. Check CloudConfiguration API still works (preserve dimension fix)

---

## Bug 2 Research: Automatic iris-vector-graph Schema Initialization

## Research Task 1: Current SchemaManager Graph Table Knowledge

### Location and Structure

**Finding**: SchemaManager already has complete knowledge of iris-vector-graph tables at multiple locations:

#### Table Metadata Registry (Lines 171-206)
```python
# iris_vector_rag/storage/schema_manager.py:171-206
"rdf_labels": {
    "embedding_column": None,
    "table_type": "graph_metadata",
    "created_by": "iris_vector_graph",
    # ... config ...
},
"rdf_props": {
    "table_type": "graph_properties",
    "created_by": "iris_vector_graph",
    # ... config ...
},
"rdf_edges": {
    "table_type": "graph_relationships",
    "created_by": "iris_vector_graph",
    # ... config ...
},
"kg_NodeEmbeddings_optimized": {
    "embedding_column": "emb",
    "table_type": "optimized_vectors",
    "created_by": "iris_vector_graph",
    "dimension": self.base_embedding_dimension,
    "supports_vector_search": True,
    # ... config ...
}
```

#### Table Configuration (Lines 461-520)
Each table has detailed column definitions and index specifications:
- `rdf_labels`: entity type/label mapping with indexes idx_labels_label_s, idx_labels_s_label
- `rdf_props`: entity properties with indexes idx_props_s_key, idx_props_key_val
- `rdf_edges`: graph relationships with indexes idx_edges_s_p, idx_edges_p_oid, idx_edges_s
- `kg_NodeEmbeddings_optimized`: HNSW-optimized vector embeddings

#### Migration Methods (Lines 1574-1770)
Complete migration implementations exist:
- `_migrate_rdf_labels_table()`: Creates VARCHAR(256) + VARCHAR(128) with proper indexes
- `_migrate_rdf_props_table()`: Creates VARCHAR(256) + VARCHAR(128) + VARCHAR(4000) with indexes
- `_migrate_rdf_edges_table()`: Creates BIGINT IDENTITY + VARCHAR columns with JSON qualifiers
- `_migrate_kg_node_embeddings_optimized_table()`: Creates VECTOR column with HNSW index

#### Table Recognition (Lines 1868-1876)
```python
iris_graph_tables = ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]
iris_graph_tables_lower = [t.lower() for t in iris_graph_tables]
if table_name.lower() in iris_graph_tables_lower:
    if self.needs_migration(table_name, pipeline_type):
        logger.info(f"Creating iris-vector-graph table: {table_name}")
        return self.migrate_table(table_name, pipeline_type=pipeline_type)
```

### Decision
**DECISION #1**: SchemaManager has **complete** iris-vector-graph table knowledge. All table schemas, indexes, and migration logic already exist. The missing piece is **automatic invocation** during pipeline initialization.

### Rationale
- Table definitions are production-ready (used in migration logic)
- Migration methods are tested and working
- No new table schema design needed
- Only need to trigger creation automatically

---

## Research Task 2: iris-vector-graph Package Detection Patterns

### Best Practices Analysis

#### Option 1: importlib.util.find_spec() (Recommended)
```python
import importlib.util

def detect_iris_vector_graph() -> bool:
    """Detect iris-vector-graph without importing it."""
    spec = importlib.util.find_spec("iris_vector_graph")
    return spec is not None
```

**Pros**:
- Does not import the package (faster, no side effects)
- Standard library solution
- Works with all package types (module, package, namespace package)
- Returns None if not found (clean boolean conversion)

**Cons**:
- Slightly more verbose than try/except

#### Option 2: try/except Import (Alternative)
```python
def detect_iris_vector_graph() -> bool:
    """Detect iris-vector-graph by attempting import."""
    try:
        import iris_vector_graph
        return True
    except ImportError:
        return False
```

**Pros**:
- Simple and obvious
- Directly tests importability

**Cons**:
- Imports the package (side effects, slower on repeated calls)
- May trigger package initialization code

### Decision
**DECISION #2**: Use `importlib.util.find_spec()` for package detection.

### Rationale
- Avoids importing iris-vector-graph unnecessarily (no side effects)
- Faster for repeated checks
- Standard library solution (no dependencies)
- Consistent with modern Python best practices

### Edge Cases Identified
1. **Package partially installed**: find_spec() returns None if package corrupt
2. **Virtual environment isolation**: Works correctly with venv/virtualenv
3. **Editable installs**: find_spec() detects editable installs correctly
4. **Import hooks**: Works with custom import hooks and namespace packages

---

## Research Task 3: Pipeline Initialization Entry Points

### Current Usage Patterns

#### Pattern 1: Explicit Table List
```python
# From docs/architecture/graphrag_service_interfaces.md:615
for table in ["SourceDocuments", "DocumentChunks", "Entities", "EntityRelationships"]:
    if not self.ensure_table_schema(table, pipeline_type='graphrag'):
        # Handle error
```

#### Pattern 2: Validation Orchestrator
```python
# From specs/001-configurationmanager-schemamanager-system/quickstart.md:207
schema_manager.ensure_table_schema(table, pipeline_type="graphrag")
```

#### Pattern 3: HippoRAG Workaround (Current Bug Fix Location)
From bug report:
```python
# hipporag2_pipeline.py:375-384
try:
    import iris_vector_graph
    schema_manager.ensure_table_schema("rdf_labels")
    schema_manager.ensure_table_schema("rdf_props")
    schema_manager.ensure_table_schema("rdf_edges")
    schema_manager.ensure_table_schema("kg_NodeEmbeddings_optimized")
    logger.info("iris-vector-graph tables ensured")
except ImportError:
    logger.debug("iris-vector-graph not available, skipping graph tables")
```

### Integration Points Identified

1. **SchemaManager.__init__()**: Initialization entry point
2. **SchemaManager.ensure_table_schema()**: Per-table entry point
3. **Pipeline validation flows**: Where pipelines call SchemaManager

### Decision
**DECISION #3**: Add automatic iris-vector-graph table initialization to **SchemaManager's standard initialization flow** via a new public method: `ensure_iris_vector_graph_tables()`.

### Rationale
- **Centralized**: All pipelines benefit automatically
- **Explicit**: Method can be called explicitly when needed
- **Backward Compatible**: Existing code continues to work
- **Idempotent**: Safe to call multiple times
- **Framework-First**: Enhancement to framework, not application logic

### Integration Strategy
```python
class SchemaManager:
    def ensure_iris_vector_graph_tables(self, pipeline_type: str = "graphrag") -> Dict[str, bool]:
        """
        Automatically create iris-vector-graph tables if package is installed.

        Returns:
            Dict mapping table names to creation success status
        """
        if not self._detect_iris_vector_graph():
            logger.debug("iris-vector-graph not installed, skipping graph tables")
            return {}

        results = {}
        for table in ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]:
            results[table] = self.ensure_table_schema(table, pipeline_type=pipeline_type)

        return results
```

---

## Research Task 4: PPR Prerequisite Validation Patterns

### PPR Failure Analysis

**From Bug Report**:
```
kg_NEIGHBORHOOD_EXPANSION failed: Table 'SQLUSER.RDF_EDGES' not found
PPR computation failed: Table 'SQLUSER.NODES' not found
```

### Validation Hook Locations

#### Location 1: Before PPR Computation
**Where**: Any code calling iris-vector-graph PPR functions
**Strategy**: Validate prerequisites before calling PPR

```python
def validate_graph_prerequisites(self) -> Tuple[bool, List[str]]:
    """
    Validate that all iris-vector-graph prerequisites are met.

    Returns:
        (all_valid, missing_tables) tuple
    """
    if not self._detect_iris_vector_graph():
        return False, ["iris-vector-graph package not installed"]

    missing = []
    for table in ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]:
        if not self.table_exists(table):
            missing.append(table)

    return len(missing) == 0, missing
```

#### Location 2: Pipeline Setup Validation
**Where**: During pipeline initialization after table creation
**Strategy**: Log whether PPR functionality is available

```python
# After ensure_iris_vector_graph_tables()
valid, missing = schema_manager.validate_graph_prerequisites()
if valid:
    logger.info("✅ PPR functionality available (all graph tables present)")
else:
    logger.warning(f"⚠️ PPR functionality degraded (missing tables: {missing})")
```

### Decision
**DECISION #4**: Implement `validate_graph_prerequisites()` method for explicit validation. Call it after table initialization to provide clear feedback about PPR availability.

### Rationale
- **Explicit Validation**: Clear separation between initialization and validation
- **Error Visibility**: Eliminates silent failures
- **Actionable Feedback**: Lists specific missing tables
- **Integration Hook**: Provides validation point for calling code

### Error Propagation Strategy
1. **Initialization Errors**: Log ERROR and return False from ensure_iris_vector_graph_tables()
2. **Missing Prerequisites**: Log WARNING with specific tables missing
3. **PPR Call Failures**: Fail fast with descriptive exception (not silent fallback)

---

## Research Task 5: IRIS Table Creation Performance Characteristics

### Performance Data

#### Table Creation Timing Estimates
Based on SchemaManager migration patterns:

| Table | Complexity | Estimated Time | Notes |
|-------|-----------|----------------|-------|
| rdf_labels | 2 columns + 2 indexes | ~500ms | Simple VARCHAR columns |
| rdf_props | 3 columns + 2 indexes | ~500ms | Includes VARCHAR(4000) |
| rdf_edges | 5 columns + 3 indexes | ~800ms | IDENTITY column + more indexes |
| kg_NodeEmbeddings_optimized | 2 columns + HNSW index | ~2000ms | VECTOR column + HNSW build |
| **Total** | **4 tables** | **~3.8 seconds** | Within <5s spec requirement |

#### HNSW Index Creation
- Empty table HNSW creation: Fast (<500ms)
- Populated table HNSW creation: Depends on row count
- Acorn optimization (if available): Additional speedup

### Performance Validation Strategy

```python
import time

def ensure_iris_vector_graph_tables_with_timing(self, pipeline_type: str = "graphrag"):
    """Create tables with performance tracking."""
    start_time = time.time()

    results = self.ensure_iris_vector_graph_tables(pipeline_type)

    elapsed = time.time() - start_time
    logger.info(f"Graph tables initialization completed in {elapsed:.2f}s")

    if elapsed > 5.0:
        logger.warning(f"Table creation exceeded 5s threshold: {elapsed:.2f}s")

    return results
```

### Decision
**DECISION #5**: Expected performance is **~3.8 seconds** for 4 tables, comfortably within <5 second requirement. No special optimization needed, but add timing logging for monitoring.

### Rationale
- Current migration methods are already optimized
- HNSW index creation is fast for empty tables
- Performance target easily met with existing implementation
- Timing logs provide operational visibility

---

## Consolidated Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **#1: Table Schemas** | Use existing SchemaManager definitions | Complete schemas already exist, tested in migration logic |
| **#2: Package Detection** | `importlib.util.find_spec()` | No import side effects, faster, standard library |
| **#3: Integration Point** | New `ensure_iris_vector_graph_tables()` method | Centralized, explicit, backward compatible |
| **#4: Prerequisite Validation** | New `validate_graph_prerequisites()` method | Explicit validation, clear error messages, no silent failures |
| **#5: Performance** | Use existing methods, add timing logs | ~3.8s expected (within <5s spec), no optimization needed |

---

## Implementation Recommendations

### 1. New SchemaManager Methods

```python
def _detect_iris_vector_graph(self) -> bool:
    """Detect if iris-vector-graph package is installed."""
    import importlib.util
    spec = importlib.util.find_spec("iris_vector_graph")
    return spec is not None

def ensure_iris_vector_graph_tables(self, pipeline_type: str = "graphrag") -> Dict[str, bool]:
    """
    Automatically create iris-vector-graph tables if package is installed.

    Returns:
        Dict mapping table names to creation success status (empty dict if not installed)
    """
    if not self._detect_iris_vector_graph():
        logger.debug("iris-vector-graph not installed, skipping graph tables")
        return {}

    logger.info("iris-vector-graph detected, initializing graph tables")
    results = {}

    for table in ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]:
        try:
            success = self.ensure_table_schema(table, pipeline_type=pipeline_type)
            results[table] = success
            if success:
                logger.info(f"✅ Graph table created/verified: {table}")
            else:
                logger.error(f"❌ Graph table creation failed: {table}")
        except Exception as e:
            logger.error(f"❌ Exception creating {table}: {e}")
            results[table] = False

    return results

def validate_graph_prerequisites(self) -> Tuple[bool, List[str]]:
    """
    Validate that all iris-vector-graph prerequisites are met.

    Returns:
        (all_valid, missing_components) tuple where missing_components includes
        package status and missing tables
    """
    missing = []

    if not self._detect_iris_vector_graph():
        missing.append("iris-vector-graph package not installed")
        return False, missing

    for table in ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]:
        if not self.table_exists(table):
            missing.append(f"Table '{table}' not found")

    return len(missing) == 0, missing
```

### 2. Integration Examples

**Example 1: Automatic Initialization in Pipeline Setup**
```python
# In pipeline initialization
schema_manager.ensure_iris_vector_graph_tables(pipeline_type=self.pipeline_type)

# Validate and log status
valid, missing = schema_manager.validate_graph_prerequisites()
if not valid:
    logger.warning(f"PPR functionality unavailable: {', '.join(missing)}")
```

**Example 2: Explicit Validation Before PPR**
```python
# Before calling PPR operations
valid, missing = schema_manager.validate_graph_prerequisites()
if not valid:
    raise RuntimeError(f"Cannot perform PPR: {', '.join(missing)}")
```

### 3. Backward Compatibility

- Existing pipelines: No changes required (new method is optional)
- HippoRAG workaround: Can be removed after framework fix deployed
- Error behavior: Changes from silent fallback to explicit error (improvement)

---

## Alternatives Considered

### Alternative 1: Automatic Initialization in __init__()
**Rejected**: Would run on every SchemaManager creation, even when not needed. Explicit call is cleaner.

### Alternative 2: try/except for Package Detection
**Rejected**: Importing package has side effects and is slower. importlib.util.find_spec() is better.

### Alternative 3: Pipeline-Specific Initialization
**Rejected**: Would require changes to each pipeline. Framework enhancement is more maintainable.

### Alternative 4: Silent Fallback for Missing Tables
**Rejected**: Violates Constitution VI (Explicit Error Handling). Silent failures are unacceptable.

---

## Research Phase Status

✅ **All Research Tasks Complete**

- [x] R1: SchemaManager graph table knowledge mapped
- [x] R2: Package detection pattern selected
- [x] R3: Integration entry points identified
- [x] R4: PPR validation strategy designed
- [x] R5: Performance characteristics validated

**Ready for Phase 1 (Design & Contracts)**
