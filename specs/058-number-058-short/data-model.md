# Data Model: Cloud Configuration Flexibility

**Feature**: 058-cloud-config-flexibility
**Date**: 2025-01-12
**Status**: Design

## Overview

This data model defines the configuration entities and their relationships for cloud deployment configuration management. All entities are configuration-time (not runtime database entities) and reside in Python classes within iris_vector_rag/config/.

## Core Entities

### 1. ConnectionConfiguration

**Purpose**: Represents IRIS database connection parameters with priority-based resolution from environment variables, config files, and defaults.

**Attributes**:
- `host: str` - IRIS database host (env: IRIS_HOST, default: "localhost")
- `port: int` - IRIS database port (env: IRIS_PORT, default: 1972)
- `username: str` - IRIS database username (env: IRIS_USERNAME, default: "_SYSTEM")
- `password: str` - IRIS database password (env: IRIS_PASSWORD, default: "SYS")
- `namespace: str` - IRIS namespace (env: IRIS_NAMESPACE, default: "USER")
- `connection_timeout: int` - Connection timeout in seconds (default: 30)

**Validation Rules**:
- host must not be empty string
- port must be in range 1-65535
- username must not be empty string
- password must not be empty string (warning only for default password)
- namespace must be valid IRIS namespace format (alphanumeric + %)

**State Transitions**: None (immutable after initialization)

**Relationships**:
- Used by: ConnectionManager (iris_vector_rag/core/connection.py)
- Sources from: ConfigurationSource

**Example**:
```python
conn_config = ConnectionConfiguration(
    host="aws-iris.example.com",
    port=1972,
    username="AppUser",
    password=os.environ['IRIS_PASSWORD'],  # From env var
    namespace="%SYS"
)
```

### 2. VectorConfiguration

**Purpose**: Encapsulates vector storage settings (dimension, distance metric, index type) with validation against supported ranges.

**Attributes**:
- `vector_dimension: int` - Embedding vector dimension (env: VECTOR_DIMENSION, default: 384)
- `distance_metric: str` - Distance calculation method (default: "COSINE", options: "COSINE", "EUCLIDEAN", "DOT")
- `index_type: str` - Vector index type (default: "HNSW", options: "HNSW", "FLAT")
- `hnsw_ef_construction: int` - HNSW build parameter (default: 200)
- `hnsw_m: int` - HNSW connectivity parameter (default: 16)

**Validation Rules**:
- vector_dimension must be in range [128, 8192]
- vector_dimension must be power of 2 or common model size (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192)
- distance_metric must be one of allowed values
- index_type must be one of allowed values
- If existing tables found: vector_dimension must match existing or error

**State Transitions**:
- UNVALIDATED → VALIDATED (after preflight check)
- VALIDATED → MISMATCHED (if existing table has different dimension)

**Relationships**:
- Used by: EntityStorageAdapter (iris_vector_rag/services/storage.py)
- Validated by: VectorDimensionValidator
- Sources from: ConfigurationSource

**Example**:
```python
vector_config = VectorConfiguration(
    vector_dimension=1024,  # NVIDIA NIM embeddings
    distance_metric="COSINE",
    index_type="HNSW"
)
```

### 3. TableConfiguration

**Purpose**: Defines table schema, names, and namespace requirements with cloud-specific overrides.

**Attributes**:
- `table_schema: str` - Schema prefix for tables (env: TABLE_SCHEMA, default: "RAG")
- `entities_table: str` - Entity table name (default: "Entities")
- `relationships_table: str` - Relationship table name (default: "EntityRelationships")
- `documents_table: str` - Document table name (default: "Documents")
- `chunks_table: str` - Chunk table name (default: "Chunks")
- `create_if_not_exists: bool` - Auto-create tables (default: True)
- `drop_if_exists: bool` - Drop existing tables (default: False)

**Validation Rules**:
- table_schema must be valid SQL identifier (alphanumeric + underscore + %)
- All table names must be valid SQL identifiers
- If drop_if_exists=True: require explicit confirmation (CLI flag or config)
- Schema must exist or have CREATE SCHEMA permissions

**State Transitions**: None (immutable after initialization)

**Relationships**:
- Used by: init_tables CLI (iris_vector_rag/cli/init_tables.py)
- Used by: EntityStorageAdapter
- Sources from: ConfigurationSource

**Computed Properties**:
- `full_entities_table -> str`: Returns f"{table_schema}.{entities_table}"
- `full_relationships_table -> str`: Returns f"{table_schema}.{relationships_table}"
- `full_documents_table -> str`: Returns f"{table_schema}.{documents_table}"
- `full_chunks_table -> str`: Returns f"{table_schema}.{chunks_table}"

**Example**:
```python
table_config = TableConfiguration(
    table_schema="SQLUser",  # AWS IRIS requirement
    entities_table="Entities",
    relationships_table="EntityRelationships",
    create_if_not_exists=True
)

# Usage
query = f"SELECT * FROM {table_config.full_entities_table}"
# Produces: SELECT * FROM SQLUser.Entities
```

### 4. ConfigurationSource

**Purpose**: Tracks where each configuration value originated (env var, file, default) for debugging and audit trails.

**Attributes**:
- `parameter_name: str` - Configuration parameter name (e.g., "database.iris.host")
- `source_type: str` - Source of value ("environment", "file", "default", "override")
- `source_location: str` - Specific location (env var name, file path, or "hardcoded")
- `resolved_value: Any` - Final resolved value (masked for passwords)
- `timestamp: datetime` - When configuration was loaded

**Validation Rules**:
- source_type must be one of allowed values
- sensitive parameters (passwords) have resolved_value masked as "***"

**State Transitions**: None (append-only log)

**Relationships**:
- Referenced by: ConnectionConfiguration, VectorConfiguration, TableConfiguration
- Aggregated by: ConfigurationManager for logging/debugging

**Example**:
```python
sources = [
    ConfigurationSource(
        parameter_name="database.iris.host",
        source_type="environment",
        source_location="IRIS_HOST",
        resolved_value="aws-iris.example.com",
        timestamp=datetime.now()
    ),
    ConfigurationSource(
        parameter_name="database.iris.password",
        source_type="environment",
        source_location="IRIS_PASSWORD",
        resolved_value="***",  # Masked
        timestamp=datetime.now()
    ),
    ConfigurationSource(
        parameter_name="storage.vector_dimension",
        source_type="file",
        source_location="/path/to/aws.yaml",
        resolved_value=1024,
        timestamp=datetime.now()
    )
]
```

## Supporting Validators

### VectorDimensionValidator

**Purpose**: Validates vector dimension configuration against existing tables.

**Methods**:
- `validate(config: VectorConfiguration, conn: Connection) -> ValidationResult`
  - Queries existing table schema
  - Compares configured vs existing dimensions
  - Returns PASS/FAIL with actionable error messages

**Validation Logic**:
```python
def validate(self, config, conn):
    configured_dim = config.vector_dimension

    # Query existing table (if exists)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT VECTOR_DIM
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'Entities'
        AND COLUMN_NAME = 'embedding'
    """)
    result = cursor.fetchone()

    if result is None:
        # No existing table - OK to create with any dimension
        return ValidationResult(
            status="PASS",
            message=f"No existing tables. Will create with dimension {configured_dim}"
        )

    existing_dim = result[0]
    if configured_dim != existing_dim:
        return ValidationResult(
            status="FAIL",
            message=f"Vector dimension mismatch: configured {configured_dim}, existing {existing_dim}",
            help_url="https://docs.iris-vector-rag.com/migration/vector-dimensions"
        )

    return ValidationResult(status="PASS")
```

### NamespaceValidator

**Purpose**: Validates namespace access and permissions.

**Methods**:
- `validate(config: ConnectionConfiguration, conn: Connection) -> ValidationResult`
  - Tests namespace access
  - Tests write permissions
  - Returns PASS/FAIL with required permissions list

**Validation Logic**:
```python
def validate(self, config, conn):
    namespace = config.namespace

    # Test namespace access
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM {namespace}.INFORMATION_SCHEMA.TABLES LIMIT 1")
    except Exception as e:
        return ValidationResult(
            status="FAIL",
            message=f"Cannot access namespace '{namespace}': {e}",
            required_permissions=["READ on %SYS.INFORMATION_SCHEMA", f"USE on {namespace}"]
        )

    # Test write permissions
    test_table = f"{namespace}.test_permissions_{int(time.time())}"
    try:
        cursor.execute(f"CREATE TABLE {test_table} (id INT)")
        cursor.execute(f"DROP TABLE {test_table}")
    except Exception as e:
        return ValidationResult(
            status="FAIL",
            message=f"Cannot create tables in namespace '{namespace}': {e}",
            required_permissions=[f"CREATE TABLE on {namespace}", f"DROP TABLE on {namespace}"]
        )

    return ValidationResult(status="PASS")
```

## Configuration Priority Flow

```
Startup
  ↓
Load default_config.yaml (defaults)
  ↓
Load user config file if --config provided (overrides defaults)
  ↓
Load environment variables (overrides config file)
  ↓
Create ConnectionConfiguration, VectorConfiguration, TableConfiguration
  ↓
Run preflight validations (VectorDimensionValidator, NamespaceValidator)
  ↓
Log all ConfigurationSource entries (for debugging)
  ↓
Initialize ConnectionManager with validated config
  ↓
Ready for operations
```

## Entity Relationships Diagram

```
ConfigurationManager
├─ loads → ConnectionConfiguration
│          └─ validated by → NamespaceValidator
├─ loads → VectorConfiguration
│          └─ validated by → VectorDimensionValidator
├─ loads → TableConfiguration
└─ tracks → ConfigurationSource[] (audit log)

ConnectionConfiguration ──uses──> ConnectionManager
VectorConfiguration ──uses──> EntityStorageAdapter
TableConfiguration ──uses──> EntityStorageAdapter
TableConfiguration ──uses──> init_tables CLI
```

## Schema Migration Considerations

**Vector Dimension Changes**:
- NOT supported in-place (requires table recreation)
- Validation prevents silent data corruption
- Error messages provide migration guidance
- Out of scope for this feature (requires separate migration tool)

**Table Schema Changes**:
- Supported via configuration (different schema prefix)
- Enables multiple deployments in same IRIS instance
- No migration needed (new schema = fresh tables)

**Backward Compatibility**:
- All default values match existing behavior (v0.4.x)
- Existing code works without any configuration changes
- New configuration is opt-in

## Validation Error Examples

### Vector Dimension Mismatch
```
ConfigValidationError: Vector dimension mismatch detected

Configured dimension: 1024 (from VECTOR_DIMENSION env var)
Existing table dimension: 384 (RAG.Entities.embedding)

This mismatch will cause data corruption during insert/query operations.

To resolve:
1. Match existing dimension: Set VECTOR_DIMENSION=384
2. Recreate tables: python -m iris_rag.cli.init_tables --drop --config your-config.yaml
   WARNING: This deletes all existing data!
3. Migrate data: Follow guide at https://docs.iris-vector-rag.com/migration/vector-dimensions

Source: VectorDimensionValidator at iris_vector_rag/config/validators.py:42
```

### Namespace Permission Error
```
ConfigValidationError: Insufficient permissions for namespace '%SYS'

Cannot create tables in namespace '%SYS': Access denied for CREATE TABLE

Required permissions:
- USE on %SYS namespace
- CREATE TABLE on %SYS namespace
- DROP TABLE on %SYS namespace (for cleanup)

AWS IRIS Configuration:
  Users in 'AppUsers' role do not have CREATE TABLE on %SYS by default.
  Contact your IRIS administrator to grant schema creation permissions.

Alternative: Use 'SQLUser' namespace (has broader permissions for application users)
  Set: IRIS_NAMESPACE=SQLUser or namespace: SQLUser in config file

Source: NamespaceValidator at iris_vector_rag/config/validators.py:98
```

## Implementation Notes

1. **Pydantic Models**: All configuration entities should use pydantic BaseModel for automatic validation
2. **Immutability**: Configuration objects are immutable after initialization (dataclasses with frozen=True)
3. **Type Hints**: Full type hints for all attributes (enables IDE autocomplete)
4. **Documentation**: Inline docstrings for all attributes with examples
5. **Testing**: Each entity has dedicated unit tests + integration tests

---
**Data Model Status**: ✅ COMPLETE - All entities and validators defined, ready for contract test generation
