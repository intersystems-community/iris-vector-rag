# Quickstart Guide: ConfigurationManager → SchemaManager System

## Overview
This guide helps developers quickly understand and use the ConfigurationManager → SchemaManager system for reliable configuration management and automated schema migrations in RAG applications.

## Basic Usage

### ConfigurationManager Quick Start

#### 1. Basic Configuration Loading
```python
from iris_rag.config.manager import ConfigurationManager

# Load default configuration
config_manager = ConfigurationManager()

# Load custom configuration file
config_manager = ConfigurationManager("/path/to/custom/config.yaml")

# Access configuration values
db_host = config_manager.get("database:iris:host", "localhost")
embedding_model = config_manager.get("embeddings:model", "all-MiniLM-L6-v2")
```

#### 2. Environment Variable Overrides
```bash
# Set environment variables with RAG_ prefix
export RAG_DATABASE__IRIS__HOST=production-db.example.com
export RAG_EMBEDDINGS__MODEL=text-embedding-ada-002
export RAG_VECTOR_INDEX__M=32

# Variables automatically override YAML configuration
```

#### 3. Specialized Configuration Access
```python
# Get database configuration with defaults
db_config = config_manager.get_database_config()
print(f"Connecting to {db_config['host']}:{db_config['port']}")

# Get embedding configuration
embedding_config = config_manager.get_embedding_config()
print(f"Using model: {embedding_config['model']} ({embedding_config['dimension']}D)")

# Get vector index configuration
vector_config = config_manager.get_vector_index_config()
print(f"HNSW config: M={vector_config['M']}, efConstruction={vector_config['efConstruction']}")
```

### SchemaManager Quick Start

#### 1. Basic Schema Management
```python
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.storage.connection_manager import ConnectionManager

# Initialize managers
connection_manager = ConnectionManager()
config_manager = ConfigurationManager()
schema_manager = SchemaManager(connection_manager, config_manager)

# Check if table needs migration
needs_migration = schema_manager.needs_migration("SourceDocuments")
print(f"Migration needed: {needs_migration}")

# Ensure table schema is correct
success = schema_manager.ensure_table_schema("SourceDocuments")
print(f"Schema ensured: {success}")
```

#### 2. Vector Dimension Management
```python
# Get vector dimension for any table
dimension = schema_manager.get_vector_dimension("SourceDocuments")
print(f"SourceDocuments vector dimension: {dimension}")

# Get dimension for specific model
dimension = schema_manager.get_vector_dimension("Entities", "text-embedding-ada-002")
print(f"Entities with ada-002: {dimension}")

# Validate dimension consistency
schema_manager.validate_vector_dimension("SourceDocuments", 384)  # OK
# schema_manager.validate_vector_dimension("SourceDocuments", 512)  # Raises ValueError
```

#### 3. Schema Status and Migration
```python
# Get comprehensive schema status
status = schema_manager.get_schema_status()
for table, info in status.items():
    print(f"{table}: {info['status']} (needs migration: {info['needs_migration']})")

# Perform manual migration if needed
if schema_manager.needs_migration("DocumentChunks"):
    success = schema_manager.migrate_table("DocumentChunks", preserve_data=False)
    print(f"Migration {'succeeded' if success else 'failed'}")
```

## Configuration File Structure

### Default Configuration (iris_rag/config/default_config.yaml)
```yaml
# Database configuration
database:
  iris:
    host: localhost
    port: "1972"
    namespace: USER
    username: _SYSTEM
    password: SYS

# Embedding model configuration
embeddings:
  model: all-MiniLM-L6-v2
  dimension: 384
  provider: sentence-transformers

# Vector index configuration
vector_index:
  type: HNSW
  M: 16
  efConstruction: 200
  Distance: COSINE

# Logging configuration
logging:
  level: INFO
  path: logs/iris_rag.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default table settings
default_table_name: SourceDocuments
default_top_k: 5
```

### Custom Configuration Example
```yaml
# Production configuration
database:
  iris:
    host: prod-iris.company.com
    port: "1972"
    namespace: PRODUCTION
    username: RAG_USER
    password: ${IRIS_PASSWORD}  # Use environment variable

embeddings:
  model: text-embedding-3-large
  dimension: 3072
  provider: openai

vector_index:
  M: 32
  efConstruction: 400
  Distance: COSINE

# Performance tuning
reconciliation:
  performance:
    max_concurrent_pipelines: 5
    batch_size_documents: 200
    memory_limit_gb: 16
```

## Environment Variables

### Database Configuration
```bash
export RAG_DATABASE__IRIS__HOST=your-iris-host.com
export RAG_DATABASE__IRIS__PORT=1972
export RAG_DATABASE__IRIS__NAMESPACE=YOUR_NAMESPACE
export RAG_DATABASE__IRIS__USERNAME=your_username
export RAG_DATABASE__IRIS__PASSWORD=your_password
```

### Embedding Configuration
```bash
export RAG_EMBEDDINGS__MODEL=text-embedding-ada-002
export RAG_EMBEDDINGS__DIMENSION=1536
export RAG_EMBEDDINGS__PROVIDER=openai
```

### Vector Index Tuning
```bash
export RAG_VECTOR_INDEX__M=32
export RAG_VECTOR_INDEX__EFCONSTRUCTION=400
export RAG_VECTOR_INDEX__DISTANCE=COSINE
```

### Logging Configuration
```bash
export RAG_LOGGING__LEVEL=DEBUG
export RAG_LOGGING__PATH=/var/log/iris_rag.log
```

## Common Use Cases

### 1. Pipeline Initialization
```python
# Standard pipeline setup
config_manager = ConfigurationManager()
schema_manager = SchemaManager(connection_manager, config_manager)

# Ensure all required tables exist with correct schema
tables = ["SourceDocuments", "DocumentChunks", "Entities", "EntityRelationships"]
for table in tables:
    schema_manager.ensure_table_schema(table, pipeline_type="graphrag")
```

### 2. Development vs Production Configuration
```python
# Development setup
dev_config = ConfigurationManager("config/development.yaml")

# Production setup with environment overrides
prod_config = ConfigurationManager("config/production.yaml")
# Environment variables like RAG_DATABASE__IRIS__HOST override YAML
```

### 3. Custom Embedding Model Registration
```python
# Register new embedding model
schema_manager.register_model("custom-bert-large", 1024)

# Use new model for table
dimension = schema_manager.get_vector_dimension("SourceDocuments", "custom-bert-large")
print(f"Custom model dimension: {dimension}")  # 1024
```

### 4. Migration Validation
```python
# Check all tables for migration needs
status = schema_manager.get_schema_status()
migration_needed = [table for table, info in status.items() if info['needs_migration']]

if migration_needed:
    print(f"Tables needing migration: {migration_needed}")
    for table in migration_needed:
        print(f"Migrating {table}...")
        success = schema_manager.migrate_table(table)
        print(f"  {'✓' if success else '✗'} {table}")
```

## Testing Integration

### Contract Test Setup
```python
import pytest
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.schema_manager import SchemaManager

@pytest.fixture
def config_manager():
    return ConfigurationManager("tests/fixtures/test_config.yaml")

@pytest.fixture
def schema_manager(config_manager, connection_manager):
    return SchemaManager(connection_manager, config_manager)

# Use in contract tests
def test_configuration_loading(config_manager):
    assert config_manager.get("database:iris:host") is not None
```

### Mock Database Testing
```python
@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.fetchone.return_value = [0]  # No existing index
    return cursor

def test_hnsw_index_creation(schema_manager, mock_cursor):
    schema_manager.ensure_vector_hnsw_index(
        mock_cursor, "RAG.SourceDocuments", "embedding", "test_index"
    )
    mock_cursor.execute.assert_called()
```

## Performance Optimization

### Configuration Caching
```python
# Configuration values are cached automatically
config_manager = ConfigurationManager()

# Multiple calls use cache
db_config_1 = config_manager.get_database_config()  # Loads and caches
db_config_2 = config_manager.get_database_config()  # Uses cache
```

### Vector Index Optimization
```python
# HNSW indexes created with ACORN=1 when available
schema_manager.ensure_all_vector_indexes()  # Creates optimized indexes

# Manual index creation with optimization
schema_manager.ensure_vector_hnsw_index(
    cursor, "RAG.SourceDocuments", "embedding", "idx_docs_embedding", try_acorn=True
)
```

## Troubleshooting

### Common Configuration Issues
```python
# Debug configuration loading
try:
    config_manager = ConfigurationManager()
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
    # Check required keys and environment variables
```

### Schema Migration Issues
```python
# Debug schema issues
try:
    schema_manager.migrate_table("SourceDocuments")
except Exception as e:
    print(f"Migration failed: {e}")

    # Check table structure
    structure = schema_manager.verify_table_structure("SourceDocuments")
    print(f"Current structure: {structure}")

    # Check schema status
    status = schema_manager.get_schema_status()["SourceDocuments"]
    print(f"Schema status: {status}")
```

### Vector Dimension Issues
```python
# Debug dimension consistency
results = schema_manager.validate_dimension_consistency()
if not results["consistent"]:
    for issue in results["issues"]:
        print(f"Issue: {issue}")
```

## Next Steps

1. **Read the contracts**: Review `contracts/` directory for detailed interface specifications
2. **Check the data model**: See `data-model.md` for comprehensive schema documentation
3. **Run tests**: Execute contract tests to validate implementation
4. **Monitor performance**: Use logging and metrics for production deployments
5. **Customize configuration**: Adapt YAML files and environment variables for your deployment