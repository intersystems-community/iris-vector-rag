# Self-Healing System

This document describes the self-healing system designed to ensure database schema integrity and data consistency for all RAG (Retrieval Augmented Generation) pipelines.

## Overview

The self-healing system has been significantly enhanced with the introduction of the [`SchemaManager`](../iris_rag/storage/schema_manager.py), which provides automatic schema validation, migration, and metadata tracking. This system ensures that database schemas remain consistent with application configuration changes, particularly for vector dimensions and embedding models.

## Key Components

### 1. SchemaManager ([`iris_rag/storage/schema_manager.py`](../iris_rag/storage/schema_manager.py))

The [`SchemaManager`](../iris_rag/storage/schema_manager.py:16) is the core component responsible for:

- **Schema Version Tracking**: Maintains schema metadata in `RAG.SchemaMetadata` table
- **Configuration Change Detection**: Automatically detects mismatches between current schema and expected configuration
- **Automated Migration**: Performs safe schema migrations when changes are detected
- **Vector Dimension Management**: Ensures vector columns match the configured embedding model dimensions

#### Key Features:

- **Auto-detection of Schema Mismatches**: Compares current table schema with expected configuration based on embedding models
- **Automated Migration Strategy**: Currently implements drop/recreate strategy for `RAG.DocumentEntities` table
- **Metadata Tracking**: Stores schema version, vector dimensions, embedding models, and configuration in `RAG.SchemaMetadata`
- **Integration with Pipelines**: Automatically called by pipelines (e.g., GraphRAG) before data operations

### 2. Legacy Self-Healing Components

The system also includes legacy components for broader data management:

*   **[`TableStatusDetector`](../scripts/table_status_detector.py)**: Analyzes RAG database tables for population status and data health
## SchemaManager Capabilities

### Automatic Schema Detection

The SchemaManager automatically detects schema mismatches by comparing current database schema with expected configuration:

```python
# Example of automatic schema detection
schema_manager = SchemaManager(connection_manager, config_manager)

# Check if migration is needed
if schema_manager.needs_migration("DocumentEntities"):
    print("Schema migration required")
    
# Get detailed status
status = schema_manager.get_schema_status()
for table, info in status.items():
    print(f"{table}: {info['status']}")
```

### Vector Dimension Management

The system automatically handles vector dimension changes when embedding models are updated:

#### Supported Embedding Models and Dimensions:
- `all-MiniLM-L6-v2`: 384 dimensions
- `all-mpnet-base-v2`: 768 dimensions  
- `text-embedding-ada-002`: 1536 dimensions
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions

#### Automatic Migration Process:

1. **Detection**: Compares current vector dimension with expected dimension for configured embedding model
2. **Validation**: Checks schema version and embedding model configuration
3. **Migration**: Drops and recreates table with correct vector dimension
4. **Metadata Update**: Updates `RAG.SchemaMetadata` with new configuration

### Schema Metadata Tracking

The system maintains comprehensive metadata in the `RAG.SchemaMetadata` table:

```sql
CREATE TABLE RAG.SchemaMetadata (
    table_name VARCHAR(255) NOT NULL,
    schema_version VARCHAR(50) NOT NULL,
    vector_dimension INTEGER,
    embedding_model VARCHAR(255),
    configuration VARCHAR(MAX),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (table_name)
)
```

### Integration with Pipelines

The SchemaManager integrates seamlessly with RAG pipelines:

```python
# Example from GraphRAG pipeline
def _store_entities(self, document_id: str, entities: List[Dict[str, Any]]):
    # Ensure schema is correct before storing entities
    if not self.schema_manager.ensure_table_schema("DocumentEntities"):
        logger.error("Failed to ensure DocumentEntities table schema")
        raise RuntimeError("Schema validation failed for DocumentEntities table")
    
    # Proceed with entity storage...
```

### Migration Strategies

#### Current Strategy: Drop/Recreate
- **Pros**: Ensures clean schema, handles any structural changes
- **Cons**: Data loss during migration
- **Use Case**: Suitable for development and when data can be regenerated

#### Future Enhancements:
- **Data Preservation**: Backup and restore data during migration
- **In-Place Migration**: Alter table structure without data loss
- **Incremental Migration**: Support for partial schema updates
*   **[`DataPopulationManager`](../scripts/data_population_manager.py)**: Manages data population tasks for RAG tables
*   **[`SelfHealingOrchestrator`](../rag_templates/validation/self_healing_orchestrator.py)**: Coordinates broader self-healing processes

These components are orchestrated via `make` targets defined in the project's [`Makefile`](../Makefile).

## Core `make` Targets & Usage

The following `make` targets are available to manage and interact with the self-healing system:

### Primary Healing Commands:

*   `make heal-data`
    *   **Description**: This is the primary command to trigger a comprehensive self-healing cycle. It aims to achieve 100% table readiness by identifying and populating any missing data.
    *   **Usage**: `make heal-data`

*   `make auto-heal-all`
    *   **Description**: Executes a complete self-healing workflow:
        1.  Checks current data readiness.
        2.  Populates any missing data.
        3.  Validates that the healing process achieved the target readiness (defaulting to 100%).
    *   **Usage**: `make auto-heal-all`

### Status & Validation:

*   `make check-readiness`
    *   **Description**: Checks and reports the current population status of all RAG tables, providing an overall readiness percentage and details for each table.
    *   **Usage**: `make check-readiness`

*   `make heal-status`
    *   **Description**: Provides a detailed status report from the `TableStatusDetector`, including record counts and health scores for each table.
    *   **Usage**: `make heal-status`

*   `make validate-healing`
    *   **Description**: Validates if the self-healing process has achieved the target data readiness (default 100%).
    *   **Usage**: `make validate-healing` (implicitly uses target 100%)
    *   **Usage with specific target**: `make validate-healing TARGET=90` (Not directly supported by this target, use `heal-to-target` or check `data_population_manager.py validate --target <percentage>`)

### Granular Population & Healing:

*   `make populate-missing`
    *   **Description**: Identifies and populates only the data for tables that are currently empty or incomplete, respecting dependencies.
    *   **Usage**: `make populate-missing`

*   `make heal-to-target TARGET=<percentage>`
    *   **Description**: Runs the self-healing process until a specific target readiness percentage is achieved. For example, `TARGET=85` will heal until 85% of tables are ready.
    *   **Usage**: `make heal-to-target TARGET=85`

*   `make heal-progressive`
    *   **Description**: Performs incremental healing with dependency-aware ordering, ensuring foundational tables are populated before their dependents. This is similar to `populate-missing` but can be part of a more controlled rollout.
    *   **Usage**: `make heal-progressive`

*   `make quick-heal`
    *   **Description**: Focuses on populating only the most essential tables quickly. This typically involves running `scripts/data_population_manager.py populate --missing`.
    *   **Usage**: `make quick-heal`

*   `make deep-heal`
    *   **Description**: Initiates a thorough healing process that populates all tables and can include additional data optimization tasks.
    *   **Usage**: `make deep-heal`

### Advanced & Emergency Operations:

*   `make heal-emergency`
    *   **Description**: **WARNING: Use with caution.** Forces a complete repopulation of all RAG tables. This is useful if data is suspected to be corrupt or if a full reset is needed.
    *   **Usage**: `make heal-emergency`

*   `make heal-monitor`
    *   **Description**: Starts a continuous monitoring process. The system periodically checks data readiness and automatically triggers healing cycles if the readiness drops below a configured threshold. Press `Ctrl+C` to stop.
    *   **Usage**: `make heal-monitor`

### Integrated Workflows:

*   `make heal-and-test`
    *   **Description**: Combines data healing with comprehensive testing. First, it runs `make heal-data` to ensure data readiness, then executes `make test-1000` (comprehensive E2E tests with 1000+ documents).
    *   **Usage**: `make heal-and-test`

*   `make heal-and-validate`
    *   **Description**: Ensures data is healed and then runs the full system validation suite. It executes `make heal-data` followed by `make validate-all`.
    *   **Usage**: `make heal-and-validate`

## Integration with RAG System

The self-healing system is crucial for the stable operation of all RAG pipelines. By ensuring that tables like `RAG.SourceDocuments`, `RAG.ChunkedDocuments`, `RAG.ColBERTTokenEmbeddings`, etc., are consistently populated, it guarantees that pipelines have the necessary data to:
*   Retrieve relevant documents.
*   Generate accurate embeddings.
*   Construct knowledge graphs.
*   Provide correct answers to user queries.

An incomplete or corrupt data backend can lead to pipeline failures or degraded performance. The self-healing mechanisms mitigate these risks.

## Basic Usage Example

1.  **Check current system status**:
    ```bash
    make check-readiness
    ```
    This will output the current percentage of populated tables and list any missing ones.

2.  **Run the self-healing process**:
    ```bash
    make heal-data
    ```
    The system will attempt to populate all missing tables.

3.  **Verify completion**:
    ```bash
    make check-readiness
    ```
    This should now report 100% readiness if the healing was successful.

For continuous operation, especially in dynamic environments, `make heal-monitor` can be used to keep the data backend healthy automatically.