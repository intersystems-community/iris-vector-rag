# GraphRAG Implementation

This document outlines the implementation of GraphRAG (Graph-based Retrieval Augmented Generation) in the RAG templates framework.

## Overview

GraphRAG has been significantly enhanced with a major overhaul that includes robust data ingestion, self-healing schema management, and improved entity embedding storage. The pipeline now automatically ensures database schema integrity before processing and provides comprehensive error handling for vector operations.

## Recent Major Enhancements

### Schema Self-Healing Integration
The GraphRAG pipeline now integrates with the [`SchemaManager`](../iris_rag/storage/schema_manager.py) to automatically ensure the `RAG.DocumentEntities` table schema matches the current embedding configuration before ingestion. This prevents schema mismatches that could cause data corruption or ingestion failures.

### Enhanced Data Ingestion Process
The updated data ingestion process includes:
- **Automatic schema validation** before entity storage
- **Robust vector embedding handling** with comprehensive error recovery
- **Detailed logging and metrics** for monitoring ingestion success rates
- **Fallback mechanisms** for entities that fail embedding storage

### Vector Dimension Management
The system now automatically detects and handles vector dimension mismatches by:
- Comparing current table schema with expected embedding dimensions
- Automatically migrating tables when embedding models change
- Preserving data integrity during schema transitions

## Overview

GraphRAG is an advanced retrieval technique that uses a knowledge graph structure to improve information retrieval for question answering. Unlike traditional vector search methods, GraphRAG leverages both the semantic similarity and the graph structure (relationships between entities) to find relevant information.

The key distinguishing features of GraphRAG are:
1. **Graph Traversal** - Starting from seed nodes, navigate through connections to discover related information
2. **Hybrid Scoring** - Combining graph proximity with semantic similarity for node ranking
3. **Path-based Relevance** - Considering the entire path to a node, not just point-to-point similarity
4. **Multi-hop Discovery** - Uncovering information that requires multiple reasoning steps

## SQL-Based Implementation Approach

Our implementation leverages SQL recursive CTEs (Common Table Expressions) for efficient in-database graph traversal. This approach offers several advantages:

1. **Performance** - Keeping traversal operations in the database reduces data transfer and leverages database optimization
2. **Scalability** - SQL engines are designed to handle large datasets efficiently
3. **Integration** - Tighter integration with IRIS database capabilities
4. **Consistency** - Following the same SQL-based pattern as our NodeRAG implementation

## Architecture Components

### 1. GraphRAG Pipeline ([`iris_rag/pipelines/graphrag.py`](../iris_rag/pipelines/graphrag.py))

The main [`GraphRAGPipeline`](../iris_rag/pipelines/graphrag.py:21) class provides:

- **Entity Extraction**: Automatic extraction of entities from document content
- **Relationship Mapping**: Discovery of relationships between entities based on co-occurrence
- **Graph-based Retrieval**: Two-stage query process for efficient document retrieval
- **Vector Fallback**: Automatic fallback to vector search when graph traversal yields no results
- **Schema Integration**: Automatic schema validation via [`SchemaManager`](../iris_rag/storage/schema_manager.py:16)

### 2. Schema Manager ([`iris_rag/storage/schema_manager.py`](../iris_rag/storage/schema_manager.py))

The [`SchemaManager`](../iris_rag/storage/schema_manager.py:16) ensures database schema integrity:

- **Automatic Schema Detection**: Compares current table schema with expected configuration
- **Vector Dimension Validation**: Ensures embedding dimensions match the configured model
- **Automated Migration**: Handles schema changes with minimal data loss
- **Metadata Tracking**: Maintains schema version history in `RAG.SchemaMetadata`

### 3. Enhanced Entity Storage

The [`_store_entities`](../iris_rag/pipelines/graphrag.py:393) method includes:

- **Pre-ingestion Schema Validation**: Calls [`schema_manager.ensure_table_schema()`](../iris_rag/storage/schema_manager.py:285) before storing entities
- **Robust Vector Handling**: Comprehensive error handling for vector embedding storage
- **Fallback Mechanisms**: Stores entities without embeddings if vector operations fail
- **Detailed Metrics**: Tracks success/failure rates for embedding storage

### 4. Database Schema

The system uses the following enhanced table structure:

#### RAG.DocumentEntities
```sql
CREATE TABLE RAG.DocumentEntities (
    entity_id VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    entity_text VARCHAR(1000) NOT NULL,
    entity_type VARCHAR(100),
    position INTEGER,
    embedding VECTOR(FLOAT, <dimension>),  -- Dimension auto-configured
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id)
)
```

#### RAG.SchemaMetadata
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

## The Knowledge Graph Structure

The GraphRAG knowledge graph is stored in two main tables:

1. **KnowledgeGraphNodes**
   ```sql
   CREATE TABLE KnowledgeGraphNodes (
       node_id VARCHAR(255) PRIMARY KEY,
       node_type VARCHAR(50),
       node_name VARCHAR(255),
       content CLOB,
       embedding VECTOR,
       metadata_json CLOB
   )
   ```

2. **KnowledgeGraphEdges**
   ```sql
   CREATE TABLE KnowledgeGraphEdges (
       edge_id VARCHAR(255) PRIMARY KEY,
       source_node_id VARCHAR(255),
       target_node_id VARCHAR(255),
       relationship_type VARCHAR(50),
       weight FLOAT,
       properties_json CLOB,
       FOREIGN KEY (source_node_id) REFERENCES KnowledgeGraphNodes(node_id),
       FOREIGN KEY (target_node_id) REFERENCES KnowledgeGraphNodes(node_id)
   )
   ```

## Recursive CTE for Graph Traversal

The core of our implementation is the SQL recursive CTE that performs graph traversal.
**Note on Parameters in CTE:** The SQL snippet below uses placeholders like `:query_embedding_str`, `:score_decay`, `:hybrid_weight`, and `:max_depth`. The `StartNodes` subquery implies initial seed nodes are provided. Due to IRIS SQL limitations where `TO_VECTOR()` does not accept direct parameter markers, these placeholders (especially `:query_embedding_str`) are substituted with validated values when the final SQL query is constructed in the Python code (e.g., using f-strings or other string formatting). The `common/vector_sql_utils.py` module provides utilities for safe construction of such dynamic SQL.

```sql
WITH RECURSIVE PathCTE (start_node, current_node, path, depth, score) AS (
    -- Base case: start with seed nodes from vector similarity
    SELECT
        n.node_id,
        n.node_id,
        CAST(n.node_id AS VARCHAR(1000)),
        0,
        VECTOR_COSINE_SIMILARITY(n.embedding, TO_VECTOR(:query_embedding_str)) AS score -- :query_embedding_str is replaced by validated string literal
    FROM KnowledgeGraphNodes n
    WHERE n.node_id IN (SELECT node_id FROM StartNodes) -- StartNodes generated dynamically
    
    UNION ALL
    
    -- Recursive case: traverse connected nodes
    SELECT
        p.start_node,
        e.target_node_id,
        p.path || ',' || e.target_node_id,
        p.depth + 1,
        -- Calculate hybrid score combining path score and direct relevance
        p.score * :score_decay + -- :score_decay is replaced by validated numeric literal
        VECTOR_COSINE_SIMILARITY(
            target_node.embedding,
            TO_VECTOR(:query_embedding_str)
        ) * :hybrid_weight -- :hybrid_weight is replaced by validated numeric literal
    FROM PathCTE p
    JOIN KnowledgeGraphEdges e ON p.current_node = e.source_node_id
    JOIN KnowledgeGraphNodes target_node ON e.target_node_id = target_node.node_id
    WHERE p.depth < :max_depth -- :max_depth is replaced by validated integer literal
      -- Avoid cycles in traversal
      AND NOT POSITION(',' || e.target_node_id || ',' IN ',' || p.path || ',') > 0
)
```

## Updated Data Ingestion Process

The GraphRAG pipeline now includes a robust data ingestion process with automatic schema management:

### 1. Schema Validation Before Ingestion

Before storing any entities, the pipeline automatically validates and ensures the database schema is correct:

```python
# Automatic schema validation in _store_entities method
if not self.schema_manager.ensure_table_schema("DocumentEntities"):
    logger.error("Failed to ensure DocumentEntities table schema")
    raise RuntimeError("Schema validation failed for DocumentEntities table")
```

### 2. Entity Embedding Storage with Error Handling

The enhanced entity storage process includes comprehensive error handling:

```python
# Enhanced embedding storage with validation
from common.vector_format_fix import format_vector_for_iris, validate_vector_for_iris

for entity in entities:
    embedding_formatted = None
    if "embedding" in entity and entity["embedding"] is not None:
        try:
            # Format and validate vector
            embedding_list = format_vector_for_iris(entity["embedding"])
            if validate_vector_for_iris(embedding_list):
                embedding_formatted = create_iris_vector_string(embedding_list)
        except VectorFormatError as e:
            logger.warning(f"Vector formatting error: {e}")
            # Continue without embedding
```

### 3. Automatic Schema Migration

When embedding models change, the system automatically migrates the schema:

```python
# Schema migration is triggered automatically when:
# - Vector dimensions don't match current embedding model
# - Embedding model configuration changes
# - Schema version is outdated

# Example: Changing from 384-dim to 768-dim embeddings
# The SchemaManager will:
# 1. Detect the dimension mismatch
# 2. Drop and recreate the DocumentEntities table
# 3. Update schema metadata
# 4. Log the migration process
```

## Usage

### 1. Basic Pipeline Setup

```python
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

# Initialize managers
connection_manager = ConnectionManager(config_manager)
config_manager = ConfigurationManager("config.yaml")

# Create pipeline (SchemaManager is initialized automatically)
pipeline = GraphRAGPipeline(
    connection_manager=connection_manager,
    config_manager=config_manager,
    llm_func=your_llm_function  # Optional
)
```

### 2. Document Ingestion with Automatic Schema Management

```python
from iris_rag.core.models import Document

# Prepare documents
documents = [
    Document(
        page_content="Your document content here...",
        metadata={"source": "document1.txt"}
    ),
    # ... more documents
]

# Ingest documents (schema validation happens automatically)
result = pipeline.ingest_documents(documents)

# Check results
print(f"Documents ingested: {result['documents_ingested']}")
print(f"Entities created: {result['entities_created']}")
print(f"Relationships created: {result['relationships_created']}")
```

### 3. Query Execution

```python
# Execute query
result = pipeline.query("What is the relationship between diabetes and insulin?")

# Process results
answer = result["answer"]
retrieved_docs = result["retrieved_documents"]
query_entities = result["query_entities"]

print(f"Answer: {answer}")
print(f"Retrieved {len(retrieved_docs)} documents")
print(f"Query entities: {query_entities}")
```

## Key Advantages of GraphRAG

- **Relationship Discovery**: Finds information connected through relationships that might not be found by direct vector similarity
- **Multi-hop Reasoning**: Can discover information that requires multiple steps or connections to reach
- **Explanatory Paths**: The traversal path provides a natural explanation of how information is connected
- **Complex Query Support**: Better equipped to handle complex queries requiring integration of multiple information sources

## Differences from NodeRAG

While both GraphRAG and NodeRAG use graph structures, they have different emphases:

- **NodeRAG** organizes nodes by type (Entity, Document, Concept) and focuses on presenting information by node type
- **GraphRAG** focuses more on the path through the graph, with less emphasis on node types
- **NodeRAG** uses node types to guide traversal behavior
- **GraphRAG** uses relationship types and weights more prominently in traversal decisions

## Implementation Notes

- The hybrid scoring approach balances both the graph structure (path from initial nodes) and direct relevance (vector similarity to the query)
- The recursive CTE approach allows efficient traversal entirely within the database
- Mock implementations are provided for testing without requiring a real knowledge graph
- The implementation includes detailed logging for better debugging and performance analysis
