# GraphRAG Schema Design

## Overview

This document describes the database schema design for GraphRAG (Graph-based Retrieval-Augmented Generation) functionality in the IRIS RAG templates project. The schema supports entity extraction, relationship mapping, and graph-based retrieval operations while maintaining consistency with the existing RAG architecture.

## Architecture Principles

### 1. Schema Manager Integration
- All tables are managed through the [`SchemaManager`](../../iris_rag/storage/schema_manager.py) class
- Automatic migration support for vector dimension changes
- Consistent vector field definitions across all tables
- Centralized configuration management

### 2. Vector Store Compatibility
- Entity embeddings use the same vector store interface as document embeddings
- Consistent with [`IVectorStore`](../../iris_rag/storage/vector_store/) architecture
- Support for HNSW indexing on entity embeddings

### 3. Modular Design
- Clean separation between entity storage and relationship storage
- Foreign key constraints ensure referential integrity
- Extensible metadata fields for future enhancements

## Table Schemas

### DocumentEntities Table

The `DocumentEntities` table stores extracted entities from documents with their vector embeddings.

#### Schema Definition
```sql
CREATE TABLE RAG.DocumentEntities (
    entity_id VARCHAR(255) NOT NULL,
    doc_id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(1000) NOT NULL,
    entity_type VARCHAR(255) NOT NULL,
    embedding VECTOR(FLOAT, 384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id) ON DELETE CASCADE
);
```

#### Column Descriptions
- **entity_id**: Unique identifier for each entity (Primary Key)
- **doc_id**: Reference to the source document (Foreign Key to `SourceDocuments.doc_id`)
- **entity_name**: The actual text/name of the entity
- **entity_type**: Classification of the entity (e.g., "PERSON", "ORGANIZATION", "LOCATION")
- **embedding**: Vector representation of the entity using document embedding model
- **created_at**: Timestamp of entity extraction

#### Configuration
```python
"DocumentEntities": {
    "embedding_column": "embedding",
    "uses_document_embeddings": True,
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "columns": {
        "entity_id": "entity_id",
        "doc_id": "doc_id",
        "entity_name": "entity_name",
        "entity_type": "entity_type",
        "embedding": "embedding"
    }
}
```

#### Indexes
- Primary key index on `entity_id`
- Foreign key index on `doc_id`
- Index on `entity_type` for filtering by entity types
- HNSW vector index on `embedding` for similarity search

### EntityRelationships Table

The `EntityRelationships` table stores relationships between entities extracted from documents.

#### Schema Definition
```sql
CREATE TABLE RAG.EntityRelationships (
    relationship_id VARCHAR(255) NOT NULL,
    source_entity_id VARCHAR(255) NOT NULL,
    target_entity_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(255) NOT NULL,
    metadata VARCHAR(MAX),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (relationship_id),
    FOREIGN KEY (source_entity_id) REFERENCES RAG.DocumentEntities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES RAG.DocumentEntities(entity_id) ON DELETE CASCADE
);
```

#### Column Descriptions
- **relationship_id**: Unique identifier for each relationship (Primary Key)
- **source_entity_id**: Reference to the source entity (Foreign Key to `DocumentEntities.entity_id`)
- **target_entity_id**: Reference to the target entity (Foreign Key to `DocumentEntities.entity_id`)
- **relationship_type**: Type of relationship (e.g., "WORKS_AT", "LOCATED_IN", "COLLABORATES_WITH")
- **metadata**: JSON metadata for additional relationship properties
- **created_at**: Timestamp of relationship extraction

#### Configuration
```python
"EntityRelationships": {
    "embedding_column": None,
    "uses_document_embeddings": False,
    "default_model": None,
    "dimension": None,
    "columns": {
        "relationship_id": "relationship_id",
        "source_entity_id": "source_entity_id",
        "target_entity_id": "target_entity_id",
        "relationship_type": "relationship_type",
        "metadata": "metadata"
    }
}
```

#### Indexes
- Primary key index on `relationship_id`
- Foreign key indexes on `source_entity_id` and `target_entity_id`
- Index on `relationship_type` for filtering by relationship types

## Schema Management

### Initialization
The GraphRAG schema is initialized through the SchemaManager:

```python
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import get_iris_connection

# Initialize components
connection_manager = type('ConnectionManager', (), {
    'get_connection': lambda: get_iris_connection()
})()
config_manager = ConfigurationManager()

# Create schema manager
schema_manager = SchemaManager(connection_manager, config_manager)

# Ensure GraphRAG tables exist
schema_manager.ensure_table_schema('DocumentEntities')
schema_manager.ensure_table_schema('EntityRelationships')
```

### Migration Support
The schema supports automatic migration when:
- Vector dimensions change in configuration
- Embedding models are updated
- Schema versions are upgraded

Migration is handled automatically by the SchemaManager:
```python
# Check if migration is needed
if schema_manager.needs_migration('DocumentEntities'):
    schema_manager.migrate_table('DocumentEntities')
```

### Vector Dimension Management
Entity embeddings use the base embedding dimension from configuration:
```python
# Get vector dimension for entities
dimension = schema_manager.get_vector_dimension('DocumentEntities')
# Returns: 384 (default base embedding dimension)
```

## Integration with GraphRAG Pipeline

### Entity Extraction
Entities are extracted from documents and stored with embeddings:

```python
from common.db_vector_utils import insert_vector

# Extract entities from document
entities = extract_entities(document_text)

for entity in entities:
    # Generate embedding for entity
    entity_embedding = embedding_func(entity.name)
    
    # Insert entity with vector
    insert_vector(
        connection=connection,
        table_name="RAG.DocumentEntities",
        vector_column="embedding",
        vector_data=entity_embedding,
        additional_data={
            "entity_id": entity.id,
            "doc_id": document.id,
            "entity_name": entity.name,
            "entity_type": entity.type
        }
    )
```

### Relationship Extraction
Relationships between entities are stored without embeddings:

```python
# Extract relationships from document
relationships = extract_relationships(document_text, entities)

for relationship in relationships:
    cursor.execute("""
        INSERT INTO RAG.EntityRelationships 
        (relationship_id, source_entity_id, target_entity_id, relationship_type, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, [
        relationship.id,
        relationship.source_entity_id,
        relationship.target_entity_id,
        relationship.type,
        json.dumps(relationship.metadata)
    ])
```

### Graph-based Retrieval
The schema supports various graph-based retrieval patterns:

#### Entity Similarity Search
```python
from iris_rag.storage.vector_store.iris_impl import IRISVectorStore

vector_store = IRISVectorStore(config_manager=config_manager)

# Find similar entities
query_embedding = embedding_func("target entity")
similar_entities = vector_store.similarity_search(
    query_embedding=query_embedding,
    k=10,
    table_name="DocumentEntities"
)
```

#### Relationship Traversal
```sql
-- Find all entities connected to a specific entity
SELECT DISTINCT 
    e2.entity_name, 
    e2.entity_type,
    r.relationship_type
FROM RAG.EntityRelationships r
JOIN RAG.DocumentEntities e1 ON r.source_entity_id = e1.entity_id
JOIN RAG.DocumentEntities e2 ON r.target_entity_id = e2.entity_id
WHERE e1.entity_name = ?
```

#### Multi-hop Graph Queries
```sql
-- Find entities connected through 2-hop relationships
WITH RECURSIVE entity_paths AS (
    -- Base case: direct relationships
    SELECT 
        source_entity_id,
        target_entity_id,
        relationship_type,
        1 as hop_count
    FROM RAG.EntityRelationships
    WHERE source_entity_id = ?
    
    UNION ALL
    
    -- Recursive case: extend paths
    SELECT 
        ep.source_entity_id,
        er.target_entity_id,
        er.relationship_type,
        ep.hop_count + 1
    FROM entity_paths ep
    JOIN RAG.EntityRelationships er ON ep.target_entity_id = er.source_entity_id
    WHERE ep.hop_count < 2
)
SELECT DISTINCT 
    e.entity_name,
    e.entity_type,
    ep.hop_count
FROM entity_paths ep
JOIN RAG.DocumentEntities e ON ep.target_entity_id = e.entity_id
```

## Performance Considerations

### Vector Indexing
- HNSW indexes are automatically created on entity embeddings
- Index parameters: M=16, efConstruction=200, Distance='COSINE'
- Indexes improve similarity search performance for large entity sets

### Query Optimization
- Foreign key indexes support efficient relationship traversal
- Entity type indexes enable fast filtering by entity categories
- Composite indexes on (doc_id, entity_type) for document-specific queries

### Scalability
- Schema supports millions of entities and relationships
- Vector operations use IRIS native vector functions for optimal performance
- Automatic cleanup through CASCADE DELETE constraints

## Testing and Validation

### Schema Validation Tests
Comprehensive tests are provided in [`test_graphrag_schema_validation.py`](../../tests/test_storage/test_graphrag_schema_validation.py):

- Table configuration validation
- Schema creation verification
- Foreign key constraint testing
- Vector dimension consistency checks
- Migration support validation

### Running Tests
```bash
# Run GraphRAG schema validation tests
uv run pytest tests/test_storage/test_graphrag_schema_validation.py -v | tee test_output/test_graphrag_schema_validation.log
```

## Future Enhancements

### Planned Extensions
1. **Temporal Relationships**: Add timestamp fields for relationship evolution tracking
2. **Weighted Relationships**: Add confidence scores for relationship strength
3. **Entity Hierarchies**: Support for entity type hierarchies and inheritance
4. **Cross-Document Entities**: Entity resolution across multiple documents

### Schema Evolution
The schema is designed to support future enhancements through:
- Extensible metadata fields (JSON format)
- Version-controlled migrations through SchemaManager
- Backward-compatible column additions
- Flexible entity and relationship type systems

## Related Documentation

- [Schema Manager Implementation](../../iris_rag/storage/schema_manager.py)
- [Vector Store Architecture](../../iris_rag/storage/vector_store/)
- [GraphRAG Pipeline Implementation](../../iris_rag/pipelines/)
- [Database Schema Management Rules](../../.clinerules)