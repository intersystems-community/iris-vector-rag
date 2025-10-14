# GraphRAG Entity Extraction System Architecture

## Executive Summary

This document defines the comprehensive architecture for fixing and extending the GraphRAG pipeline with a proper entity extraction system. The current implementation is fundamentally broken - it attempts to read from empty knowledge graph tables with no extraction pipeline to populate them.

## Current State Analysis

### Root Causes Identified
1. **Connection API Misuse**: Uses `self.connection_manager.connection` (doesn't exist) instead of `get_connection()`
2. **Schema Gap**: No management for `RAG.Entities`, `RAG.EntityRelationships` tables
3. **Validation Gap**: GraphRAG not in requirements registry, so validation never runs  
4. **Missing Core Component**: No entity extraction pipeline exists anywhere

### Critical Issues
- GraphRAG pipeline reads from empty knowledge graph tables
- Always falls back to vector search (defeating the purpose)
- No entity extraction service to populate the knowledge graph
- Schema manager unaware of graph tables
- No validation ensures proper setup

## Proposed System Architecture

### 1. Entity Extraction Service

```
┌─────────────────────────────────────────────────────────────┐
│                   Entity Extraction Service                 │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │   LLM-Based     │ │   NLP-Based     │ │   Pattern-Based │ │
│ │   Extractor     │ │   Extractor     │ │   Extractor     │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │            Entity Extraction Engine                     │ │
│ │ • Configurable extraction strategies                    │ │
│ │ • Domain-aware entity type mapping                      │ │
│ │ • Batch processing with rate limiting                   │ │
│ │ • Error handling and retry logic                        │ │
│ │ • Performance monitoring and metrics                    │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │            Relationship Extraction Engine               │ │
│ │ • Dependency parsing for relationships                  │ │
│ │ • Co-occurrence analysis                                │ │
│ │ • Semantic relationship detection                       │ │
│ │ • Cross-document relationship linking                   │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2. Knowledge Graph Storage Layer

```
┌─────────────────────────────────────────────────────────────┐
│                Knowledge Graph Storage                       │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │  RAG.Entities   │ │RAG.EntityRela-  │ │ RAG.EntityEm-   │ │
│ │                 │ │  tionships      │ │   beddings      │ │
│ │ • entity_id     │ │ • relationship_ │ │ • entity_id     │ │
│ │ • entity_name   │ │   id            │ │ • embedding     │ │
│ │ • entity_type   │ │ • source_entity │ │ • model_version │ │
│ │ • source_doc_id │ │ • target_entity │ │ • created_at    │ │
│ │ • confidence    │ │ • relation_type │ │                 │ │
│ │ • metadata      │ │ • confidence    │ │                 │ │
│ │ • created_at    │ │ • metadata      │ │                 │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3. Extended Schema Management

```
┌─────────────────────────────────────────────────────────────┐
│                  Extended Schema Manager                     │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │            Existing Schema Manager                      │ │
│ │ • SourceDocuments table management                      │ │
│ │ • DocumentChunks table management                       │ │
│ │ • Vector dimension authority                            │ │
│ │ • Migration support                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │         GraphRAG Schema Extensions                      │ │
│ │ • RAG.Entities table DDL and migration                  │ │
│ │ • RAG.EntityRelationships table DDL and migration      │ │
│ │ • RAG.EntityEmbeddings table DDL and migration         │ │
│ │ • Graph-specific indexing strategies                    │ │
│ │ • Performance optimization for graph traversal         │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4. Enhanced Validation Framework

```
┌─────────────────────────────────────────────────────────────┐
│                Enhanced Validation Framework                │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │              Existing Validators                        │ │
│ │ • BasicRAGRequirements                                  │ │
│ │ • CRAGRequirements                                      │ │
│ │ • BasicRAGRerankingRequirements                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │           GraphRAG Requirements                         │ │
│ │ • Required tables: Entities, EntityRelationships       │ │
│ │ • Required embeddings: Entity embeddings               │ │
│ │ • Minimum entity/relationship counts                    │ │
│ │ • Graph connectivity validation                         │ │
│ │ • Performance threshold validation                      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 5. Complete System Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Document   │───▶│     Entity      │───▶│  Knowledge  │ │
│  │   Loading    │    │   Extraction    │    │   Graph     │ │
│  │              │    │                 │    │  Population │ │
│  └──────────────┘    └─────────────────┘    └─────────────┘ │
│                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Query      │───▶│     Graph       │───▶│   Answer    │ │
│  │ Processing   │    │   Traversal     │    │ Generation  │ │
│  │              │    │                 │    │             │ │
│  └──────────────┘    └─────────────────┘    └─────────────┘ │
│                                                             │
│            ┌─────────────────────────────────┐              │
│            │        Fallback to Vector       │              │
│            │        Search if Graph          │              │
│            │        Query Fails              │              │
│            └─────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Component Design Details

### Entity Extraction Service Interface

```python
class IEntityExtractor(ABC):
    """Interface for entity extraction services."""
    
    @abstractmethod
    def extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities from a document."""
        pass
    
    @abstractmethod
    def extract_relationships(self, document: Document, 
                            entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities."""
        pass
    
    @abstractmethod
    def set_domain_config(self, domain_config: DomainConfig) -> None:
        """Configure for specific domain (medical, legal, etc.)."""
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """Get supported entity types."""
        pass
```

### Schema Extensions

```sql
-- RAG.Entities table
CREATE TABLE RAG.Entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    source_doc_id VARCHAR(255) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    canonical_name VARCHAR(500),
    metadata VARCHAR(MAX),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- RAG.EntityRelationships table  
CREATE TABLE RAG.EntityRelationships (
    relationship_id VARCHAR(255) PRIMARY KEY,
    source_entity_id VARCHAR(255) NOT NULL,
    target_entity_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source_doc_id VARCHAR(255),
    metadata VARCHAR(MAX),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES RAG.Entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES RAG.Entities(entity_id),
    FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Performance indexes
CREATE INDEX idx_entities_type ON RAG.Entities(entity_type);
CREATE INDEX idx_entities_source ON RAG.Entities(source_doc_id);
CREATE INDEX idx_entities_name ON RAG.Entities(entity_name);
CREATE INDEX idx_relationships_source ON RAG.EntityRelationships(source_entity_id);
CREATE INDEX idx_relationships_target ON RAG.EntityRelationships(target_entity_id);
CREATE INDEX idx_relationships_type ON RAG.EntityRelationships(relationship_type);
```

### GraphRAG Requirements Class

```python
class GraphRAGRequirements(PipelineRequirements):
    """Requirements for GraphRAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "graphrag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments", 
                schema="RAG", 
                description="Document storage",
                min_rows=1
            ),
            TableRequirement(
                name="Entities", 
                schema="RAG", 
                description="Extracted entities",
                min_rows=5  # Require minimum entities
            ),
            TableRequirement(
                name="EntityRelationships", 
                schema="RAG", 
                description="Entity relationships",
                min_rows=2  # Require minimum relationships  
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments", 
                column="embedding",
                description="Document-level embeddings"
            ),
            EmbeddingRequirement(
                name="entity_embeddings",
                table="RAG.Entities",
                column="embedding", 
                description="Entity embeddings for similarity"
            )
        ]
```

## Integration Strategy

### 1. Document Loading Integration

```python
def load_documents(self, documents_path: str, **kwargs) -> None:
    """Enhanced document loading with entity extraction."""
    
    # Step 1: Store documents (existing functionality)
    documents = self._load_documents_from_path(documents_path)
    self.vector_store.add_documents(documents, auto_chunk=True)
    
    # Step 2: Extract entities (NEW)
    entity_service = self._get_entity_extraction_service()
    
    for document in documents:
        # Extract entities
        entities = entity_service.extract_entities(document)
        
        # Extract relationships
        relationships = entity_service.extract_relationships(document, entities)
        
        # Store in knowledge graph
        self._store_entities(entities, document.id)
        self._store_relationships(relationships, document.id)
```

### 2. Query Processing Integration

```python
def _retrieve_via_kg(self, query_text: str, top_k: int) -> Tuple[List[Document], str]:
    """Fixed knowledge graph retrieval."""
    
    # Step 1: Find seed entities (FIXED CONNECTION)
    connection = self.connection_manager.get_connection()  # FIX: Use get_connection()
    seed_entities = self._find_seed_entities(query_text, connection)
    
    if not seed_entities:
        return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
    
    # Step 2: Traverse graph
    relevant_entities = self._traverse_graph(seed_entities, connection)
    
    # Step 3: Get documents
    docs = self._get_documents_from_entities(relevant_entities, top_k, connection)
    
    return docs, "knowledge_graph_traversal"
```

## Performance Considerations

### 1. Entity Extraction Performance

- **Batch Processing**: Process documents in configurable batches (default: 10)
- **Parallel Processing**: Use async/await for I/O-bound operations
- **Caching**: Cache entity extraction results for identical content
- **Rate Limiting**: Configurable delays for LLM-based extraction

### 2. Graph Traversal Performance

- **Depth Limiting**: Configurable max traversal depth (default: 2)
- **Entity Limiting**: Configurable max entities per query (default: 50)
- **Index Optimization**: Proper indexing on entity names, types, relationships
- **Connection Pooling**: Reuse database connections

### 3. Memory Management

- **Streaming Processing**: Process large document collections in streams
- **Garbage Collection**: Clean up temporary objects during processing
- **Memory Monitoring**: Track memory usage during extraction

## Error Handling Strategy

### 1. Fail-Fast Validation

```python
def validate_graphrag_prerequisites(self) -> ValidationResult:
    """Validate GraphRAG can run successfully."""
    
    errors = []
    
    # Check table existence
    if not self._table_exists("RAG.Entities"):
        errors.append("RAG.Entities table does not exist")
    
    # Check minimum data requirements
    entity_count = self._count_entities()
    if entity_count < 5:
        errors.append(f"Insufficient entities: {entity_count} < 5")
    
    # Check graph connectivity
    if not self._has_relationships():
        errors.append("No relationships found - graph is disconnected")
    
    return ValidationResult(errors=errors)
```

### 2. Graceful Degradation

- **Vector Fallback**: Always fall back to vector search if graph queries fail
- **Partial Results**: Return partial results rather than complete failure
- **Error Logging**: Comprehensive logging for debugging
- **Retry Logic**: Configurable retry attempts for transient failures

### 3. Circuit Breaker Pattern

- **Failure Threshold**: Stop using graph after N consecutive failures
- **Recovery Testing**: Periodically test if graph queries are working again
- **Fallback Metrics**: Track fallback usage for monitoring

## File Organization

```
iris_rag/
├── storage/
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── interfaces.py           # Service interfaces
│   │   ├── models.py               # Entity/Relationship models  
│   │   ├── entity_extractor.py     # Entity extraction service
│   │   ├── graph_storage.py        # Knowledge graph storage
│   │   └── graph_traversal.py      # Graph traversal engine
│   ├── schema_manager.py           # EXTENDED with graph tables
│   └── vector_store_iris.py
├── validation/
│   └── requirements.py             # EXTENDED with GraphRAGRequirements
└── pipelines/
    └── graphrag.py                 # FIXED implementation
```

## Implementation Phases

### Phase 1: Foundation (1-2 weeks)
- [ ] Create knowledge graph interfaces and models
- [ ] Extend schema manager for graph tables  
- [ ] Add GraphRAG to validation requirements registry
- [ ] Fix connection manager usage in GraphRAG pipeline

### Phase 2: Entity Extraction (2-3 weeks)  
- [ ] Implement entity extraction service
- [ ] Add relationship extraction capabilities
- [ ] Create domain configuration system
- [ ] Add batch processing and error handling

### Phase 3: Integration (1-2 weeks)
- [ ] Integrate entity extraction into document loading
- [ ] Fix graph traversal implementation
- [ ] Add performance monitoring
- [ ] Implement circuit breaker pattern

### Phase 4: Testing & Optimization (1-2 weeks)
- [ ] Comprehensive test suite
- [ ] Performance benchmarking
- [ ] Memory optimization
- [ ] Documentation completion

## Success Metrics

### Functional Metrics
- [ ] Entity extraction accuracy > 80%
- [ ] Relationship extraction accuracy > 70% 
- [ ] Graph connectivity (entities connected by relationships)
- [ ] Query success rate > 95%

### Performance Metrics
- [ ] Entity extraction: < 5 seconds per document
- [ ] Graph traversal: < 500ms per query
- [ ] Memory usage: < 2GB for 1000 documents
- [ ] Fallback rate: < 10% of queries

### Quality Metrics
- [ ] Code coverage > 80%
- [ ] All validation tests pass
- [ ] Documentation complete
- [ ] Architecture review approval

This architecture provides a comprehensive solution to fix the broken GraphRAG implementation while establishing a solid foundation for knowledge graph operations within the IRIS RAG framework.