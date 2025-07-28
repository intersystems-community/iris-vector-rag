# GraphRAG Knowledge Graph Infrastructure Architecture

## Executive Summary

This document defines the comprehensive knowledge graph infrastructure for GraphRAG, designed to integrate seamlessly with the existing RAG templates architecture while providing robust entity extraction, relationship mapping, and graph-based retrieval capabilities. The design incorporates IRIS's unique globals architecture for high-performance graph traversal optimization.

## Current Implementation Analysis

### Existing Components
- **Basic GraphRAG Pipeline**: [`iris_rag/pipelines/graphrag.py`](../../iris_rag/pipelines/graphrag.py) - Partial implementation with basic entity extraction
- **NodeRAG Pipeline**: [`iris_rag/pipelines/noderag.py`](../../iris_rag/pipelines/noderag.py) - Graph traversal capabilities
- **Schema Tables**: Basic knowledge graph tables (`DocumentEntities`, `KnowledgeGraphNodes`, `KnowledgeGraphEdges`)
- **Vector Store Integration**: [`IRISVectorStore`](../../iris_rag/storage/vector_store_iris.py) interface compliance

### Identified Gaps
1. **Missing Comprehensive Entity Extraction**: Current implementation uses basic keyword extraction
2. **Lack of Relationship Mapping**: No sophisticated relationship identification between entities
3. **No Knowledge Graph Manager**: Missing centralized graph operations interface
4. **Limited Graph Query Engine**: Basic traversal without advanced graph algorithms
5. **No Entity Linking/Disambiguation**: Entities not properly linked or deduplicated
6. **Missing Performance Optimization**: No caching or indexing strategies for graph operations

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    GraphRAG Knowledge Graph Infrastructure      │
├─────────────────────────────────────────────────────────────────┤
│  GraphRAG Pipeline  │  Knowledge Graph  │  Graph Query Engine  │
│                     │     Manager       │                      │
├─────────────────────────────────────────────────────────────────┤
│                    Core Knowledge Graph Services                │
├─────────────────────────────────────────────────────────────────┤
│ Entity Extractor │ Relationship │ Entity Linker │ Graph Builder │
│                  │   Extractor  │               │               │
├─────────────────────────────────────────────────────────────────┤
│                    Storage & Query Layer                        │
├─────────────────────────────────────────────────────────────────┤
│ SQL Tables       │ IRIS Globals  │ Vector Store │ Query Engine  │
│ (Basic Triples)  │ (Optimized    │ (Embeddings) │ (Path Queries)│
│                  │  Traversal)   │              │               │
├─────────────────────────────────────────────────────────────────┤
│                    IRIS Database Layer                          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Models and Schema Design

### Phase 1: Basic Triple Storage (Initial Implementation)

We start with simple, efficient triple storage using basic SQL tables with indexes. This provides a solid foundation that can be optimized later.

#### Core Entity Model

```python
@dataclass
class Entity:
    entity_id: str
    entity_name: str
    entity_type: str
    canonical_name: str  # For disambiguation
    description: Optional[str]
    properties: Dict[str, Any]
    embedding: Optional[List[float]]
    confidence_score: float
    source_documents: List[str]
    created_at: datetime
    updated_at: datetime
```

#### Core Relationship Model

```python
@dataclass
class Relationship:
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    strength: float
    confidence: float
    source_documents: List[str]
    created_at: datetime
```

### Basic Triple Tables Schema

#### Entities Table
```sql
CREATE TABLE RAG.Entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_name VARCHAR(500) NOT NULL,
    canonical_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    description TEXT,
    properties TEXT, -- JSON object
    embedding VECTOR(FLOAT, 384), -- Vector column for IRIS
    confidence_score FLOAT DEFAULT 1.0,
    source_documents TEXT, -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Relationships Table (Triple Store)
```sql
CREATE TABLE RAG.Relationships (
    relationship_id VARCHAR(255) PRIMARY KEY,
    source_entity_id VARCHAR(255) NOT NULL,
    target_entity_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    confidence FLOAT DEFAULT 1.0,
    source_documents TEXT, -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES RAG.Entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES RAG.Entities(entity_id)
);
```

#### Basic Indexes for Performance
```sql
-- Entity indexes
CREATE INDEX idx_entities_type ON RAG.Entities(entity_type);
CREATE INDEX idx_entities_canonical ON RAG.Entities(canonical_name);
CREATE VECTOR INDEX idx_entities_embedding ON RAG.Entities(embedding);

-- Relationship indexes for basic traversal
CREATE INDEX idx_relationships_source ON RAG.Relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON RAG.Relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON RAG.Relationships(relationship_type);
CREATE INDEX idx_relationships_strength ON RAG.Relationships(strength);

-- Composite indexes for common query patterns
CREATE INDEX idx_rel_source_type ON RAG.Relationships(source_entity_id, relationship_type);
CREATE INDEX idx_rel_target_type ON RAG.Relationships(target_entity_id, relationship_type);
```

## Performance Optimization Strategy: IRIS Globals Architecture

### Phase 2: IRIS Globals Optimization (Future Enhancement)

When basic SQL table performance becomes insufficient for complex path queries, we can leverage IRIS's unique globals architecture for "pointer chasing" operations directly on the underlying globals.

#### IRIS Globals Design for Graph Traversal

```objectscript
// Entity Global Structure
^RAG.Entity(entityId) = entityData
^RAG.Entity(entityId, "type") = entityType
^RAG.Entity(entityId, "name") = entityName
^RAG.Entity(entityId, "embedding") = embeddingVector

// Relationship Globals for Fast Traversal
^RAG.Graph("out", sourceEntityId, relationshipType, targetEntityId) = strength
^RAG.Graph("in", targetEntityId, relationshipType, sourceEntityId) = strength

// Path Index for Multi-hop Queries
^RAG.Path(startEntity, depth, endEntity) = pathData
^RAG.PathIndex(relationshipType, depth) = ""
```

#### ObjectScript Graph Traversal Functions

```objectscript
/// High-performance graph traversal using globals
ClassMethod TraverseGraph(startEntity As %String, maxDepth As %Integer = 3, 
                         relationshipTypes As %List = "") As %List
{
    Set result = ##class(%ListOfDataTypes).%New()
    Set visited = ##class(%ArrayOfDataTypes).%New()
    
    // Use globals for direct pointer chasing
    Do ..TraverseRecursive(startEntity, 0, maxDepth, relationshipTypes, .visited, result)
    
    Quit result
}

/// Recursive traversal with globals optimization
ClassMethod TraverseRecursive(entityId As %String, currentDepth As %Integer,
                             maxDepth As %Integer, relationshipTypes As %List,
                             ByRef visited, result As %List)
{
    If (currentDepth > maxDepth) || (visited.IsDefined(entityId)) Quit
    
    Set visited(entityId) = 1
    Do result.Insert(entityId)
    
    // Direct global traversal - much faster than SQL JOINs
    Set relType = ""
    For {
        Set relType = $Order(^RAG.Graph("out", entityId, relType))
        Quit:relType=""
        
        // Filter by relationship types if specified
        If (relationshipTypes.Count() > 0) && ('relationshipTypes.Find(relType)) Continue
        
        Set targetEntity = ""
        For {
            Set targetEntity = $Order(^RAG.Graph("out", entityId, relType, targetEntity))
            Quit:targetEntity=""
            
            Do ..TraverseRecursive(targetEntity, currentDepth + 1, maxDepth, 
                                  relationshipTypes, .visited, result)
        }
    }
}
```

#### Python Interface to ObjectScript Optimization

```python
class IRISGraphOptimizer:
    """High-performance graph operations using IRIS globals."""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self._globals_enabled = self._check_globals_support()
    
    def traverse_graph_optimized(self, start_entity: str, max_depth: int = 3,
                                relationship_types: List[str] = None) -> List[str]:
        """Use ObjectScript globals for high-performance traversal."""
        if not self._globals_enabled:
            return self._fallback_sql_traversal(start_entity, max_depth, relationship_types)
        
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Call ObjectScript method directly
            rel_types_list = ",".join(relationship_types) if relationship_types else ""
            
            cursor.execute("""
                SELECT RAG.GraphTraversal_TraverseGraph(?, ?, ?)
            """, [start_entity, max_depth, rel_types_list])
            
            result = cursor.fetchone()[0]
            return self._parse_traversal_result(result)
            
        except Exception as e:
            logger.warning(f"Globals optimization failed, falling back to SQL: {e}")
            return self._fallback_sql_traversal(start_entity, max_depth, relationship_types)
        finally:
            cursor.close()
    
    def _fallback_sql_traversal(self, start_entity: str, max_depth: int,
                               relationship_types: List[str]) -> List[str]:
        """Fallback to SQL-based traversal."""
        # Standard SQL recursive CTE or iterative approach
        pass
```

## Service Architecture

### 1. Knowledge Graph Manager Interface

```python
class IKnowledgeGraphManager(ABC):
    """Core interface for knowledge graph operations."""
    
    @abstractmethod
    def build_graph(self, documents: List[Document]) -> GraphBuildResult:
        """Build knowledge graph from documents."""
        pass
    
    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """Add entity to graph."""
        pass
    
    @abstractmethod
    def link_entities(self, entity_id1: str, entity_id2: str, 
                     relationship: Relationship) -> str:
        """Create relationship between entities."""
        pass
    
    @abstractmethod
    def find_subgraph(self, seed_entities: List[str], 
                     max_depth: int = 2) -> SubGraph:
        """Extract subgraph around seed entities."""
        pass
    
    @abstractmethod
    def traverse_graph(self, start_entity: str, 
                      traversal_config: TraversalConfig) -> List[Entity]:
        """Traverse graph with specified strategy."""
        pass
```

### 2. Entity Extraction Service

```python
class EntityExtractionService:
    """Entity extraction with configurable complexity."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.extraction_mode = config_manager.get("graphrag.entity_extraction.mode", "basic")
        
        if self.extraction_mode == "advanced":
            self.ner_models = self._load_ner_models()
        
    def extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities using configured strategy."""
        if self.extraction_mode == "basic":
            return self._extract_basic_entities(document)
        elif self.extraction_mode == "advanced":
            return self._extract_advanced_entities(document)
        else:
            return self._extract_pattern_entities(document)
    
    def _extract_basic_entities(self, document: Document) -> List[Entity]:
        """Basic entity extraction using simple patterns."""
        # Start with simple, reliable extraction
        entities = []
        text = document.page_content
        
        # Extract capitalized words as potential entities
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for i, word in enumerate(set(capitalized_words)):
            if len(word) > 3:  # Filter short words
                entity = Entity(
                    entity_id=f"{document.id}_entity_{i}",
                    entity_name=word,
                    entity_type="UNKNOWN",
                    canonical_name=word.lower(),
                    description=None,
                    properties={},
                    embedding=None,
                    confidence_score=0.7,
                    source_documents=[document.id],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                entities.append(entity)
        
        return entities
```

### 3. Relationship Extraction Service

```python
class RelationshipExtractionService:
    """Extract relationships between entities."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.extraction_mode = config_manager.get("graphrag.relationship_extraction.mode", "cooccurrence")
    
    def extract_relationships(self, document: Document, 
                            entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using configured strategy."""
        if self.extraction_mode == "cooccurrence":
            return self._extract_cooccurrence_relationships(document, entities)
        elif self.extraction_mode == "pattern":
            return self._extract_pattern_relationships(document, entities)
        else:
            return []
    
    def _extract_cooccurrence_relationships(self, document: Document,
                                          entities: List[Entity]) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        relationships = []
        
        # Simple co-occurrence within sentence boundaries
        sentences = document.page_content.split('.')
        
        for sentence in sentences:
            sentence_entities = [e for e in entities if e.entity_name in sentence]
            
            # Create relationships between co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    relationship = Relationship(
                        relationship_id=f"{entity1.entity_id}_{entity2.entity_id}_cooccur",
                        source_entity_id=entity1.entity_id,
                        target_entity_id=entity2.entity_id,
                        relationship_type="CO_OCCURS",
                        strength=1.0,
                        confidence=0.6,
                        source_documents=[document.id],
                        created_at=datetime.now()
                    )
                    relationships.append(relationship)
        
        return relationships
```

### 4. Adaptive Graph Query Engine

```python
class GraphQueryEngine:
    """Adaptive graph querying with performance optimization."""
    
    def __init__(self, knowledge_graph_manager: IKnowledgeGraphManager):
        self.kg_manager = knowledge_graph_manager
        self.optimizer = IRISGraphOptimizer(knowledge_graph_manager.connection_manager)
        self.cache = GraphQueryCache()
    
    def find_relevant_subgraph(self, query_entities: List[str], 
                              query_context: str,
                              max_depth: int = 2) -> SubGraph:
        """Find most relevant subgraph with adaptive optimization."""
        
        # Check cache first
        cache_key = self._generate_cache_key(query_entities, max_depth)
        cached_result = self.cache.get_cached_subgraph(cache_key)
        if cached_result:
            return cached_result
        
        # Determine optimal traversal strategy
        if len(query_entities) > 10 or max_depth > 3:
            # Use IRIS globals optimization for complex queries
            traversal_result = self.optimizer.traverse_graph_optimized(
                query_entities[0], max_depth)
        else:
            # Use standard SQL for simple queries
            traversal_result = self._sql_based_traversal(query_entities, max_depth)
        
        # Build subgraph from traversal result
        subgraph = self._build_subgraph(traversal_result, query_context)
        
        # Cache result
        self.cache.cache_subgraph(cache_key, subgraph)
        
        return subgraph
```

## Integration with Existing Architecture

### 1. RAGPipeline Integration

```python
class EnhancedGraphRAGPipeline(RAGPipeline):
    """Enhanced GraphRAG with comprehensive knowledge graph."""
    
    def __init__(self, config_manager: ConfigurationManager, **kwargs):
        super().__init__(config_manager, **kwargs)
        
        # Initialize knowledge graph components
        self.kg_manager = KnowledgeGraphManager(
            config_manager=config_manager,
            schema_manager=self.schema_manager,
            vector_store=self.vector_store
        )
        
        self.entity_extractor = EntityExtractionService(config_manager)
        self.relationship_extractor = RelationshipExtractionService(config_manager)
        self.graph_query_engine = GraphQueryEngine(self.kg_manager)
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Enhanced document loading with graph construction."""
        # Use base class for vector storage
        super().load_documents(documents_path, **kwargs)
        
        # Build knowledge graph
        documents = self._get_documents(documents_path, **kwargs)
        graph_result = self.kg_manager.build_graph(documents)
        
        logger.info(f"Built knowledge graph: {graph_result.entities_created} entities, "
                   f"{graph_result.relationships_created} relationships")
```

### 2. Schema Manager Integration

```python
# Extension to SchemaManager for knowledge graph tables
def _build_table_configurations(self):
    """Extended table configurations including knowledge graph tables."""
    # ... existing configurations ...
    
    # Knowledge graph table configurations
    self._table_configs.update({
        "Entities": {
            "embedding_column": "embedding",
            "content_column": "description",
            "id_column": "entity_id",
            "uses_document_embeddings": True,
            "default_model": self.base_embedding_model,
            "dimension": self.base_embedding_dimension,
            "supports_vector_search": True,
            "supports_globals_optimization": True
        },
        "Relationships": {
            "embedding_column": None,
            "uses_document_embeddings": False,
            "supports_graph_traversal": True,
            "supports_globals_optimization": True,
            "optimization_strategy": "iris_globals"
        }
    })
```

## Performance Optimization Strategy

### Phase 1: Basic SQL Performance (Immediate)
- **Proper Indexing**: Composite indexes for common query patterns
- **Query Optimization**: Efficient SQL for basic graph operations
- **Connection Pooling**: Reuse database connections
- **Basic Caching**: Cache frequently accessed entities and relationships

### Phase 2: IRIS Globals Optimization (Future)
- **Globals-based Traversal**: Direct pointer chasing for path queries
- **ObjectScript Integration**: High-performance graph algorithms
- **Adaptive Query Planning**: Choose optimal strategy based on query complexity
- **Advanced Caching**: Multi-level caching with globals support

### Performance Thresholds for Optimization
```python
class PerformanceThresholds:
    """Define when to trigger globals optimization."""
    
    # Trigger globals optimization when:
    MAX_SQL_TRAVERSAL_DEPTH = 3
    MAX_SQL_ENTITY_COUNT = 1000
    MAX_SQL_RELATIONSHIP_COUNT = 10000
    SQL_QUERY_TIMEOUT_MS = 5000
    
    @classmethod
    def should_use_globals_optimization(cls, query_params: Dict) -> bool:
        """Determine if globals optimization should be used."""
        return (
            query_params.get('max_depth', 0) > cls.MAX_SQL_TRAVERSAL_DEPTH or
            query_params.get('entity_count', 0) > cls.MAX_SQL_ENTITY_COUNT or
            query_params.get('relationship_count', 0) > cls.MAX_SQL_RELATIONSHIP_COUNT
        )
```

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Basic entity and relationship models
- [ ] Knowledge graph manager interface
- [ ] Simple entity extraction service
- [ ] Basic SQL-based graph operations
- [ ] Schema manager extensions

### Phase 2: Enhanced Extraction (Weeks 3-4)
- [ ] Improved entity extraction algorithms
- [ ] Relationship extraction service
- [ ] Basic entity linking
- [ ] Graph construction pipeline

### Phase 3: Query Engine & Optimization (Weeks 5-6)
- [ ] Graph query engine with SQL backend
- [ ] Performance monitoring and thresholds
- [ ] Basic caching implementation
- [ ] Subgraph extraction algorithms

### Phase 4: IRIS Globals Optimization (Future Enhancement)
- [ ] ObjectScript graph traversal functions
- [ ] Python-ObjectScript integration layer
- [ ] Adaptive query planning
- [ ] Advanced performance optimization

### Phase 5: Integration & Testing (Weeks 7-8)
- [ ] Enhanced GraphRAG pipeline
- [ ] Comprehensive testing suite
- [ ] Performance benchmarking
- [ ] Documentation and examples

## Success Metrics

### Functional Metrics
- **Entity Extraction Accuracy**: >80% precision/recall (basic mode)
- **Relationship Extraction Quality**: >70% accuracy (co-occurrence mode)
- **Graph Completeness**: >85% of relevant entities captured
- **Query Response Accuracy**: >85% relevant subgraph retrieval

### Performance Metrics (Phase 1 - SQL)
- **Graph Construction Time**: <15 minutes for 1000 documents
- **Query Response Time**: <5 seconds for basic graph queries
- **Memory Usage**: <2GB for 5K entities, 25K relationships
- **Scalability**: Handle up to 10K entities efficiently

### Performance Metrics (Phase 2 - Globals Optimization)
- **Complex Query Response Time**: <2 seconds for multi-hop queries
- **Deep Traversal Performance**: <1 second for 5-hop traversals
- **Large Graph Scalability**: Handle 100K+ entities efficiently
- **Memory Efficiency**: <8GB for 100K entities, 500K relationships

## Conclusion

This architecture provides a pragmatic approach to GraphRAG knowledge graph infrastructure:

1. **Start Simple**: Basic triple storage with SQL tables and indexes provides immediate functionality
2. **Measure Performance**: Monitor query performance and identify bottlenecks
3. **Optimize Strategically**: Leverage IRIS globals architecture when SQL performance becomes insufficient
4. **Maintain Compatibility**: All optimizations maintain API compatibility with existing components

The design ensures that we can deliver working GraphRAG functionality quickly while having a clear path to high-performance optimization using IRIS's unique capabilities for pointer chasing and direct global access.