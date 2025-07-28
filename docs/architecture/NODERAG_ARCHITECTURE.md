# NodeRAG Architecture Documentation

## Executive Summary

This document defines the comprehensive NodeRAG (Node-based Retrieval Augmented Generation) architecture that extends the existing GraphRAG knowledge graph infrastructure. NodeRAG provides hierarchical document representation and multi-level retrieval capabilities while maintaining full compatibility with the existing RAG templates architecture.

## 1. Architecture Overview

### 1.1 System Design Philosophy

NodeRAG extends the solid GraphRAG foundation with hierarchical document processing capabilities:

- **Hierarchical Structure**: Document → Section → Paragraph → Sentence hierarchy
- **GraphRAG Extension**: Builds upon existing [`Entity`](iris_rag/storage/knowledge_graph/models.py:143) and [`Relationship`](iris_rag/storage/knowledge_graph/models.py:198) models
- **Multi-Level Retrieval**: Context-aware retrieval at different granularity levels
- **IRIS Optimization**: Leverages IRIS's unique capabilities for hierarchical queries
- **Modular Design**: Clean separation of concerns with files under 500 lines

### 1.2 Core Architecture Components

```
NodeRAG Architecture (Extending GraphRAG)
┌─────────────────────────────────────────────────────────────────┐
│                    NodeRAG Hierarchical Layer                   │
├─────────────────────────────────────────────────────────────────┤
│  Enhanced NodeRAG  │  Hierarchical    │  Multi-Level Context   │
│     Pipeline       │   Retriever      │     Expansion          │
├─────────────────────────────────────────────────────────────────┤
│                    Hierarchical Services Layer                  │
├─────────────────────────────────────────────────────────────────┤
│ NodeHierarchy │ DocumentStructure │ HierarchicalRetriever │    │
│   Manager     │    Analyzer       │                       │    │
├─────────────────────────────────────────────────────────────────┤
│                    Hierarchical Data Layer                      │
├─────────────────────────────────────────────────────────────────┤
│ HierarchicalNode │ NodeContext │ HierarchicalSubGraph │        │
│ (extends Entity) │             │                       │        │
├─────────────────────────────────────────────────────────────────┤
│                    GraphRAG Foundation Layer                    │
├─────────────────────────────────────────────────────────────────┤
│ Entity/Relationship │ KnowledgeGraph │ Vector Store │ Schema   │
│     Models          │   Interfaces   │  Interface   │ Manager  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Architecture

### 2.1 Hierarchical Data Models

**Location**: [`iris_rag/storage/knowledge_graph/hierarchical_models.py`](iris_rag/storage/knowledge_graph/hierarchical_models.py)

#### Core Hierarchical Models

```python
# Base hierarchical node extending GraphRAG Entity
@dataclass
class HierarchicalNode(Entity):
    node_type: NodeType
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    depth_level: int = 0
    content: str = ""
    # Inherits: entity_id, entity_name, entity_type, etc. from Entity

# Specialized node types
class DocumentNode(HierarchicalNode)    # depth_level = 0
class SectionNode(HierarchicalNode)     # depth_level = 1  
class ParagraphNode(HierarchicalNode)   # depth_level = 2
class SentenceNode(HierarchicalNode)    # depth_level = 3
```

#### Hierarchical Relationships

```python
@dataclass
class HierarchicalRelationship(Relationship):
    relationship_type: HierarchicalRelationType
    depth_difference: int = 0
    path: List[str] = field(default_factory=list)
    # Inherits: relationship_id, source_entity_id, target_entity_id, etc.
```

#### Context and Subgraph Models

```python
@dataclass
class NodeContext:
    node: HierarchicalNode
    parent_nodes: List[HierarchicalNode]
    child_nodes: List[HierarchicalNode]
    sibling_nodes: List[HierarchicalNode]
    context_strategy: ContextStrategy
    relevance_score: float

@dataclass
class HierarchicalSubGraph:
    root_nodes: List[HierarchicalNode]
    all_nodes: List[HierarchicalNode]
    relationships: List[HierarchicalRelationship]
    node_contexts: Dict[str, NodeContext]
```

### 2.2 Hierarchical Interfaces

**Location**: [`iris_rag/storage/knowledge_graph/hierarchical_interfaces.py`](iris_rag/storage/knowledge_graph/hierarchical_interfaces.py)

#### Core Interfaces Extending GraphRAG

```python
class INodeHierarchyManager(IKnowledgeGraphManager):
    """Extends GraphRAG manager with hierarchical capabilities"""
    
    @abstractmethod
    def create_document_hierarchy(self, document: Document) -> DocumentNode
    
    @abstractmethod
    def get_node_children(self, node_id: str) -> List[HierarchicalNode]
    
    @abstractmethod
    def get_node_ancestors(self, node_id: str) -> List[HierarchicalNode]

class IHierarchicalRetriever(IGraphQueryEngine):
    """Extends GraphRAG query engine with hierarchical retrieval"""
    
    @abstractmethod
    def retrieve_with_context(self, query: str, node_ids: List[str],
                            context_strategy: ContextStrategy) -> List[HierarchicalNode]
    
    @abstractmethod
    def retrieve_multi_level(self, query: str,
                           level_weights: Dict[NodeType, float]) -> List[Tuple[HierarchicalNode, float]]

class IDocumentStructureAnalyzer(ABC):
    """New interface for document structure analysis"""
    
    @abstractmethod
    def analyze_document_structure(self, document: Document) -> DocumentStructure
    
    @abstractmethod
    def detect_sections(self, content: str) -> List[SectionInfo]
```

### 2.3 Database Schema Extensions

**Location**: [`iris_rag/storage/knowledge_graph/hierarchical_schema.py`](iris_rag/storage/knowledge_graph/hierarchical_schema.py)

#### Hierarchical Tables

```sql
-- Main hierarchical nodes table extending GraphRAG
CREATE TABLE RAG.HierarchicalNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    entity_id VARCHAR(255) REFERENCES RAG.Entities(entity_id),
    node_type VARCHAR(50) NOT NULL,
    parent_id VARCHAR(255) REFERENCES RAG.HierarchicalNodes(node_id),
    depth_level INTEGER NOT NULL DEFAULT 0,
    sibling_order INTEGER DEFAULT 0,
    content CLOB,
    node_metadata JSON,
    embeddings VECTOR(DOUBLE, 1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hierarchy optimization table for fast traversal
CREATE TABLE RAG.NodeHierarchy (
    ancestor_id VARCHAR(255) NOT NULL,
    descendant_id VARCHAR(255) NOT NULL,
    depth INTEGER NOT NULL,
    path VARCHAR(2000),  -- JSON array of node IDs
    PRIMARY KEY (ancestor_id, descendant_id)
);

-- Document structure metadata
CREATE TABLE RAG.DocumentStructure (
    document_id VARCHAR(255) PRIMARY KEY,
    root_node_id VARCHAR(255) NOT NULL,
    total_nodes INTEGER DEFAULT 0,
    max_depth INTEGER DEFAULT 0,
    node_type_counts JSON,
    structure_metadata JSON,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Performance Optimization

- **Indexes**: Optimized for parent-child traversal, depth queries, and type filtering
- **Views**: Pre-computed context views for common query patterns
- **Procedures**: IRIS ObjectScript procedures for complex hierarchical operations
- **Caching**: Node context cache table for performance optimization

## 3. Service Architecture

### 3.1 Service Boundaries and Responsibilities

#### NodeHierarchyManager Service
- **Responsibility**: Hierarchical node lifecycle management
- **Extends**: [`IKnowledgeGraphManager`](iris_rag/storage/knowledge_graph/interfaces.py:270)
- **Key Operations**:
  - Create document hierarchies from source documents
  - Manage parent-child relationships
  - Update hierarchy paths for optimization
  - Handle node lifecycle (create, update, delete)

#### DocumentStructureAnalyzer Service
- **Responsibility**: Automatic document structure detection
- **Key Operations**:
  - Analyze document content for hierarchical structure
  - Detect sections, headings, and content boundaries
  - Split content into paragraphs and sentences
  - Extract structural metadata

#### HierarchicalRetriever Service
- **Responsibility**: Multi-level context-aware retrieval
- **Extends**: [`IGraphQueryEngine`](iris_rag/storage/knowledge_graph/interfaces.py:125)
- **Key Operations**:
  - Vector search at multiple hierarchy levels
  - Context expansion using hierarchy relationships
  - Multi-level relevance scoring and aggregation
  - Intelligent subgraph extraction

#### HierarchicalStorage Service
- **Responsibility**: Optimized hierarchical data persistence
- **Extends**: [`IKnowledgeGraphStorage`](iris_rag/storage/knowledge_graph/interfaces.py:189)
- **Key Operations**:
  - Efficient bulk storage of hierarchical structures
  - Hierarchy path maintenance and optimization
  - Context caching for performance
  - IRIS-optimized query execution

### 3.2 Service Integration Patterns

```python
# Service composition pattern
class NodeRAGPipeline(RAGPipeline):
    def __init__(self, config_manager: ConfigurationManager):
        super().__init__(config_manager)
        
        # Initialize hierarchical services
        self.hierarchy_manager = NodeHierarchyManager(
            storage=self.hierarchical_storage,
            structure_analyzer=self.structure_analyzer,
            embedding_manager=self.embedding_manager
        )
        
        self.hierarchical_retriever = HierarchicalRetriever(
            storage=self.hierarchical_storage,
            vector_store=self.vector_store,
            hierarchy_manager=self.hierarchy_manager
        )
    
    def retrieve_documents(self, query: str, **kwargs) -> List[Document]:
        # Multi-level hierarchical retrieval
        return self.hierarchical_retriever.retrieve_with_context(
            query=query,
            context_strategy=ContextStrategy.SMART_EXPANSION,
            **kwargs
        )
```

## 4. Integration Architecture

### 4.1 GraphRAG Foundation Integration

#### Model Extension Strategy
- [`HierarchicalNode`](iris_rag/storage/knowledge_graph/hierarchical_models.py) extends [`Entity`](iris_rag/storage/knowledge_graph/models.py:143)
- [`HierarchicalRelationship`](iris_rag/storage/knowledge_graph/hierarchical_models.py) extends [`Relationship`](iris_rag/storage/knowledge_graph/models.py:198)
- Maintains full compatibility with existing GraphRAG entity/relationship system
- Leverages existing type registries and validation

#### Interface Extension Strategy
- Hierarchical interfaces extend existing GraphRAG interfaces
- Maintains Liskov Substitution Principle for backward compatibility
- Adds hierarchical-specific methods without breaking existing contracts

#### Storage Integration Strategy
- Hierarchical tables complement existing GraphRAG tables
- Foreign key relationships maintain referential integrity
- Shared use of [`SchemaManager`](iris_rag/storage/schema_manager.py) for consistent schema management

### 4.2 RAG Pipeline Integration

#### Enhanced NodeRAG Pipeline
- Extends existing [`NodeRAGPipeline`](iris_rag/pipelines/noderag.py:18)
- Maintains [`RAGPipeline`](iris_rag/core/base.py:6) interface compatibility
- Adds hierarchical retrieval while preserving existing functionality

#### Vector Store Integration
- Uses existing [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py) interface
- Hierarchical nodes stored with embeddings in vector store
- Multi-level similarity search capabilities

#### Configuration Integration
- Extends [`ConfigurationManager`](iris_rag/config/manager.py) with hierarchical settings
- Backward-compatible configuration schema
- Hierarchical-specific parameters with sensible defaults

### 4.3 Chunking Architecture Integration

#### Hierarchical Chunking Strategy
- Replaces fixed-size chunking with structure-aware chunking
- Creates chunks at multiple hierarchy levels simultaneously
- Maintains parent-child relationships between chunks
- Preserves document structure for enhanced context

## 5. API Architecture

### 5.1 Hierarchical Operations API

```python
# Node hierarchy operations
POST /api/noderag/documents/{doc_id}/hierarchy
{
    "max_depth": 3,
    "structure_strategy": "auto_detect",
    "embedding_strategy": "per_node"
}

GET /api/noderag/nodes/{node_id}/hierarchy
{
    "include_ancestors": true,
    "include_descendants": true,
    "max_depth": 2
}

# Context expansion operations
POST /api/noderag/nodes/expand-context
{
    "node_ids": ["node1", "node2"],
    "strategy": "smart_expansion",
    "max_nodes": 20
}
```

### 5.2 Multi-Level Retrieval API

```python
# Hierarchical search
POST /api/noderag/search/hierarchical
{
    "query": "search query",
    "node_types": ["paragraph", "section"],
    "context_strategy": "expand_up_down",
    "level_weights": {
        "document": 0.2,
        "section": 0.3,
        "paragraph": 0.4,
        "sentence": 0.1
    }
}

# Multi-level retrieval
POST /api/noderag/retrieve/multi-level
{
    "query": "search query",
    "top_k": 10,
    "similarity_threshold": 0.7,
    "context_expansion": true
}
```

## 6. Performance Architecture

### 6.1 IRIS Optimization Strategy

#### Hierarchical Query Optimization
- **NodeHierarchy Table**: Pre-computed ancestor-descendant relationships
- **Path Materialization**: Stored paths for fast traversal
- **Depth Indexing**: Optimized indexes for level-based queries
- **IRIS Globals**: Leverage IRIS's unique pointer-chasing capabilities

#### Caching Strategy
- **Node Context Cache**: Frequently accessed context patterns
- **Subgraph Cache**: Common hierarchical subgraphs
- **Query Result Cache**: Multi-level search results
- **Embedding Cache**: Node-level embeddings for fast similarity search

### 6.2 Scalability Architecture

#### Horizontal Scaling
- **Partition by Document**: Large documents can be processed independently
- **Parallel Hierarchy Creation**: Multiple documents processed concurrently
- **Distributed Context Expansion**: Context expansion across multiple nodes
- **Batch Processing**: Efficient bulk operations for large-scale ingestion

#### Vertical Scaling
- **Memory Optimization**: Efficient in-memory hierarchy representations
- **Query Optimization**: Optimized SQL queries for hierarchical operations
- **Index Strategy**: Strategic indexing for common access patterns
- **Connection Pooling**: Efficient database connection management

## 7. Testing Architecture

### 7.1 Testing Strategy

#### Unit Testing
- **Model Testing**: Hierarchical node creation and relationships
- **Interface Testing**: Service interface compliance and behavior
- **Schema Testing**: Database schema creation and migration
- **Algorithm Testing**: Context expansion and retrieval algorithms

#### Integration Testing
- **Pipeline Testing**: End-to-end hierarchical retrieval
- **Performance Testing**: Large-scale document processing (1000+ docs)
- **Compatibility Testing**: GraphRAG integration and backward compatibility
- **Real Data Testing**: PMC document corpus validation

#### Performance Testing
- **Scalability Testing**: Performance with deep hierarchies
- **Throughput Testing**: Concurrent hierarchical operations
- **Memory Testing**: Memory usage with large document structures
- **Query Performance**: Hierarchical query optimization validation

### 7.2 Test Data Strategy

#### Real PMC Data
- **Minimum 1000 Documents**: Meaningful performance testing
- **Hierarchical Complexity**: Documents with varying structure complexity
- **Domain Diversity**: Multiple biomedical domains and document types
- **Size Variation**: Small to large documents for scalability testing

## 8. Deployment Architecture

### 8.1 Environment Requirements

#### IRIS Database
- **Enterprise Edition**: Required for large-scale testing (>10GB data)
- **Vector Support**: IRIS vector capabilities for embeddings
- **JSON Support**: Native JSON support for metadata storage
- **Performance Tuning**: Optimized for hierarchical query patterns

#### Python Environment
- **UV Package Manager**: All Python commands use `uv run` prefix
- **Dependency Management**: Clean separation of hierarchical dependencies
- **Configuration Management**: Environment-specific hierarchical settings

### 8.2 Migration Strategy

#### Backward Compatibility
- **Existing Data**: No impact on existing GraphRAG data
- **API Compatibility**: Existing APIs continue to function
- **Configuration**: Hierarchical features opt-in by default
- **Performance**: No degradation of existing functionality

#### Gradual Rollout
- **Phase 1**: Core hierarchical infrastructure
- **Phase 2**: Enhanced retrieval capabilities
- **Phase 3**: Performance optimization and caching
- **Phase 4**: Advanced hierarchical features

## 9. Security Architecture

### 9.1 Data Security

#### Hierarchical Data Protection
- **Access Control**: Node-level access control inheritance
- **Data Encryption**: Hierarchical content encryption at rest
- **Audit Logging**: Hierarchical operation audit trails
- **Privacy Compliance**: Document structure privacy considerations

### 9.2 API Security

#### Authentication and Authorization
- **Role-Based Access**: Hierarchical operation permissions
- **API Rate Limiting**: Protection against hierarchical query abuse
- **Input Validation**: Hierarchical parameter validation
- **Output Sanitization**: Secure hierarchical data responses

## 10. Monitoring Architecture

### 10.1 Performance Monitoring

#### Hierarchical Metrics
- **Hierarchy Depth**: Average and maximum hierarchy depths
- **Context Expansion**: Context expansion performance metrics
- **Query Performance**: Multi-level query execution times
- **Cache Hit Rates**: Hierarchical cache effectiveness

#### System Health Monitoring
- **Database Performance**: Hierarchical table query performance
- **Memory Usage**: Hierarchical structure memory consumption
- **Error Rates**: Hierarchical operation error tracking
- **Throughput**: Hierarchical processing throughput metrics

## Conclusion

The NodeRAG architecture provides a comprehensive hierarchical document representation system that seamlessly extends the existing GraphRAG knowledge graph infrastructure. By building upon the solid GraphRAG foundation, NodeRAG delivers enhanced retrieval capabilities while maintaining full backward compatibility and leveraging IRIS's unique performance characteristics.

The modular design ensures clean separation of concerns, with each component under 500 lines and following established architectural patterns. The hierarchical approach enables context-aware retrieval at multiple granularity levels, significantly improving the quality and relevance of retrieved content for RAG applications.

This architecture is ready for implementation following the TDD methodology, with comprehensive testing strategies and performance optimization built into the design from the ground up.