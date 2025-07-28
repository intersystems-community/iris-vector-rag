# NodeRAG Implementation Plan

## Overview

NodeRAG (Node-based Retrieval Augmented Generation) extends the existing GraphRAG knowledge graph infrastructure to provide hierarchical node-based document representation and retrieval. This implementation builds upon the solid GraphRAG foundation while adding document-level and chunk-level node relationships for enhanced contextual retrieval.

## 1. Architecture Overview

### 1.1 System Design Principles

- **Hierarchical Node Structure**: Document → Section → Paragraph → Sentence hierarchy
- **GraphRAG Foundation**: Leverages existing knowledge graph infrastructure
- **Multi-Level Retrieval**: Support for retrieval at different granularity levels
- **Context Expansion**: Parent-child traversal for enhanced context
- **Modular Design**: Clean separation of concerns with <500 line files

### 1.2 Core Components

```
NodeRAG Architecture (Extending GraphRAG)
├── Hierarchical Node Models (extends GraphRAG Entity/Relationship)
│   ├── HierarchicalNode (base class extending Entity)
│   ├── DocumentNode (top-level document representation)
│   ├── SectionNode (document sections/chapters)
│   ├── ParagraphNode (paragraph-level content)
│   └── SentenceNode (sentence-level granularity)
├── Hierarchical Interfaces (extends GraphRAG interfaces)
│   ├── INodeHierarchyManager (extends IKnowledgeGraphManager)
│   ├── IDocumentStructureAnalyzer (new interface)
│   └── IHierarchicalRetriever (extends IGraphQueryEngine)
├── Service Layer
│   ├── NodeHierarchyManager (implements INodeHierarchyManager)
│   ├── DocumentStructureAnalyzer (implements IDocumentStructureAnalyzer)
│   └── HierarchicalRetriever (implements IHierarchicalRetriever)
└── Enhanced Pipeline
    └── NodeRAGPipeline (enhanced existing implementation)
```

### 1.3 Integration with Existing GraphRAG Infrastructure

NodeRAG extends the existing GraphRAG components:

- **Models**: [`HierarchicalNode`](iris_rag/storage/knowledge_graph/hierarchical_models.py) extends [`Entity`](iris_rag/storage/knowledge_graph/models.py:143)
- **Interfaces**: New hierarchical interfaces extend existing [`IKnowledgeGraphManager`](iris_rag/storage/knowledge_graph/interfaces.py:270)
- **Storage**: Reuses [`IKnowledgeGraphStorage`](iris_rag/storage/knowledge_graph/interfaces.py:189) with hierarchical extensions
- **Pipeline**: Enhances existing [`NodeRAGPipeline`](iris_rag/pipelines/noderag.py:18) with hierarchical capabilities

## 2. Data Model Extensions

### 2.1 Hierarchical Node Models

Building upon existing GraphRAG [`Entity`](iris_rag/storage/knowledge_graph/models.py:143) and [`Relationship`](iris_rag/storage/knowledge_graph/models.py:198) models:

```python
# Extends existing GraphRAG Entity model
@dataclass
class HierarchicalNode(Entity):
    """Base class for hierarchical document nodes extending GraphRAG Entity"""
    node_type: NodeType
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    sibling_order: int = 0
    depth_level: int = 0
    content: str = ""
    node_metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    
    # Inherited from Entity: entity_id, entity_name, entity_type, etc.

@dataclass 
class DocumentNode(HierarchicalNode):
    """Top-level document representation"""
    document_id: str
    title: str
    source_path: str
    total_sections: int = 0
    document_type: str = "document"
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.DOCUMENT
        self.depth_level = 0

@dataclass
class SectionNode(HierarchicalNode):
    """Document section (chapters, headings)"""
    section_title: str
    section_type: str = "section"  # heading, chapter, abstract, etc.
    heading_level: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.SECTION
        self.depth_level = 1

@dataclass
class ParagraphNode(HierarchicalNode):
    """Paragraph-level content"""
    paragraph_index: int
    sentence_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.PARAGRAPH
        self.depth_level = 2

@dataclass
class SentenceNode(HierarchicalNode):
    """Sentence-level granularity"""
    sentence_index: int
    word_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.SENTENCE
        self.depth_level = 3

enum NodeType:
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
```

### 2.2 Hierarchical Relationship Extensions

```python
@dataclass
class HierarchicalRelationship(Relationship):
    """Extends GraphRAG relationships for hierarchical structure"""
    relationship_type: HierarchicalRelationType
    depth_difference: int = 0
    path: List[str] = field(default_factory=list)  # Full path from root to target
    
    # Inherited from Relationship: relationship_id, source_entity_id, target_entity_id, etc.

enum HierarchicalRelationType:
    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    ANCESTOR_DESCENDANT = "ancestor_descendant"
    CONTAINS = "contains"
    PART_OF = "part_of"
```

## 3. Database Schema Extensions

### 3.1 Hierarchical Node Storage Tables

Extending existing GraphRAG schema while maintaining compatibility:

```sql
-- Extends existing KnowledgeGraphNodes table structure
CREATE TABLE HierarchicalNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    entity_id VARCHAR(255) REFERENCES Entities(entity_id),
    node_type VARCHAR(50) NOT NULL,
    parent_id VARCHAR(255) REFERENCES HierarchicalNodes(node_id),
    depth_level INTEGER NOT NULL DEFAULT 0,
    sibling_order INTEGER DEFAULT 0,
    content CLOB,
    node_metadata JSON,
    embeddings VECTOR(DOUBLE, 1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_hierarchical_parent (parent_id),
    INDEX idx_hierarchical_depth (depth_level),
    INDEX idx_hierarchical_type (node_type),
    INDEX idx_hierarchical_embedding (embeddings)
);

-- Hierarchical path optimization table for fast ancestor/descendant queries
CREATE TABLE NodeHierarchy (
    ancestor_id VARCHAR(255) NOT NULL,
    descendant_id VARCHAR(255) NOT NULL,
    depth INTEGER NOT NULL,
    path VARCHAR(2000),  -- JSON array of node IDs in path
    PRIMARY KEY (ancestor_id, descendant_id),
    
    FOREIGN KEY (ancestor_id) REFERENCES HierarchicalNodes(node_id),
    FOREIGN KEY (descendant_id) REFERENCES HierarchicalNodes(node_id),
    
    INDEX idx_hierarchy_ancestor (ancestor_id),
    INDEX idx_hierarchy_descendant (descendant_id),
    INDEX idx_hierarchy_depth (depth)
);

-- Document structure metadata table
CREATE TABLE DocumentStructure (
    document_id VARCHAR(255) PRIMARY KEY,
    root_node_id VARCHAR(255) NOT NULL,
    total_nodes INTEGER DEFAULT 0,
    max_depth INTEGER DEFAULT 0,
    structure_metadata JSON,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (root_node_id) REFERENCES HierarchicalNodes(node_id),
    INDEX idx_doc_structure_root (root_node_id)
);
```

### 3.2 Schema Integration Strategy

- **Backward Compatibility**: Existing [`KnowledgeGraphNodes`](iris_rag/storage/knowledge_graph/) table remains unchanged
- **Gradual Migration**: [`HierarchicalNodes`](iris_rag/storage/knowledge_graph/hierarchical_models.py) table extends functionality
- **Unified Access**: [`SchemaManager`](iris_rag/storage/schema_manager.py) handles both table types
- **Performance Optimization**: [`NodeHierarchy`](iris_rag/storage/knowledge_graph/hierarchical_models.py) table enables fast traversal queries

## 4. Interface Extensions

### 4.1 Hierarchical Node Management Interface

```python
class INodeHierarchyManager(IKnowledgeGraphManager):
    """Extends GraphRAG manager with hierarchical node capabilities"""
    
    @abstractmethod
    def create_document_hierarchy(self, document: Document) -> DocumentNode:
        """Create hierarchical node structure from document"""
        pass
    
    @abstractmethod
    def get_node_children(self, node_id: str) -> List[HierarchicalNode]:
        """Get direct children of a node"""
        pass
    
    @abstractmethod
    def get_node_parent(self, node_id: str) -> Optional[HierarchicalNode]:
        """Get parent of a node"""
        pass
    
    @abstractmethod
    def get_node_ancestors(self, node_id: str) -> List[HierarchicalNode]:
        """Get all ancestors of a node"""
        pass
    
    @abstractmethod
    def get_node_descendants(self, node_id: str, max_depth: int = None) -> List[HierarchicalNode]:
        """Get all descendants of a node"""
        pass
    
    @abstractmethod
    def get_node_siblings(self, node_id: str) -> List[HierarchicalNode]:
        """Get sibling nodes at the same level"""
        pass
```

### 4.2 Document Structure Analysis Interface

```python
class IDocumentStructureAnalyzer(ABC):
    """Interface for analyzing and creating document hierarchical structure"""
    
    @abstractmethod
    def analyze_document_structure(self, document: Document) -> DocumentStructure:
        """Analyze document and identify hierarchical structure"""
        pass
    
    @abstractmethod
    def create_hierarchical_nodes(self, document: Document, 
                                structure: DocumentStructure) -> List[HierarchicalNode]:
        """Create hierarchical nodes from document structure"""
        pass
    
    @abstractmethod
    def detect_sections(self, content: str) -> List[SectionInfo]:
        """Detect sections in document content"""
        pass
    
    @abstractmethod
    def split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs"""
        pass
    
    @abstractmethod
    def split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences"""
        pass
```

### 4.3 Hierarchical Retrieval Interface

```python
class IHierarchicalRetriever(IGraphQueryEngine):
    """Extends GraphRAG query engine with hierarchical retrieval"""
    
    @abstractmethod
    def retrieve_with_context(self, query: str, node_ids: List[str], 
                            context_strategy: ContextStrategy) -> List[HierarchicalNode]:
        """Retrieve nodes with hierarchical context expansion"""
        pass
    
    @abstractmethod
    def expand_context_up(self, node_ids: List[str], levels: int = 1) -> List[HierarchicalNode]:
        """Expand context by traversing up the hierarchy"""
        pass
    
    @abstractmethod
    def expand_context_down(self, node_ids: List[str], levels: int = 1) -> List[HierarchicalNode]:
        """Expand context by traversing down the hierarchy"""
        pass
    
    @abstractmethod
    def retrieve_at_level(self, query: str, node_type: NodeType, 
                         top_k: int = 10) -> List[HierarchicalNode]:
        """Retrieve nodes at specific hierarchy level"""
        pass
```

## 5. Service Architecture

### 5.1 NodeHierarchyManager Service

```python
class NodeHierarchyManager(INodeHierarchyManager):
    """Implementation of hierarchical node management"""
    
    def __init__(self, storage: IKnowledgeGraphStorage, 
                 structure_analyzer: IDocumentStructureAnalyzer,
                 embedding_manager: EmbeddingManager):
        self.storage = storage
        self.structure_analyzer = structure_analyzer
        self.embedding_manager = embedding_manager
    
    def create_document_hierarchy(self, document: Document) -> DocumentNode:
        """Create complete hierarchical structure from document"""
        # 1. Analyze document structure
        # 2. Create hierarchical nodes
        # 3. Generate embeddings for each node
        # 4. Store nodes and relationships
        # 5. Update hierarchy optimization table
```

### 5.2 DocumentStructureAnalyzer Service

```python
class DocumentStructureAnalyzer(IDocumentStructureAnalyzer):
    """Implementation of document structure analysis"""
    
    def analyze_document_structure(self, document: Document) -> DocumentStructure:
        """Analyze document and create hierarchical structure"""
        # 1. Detect document sections using heading patterns
        # 2. Split sections into paragraphs
        # 3. Split paragraphs into sentences
        # 4. Create hierarchical relationships
        # 5. Generate structure metadata
```

### 5.3 HierarchicalRetriever Service

```python
class HierarchicalRetriever(IHierarchicalRetriever):
    """Implementation of hierarchical retrieval strategies"""
    
    def retrieve_with_context(self, query: str, node_ids: List[str], 
                            context_strategy: ContextStrategy) -> List[HierarchicalNode]:
        """Retrieve nodes with intelligent context expansion"""
        # 1. Initial vector search for relevant nodes
        # 2. Apply context expansion strategy
        # 3. Traverse hierarchy based on strategy
        # 4. Aggregate and rank results
        # 5. Return contextualized node set
```

## 6. Enhanced Pipeline Integration

### 6.1 NodeRAG Pipeline Enhancements

The existing [`NodeRAGPipeline`](iris_rag/pipelines/noderag.py:18) will be enhanced with hierarchical capabilities:

```python
class NodeRAGPipeline(RAGPipeline):
    """Enhanced NodeRAG pipeline with hierarchical node support"""
    
    def __init__(self, config_manager: ConfigurationManager,
                 hierarchy_manager: INodeHierarchyManager,
                 hierarchical_retriever: IHierarchicalRetriever,
                 **kwargs):
        super().__init__(config_manager, **kwargs)
        self.hierarchy_manager = hierarchy_manager
        self.hierarchical_retriever = hierarchical_retriever
    
    def retrieve_documents(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """Enhanced retrieval with hierarchical context"""
        # 1. Multi-level vector search
        # 2. Hierarchical context expansion
        # 3. Cross-level relevance scoring
        # 4. Intelligent content aggregation
```

## 7. API Specifications

### 7.1 Node Operations API

```python
# Create hierarchical structure
POST /api/noderag/documents/{doc_id}/hierarchy
{
    "structure_strategy": "auto_detect",
    "max_depth": 3,
    "embedding_strategy": "per_node"
}

# Get node hierarchy
GET /api/noderag/nodes/{node_id}/hierarchy
{
    "include_ancestors": true,
    "include_descendants": true,
    "max_depth": 2
}

# Hierarchical search
POST /api/noderag/search/hierarchical
{
    "query": "search query",
    "node_types": ["paragraph", "section"],
    "context_strategy": "expand_up_down",
    "context_levels": 1
}
```

### 7.2 Context Expansion API

```python
# Expand context around nodes
POST /api/noderag/nodes/expand-context
{
    "node_ids": ["node1", "node2"],
    "strategy": "smart_expansion",
    "max_nodes": 20,
    "relevance_threshold": 0.7
}

# Multi-level retrieval
POST /api/noderag/retrieve/multi-level
{
    "query": "search query",
    "levels": [
        {"type": "document", "weight": 0.3},
        {"type": "section", "weight": 0.4},
        {"type": "paragraph", "weight": 0.3}
    ]
}
```

## 8. Integration Points

### 8.1 With Existing GraphRAG Infrastructure

- **Entity Integration**: [`HierarchicalNode`](iris_rag/storage/knowledge_graph/hierarchical_models.py) extends [`Entity`](iris_rag/storage/knowledge_graph/models.py:143)
- **Relationship Integration**: [`HierarchicalRelationship`](iris_rag/storage/knowledge_graph/hierarchical_models.py) extends [`Relationship`](iris_rag/storage/knowledge_graph/models.py:198)
- **Storage Integration**: Reuses [`IKnowledgeGraphStorage`](iris_rag/storage/knowledge_graph/interfaces.py:189) interface
- **Query Integration**: Extends [`IGraphQueryEngine`](iris_rag/storage/knowledge_graph/interfaces.py:125) capabilities

### 8.2 With Existing RAG Architecture

- **Pipeline Integration**: Enhances [`RAGPipeline`](iris_rag/core/base.py:6) base class
- **Vector Store Integration**: Uses [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py) for embeddings
- **Configuration Integration**: Extends [`ConfigurationManager`](iris_rag/config/manager.py) system
- **Schema Integration**: Uses [`SchemaManager`](iris_rag/storage/schema_manager.py) for table management

### 8.3 With Chunking Architecture

- **Hierarchical Chunking**: Replaces fixed-size chunking with structure-aware chunking
- **Multi-Level Chunks**: Creates chunks at different hierarchy levels
- **Context-Aware Splitting**: Uses document structure for intelligent splitting
- **Chunk Relationships**: Maintains parent-child relationships between chunks

## 9. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create hierarchical data models extending GraphRAG
- [ ] Design and implement hierarchical interfaces
- [ ] Extend database schema with hierarchical tables
- [ ] Implement basic NodeHierarchyManager

### Phase 2: Core Services (Weeks 3-4)
- [ ] Implement DocumentStructureAnalyzer
- [ ] Implement HierarchicalRetriever
- [ ] Create hierarchical storage layer
- [ ] Develop context expansion algorithms

### Phase 3: Pipeline Integration (Weeks 5-6)
- [ ] Enhance NodeRAGPipeline with hierarchical capabilities
- [ ] Implement multi-level retrieval strategies
- [ ] Create hierarchical answer synthesis
- [ ] Develop performance optimization

### Phase 4: Testing & Optimization (Weeks 7-8)
- [ ] Comprehensive unit testing
- [ ] Integration testing with real PMC data
- [ ] Performance benchmarking
- [ ] Documentation and examples

## 10. Testing Strategy

### 10.1 Unit Testing
- Test hierarchical node creation and relationships
- Test document structure analysis algorithms
- Test context expansion strategies
- Test multi-level retrieval logic

### 10.2 Integration Testing
- Test with real PMC document corpus (1000+ documents)
- Test hierarchical pipeline end-to-end
- Test performance with large document hierarchies
- Test backward compatibility with existing GraphRAG

### 10.3 Performance Testing
- Benchmark hierarchical vs. flat retrieval
- Test scalability with deep document hierarchies
- Measure context expansion performance
- Compare against published NodeRAG benchmarks

## 11. Success Criteria

- [ ] Complete hierarchical node architecture extending GraphRAG
- [ ] Seamless integration with existing knowledge graph infrastructure
- [ ] Scalable hierarchical storage and retrieval design
- [ ] Comprehensive API for hierarchical operations
- [ ] Performance improvements over flat document retrieval
- [ ] All files under 500 lines following SPARC methodology
- [ ] Full test coverage with real PMC data
- [ ] Documentation and implementation guides

## Conclusion

This NodeRAG implementation plan provides a comprehensive architecture that extends the existing GraphRAG knowledge graph foundation with hierarchical document representation capabilities. The design maintains full compatibility with existing infrastructure while adding powerful new capabilities for context-aware retrieval and multi-level document understanding.

The modular architecture ensures clean separation of concerns, with each component under 500 lines and following established patterns. The implementation leverages IRIS's unique capabilities for high-performance hierarchical queries while maintaining the flexibility to work with various document types and domains.