# RAG Memory Component Architecture

## Executive Summary

The RAG Memory Component Architecture provides generic, reusable memory management patterns for RAG applications. This architecture demonstrates how to add memory capabilities to RAG pipelines, enabling knowledge retention, pattern extraction, and incremental learning that any downstream application can adapt for their specific needs.

**Core Purpose**: Demonstrate extensible memory patterns for RAG techniques, not application-specific constructs.

---

## 1. Architecture Principles

### 1.1 Reusability-First Design
- **Generic Memory Patterns**: Adaptable to any RAG application domain
- **RAG-Technique Agnostic**: Works with BasicRAG, CRAG, GraphRAG, and future techniques
- **Pluggable Components**: Modular architecture for easy integration
- **Configuration-Driven**: Behavior controlled through external configuration

### 1.2 Demonstration Scope
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAG-TEMPLATES MEMORY COMPONENTS                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Knowledge       │  │ Temporal        │  │ Incremental     │            │
│  │ Pattern         │  │ Memory          │  │ Learning        │            │
│  │ Extraction      │  │ Storage         │  │ Patterns        │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                     INTEGRATION LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ RAG Bridge      │  │ CDC & Graph     │  │ Memory          │            │
│  │ Integration     │  │ Union (M2)      │  │ Persistence     │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                      FOUNDATION LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ IRIS Vector     │  │ RAG Pipeline    │  │ Configuration   │            │
│  │ Storage         │  │ Framework       │  │ Management      │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Generic Memory Components

### 2.1 Knowledge Pattern Extractor
**File**: [`iris_rag/memory/knowledge_extractor.py`](../iris_rag/memory/knowledge_extractor.py:1)

**Purpose**: Extract reusable knowledge patterns from any RAG technique response

```python
class KnowledgePatternExtractor:
    """Extract knowledge patterns from RAG responses for memory storage."""
    
    def extract_patterns(self, rag_response: RAGResponse) -> List[KnowledgePattern]:
        """Extract patterns that can be reused across queries"""
        
    def extract_entities(self, content: str) -> List[Entity]:
        """Extract entities using configurable NER approaches"""
        
    def extract_relationships(self, content: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
```

### 2.2 Temporal Memory Manager
**File**: [`iris_rag/memory/temporal_manager.py`](../iris_rag/memory/temporal_manager.py:1)

**Purpose**: Generic temporal storage patterns adaptable to any time-based memory needs

```python
class TemporalMemoryManager:
    """Generic temporal memory patterns for RAG applications."""
    
    def store_with_window(self, content: Any, window_config: TemporalWindow) -> str:
        """Store content with configurable temporal windows"""
        
    def retrieve_temporal_context(self, query: str, window: TemporalWindow) -> List[MemoryItem]:
        """Retrieve relevant memories within temporal context"""
        
    def cleanup_expired_memories(self, retention_policy: RetentionPolicy) -> int:
        """Clean up memories based on configurable retention policies"""
```

### 2.3 Incremental Learning Manager
**File**: [`iris_rag/memory/incremental_manager.py`](../iris_rag/memory/incremental_manager.py:1)

**Purpose**: Demonstrate incremental learning patterns using existing M2 infrastructure

```python
class IncrementalLearningManager:
    """Incremental learning patterns for RAG memory systems."""
    
    def process_knowledge_updates(self, changes: ChangeBatch) -> LearningResult:
        """Process incremental knowledge updates using CDC and graph union"""
        
    def merge_knowledge_graphs(self, existing: KnowledgeGraph, new: KnowledgeGraph) -> KnowledgeGraph:
        """Merge knowledge graphs using V̂ ∪ V̂' and Ê ∪ Ê' operations"""
        
    def update_embeddings_incrementally(self, changed_docs: List[Document]) -> EmbeddingResult:
        """Update vector embeddings incrementally without full recomputation"""
```

---

## 3. Integration Patterns

### 3.1 RAG Pipeline Memory Integration
**File**: [`iris_rag/memory/rag_integration.py`](../iris_rag/memory/rag_integration.py:1)

**Shows how to add memory to any RAG pipeline**:

```python
class MemoryEnabledRAGPipeline:
    """Base pattern for adding memory to any RAG technique."""
    
    def __init__(self, base_pipeline: RAGPipeline, memory_config: MemoryConfig):
        self.base_pipeline = base_pipeline
        self.memory_manager = self._create_memory_manager(memory_config)
    
    async def query_with_memory(self, query: str) -> EnrichedRAGResponse:
        """Execute RAG query with memory enrichment"""
        
        # 1. Check memory for relevant context
        memory_context = await self.memory_manager.get_relevant_context(query)
        
        # 2. Execute base RAG pipeline
        rag_response = await self.base_pipeline.query(query)
        
        # 3. Extract and store new knowledge patterns
        patterns = self.knowledge_extractor.extract_patterns(rag_response)
        await self.memory_manager.store_patterns(patterns)
        
        # 4. Return enriched response
        return EnrichedRAGResponse(
            base_response=rag_response,
            memory_context=memory_context,
            extracted_patterns=patterns
        )
```

### 3.2 Configuration-Driven Behavior
**File**: [`config/memory_config.yaml`](../config/memory_config.yaml:1)

```yaml
rag_memory_config:
  # Generic memory patterns
  temporal_windows:
    - name: "short_term"
      duration_days: 7
      cleanup_frequency: "daily"
    - name: "long_term" 
      duration_days: 90
      cleanup_frequency: "weekly"
  
  # Knowledge extraction patterns
  knowledge_extraction:
    entity_extraction:
      method: "spacy"  # configurable: spacy, nltk, custom
      confidence_threshold: 0.8
    relationship_extraction:
      method: "dependency_parsing"
      max_distance: 3
  
  # Integration patterns
  rag_integration:
    techniques:
      - basic_rag
      - crag
      - graphrag
    memory_enrichment: true
    pattern_storage: true
```

---

## 4. Incremental Learning Architecture

### 4.1 LightRAG-Inspired Patterns
**Building on existing M2 infrastructure**:

```python
class RAGIncrementalLearning:
    """Demonstrate incremental learning for RAG systems."""
    
    def __init__(self, cdc_detector: CDCDetector, graph_union: GraphUnionOperator):
        self.cdc_detector = cdc_detector  # From existing M2
        self.graph_union = graph_union    # From existing M2
        
    async def learn_incrementally(self, new_documents: List[Document]) -> LearningResult:
        """Demonstrate incremental learning workflow"""
        
        # 1. Detect changes using existing CDC
        changes = self.cdc_detector.detect_changes(new_documents)
        
        # 2. Extract new knowledge patterns
        new_patterns = await self._extract_knowledge_patterns(changes.new_documents)
        
        # 3. Merge with existing knowledge using graph union
        learning_result = await self._merge_knowledge(new_patterns)
        
        return learning_result
```

### 4.2 Vector Embedding Updates
```python
class IncrementalEmbeddingManager:
    """Demonstrate incremental embedding updates."""
    
    async def update_embeddings(self, changed_docs: List[Document]) -> EmbeddingUpdateResult:
        """Update embeddings without full recomputation"""
        
        # Only recompute embeddings for changed documents
        # Preserve existing embeddings for unchanged content
        # Demonstrate efficient vector store updates
```

---

## 5. Generic Data Models

### 5.1 Extensible Memory Models
```python
@dataclass
class GenericMemoryItem:
    """Generic memory item that applications can extend."""
    memory_id: str
    content: Dict[str, Any]
    memory_type: str  # "knowledge_pattern", "temporal_context", etc.
    confidence_score: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Applications can extend this base structure
    application_data: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class KnowledgePattern:
    """Generic knowledge pattern extraction."""
    pattern_id: str
    pattern_type: str  # "entity", "relationship", "concept"
    source_rag_technique: str
    extraction_confidence: float
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    context: str = ""
```

### 5.2 Configuration Models
```python
@dataclass
class MemoryConfig:
    """Configuration for memory behavior."""
    temporal_windows: List[TemporalWindow]
    knowledge_extraction: KnowledgeExtractionConfig
    persistence: PersistenceConfig
    performance: PerformanceConfig
    
    # Allow applications to add custom config
    custom_config: Dict[str, Any] = field(default_factory=dict)
```

---

## 6. Performance Architecture

### 6.1 Caching Patterns
```python
class MemoryCacheManager:
    """Generic caching patterns for RAG memory systems."""
    
    # L1: In-memory cache for hot patterns
    # L2: Vector similarity cache for embeddings  
    # L3: Database with optimized indexes
    
    async def get_cached_patterns(self, query_embedding: List[float]) -> List[KnowledgePattern]:
        """Multi-level cache lookup with vector similarity"""
```

### 6.2 Performance Targets
- **Knowledge Pattern Extraction**: <50ms per RAG response
- **Temporal Memory Retrieval**: <100ms for time-window queries
- **Incremental Learning**: <30s for 1K document updates
- **Memory Cache Hit Rate**: >80% for frequently accessed patterns

---

## 7. Integration Examples

### 7.1 BasicRAG with Memory
```python
# Example: Adding memory to BasicRAG
basic_rag = BasicRAGPipeline(vector_store, llm_func)
memory_config = MemoryConfig.from_yaml("config/memory_config.yaml")
memory_enabled_rag = MemoryEnabledRAGPipeline(basic_rag, memory_config)

response = await memory_enabled_rag.query_with_memory("What are the symptoms of diabetes?")
# Returns: EnrichedRAGResponse with memory context and extracted patterns
```

### 7.2 GraphRAG with Incremental Learning
```python
# Example: Adding incremental learning to GraphRAG  
graphrag = GraphRAGPipeline(vector_store, knowledge_graph, llm_func)
learning_manager = IncrementalLearningManager(cdc_detector, graph_union)

# Process new documents incrementally
new_docs = load_new_documents()
learning_result = await learning_manager.learn_incrementally(new_docs)
```

---

## 8. Extensibility Patterns

### 8.1 Plugin Architecture
```python
class MemoryPlugin(ABC):
    """Base class for memory plugins."""
    
    @abstractmethod
    def extract_knowledge(self, rag_response: RAGResponse) -> List[KnowledgePattern]:
        """Extract domain-specific knowledge patterns"""
        
    @abstractmethod  
    def enrich_query(self, query: str, memory_context: List[MemoryItem]) -> str:
        """Enrich query with memory context"""

# Applications can create domain-specific plugins
class BiomedicalMemoryPlugin(MemoryPlugin):
    """Biomedical domain knowledge extraction"""
    
class LegalMemoryPlugin(MemoryPlugin):
    """Legal domain knowledge extraction"""
```

### 8.2 Storage Adapters
```python
class MemoryStorageAdapter(ABC):
    """Pluggable storage for different backends."""
    
    @abstractmethod
    async def store_memory(self, item: GenericMemoryItem) -> str:
        """Store memory item"""
        
    @abstractmethod
    async def retrieve_memory(self, query: str, filters: Dict) -> List[GenericMemoryItem]:
        """Retrieve memory items"""

# Default IRIS implementation
class IRISMemoryAdapter(MemoryStorageAdapter):
    """IRIS-based memory storage"""

# Alternative implementations  
class PostgresMemoryAdapter(MemoryStorageAdapter):
    """PostgreSQL with pgvector"""
```

---

## 9. Success Criteria

### 9.1 Reusability
- ✅ **Zero application-specific assumptions** in memory components
- ✅ **Configuration-driven behavior** for different use cases
- ✅ **Pluggable architecture** for custom extensions
- ✅ **Clear integration patterns** for any RAG technique

### 9.2 Performance
- ✅ **Sub-100ms memory retrieval** for temporal queries
- ✅ **Efficient incremental learning** using existing M2 infrastructure
- ✅ **Scalable caching patterns** with configurable cache layers
- ✅ **Vector similarity optimization** for knowledge pattern matching

### 9.3 Integration
- ✅ **Works with all RAG techniques** in rag-templates
- ✅ **Leverages existing infrastructure** (M1 bridge, M2 incremental)
- ✅ **Minimal dependencies** and clean interfaces
- ✅ **Easy adoption** for downstream applications

---

## 10. Implementation Roadmap

### Phase 1: Core Memory Components (Days 1-2)
- [ ] **Knowledge Pattern Extractor**: Generic entity/relationship extraction
- [ ] **Temporal Memory Manager**: Time-window storage and retrieval
- [ ] **Base Memory Models**: Extensible data structures
- [ ] **Configuration System**: YAML-driven behavior

### Phase 2: RAG Integration (Days 3-4)  
- [ ] **Memory-Enabled Pipeline Base**: Generic RAG memory integration
- [ ] **BasicRAG + Memory**: Demonstration with BasicRAG
- [ ] **GraphRAG + Memory**: Demonstration with GraphRAG
- [ ] **Performance Optimization**: Caching and indexing

### Phase 3: Incremental Learning (Days 5-6)
- [ ] **Incremental Learning Manager**: Using M2 CDC and graph union
- [ ] **Vector Embedding Updates**: Efficient embedding management
- [ ] **Plugin Architecture**: Extensibility patterns
- [ ] **Documentation and Examples**: Usage patterns for applications

---

## 11. Conclusion

This RAG Memory Component Architecture provides generic, reusable memory patterns that demonstrate how to add memory capabilities to any RAG system. By focusing on configuration-driven, pluggable components rather than application-specific constructs, it serves as a foundation that downstream applications (like kg-ticket-resolver) can adapt for their specific needs.

The architecture leverages the existing rag-templates infrastructure (M1 RAG Bridge, M2 Incremental Indexing) while remaining completely generic and domain-agnostic. Each component stays under 500 lines and follows SPARC methodology for maximum reusability.