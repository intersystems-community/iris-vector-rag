# Chunking Architecture Design

## Current State Analysis

The chunking system is well-designed with:
- ✅ [`DocumentChunkingService`](../tools/chunking/chunking_service.py) with multiple strategies
- ✅ Configuration-driven chunking (fixed_size, semantic, hybrid)
- ✅ Good defaults and extensible design
- ✅ Proper metadata handling and chunk relationships

## Problem: Incomplete Integration

**Current Issues:**
1. **BasicRAG**: Manually uses DocumentChunkingService (correct approach, but should be automatic)
2. **All Other Pipelines**: Don't use chunking at all
3. **IRISVectorStore**: Doesn't handle chunking automatically
4. **Inconsistent Interface**: Developers must know about chunking implementation

## Architectural Solution

### 1. Enhanced IRISVectorStore with Automatic Chunking

```python
class IRISVectorStore(VectorStore):
    def __init__(self, config_manager, schema_manager=None, connection_manager=None):
        # ... existing initialization ...
        
        # Initialize chunking service based on configuration
        self.chunking_config = self.config_manager.get("storage:chunking", {})
        self.auto_chunk = self.chunking_config.get("enabled", True)
        
        if self.auto_chunk:
            self.chunking_service = DocumentChunkingService(
                embedding_func=self._get_embedding_func()
            )
            self.default_strategy = self.chunking_config.get("strategy", "fixed_size")
            self.chunk_size_threshold = self.chunking_config.get("threshold", 1000)
    
    def add_documents(self, documents, embeddings=None, chunking_strategy=None, auto_chunk=None):
        """
        Add documents with automatic chunking support.
        
        Args:
            documents: List of Document objects
            embeddings: Optional pre-computed embeddings
            chunking_strategy: Override default chunking strategy
            auto_chunk: Override auto-chunking setting
        """
        # Determine if we should chunk
        should_chunk = auto_chunk if auto_chunk is not None else self.auto_chunk
        strategy = chunking_strategy or self.default_strategy
        
        if should_chunk and self.chunking_service:
            # Automatically chunk large documents
            processed_docs = []
            for doc in documents:
                if len(doc.page_content) > self.chunk_size_threshold:
                    # Chunk the document
                    chunks = self._chunk_document(doc, strategy)
                    processed_docs.extend(chunks)
                else:
                    # Keep small documents as-is
                    processed_docs.append(doc)
            
            # Generate embeddings for processed documents if needed
            if embeddings is None:
                embeddings = self._generate_embeddings(processed_docs)
            
            return self._store_documents(processed_docs, embeddings)
        else:
            # Store documents without chunking
            return self._store_documents(documents, embeddings)
```

### 2. Configuration-Driven Chunking

```yaml
# config/default.yaml
storage:
  chunking:
    enabled: true
    strategy: "fixed_size"  # fixed_size, semantic, hybrid
    threshold: 1000  # Chunk documents larger than this
    chunk_size: 512
    overlap: 50
    preserve_sentences: true
```

### 3. Pipeline Simplification

```python
# All pipelines become chunking-agnostic
class HyDERAGPipeline(RAGPipeline):
    def load_documents(self, documents_path, **kwargs):
        documents = self._load_documents_from_path(documents_path)
        
        # IRISVectorStore handles chunking automatically
        self.vector_store.add_documents(documents)
        
        # No manual chunking needed!
```

### 4. Service Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Layer                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   BasicRAG  │ │    HyDE     │ │    CRAG     │  ...      │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 IRISVectorStore Layer                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              add_documents()                        │   │
│  │  • Auto-detects large documents                     │   │
│  │  • Applies configured chunking strategy             │   │
│  │  • Generates embeddings                             │   │
│  │  • Stores with proper metadata                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                DocumentChunkingService                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ FixedSize   │ │  Semantic   │ │   Hybrid    │           │
│  │ Strategy    │ │  Strategy   │ │  Strategy   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Enhance IRISVectorStore
1. Add chunking configuration support
2. Integrate DocumentChunkingService
3. Implement automatic chunking logic
4. Add size threshold detection

### Phase 2: Update Pipeline Base Class
1. Remove manual chunking from BasicRAG
2. Update base RAGPipeline to use enhanced vector store
3. Ensure all pipelines inherit automatic chunking

### Phase 3: Configuration Integration
1. Add chunking configuration to default.yaml
2. Update ConfigurationManager to handle chunking settings
3. Add validation for chunking parameters

### Phase 4: Special Cases
1. **ColBERT**: Disable auto-chunking (uses token-level embeddings)
2. **SQL RAG**: Optional chunking (may work with full documents)
3. **Advanced Use Cases**: Allow manual override of chunking behavior

## Developer Experience

### Simple Case (90% of use cases)
```python
# Just works with good defaults
vector_store = IRISVectorStore(config_manager)
vector_store.add_documents(large_documents)  # Automatically chunked
```

### Advanced Case (10% of use cases)
```python
# Explicit control when needed
vector_store.add_documents(
    documents, 
    chunking_strategy="semantic",
    auto_chunk=True
)
```

### Configuration Override
```python
# Per-pipeline configuration
config = {
    "storage": {
        "chunking": {
            "strategy": "hybrid",
            "chunk_size": 1000
        }
    }
}
```

## Benefits

1. **Clean Separation of Concerns**: Chunking logic in storage layer
2. **Consistent Interface**: All pipelines get chunking automatically
3. **Configuration-Driven**: Easy to experiment with strategies
4. **Backward Compatible**: Existing code continues to work
5. **Developer Friendly**: Matches LlamaIndex/Haystack patterns
6. **Extensible**: Easy to add new chunking strategies

## Migration Path

1. **Phase 1**: Enhance IRISVectorStore (non-breaking)
2. **Phase 2**: Update BasicRAG to use automatic chunking
3. **Phase 3**: All other pipelines get chunking for free
4. **Phase 4**: Remove manual chunking code from BasicRAG

This approach leverages the existing well-designed chunking system and integrates it properly into the storage layer, creating a clean, developer-friendly interface.