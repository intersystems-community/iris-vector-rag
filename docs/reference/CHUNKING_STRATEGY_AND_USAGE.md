# Chunking Strategy and Usage Guide

## Overview

This document provides a comprehensive guide to document chunking strategies implemented in the RAG templates project. Chunking is a critical preprocessing step that breaks down large documents into smaller, semantically coherent segments to improve retrieval accuracy and generation quality in RAG systems.

## Table of Contents

1. [Introduction](#introduction)
2. [Current Implementation Architecture](#current-implementation-architecture)
3. [Chunking Strategies](#chunking-strategies)
4. [Configuration Options](#configuration-options)
5. [Integration with RAG Pipelines](#integration-with-rag-pipelines)
6. [Performance Considerations](#performance-considerations)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

### Why Chunking Matters

Document chunking significantly impacts RAG system performance across multiple dimensions:

- **Retrieval Quality**: Smaller, focused chunks often lead to more precise retrieval results
- **Context Relevance**: Well-segmented chunks provide better context for language model generation
- **Performance**: Optimized chunk sizes balance information density with processing efficiency
- **Memory Usage**: Smaller chunks reduce memory requirements during vector operations
- **Embedding Quality**: Chunks that respect semantic boundaries produce more meaningful embeddings

### Project Context

The RAG templates project implements multiple chunking approaches to handle diverse document types, particularly biomedical literature from PMC (PubMed Central). The system supports both simple and advanced chunking strategies depending on the specific RAG technique and use case requirements.

## Current Implementation Architecture

### Two-Tier Chunking System

The project implements a two-tier chunking architecture:

1. **Basic Chunking** ([`iris_rag/pipelines/basic.py`](iris_rag/pipelines/basic.py:182-200)) - Simple character-based splitting with overlap
2. **Enhanced Chunking** ([`tools/chunking/enhanced_chunking_service.py`](tools/chunking/enhanced_chunking_service.py)) - Advanced biomedical-optimized strategies

### Core Components

#### Basic Pipeline Chunking

The basic RAG pipeline implements simple text splitting:

```python
def _split_text(self, text: str) -> List[str]:
    """Split text into chunks with overlap."""
    if len(text) <= self.chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + self.chunk_size
        # Character-based splitting with overlap
        chunk = text[start:end]
        chunks.append(chunk)
        start += self.chunk_size - self.chunk_overlap
    
    return chunks
```

**Configuration**: Uses [`config.yaml`](config/config.yaml:15-17) settings:
- `chunk_size`: 1000 characters (default)
- `chunk_overlap`: 200 characters (default)

#### Enhanced Chunking Service

The enhanced service ([`tools/chunking/enhanced_chunking_service.py`](tools/chunking/enhanced_chunking_service.py)) provides sophisticated biomedical-optimized chunking with multiple strategies.

## Chunking Strategies

### 1. Fixed-Size Chunking (Basic)

**Implementation**: [`iris_rag/pipelines/basic.py`](iris_rag/pipelines/basic.py:150-180)

**How it works**: Splits text into fixed-size chunks with configurable overlap using character-based boundaries.

**Configuration**:
```yaml
# config/config.yaml
chunking:
  chunk_size: 1000      # Characters
  chunk_overlap: 200    # Characters

# Pipeline-specific overrides
pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
```

**When to use**:
- Simple documents with uniform structure
- Fast processing requirements
- When semantic boundaries are less critical

**Trade-offs**:
- ✅ Fast and predictable
- ✅ Simple configuration
- ❌ May break semantic boundaries
- ❌ No domain-specific optimization

### 2. Recursive Chunking (Enhanced)

**Implementation**: [`tools/chunking/enhanced_chunking_service.py`](tools/chunking/enhanced_chunking_service.py:359-450)

**How it works**: Hierarchically splits text using biomedical separator hierarchy, starting with major separators (section headers) and progressively using finer separators until target chunk sizes are achieved.

**Key Features**:
- Biomedical separator hierarchy
- Token-based size estimation
- Quality-driven processing levels

**Configuration**:
```python
strategy = RecursiveChunkingStrategy(
    chunk_size=512,           # Target tokens
    chunk_overlap=50,         # Token overlap
    quality=ChunkingQuality.BALANCED,
    model='default'
)
```

**Separator Hierarchy**:
```python
# High Quality (9 levels)
separators = [
    "\n\n## ",     # Section headers
    "\n\n### ",    # Subsection headers  
    "\n\n#### ",   # Sub-subsection headers
    "\n\n**",      # Bold text (important concepts)
    "\n\n",        # Paragraph breaks
    "\n",          # Line breaks
    ". ",          # Sentence endings
    "? ",          # Question endings
    "! ",          # Exclamation endings
]
```

**When to use**:
- Documents with clear hierarchical structure
- Scientific papers and reports
- When preserving document structure is important

### 3. Semantic Chunking (Enhanced)

**Implementation**: [`tools/chunking/enhanced_chunking_service.py`](tools/chunking/enhanced_chunking_service.py:512-680)

**How it works**: Groups sentences based on semantic coherence using biomedical semantic analysis. Creates chunk boundaries where coherence drops below a threshold.

**Key Features**:
- Biomedical semantic analysis
- Coherence-based boundary detection
- Adaptive chunk sizing

**Configuration**:
```python
strategy = SemanticChunkingStrategy(
    target_chunk_size=512,    # Preferred tokens
    min_chunk_size=100,       # Minimum tokens
    max_chunk_size=1024,      # Maximum tokens
    overlap_sentences=1,      # Sentence overlap
    quality=ChunkingQuality.HIGH_QUALITY
)
```

**When to use**:
- Complex scientific texts with varied structures
- When semantic coherence is prioritized over speed
- Documents with inconsistent formatting

### 4. Adaptive Chunking (Enhanced)

**Implementation**: [`tools/chunking/enhanced_chunking_service.py`](tools/chunking/enhanced_chunking_service.py:682-780)

**How it works**: Dynamically analyzes document characteristics and selects between recursive and semantic approaches based on content analysis.

**Document Analysis Factors**:
- Word and sentence count
- Biomedical content density
- Structural clarity
- Topic coherence

**Configuration**:
```python
strategy = AdaptiveChunkingStrategy(model='default')
# Automatically configures based on document analysis
```

**When to use**:
- Mixed document types in large-scale ingestion
- Production environments requiring consistent quality
- When optimal strategy is unknown beforehand

### 5. Hybrid Chunking (Enhanced)

**Implementation**: [`tools/chunking/enhanced_chunking_service.py`](tools/chunking/enhanced_chunking_service.py:825-900)

**How it works**: Combines recursive and semantic approaches by first using recursive chunking, then applying semantic analysis to refine boundaries.

**Configuration**:
```python
strategy = HybridChunkingStrategy(
    primary_chunk_size=512,      # Initial recursive target
    secondary_chunk_size=384,    # Semantic refinement target
    overlap=50,                  # Token overlap
    semantic_threshold=0.7       # Coherence threshold
)
```

**When to use**:
- High-quality chunking requirements
- Complex biomedical literature
- When both structure and semantics matter

## Configuration Options

### Global Configuration

**File**: [`config/config.yaml`](config/config.yaml)

```yaml
# Basic chunking parameters
chunking:
  chunk_size: 1000             # Characters for basic chunking
  chunk_overlap: 200           # Character overlap

# Pipeline-specific configurations
pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
  colbert:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
  crag:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
```

### Environment Variables

```bash
# Override chunking configuration
export CHUNK_SIZE=512
export CHUNK_OVERLAP=50
export CHUNKING_METHOD=fixed_size
```

### Enhanced Chunking Configuration

**Quality Levels**:
- `FAST`: 3 separator levels, minimal analysis
- `BALANCED`: 6 separator levels, moderate analysis  
- `HIGH_QUALITY`: 9 separator levels, comprehensive analysis

**Token Estimation Models**:
```python
TOKEN_RATIOS = {
    'gpt-4': 0.75,
    'gpt-3.5-turbo': 0.75,
    'claude': 0.8,
    'claude-3': 0.8,
    'text-embedding-ada-002': 0.75,
    'default': 0.75
}
```

## Integration with RAG Pipelines

### Current Usage Patterns

#### Basic RAG Pipeline

**File**: [`iris_rag/pipelines/basic.py`](iris_rag/pipelines/basic.py:150-180)

```python
def _chunk_documents(self, documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    chunked_documents = []
    
    for doc in documents:
        chunks = self._split_text(doc.page_content)
        
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "parent_document_id": doc.id,
                "chunk_size": len(chunk_text)
            })
            
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            chunked_documents.append(chunk_doc)
    
    return chunked_documents
```

#### Enhanced Chunking Integration

To use enhanced chunking in pipelines:

```python
from tools.chunking.enhanced_chunking_service import (
    EnhancedDocumentChunkingService,
    ChunkingQuality
)

# Initialize service
chunking_service = EnhancedDocumentChunkingService()

# Configure strategy
chunks = chunking_service.chunk_document(
    text=document.page_content,
    doc_id=document.id,
    strategy="adaptive",
    quality=ChunkingQuality.BALANCED
)
```

### Pipeline-Specific Considerations

#### ColBERT Pipeline
- Uses document-level embeddings primarily
- Chunking may be applied for token-level embeddings
- Configuration: [`config/config.yaml`](config/config.yaml:56-59)

#### CRAG Pipeline  
- Implements internal decomposition
- May benefit from pre-chunking for better retrieval
- Configuration: [`config/config.yaml`](config/config.yaml:60-63)

#### GraphRAG/NodeRAG
- Operates on knowledge graph nodes
- Chunking affects node granularity
- May use chunks as input for graph construction

## Performance Considerations

### Chunk Size Impact

**Small Chunks (256-512 tokens)**:
- ✅ More precise retrieval
- ✅ Better semantic coherence
- ❌ Higher storage overhead
- ❌ More embedding computations

**Medium Chunks (512-1024 tokens)**:
- ✅ Balanced performance/quality
- ✅ Good for most use cases
- ✅ Reasonable storage requirements

**Large Chunks (1024+ tokens)**:
- ✅ Lower storage overhead
- ✅ Fewer embeddings to compute
- ❌ May lose retrieval precision
- ❌ Risk of semantic drift

### Memory and Storage

**Estimation Formula**:
```
Total Chunks ≈ (Total Document Length) / (Chunk Size - Overlap)
Storage Requirements ≈ Total Chunks × (Embedding Dimension × 4 bytes + Metadata)
```

**Example for 1000 documents**:
- Average document: 5000 tokens
- Chunk size: 512 tokens, overlap: 50 tokens
- Estimated chunks: ~11,000
- Storage (384-dim embeddings): ~17MB vectors + metadata

### Processing Performance

**Basic Chunking**: ~1000 documents/second
**Enhanced Chunking**: 
- Recursive: ~500 documents/second
- Semantic: ~100 documents/second  
- Adaptive: ~200 documents/second
- Hybrid: ~50 documents/second

## Best Practices

### Choosing a Chunking Strategy

1. **For Production Systems**: Use adaptive chunking for mixed content
2. **For Speed**: Use basic fixed-size chunking
3. **For Quality**: Use semantic or hybrid chunking
4. **For Scientific Literature**: Use recursive with biomedical separators

### Configuration Guidelines

1. **Start with defaults**: 512 tokens, 50 token overlap
2. **Adjust based on document type**:
   - Short articles: 256-512 tokens
   - Long papers: 512-1024 tokens
   - Technical documents: Use semantic chunking
3. **Monitor retrieval quality**: Adjust chunk size if precision drops
4. **Consider embedding model**: Larger models can handle bigger chunks

### Optimization Tips

1. **Batch Processing**: Process documents in batches for better memory usage
2. **Quality vs Speed**: Use BALANCED quality for most use cases
3. **Overlap Strategy**: 10-20% overlap typically optimal
4. **Monitoring**: Track chunk size distribution and retrieval metrics

### Integration Patterns

```python
# Recommended pattern for new pipelines
class CustomRAGPipeline(RAGPipeline):
    def __init__(self, connection_manager, config_manager):
        super().__init__(connection_manager, config_manager)
        
        # Initialize chunking based on configuration
        chunking_method = config_manager.get("chunking:method", "basic")
        
        if chunking_method == "enhanced":
            from tools.chunking.enhanced_chunking_service import EnhancedDocumentChunkingService
            self.chunking_service = EnhancedDocumentChunkingService()
        else:
            # Use built-in basic chunking
            self.chunk_size = config_manager.get("chunking:chunk_size", 1000)
            self.chunk_overlap = config_manager.get("chunking:chunk_overlap", 200)
    
    def _chunk_documents(self, documents):
        if hasattr(self, 'chunking_service'):
            # Use enhanced chunking
            return self._enhanced_chunk_documents(documents)
        else:
            # Use basic chunking
            return self._basic_chunk_documents(documents)
```

## Troubleshooting

### Common Issues

#### 1. Chunks Too Large/Small

**Symptoms**: Poor retrieval quality, memory issues
**Solutions**:
- Adjust `chunk_size` parameter
- Check token estimation accuracy
- Consider different chunking strategy

#### 2. Poor Semantic Boundaries

**Symptoms**: Chunks break mid-sentence or mid-concept
**Solutions**:
- Use recursive or semantic chunking
- Increase quality level
- Adjust separator hierarchy

#### 3. Performance Issues

**Symptoms**: Slow chunking, high memory usage
**Solutions**:
- Use basic chunking for speed
- Reduce quality level
- Process in smaller batches
- Use FAST quality setting

#### 4. Inconsistent Chunk Sizes

**Symptoms**: Wide variation in chunk token counts
**Solutions**:
- Use adaptive chunking
- Adjust min/max chunk size parameters
- Check document preprocessing

### Debugging Tools

```python
# Analyze chunking results
def analyze_chunks(chunks):
    sizes = [chunk.metrics.token_count for chunk in chunks]
    print(f"Chunk count: {len(chunks)}")
    print(f"Average size: {sum(sizes)/len(sizes):.1f} tokens")
    print(f"Size range: {min(sizes)}-{max(sizes)} tokens")
    print(f"Size std dev: {statistics.stdev(sizes):.1f}")

# Test different strategies
def compare_strategies(text, doc_id):
    strategies = {
        'recursive': RecursiveChunkingStrategy(),
        'semantic': SemanticChunkingStrategy(),
        'adaptive': AdaptiveChunkingStrategy()
    }
    
    for name, strategy in strategies.items():
        chunks = strategy.chunk(text, doc_id)
        print(f"{name}: {len(chunks)} chunks")
        analyze_chunks(chunks)
```

### Performance Monitoring

```python
# Monitor chunking performance
import time

def monitor_chunking_performance(documents, strategy):
    start_time = time.time()
    total_chunks = 0
    
    for doc in documents:
        chunks = strategy.chunk(doc.page_content, doc.id)
        total_chunks += len(chunks)
    
    elapsed = time.time() - start_time
    print(f"Processed {len(documents)} documents")
    print(f"Generated {total_chunks} chunks")
    print(f"Time: {elapsed:.2f}s ({len(documents)/elapsed:.1f} docs/sec)")
```

## Future Considerations

### Planned Enhancements

1. **Dynamic Chunk Sizing**: Automatic optimization based on retrieval metrics
2. **Multi-Modal Chunking**: Support for documents with images and tables
3. **Domain-Specific Strategies**: Specialized chunking for different scientific domains
4. **Hierarchical Chunking**: Multi-level chunk relationships for better context

### Research Directions

1. **Embedding-Aware Chunking**: Optimize chunks based on embedding model characteristics
2. **Query-Aware Chunking**: Adapt chunking strategy based on expected query types
3. **Cross-Document Chunking**: Chunk boundaries that span related documents
4. **Real-Time Adaptation**: Dynamic strategy selection based on retrieval performance

---

## Related Documentation

- [Basic RAG Pipeline Guide](../guides/BASIC_RAG_PIPELINE.md)
- [Configuration Management](../reference/CONFIGURATION.md)
- [Performance Optimization](../guides/PERFORMANCE_OPTIMIZATION.md)
- [Vector Storage Guide](../reference/VECTOR_STORAGE.md)

## Contributing

When modifying chunking strategies:

1. Follow the existing interface patterns
2. Add comprehensive tests for new strategies
3. Update this documentation
4. Benchmark performance impact
5. Consider backward compatibility

For questions or contributions, see the [project contribution guidelines](../../CONTRIBUTING.md).