# Chunking Implementation Specification

## Overview

This document specifies the implementation requirements for automatic chunking in the RAG templates project. The goal is to move chunking logic into the [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py) layer to provide a unified developer experience.

## Architecture Goals

1. **Automatic Chunking**: [`IRISVectorStore.add_documents()`](iris_rag/storage/vector_store_iris.py) handles chunking automatically based on configuration
2. **Pipeline Simplification**: RAG pipelines become chunking-agnostic - they just call [`vector_store.add_documents()`](iris_rag/storage/vector_store_iris.py)
3. **Configuration-Driven**: Chunking strategy controlled via [`config/default.yaml`](config/default.yaml)
4. **Unified Interface**: Developer experience matches LlamaIndex/Haystack patterns

## Current State Analysis

### Working Components
- [`tools/chunking/chunking_service.py`](tools/chunking/chunking_service.py): Well-designed chunking service with multiple strategies
- [`tests/conftest_1000docs.py`](tests/conftest_1000docs.py): Enterprise-scale testing fixtures for 1000+ documents
- [`iris_rag/pipelines/basic.py`](iris_rag/pipelines/basic.py): Only pipeline using [`DocumentChunkingService`](tools/chunking/chunking_service.py) (lines 55-57, 171-177)

### Missing Integration
- [`IRISVectorStore.add_documents()`](iris_rag/storage/vector_store_iris.py) doesn't handle chunking automatically
- 7 out of 8 pipelines lack chunking entirely
- No chunking configuration in [`config/default.yaml`](config/default.yaml)

## Implementation Requirements

### 1. IRISVectorStore Enhancement

**File**: [`iris_rag/storage/vector_store_iris.py`](iris_rag/storage/vector_store_iris.py)

**Required Changes**:
```python
class IRISVectorStore:
    def __init__(self, config_manager, schema_manager=None):
        # Existing initialization...
        
        # NEW: Chunking configuration
        self.chunking_config = config_manager.get("storage:chunking", {})
        self.auto_chunk = self.chunking_config.get("enabled", False)
        
        # NEW: Initialize chunking service if enabled
        if self.auto_chunk:
            from tools.chunking.chunking_service import DocumentChunkingService
            self.chunking_service = DocumentChunkingService(
                strategy=self.chunking_config.get("strategy", "fixed_size"),
                **self.chunking_config.get(self.chunking_config.get("strategy", "fixed_size"), {})
            )
    
    def add_documents(self, documents: List[Document], 
                     auto_chunk: bool = None, 
                     chunking_strategy: str = None) -> List[str]:
        """Add documents with automatic chunking."""
        
        # Allow per-call override of auto-chunking
        should_chunk = auto_chunk if auto_chunk is not None else self.auto_chunk
        
        if should_chunk:
            chunked_documents = []
            strategy = chunking_strategy or self.chunking_config.get("strategy", "fixed_size")
            
            for doc in documents:
                if self._should_chunk_document(doc):
                    chunks = self._chunk_document(doc, strategy)
                    chunked_documents.extend(chunks)
                else:
                    chunked_documents.append(doc)
            
            documents = chunked_documents
        
        # Continue with existing add_documents logic...
        return self._store_documents(documents, embeddings)
    
    def _should_chunk_document(self, document: Document) -> bool:
        """Determine if document should be chunked based on size threshold."""
        threshold = self.chunking_config.get("threshold", 1000)
        return len(document.page_content) > threshold
    
    def _chunk_document(self, document: Document, strategy: str) -> List[Document]:
        """Chunk a single document using specified strategy."""
        return self.chunking_service.chunk_document(document, strategy)
```

### 2. Configuration Integration

**File**: [`config/default.yaml`](config/default.yaml)

**Required Addition**:
```yaml
storage:
  iris:
    # Existing IRIS settings...
    
  # NEW: Automatic chunking configuration
  chunking:
    enabled: true
    strategy: fixed_size
    threshold: 1000
    
    fixed_size:
      chunk_size: 512
      overlap: 50
      
    semantic:
      min_chunk_size: 200
      max_chunk_size: 800
      coherence_threshold: 0.7
      
    hybrid:
      base_chunk_size: 512
      semantic_adjustment: true
      max_deviation: 200

# Pipeline-specific overrides
pipeline_overrides:
  colbert:
    chunking:
      enabled: false  # ColBERT uses token-level embeddings
      
  sql_rag:
    chunking:
      enabled: false  # SQL RAG may not need chunking
```

### 3. Pipeline Refactoring

**Files to Update**:
- [`iris_rag/pipelines/basic.py`](iris_rag/pipelines/basic.py): Remove manual chunking (lines 55-57, 171-177)
- [`iris_rag/pipelines/hyde.py`](iris_rag/pipelines/hyde.py): Add chunking via vector store
- [`iris_rag/pipelines/crag.py`](iris_rag/pipelines/crag.py): Add chunking via vector store
- [`iris_rag/pipelines/graphrag.py`](iris_rag/pipelines/graphrag.py): Add chunking via vector store
- [`iris_rag/pipelines/hybrid_ifind.py`](iris_rag/pipelines/hybrid_ifind.py): Add chunking via vector store
- [`iris_rag/pipelines/noderag.py`](iris_rag/pipelines/noderag.py): Add chunking via vector store
- [`iris_rag/pipelines/colbert/pipeline.py`](iris_rag/pipelines/colbert/pipeline.py): Disable auto-chunking
- [`iris_rag/pipelines/sql_rag.py`](iris_rag/pipelines/sql_rag.py): Configure appropriately

**Pattern for Pipeline Updates**:
```python
# BEFORE (BasicRAG example):
def load_documents(self, data_path: str, documents: List[Document] = None):
    # Manual chunking
    chunking_service = DocumentChunkingService(strategy="fixed_size")
    chunked_documents = []
    for doc in documents:
        chunks = chunking_service.chunk_document(doc, "fixed_size")
        chunked_documents.extend(chunks)
    
    # Store chunks
    self.vector_store.add_documents(chunked_documents)

# AFTER (all pipelines):
def load_documents(self, data_path: str, documents: List[Document] = None):
    # Automatic chunking handled by vector store
    self.vector_store.add_documents(documents)
```

### 4. Special Cases

**ColBERT Pipeline**:
```python
# ColBERT uses token-level embeddings - disable chunking
def load_documents(self, data_path: str, documents: List[Document] = None):
    self.vector_store.add_documents(documents, auto_chunk=False)
```

**GraphRAG Pipeline**:
```python
# GraphRAG might benefit from semantic chunking for entity extraction
def load_documents(self, data_path: str, documents: List[Document] = None):
    self.vector_store.add_documents(documents, chunking_strategy="semantic")
```

## Testing Strategy

### TDD Implementation Order

1. **RED Phase**: Run [`tests/test_chunking_architecture_integration.py`](tests/test_chunking_architecture_integration.py) - all tests should FAIL
2. **GREEN Phase**: Implement features incrementally to make tests pass:
   - Add chunking configuration to [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py)
   - Implement automatic chunking logic
   - Add configuration to [`default.yaml`](config/default.yaml)
   - Refactor pipelines to use automatic chunking
3. **REFACTOR Phase**: Clean up implementations while keeping tests passing

### Test Execution Commands

```bash
# Run chunking architecture tests
uv run pytest tests/test_chunking_architecture_integration.py -v | tee test_output/chunking_architecture.log

# Run with 1000+ documents for scale testing
uv run pytest tests/test_chunking_architecture_integration.py::TestChunkingPerformanceIntegration -v | tee test_output/chunking_performance_1000.log

# Run complete end-to-end test
uv run pytest tests/test_chunking_architecture_integration.py::TestCompleteChunkingArchitecture::test_end_to_end_chunking_architecture -v | tee test_output/chunking_e2e.log
```

## Success Criteria

### Functional Requirements
- [ ] [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py) automatically chunks documents above threshold
- [ ] All 8 RAG pipelines work with automatic chunking
- [ ] Chunking strategy configurable via YAML
- [ ] Per-call chunking overrides work
- [ ] ColBERT and SQL RAG can disable chunking appropriately

### Performance Requirements
- [ ] Chunking performance scales to 1000+ documents
- [ ] Memory usage remains reasonable during chunking
- [ ] Chunking adds < 10% overhead to document ingestion

### Developer Experience Requirements
- [ ] Pipelines become chunking-agnostic (just call [`add_documents()`](iris_rag/storage/vector_store_iris.py))
- [ ] Configuration drives chunking behavior consistently
- [ ] Interface matches LlamaIndex/Haystack patterns
- [ ] Clear error messages for chunking failures

## Migration Path

### Phase 1: Foundation (TDD RED)
1. Create failing tests in [`tests/test_chunking_architecture_integration.py`](tests/test_chunking_architecture_integration.py)
2. Run tests to confirm failures
3. Document current failure points

### Phase 2: Core Implementation (TDD GREEN)
1. Add chunking configuration to [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py)
2. Implement automatic chunking logic
3. Add configuration to [`default.yaml`](config/default.yaml)
4. Make basic tests pass

### Phase 3: Pipeline Integration (TDD GREEN continued)
1. Refactor [`BasicRAG`](iris_rag/pipelines/basic.py) to use automatic chunking
2. Add chunking to remaining 7 pipelines
3. Handle special cases (ColBERT, SQL RAG)
4. Make all pipeline tests pass

### Phase 4: Performance & Polish (TDD REFACTOR)
1. Optimize chunking performance for 1000+ documents
2. Add comprehensive error handling
3. Clean up code while maintaining test coverage
4. Document final architecture

## Risk Mitigation

### Backward Compatibility
- Keep existing [`DocumentChunkingService`](tools/chunking/chunking_service.py) interface unchanged
- Add new functionality as optional parameters
- Maintain existing test fixtures

### Performance Concerns
- Implement chunking lazily (only when needed)
- Add configuration to disable chunking globally
- Monitor memory usage during large-scale tests

### Error Handling
- Graceful fallback when chunking fails
- Clear error messages for configuration issues
- Validation of chunking parameters

## Implementation Timeline

**Estimated Effort**: 2-3 development cycles

1. **Cycle 1**: TDD setup and [`IRISVectorStore`](iris_rag/storage/vector_store_iris.py) enhancement
2. **Cycle 2**: Pipeline refactoring and integration
3. **Cycle 3**: Performance optimization and documentation

This specification drives toward a clean, extensible chunking architecture that provides a unified developer experience across all RAG pipelines.