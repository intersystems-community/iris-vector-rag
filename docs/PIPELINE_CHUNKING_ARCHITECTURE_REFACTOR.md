# Pipeline Chunking Architecture Refactor

## Problem Analysis

The current pipeline migration approach violates DRY principles by requiring each pipeline to implement its own `load_documents` method with chunking logic. This creates:

1. **Code Duplication**: Every pipeline reimplements the same chunking integration
2. **Maintenance Burden**: Changes to chunking logic require updates across all pipelines
3. **Inconsistency Risk**: Different pipelines may implement chunking differently
4. **Testing Complexity**: Each pipeline needs separate chunking tests

## Current Architecture Issues

### Before (Current State)
```python
# Each pipeline implements its own load_documents with chunking
class CRAGPipeline(RAGPipeline):
    def load_documents(self, documents_path: str, **kwargs):
        # Custom chunking configuration reading
        chunking_config = self.config_manager.get_config("pipeline_overrides:crag:chunking", {})
        # Custom chunking logic
        self.vector_store.add_documents(documents, **chunking_config)

class GraphRAGPipeline(RAGPipeline):
    def load_documents(self, documents_path: str, **kwargs):
        # Duplicate chunking configuration reading
        chunking_config = self.config_manager.get_config("pipeline_overrides:graphrag:chunking", {})
        # Duplicate chunking logic
        self.vector_store.add_documents(documents, **chunking_config)
```

### After (Proposed Architecture)
```python
# Base class provides default chunking-aware load_documents
class RAGPipeline(ABC):
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Default implementation with automatic chunking via vector store."""
        # Load documents from path or kwargs
        documents = self._get_documents(documents_path, **kwargs)
        
        # Get pipeline-specific chunking configuration
        pipeline_name = self._get_pipeline_name()
        chunking_config = self._get_chunking_config(pipeline_name)
        
        # Delegate to vector store with automatic chunking
        self.vector_store.add_documents(documents, **chunking_config)
    
    def _get_pipeline_name(self) -> str:
        """Extract pipeline name from class name."""
        return self.__class__.__name__.lower().replace('pipeline', '').replace('rag', '')
    
    def _get_chunking_config(self, pipeline_name: str) -> dict:
        """Get pipeline-specific chunking configuration."""
        return self.config_manager.get_config(f"pipeline_overrides:{pipeline_name}:chunking", {})

# Pipelines inherit default behavior unless they need special processing
class CRAGPipeline(RAGPipeline):
    # No load_documents override needed - uses base implementation
    pass

class GraphRAGPipeline(RAGPipeline):
    # No load_documents override needed - uses base implementation
    pass

class SQLRAGPipeline(RAGPipeline):
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Override for conditional chunking logic."""
        documents = self._get_documents(documents_path, **kwargs)
        
        # Special SQL RAG logic: conditional chunking based on document type
        for doc in documents:
            if self._should_chunk_document(doc):
                chunking_config = self._get_chunking_config("sql_rag")
                self.vector_store.add_documents([doc], **chunking_config)
            else:
                self.vector_store.add_documents([doc], enabled=False)
```

## Proposed Solution

### 1. Enhance Base RAGPipeline Class

Add default `load_documents` implementation to [`iris_rag/core/base.py`](iris_rag/core/base.py):

```python
class RAGPipeline(ABC):
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Default implementation with automatic chunking via vector store.
        
        Pipelines can override this method if they need special processing,
        but most should use this default implementation.
        """
        # Load documents from path or direct input
        documents = self._get_documents(documents_path, **kwargs)
        
        # Get pipeline-specific chunking configuration
        pipeline_name = self._get_pipeline_name()
        chunking_config = self._get_chunking_config(pipeline_name)
        
        # Apply any pipeline-specific document preprocessing
        documents = self._preprocess_documents(documents, **kwargs)
        
        # Delegate to vector store with automatic chunking
        self.vector_store.add_documents(documents, **chunking_config)
    
    def _get_documents(self, documents_path: str, **kwargs) -> List[Document]:
        """Load documents from path or kwargs."""
        if "documents" in kwargs:
            return kwargs["documents"]
        return self._load_documents_from_path(documents_path)
    
    def _get_pipeline_name(self) -> str:
        """Extract pipeline name from class name for configuration lookup."""
        class_name = self.__class__.__name__.lower()
        # Remove common suffixes to get clean pipeline name
        for suffix in ['pipeline', 'ragpipeline', 'rag']:
            if class_name.endswith(suffix):
                class_name = class_name[:-len(suffix)]
                break
        return class_name
    
    def _get_chunking_config(self, pipeline_name: str) -> dict:
        """Get pipeline-specific chunking configuration."""
        config_key = f"pipeline_overrides:{pipeline_name}:chunking"
        return self.config_manager.get_config(config_key, {})
    
    def _preprocess_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Hook for pipeline-specific document preprocessing.
        Override in subclasses if needed.
        """
        return documents
    
    def _load_documents_from_path(self, documents_path: str) -> List[Document]:
        """Load documents from file or directory path."""
        # Move implementation from BasicRAG to base class
        # ... (implementation details)
```

### 2. Simplify Pipeline Implementations

Most pipelines become much simpler:

```python
class CRAGPipeline(RAGPipeline):
    """CRAG pipeline - uses default chunking behavior."""
    # No load_documents override needed
    pass

class GraphRAGPipeline(RAGPipeline):
    """GraphRAG pipeline - uses semantic chunking via configuration."""
    # No load_documents override needed
    pass

class HybridIFindRAGPipeline(RAGPipeline):
    """Hybrid IFind pipeline - uses hybrid chunking via configuration."""
    # No load_documents override needed
    pass
```

### 3. Special Cases Override When Needed

Only pipelines with special requirements override:

```python
class SQLRAGPipeline(RAGPipeline):
    """SQL RAG pipeline with conditional chunking."""
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Override for conditional chunking based on document type."""
        documents = self._get_documents(documents_path, **kwargs)
        
        # Special logic: chunk based on document type and size
        for doc in documents:
            if self._should_chunk_document(doc):
                chunking_config = self._get_chunking_config("sql_rag")
                self.vector_store.add_documents([doc], **chunking_config)
            else:
                # Store without chunking
                self.vector_store.add_documents([doc], enabled=False)
    
    def _should_chunk_document(self, doc: Document) -> bool:
        """Determine if document should be chunked."""
        doc_type = self._determine_document_type(doc)
        doc_size = len(doc.page_content)
        
        # Only chunk large text documents
        return doc_type == "text" and doc_size > 2000

class ColBERTRAGPipeline(RAGPipeline):
    """ColBERT pipeline - disables chunking due to token-level embeddings."""
    
    def _get_chunking_config(self, pipeline_name: str) -> dict:
        """Override to disable chunking for ColBERT."""
        config = super()._get_chunking_config(pipeline_name)
        config["enabled"] = False  # Force disable chunking
        return config
```

## Benefits of This Architecture

### 1. DRY Compliance
- Single implementation of chunking logic in base class
- Configuration reading centralized
- Document loading logic shared

### 2. Maintainability
- Changes to chunking behavior only require base class updates
- Pipeline-specific configurations remain isolated
- Clear separation of concerns

### 3. Extensibility
- Easy to add new pipelines (just inherit, no chunking code needed)
- Special cases can override specific methods
- Hook methods allow customization without full override

### 4. Testing Simplification
- Base chunking behavior tested once in base class tests
- Pipeline-specific tests focus on their unique logic
- Reduced test duplication

## Migration Strategy

### Phase 1: Enhance Base Class
1. Move `load_documents` implementation from BasicRAG to base class
2. Add helper methods for configuration and document loading
3. Update BasicRAG to use inherited implementation

### Phase 2: Simplify Pipelines
1. Remove `load_documents` overrides from simple pipelines (CRAG, GraphRAG, NodeRAG, HyDE)
2. Keep overrides only for special cases (SQL RAG, ColBERT)
3. Update configuration to use consistent pipeline naming

### Phase 3: Update Tests
1. Create base class tests for default chunking behavior
2. Simplify pipeline-specific tests to focus on unique logic
3. Remove duplicated chunking tests

## Configuration Impact

Pipeline naming becomes consistent:
```yaml
pipeline_overrides:
  crag:
    chunking:
      enabled: true
      strategy: "fixed_size"
  graphrag:
    chunking:
      enabled: true
      strategy: "semantic"
  colbert:
    chunking:
      enabled: false
```

Class name mapping:
- `CRAGPipeline` → `crag`
- `GraphRAGPipeline` → `graphrag`
- `ColBERTRAGPipeline` → `colbert`
- `SQLRAGPipeline` → `sqlrag`

## Conclusion

This refactoring eliminates DRY violations while maintaining flexibility for special cases. The architecture becomes more maintainable, testable, and extensible while preserving all existing functionality.