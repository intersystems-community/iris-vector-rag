# Service Boundary Interfaces - Chunking Architecture

## Overview

This document defines the formal interfaces and contracts between services in the chunking architecture, ensuring clean separation of concerns and maintainable code boundaries.

## Interface Definitions

### 1. IRISVectorStore Interface

**Primary Interface**: Document storage with automatic chunking capabilities

```python
class IRISVectorStore:
    """Vector store with automatic chunking integration."""
    
    def __init__(self, config_manager: ConfigurationManager, 
                 schema_manager: SchemaManager = None):
        """Initialize vector store with chunking configuration.
        
        Args:
            config_manager: Configuration management service
            schema_manager: Optional schema manager (auto-created if None)
        """
    
    def add_documents(self, 
                     documents: List[Document], 
                     auto_chunk: bool = None,
                     chunking_strategy: str = None) -> List[str]:
        """Add documents with optional automatic chunking.
        
        Args:
            documents: List of documents to store
            auto_chunk: Override auto-chunking behavior (None = use config)
            chunking_strategy: Override chunking strategy (None = use config)
            
        Returns:
            List of document IDs (may include chunk IDs if chunked)
            
        Raises:
            ChunkingError: If chunking fails
            StorageError: If document storage fails
        """
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         k: int = 10) -> List[Document]:
        """Search for similar documents/chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of similar documents (may include chunks)
        """
```

**Configuration Contract**:
```python
# Expected configuration structure
chunking_config = {
    "enabled": bool,
    "default_strategy": str,
    "auto_chunk_threshold": int,
    "strategies": {
        "fixed_size": {"chunk_size": int, "overlap": int},
        "semantic": {"model": str},
        "hybrid": {"chunk_size": int, "overlap": int}
    }
}
```

### 2. DocumentChunkingService Interface

**Primary Interface**: Document chunking with multiple strategies

```python
class DocumentChunkingService:
    """Service for chunking documents using various strategies."""
    
    def __init__(self):
        """Initialize chunking service with available strategies."""
    
    def chunk_document(self, 
                      document: Document, 
                      strategy: str = "fixed_size",
                      **kwargs) -> List[Document]:
        """Chunk a document using specified strategy.
        
        Args:
            document: Document to chunk
            strategy: Chunking strategy ("fixed_size", "semantic", "hybrid")
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of document chunks
            
        Raises:
            ValueError: If strategy is unknown
            ChunkingError: If chunking fails
        """
    
    @property
    def strategies(self) -> Dict[str, Any]:
        """Available chunking strategies."""
```

**Strategy Interface Contract**:
```python
# Each strategy must implement:
def chunk_text(text: str, **params) -> List[str]:
    """Chunk text into segments.
    
    Args:
        text: Text to chunk
        **params: Strategy-specific parameters
        
    Returns:
        List of text chunks
    """
```

### 3. ConfigurationManager Interface

**Primary Interface**: Hierarchical configuration management

```python
class ConfigurationManager:
    """Centralized configuration management."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using hierarchical key.
        
        Args:
            key: Hierarchical key (e.g., "storage:chunking:enabled")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
    
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (None = default)
            
        Returns:
            Loaded configuration dictionary
        """
```

**Configuration Schema Contract**:
```yaml
# Required configuration structure
storage:
  chunking:
    enabled: boolean
    default_strategy: string
    auto_chunk_threshold: integer
    strategies:
      fixed_size:
        chunk_size: integer
        overlap: integer
      semantic:
        model: string
      hybrid:
        chunk_size: integer
        overlap: integer
```

### 4. Pipeline Interface

**Primary Interface**: RAG pipeline with chunking delegation

```python
class RAGPipeline:
    """Base class for RAG pipelines with automatic chunking."""
    
    def __init__(self, 
                 config_manager: ConfigurationManager,
                 llm_func: Callable,
                 embedding_func: Callable = None):
        """Initialize pipeline with dependencies.
        
        Args:
            config_manager: Configuration management service
            llm_func: Language model function
            embedding_func: Embedding function (optional)
        """
    
    def load_documents(self, 
                      data_path: str = "", 
                      documents: List[Document] = None) -> None:
        """Load documents with automatic chunking.
        
        Args:
            data_path: Path to document data (optional)
            documents: Pre-loaded documents (optional)
            
        Note:
            Chunking is handled automatically by IRISVectorStore
        """
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with query, answer, and retrieved_documents
        """
```

**Pipeline Contract**:
- Pipelines MUST delegate chunking to IRISVectorStore
- Pipelines MUST NOT implement manual chunking logic
- Pipelines MUST use standard return format for queries

## Service Dependencies

### Dependency Graph
```
ConfigurationManager (no dependencies)
    ↓
DocumentChunkingService (no dependencies)
    ↓
IRISVectorStore (depends on: ConfigurationManager, DocumentChunkingService)
    ↓
RAGPipeline (depends on: ConfigurationManager, IRISVectorStore)
```

### Dependency Injection Pattern
```python
# Recommended initialization pattern
config_manager = ConfigurationManager()
vector_store = IRISVectorStore(config_manager=config_manager)
pipeline = BasicRAGPipeline(
    config_manager=config_manager,
    llm_func=llm_function,
    embedding_func=embedding_function
)
```

## Error Handling Contracts

### ChunkingError Hierarchy
```python
class ChunkingError(Exception):
    """Base exception for chunking operations."""
    pass

class StrategyNotFoundError(ChunkingError):
    """Raised when chunking strategy is not available."""
    pass

class DocumentTooLargeError(ChunkingError):
    """Raised when document exceeds maximum size limits."""
    pass

class ChunkingConfigurationError(ChunkingError):
    """Raised when chunking configuration is invalid."""
    pass
```

### Error Handling Contract
- Services MUST raise specific exceptions for different error conditions
- Services MUST provide meaningful error messages
- Services MUST handle dependency failures gracefully
- Services MUST log errors at appropriate levels

## Performance Contracts

### IRISVectorStore Performance
- `add_documents()` MUST complete in O(n*m) where n=documents, m=avg_chunks
- Chunking decision MUST complete in O(1) per document
- Configuration loading MUST be cached and reused

### DocumentChunkingService Performance
- `chunk_document()` MUST complete in O(n) where n=document_length
- Strategy selection MUST complete in O(1)
- Memory usage MUST be bounded by chunk_size * max_chunks

### Configuration Performance
- `get()` operations MUST complete in O(1) for cached values
- Configuration loading MUST be performed once at startup
- Configuration changes MUST not require service restart

## Backward Compatibility Contracts

### IRISVectorStore Compatibility
- Existing `add_documents(documents)` calls MUST continue to work
- Default behavior MUST match existing functionality
- New parameters MUST be optional with sensible defaults

### Pipeline Compatibility
- Existing pipeline initialization MUST continue to work
- Existing `load_documents()` calls MUST continue to work
- Performance characteristics MUST not degrade significantly

## Testing Contracts

### Unit Testing Requirements
- Each service MUST have isolated unit tests
- Dependencies MUST be mockable
- Error conditions MUST be testable

### Integration Testing Requirements
- Service interactions MUST be tested end-to-end
- Configuration loading MUST be tested with real files
- Performance characteristics MUST be validated

### Contract Testing Requirements
- Interface contracts MUST be validated with contract tests
- Breaking changes MUST be detected by contract tests
- API compatibility MUST be maintained across versions

## Monitoring and Observability Contracts

### Logging Requirements
- Services MUST log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Log messages MUST include relevant context
- Sensitive information MUST NOT be logged

### Metrics Requirements
- Services MUST expose performance metrics
- Chunking operations MUST be measurable
- Error rates MUST be trackable

### Health Check Requirements
- Services MUST provide health check endpoints
- Dependencies MUST be validated in health checks
- Configuration validity MUST be checkable

## Versioning and Evolution Contracts

### API Versioning
- Interface changes MUST follow semantic versioning
- Breaking changes MUST increment major version
- Deprecation warnings MUST be provided for removed features

### Configuration Evolution
- Configuration schema changes MUST be backward compatible
- New configuration options MUST have sensible defaults
- Configuration migration MUST be automated

### Service Evolution
- New chunking strategies MUST be addable without code changes
- Service implementations MUST be replaceable
- Interface extensions MUST not break existing clients

This interface specification ensures that the chunking architecture maintains clean boundaries, supports evolution, and provides reliable contracts for all consumers.