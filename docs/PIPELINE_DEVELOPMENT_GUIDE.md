# Pipeline Development Guide

This guide helps developers create new RAG pipelines that follow project architecture patterns and avoid common pitfalls.

## Quick Start: Creating a New Pipeline

### 1. Choose Your Base Class

**Always inherit from an existing pipeline** instead of duplicating code:

```python
from iris_rag.pipelines.basic import BasicRAGPipeline

class MyCustomPipeline(BasicRAGPipeline):
    def __init__(self, connection_manager, config_manager, **kwargs):
        super().__init__(connection_manager, config_manager, **kwargs)
        # Add custom initialization here
```

**Common Base Classes:**
- `BasicRAGPipeline` - Standard retrieval + generation
- `HyDERAGPipeline` - Hypothetical document generation  
- `CRAGPipeline` - Corrective RAG with validation

### 2. Override Only What You Need

**✅ DO: Override specific methods**
```python
class ReRankingPipeline(BasicRAGPipeline):
    def query(self, query_text: str, top_k: int = 5, **kwargs):
        # Get more candidates for reranking
        initial_k = top_k * self.rerank_factor
        parent_result = super().query(query_text, top_k=initial_k, **kwargs)
        
        # Apply reranking logic
        return self._rerank_documents(parent_result, top_k)
```

**❌ DON'T: Copy entire parent class**
```python
class BadPipeline(RAGPipeline):  # Starting from scratch
    def __init__(self, ...):
        # 300 lines of duplicated code from BasicRAGPipeline
```

### 3. Use Lazy Loading for Heavy Dependencies

**✅ DO: Import inside functions**
```python
def my_reranker(query: str, docs: List[Document]):
    # Lazy import to avoid module-level loading
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("my-model")
    return model.predict([(query, doc.page_content) for doc in docs])
```

**❌ DON'T: Import at module level**
```python
from sentence_transformers import CrossEncoder  # Loads immediately
model = CrossEncoder("my-model")  # Even worse - global loading
```

## Configuration Patterns

### 1. Create Dedicated Config Section

Add to `config/pipelines.yaml`:
```yaml
pipeline_configs:
  my_custom_pipeline:
    custom_param: "value"
    model_name: "my-model"
    batch_size: 32
```

### 2. Access Config in Pipeline

```python
class MyCustomPipeline(BasicRAGPipeline):
    def __init__(self, connection_manager, config_manager, **kwargs):
        super().__init__(connection_manager, config_manager, **kwargs)
        
        # Get dedicated config with fallback to basic config
        self.custom_config = self.config_manager.get(
            "pipelines:my_custom_pipeline", 
            self.config_manager.get("pipelines:basic", {})
        )
        
        self.custom_param = self.custom_config.get("custom_param", "default")
```

## Registration System

### 1. Add to Pipeline Registry

Only modify `config/pipelines.yaml` - **NO source code changes needed**:

```yaml
pipelines:
  - name: "MyCustomPipeline"
    module: "iris_rag.pipelines.my_custom"  # or external package
    class: "MyCustomPipeline"
    enabled: true
    params:
      top_k: 5
      custom_param: "value"
```

### 2. External Package Registration

For pipelines in separate repositories:
```yaml
pipelines:
  - name: "ExternalPipeline"
    module: "external_package.rag_pipelines"  # Will be imported dynamically
    class: "AdvancedRAGPipeline"
    enabled: true
```

The registration system uses `ModuleLoader` with `importlib.import_module()` for dynamic loading.

## Common Patterns & Anti-Patterns

### ✅ Good Pipeline Structure

```python
class WellDesignedPipeline(BasicRAGPipeline):
    """Brief description of what this pipeline does differently."""
    
    def __init__(self, connection_manager, config_manager, 
                 custom_func: Optional[Callable] = None, **kwargs):
        super().__init__(connection_manager, config_manager, **kwargs)
        
        # Custom configuration
        self.pipeline_config = self.config_manager.get("pipelines:my_pipeline", {})
        self.custom_param = self.pipeline_config.get("custom_param", "default")
        
        # Custom function with default
        self.custom_func = custom_func or self._default_custom_func
    
    def query(self, query_text: str, top_k: int = 5, **kwargs):
        """Override only the parts that need customization."""
        # Use parent for standard retrieval
        parent_result = super().query(query_text, top_k=initial_k, **kwargs)
        
        # Add custom processing
        enhanced_result = self._apply_custom_logic(parent_result)
        
        return enhanced_result
    
    def _apply_custom_logic(self, result):
        """Private method for custom processing."""
        # Custom logic here
        return result
    
    def _default_custom_func(self, data):
        """Default implementation with lazy loading."""
        from some_library import SomeModel
        model = SomeModel()
        return model.process(data)
```

### ❌ Anti-Patterns to Avoid

```python
# DON'T: Duplicate parent class code
class BadPipeline(RAGPipeline):
    def __init__(self, ...):
        # 200+ lines copied from BasicRAGPipeline
    
    def query(self, ...):
        # Another 100+ lines copied from BasicRAGPipeline
        # with tiny modifications

# DON'T: Module-level heavy imports
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder
big_model = AutoModel.from_pretrained("huge-model")  # Loads on import!

# DON'T: Hard-coded configuration
class BadConfigPipeline(BasicRAGPipeline):
    def __init__(self, ...):
        self.model_name = "hard-coded-model"  # Should be configurable
        self.batch_size = 32  # Should come from config
```

## Testing Your Pipeline

### 1. Unit Tests
```python
def test_my_pipeline():
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager()
    
    pipeline = MyCustomPipeline(connection_manager, config_manager)
    
    # Test with mock data
    result = pipeline.query("test query", top_k=3)
    assert "retrieved_documents" in result
    assert len(result["retrieved_documents"]) <= 3
```

### 2. Integration Tests
```python
@pytest.mark.integration
def test_pipeline_e2e():
    # Test with real database connection
    pipeline = MyCustomPipeline()
    pipeline.load_documents(["test_doc.txt"])
    
    result = pipeline.execute("What is the main topic?")
    assert result["answer"] is not None
```

## Performance Considerations

### 1. Lazy Loading Heavy Models
```python
class PerformantPipeline(BasicRAGPipeline):
    def __init__(self, ...):
        super().__init__(...)
        self._model = None  # Don't load until needed
    
    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.model_name)
        return self._model
```

### 2. Caching Expensive Operations
```python
from functools import lru_cache

class CachingPipeline(BasicRAGPipeline):
    @lru_cache(maxsize=100)
    def _expensive_operation(self, input_hash):
        # Expensive computation here
        return result
```

### 3. Batch Processing
```python
def process_batch(self, queries: List[str]) -> List[Dict]:
    # Process multiple queries efficiently
    embeddings = self.embedding_manager.embed_texts(queries)  # Batch embed
    results = []
    for query, embedding in zip(queries, embeddings):
        result = self._search_with_embedding(embedding)
        results.append(result)
    return results
```

## Documentation Requirements

Every new pipeline should include:

1. **Class docstring** explaining the technique and differences from base class
2. **Method docstrings** for overridden methods
3. **Configuration documentation** in the class docstring
4. **Usage example** in the module docstring

```python
"""
Custom Reranking Pipeline

This pipeline extends BasicRAGPipeline by adding a reranking step after initial
vector retrieval using cross-encoder models.

Configuration:
    rerank_factor (int): Multiplier for initial retrieval (default: 2)
    reranker_model (str): HuggingFace model name for reranking

Usage:
    pipeline = BasicRAGRerankingPipeline(conn_mgr, config_mgr)
    result = pipeline.execute("What is machine learning?")
"""
```

## Common Mistakes Checklist

Before submitting your pipeline, verify:

- [ ] **Inherits from appropriate base class** (not RAGPipeline directly)
- [ ] **No code duplication** from parent classes
- [ ] **Heavy imports are lazy-loaded** (inside methods, not module-level)
- [ ] **Configuration uses dedicated config section** 
- [ ] **Registration only requires config changes** (no __init__.py modifications)
- [ ] **Docstrings explain the unique technique**
- [ ] **Tests cover custom functionality**
- [ ] **Error handling for custom components**

## Getting Help

1. **Study existing pipelines** in `iris_rag/pipelines/` for patterns
2. **Check base class methods** to understand what you can override
3. **Review configuration examples** in `config/pipelines.yaml`
4. **Run tests** to ensure compatibility: `make test-unit`

## Example: Complete Minimal Pipeline

```python
"""
Example Custom Pipeline

Demonstrates proper inheritance and configuration patterns.
"""

import logging
from typing import List, Dict, Any, Optional
from .basic import BasicRAGPipeline
from ..core.models import Document

logger = logging.getLogger(__name__)


class ExampleCustomPipeline(BasicRAGPipeline):
    """
    Example pipeline showing proper development patterns.
    
    This pipeline adds a simple post-processing step to BasicRAGPipeline
    results while following all architectural best practices.
    
    Configuration (config/pipelines.yaml):
        pipeline_configs:
          example_custom:
            post_process: true
            custom_threshold: 0.8
    """
    
    def __init__(self, connection_manager, config_manager, **kwargs):
        """Initialize with custom configuration."""
        super().__init__(connection_manager, config_manager, **kwargs)
        
        # Get dedicated config section
        self.custom_config = self.config_manager.get(
            "pipelines:example_custom",
            self.config_manager.get("pipelines:basic", {})
        )
        
        self.post_process = self.custom_config.get("post_process", True)
        self.custom_threshold = self.custom_config.get("custom_threshold", 0.8)
        
        logger.info(f"Initialized ExampleCustomPipeline with post_process={self.post_process}")
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Override query to add post-processing."""
        # Use parent for standard functionality
        result = super().query(query_text, top_k, **kwargs)
        
        # Add custom post-processing
        if self.post_process:
            result = self._post_process_result(result)
        
        # Update metadata
        result["metadata"]["post_processed"] = self.post_process
        result["metadata"]["pipeline_type"] = "example_custom"
        
        return result
    
    def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom post-processing logic."""
        # Example: filter documents by custom criteria
        documents = result.get("retrieved_documents", [])
        
        # Apply custom filtering logic here
        filtered_docs = [doc for doc in documents if self._meets_criteria(doc)]
        
        result["retrieved_documents"] = filtered_docs
        result["metadata"]["filtered_count"] = len(documents) - len(filtered_docs)
        
        return result
    
    def _meets_criteria(self, document: Document) -> bool:
        """Custom criteria for document filtering."""
        # Example custom logic
        return len(document.page_content) > 100  # Simple length check
```

This guide would have helped your intern avoid the major pitfalls and create a proper, maintainable pipeline from the start!