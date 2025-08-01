# Pipeline Development Guide

This guide helps developers create new RAG pipelines that follow project architecture patterns and avoid common pitfalls.

## ⚡ IMPORTANT: Unified API Architecture

**All pipelines now use a single `query()` method** - the old `execute()` and `run()` methods are deprecated for consistency and performance.

### The New Standard Pattern
```python
# ✅ CORRECT: Use the unified query() method
result = pipeline.query("What is machine learning?", top_k=5, include_sources=True)

# ❌ DEPRECATED: Don't use execute() or run()
result = pipeline.query("What is machine learning?")  # Still works but deprecated
result = pipeline.query("What is machine learning?")     # Still works but deprecated
```

### Standard Response Format
All pipelines return the same structured response:
```python
{
    "query": str,           # The original query
    "answer": str,          # Generated answer (or None if generate_answer=False)
    "retrieved_documents": List[Document],  # Retrieved documents
    "contexts": List[str],  # Document content as strings (for RAGAS compatibility)
    "execution_time": float,               # Processing time in seconds
    "metadata": {           # Pipeline-specific metadata
        "num_retrieved": int,
        "pipeline_type": str,
        "generated_answer": bool,
        # ... custom fields
    }
}
```

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

**✅ DO: Override the unified query() method**
```python
class ReRankingPipeline(BasicRAGPipeline):
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        # Get more candidates for reranking
        initial_k = min(top_k * self.rerank_factor, 100)  # Cap for performance
        
        # Use parent method for initial retrieval (don't generate answer yet)
        parent_kwargs = kwargs.copy()
        parent_kwargs['generate_answer'] = False
        parent_result = super().query(query_text, top_k=initial_k, **parent_kwargs)
        
        # Apply reranking to retrieved documents
        candidates = parent_result.get("retrieved_documents", [])
        if len(candidates) > 1 and self.reranker_func:
            final_documents = self._rerank_documents(query_text, candidates, top_k)
        else:
            final_documents = candidates[:top_k]
        
        # Generate answer with reranked documents if requested
        generate_answer = kwargs.get("generate_answer", True)
        if generate_answer and self.llm_func and final_documents:
            answer = self._generate_answer(query_text, final_documents)
        else:
            answer = parent_result.get("answer")
        
        # Return complete response in standard format
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": final_documents,
            "contexts": [doc.page_content for doc in final_documents],
            "execution_time": parent_result.get("execution_time", 0.0),
            "metadata": {
                "num_retrieved": len(final_documents),
                "pipeline_type": "reranking",
                "reranked": len(candidates) > 1,
                "generated_answer": generate_answer and answer is not None
            }
        }
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

## Requirements-Driven Orchestrator Architecture

### Overview

The orchestrator uses an elegant **requirements-driven architecture** that automatically sets up pipelines based on their declared requirements, eliminating hardcoded setup logic.

### How It Works

```python
# 1. Define requirements (in requirements.py)
class MyPipelineRequirements(PipelineRequirements):
    @property
    def pipeline_name(self) -> str:
        return "my_pipeline"
    
    @property 
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments", 
                schema="RAG",
                description="Main document storage"
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments", 
                column="embedding",
                description="Document-level embeddings"
            )
        ]

# 2. Register requirements (in requirements.py)
PIPELINE_REQUIREMENTS_REGISTRY = {
    "my_pipeline": MyPipelineRequirements,
    # ... other pipelines
}

# 3. Orchestrator automatically fulfills requirements
orchestrator.setup_pipeline("my_pipeline")  # Just works!
```

### Architecture Benefits

#### ✅ **Self-Documenting**
Requirements clearly define what each pipeline needs:
```python
# Before: Hidden in hardcoded setup methods
def _setup_my_pipeline(self, requirements):
    # What does this pipeline actually need? 🤷‍♂️
    self._ensure_document_embeddings()
    self._generate_chunks()

# After: Clear, declarative requirements  
class MyPipelineRequirements(PipelineRequirements):
    required_tables = [TableRequirement("SourceDocuments", ...)]
    required_embeddings = [EmbeddingRequirement("document_embeddings", ...)]
```

#### ✅ **Zero Duplication**
Similar pipelines share setup logic automatically:
```python
# Before: Duplicate hardcoded methods
def _setup_basic_pipeline(self):      # Duplicate
    self._ensure_document_embeddings()

def _setup_basic_rerank_pipeline(self): # Duplicate  
    self._ensure_document_embeddings()  # Same logic!

# After: Single generic fulfillment
if pipeline_type in ["basic", "basic_rerank"]:
    self._fulfill_requirements(requirements)  # Shared logic!
```

#### ✅ **Automatic Setup**
New pipelines work without touching orchestrator code:
```python
# Before: Must add hardcoded method to orchestrator
def setup_pipeline(self, pipeline_type):
    if pipeline_type == "my_new_pipeline":
        self._setup_my_new_pipeline()  # Must add this method!

# After: Just define requirements, setup works automatically
class MyNewPipelineRequirements(PipelineRequirements):
    # Define what you need
    required_tables = [...]
    
# Orchestrator automatically fulfills requirements - no code changes!
```

### Migration Strategy

The architecture supports **gradual migration**:

```python
def setup_pipeline(self, pipeline_type):
    # NEW: Generic approach (recommended)
    if pipeline_type in ["basic", "basic_rerank"]:
        self._fulfill_requirements(requirements)  # Requirements-driven!
        
    # LEGACY: Hardcoded methods (still supported)  
    elif pipeline_type == "colbert":
        self._setup_colbert_pipeline(requirements)  # Will migrate later
        
    # FALLBACK: Try generic for unknown pipelines
    else:
        self._fulfill_requirements(requirements)  # Always try generic first
```

### Adding Your Pipeline

**Step 1**: Define requirements (copy existing similar pipeline):
```python
class MyPipelineRequirements(PipelineRequirements):
    @property
    def pipeline_name(self) -> str:
        return "my_pipeline"
    
    # Copy requirements from similar pipeline, modify as needed
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding", 
                description="Document embeddings for vector search"
            )
        ]
```

**Step 2**: Register requirements:
```python
PIPELINE_REQUIREMENTS_REGISTRY = {
    "my_pipeline": MyPipelineRequirements,
    # ... existing pipelines
}
```

**Step 3**: Add to factory (in factory.py):
```python
elif pipeline_type == "my_pipeline":
    return MyPipeline(connection_manager, config_manager, llm_func)
```

**That's it!** Orchestrator setup works automatically based on your requirements.

### Best Practices

#### ✅ **Reuse Existing Requirements**
```python
# Good: Reuse existing requirement patterns
required_embeddings = [
    EmbeddingRequirement(
        name="document_embeddings",           # Standard name
        table="RAG.SourceDocuments",          # Standard table
        column="embedding",                   # Standard column
        description="Document-level embeddings"
    )
]

# Avoid: Creating new requirement types unless necessary
```

#### ✅ **Clear Descriptions**
```python
# Good: Descriptive, helpful
TableRequirement(
    name="SourceDocuments",
    schema="RAG", 
    description="Main document storage with embeddings for vector search",
    min_rows=1
)

# Avoid: Vague descriptions
TableRequirement(name="SourceDocuments", description="Documents")
```

#### ✅ **Optional vs Required**
```python
class MyPipelineRequirements(PipelineRequirements):
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [...]  # Must exist for pipeline to work
    
    @property  
    def optional_tables(self) -> List[TableRequirement]:
        return [...]  # Nice to have, enhances functionality
```

### TDD Benefits

The requirements-driven architecture makes **Test-Driven Development much more elegant**:

#### ✅ **Generic Tests Replace Duplicate Tests**

```python
# BEFORE: Duplicate test methods for similar pipelines
def test_basic_pipeline_setup():
    """Test basic pipeline setup."""
    orchestrator._setup_basic_pipeline(requirements)
    assert_embeddings_exist()
    assert_tables_exist()

def test_basic_rerank_pipeline_setup():  # Duplicate!
    """Test basic rerank pipeline setup.""" 
    orchestrator._setup_basic_rerank_pipeline(requirements)
    assert_embeddings_exist()  # Same assertions!
    assert_tables_exist()      # Same logic!

# AFTER: One generic test for all similar pipelines
@pytest.mark.parametrize("pipeline_type", ["basic", "basic_rerank", "future_basic_variant"])
def test_basic_pipeline_family_setup(pipeline_type):
    """Test that all basic pipeline variants work automatically."""
    requirements = get_pipeline_requirements(pipeline_type)
    orchestrator._fulfill_requirements(requirements)
    
    # Validate based on declared requirements (not hardcoded logic)
    for table_req in requirements.required_tables:
        assert_table_exists(table_req.name)
    
    for embedding_req in requirements.required_embeddings:
        assert_embeddings_exist(embedding_req.table, embedding_req.column)
```

#### ✅ **Requirements Are Unit Testable**

```python
# Test requirement definitions themselves
def test_basic_rerank_requirements():
    """Test that basic_rerank declares correct requirements."""
    req = get_pipeline_requirements("basic_rerank")
    
    # Validate requirements structure
    assert req.pipeline_name == "basic_rerank"
    assert len(req.required_tables) == 1
    assert req.required_tables[0].name == "SourceDocuments"
    assert len(req.required_embeddings) == 1
    assert req.required_embeddings[0].table == "RAG.SourceDocuments"

# Test requirement fulfillment generically  
def test_embedding_requirement_fulfillment():
    """Test that embedding requirements are fulfilled correctly."""
    embedding_req = EmbeddingRequirement(
        name="test_embeddings",
        table="RAG.SourceDocuments", 
        column="embedding",
        description="Test embeddings"
    )
    
    orchestrator._fulfill_embedding_requirement(embedding_req)
    assert_embeddings_exist("RAG.SourceDocuments", "embedding")
```

#### ✅ **Test Coverage Scales Automatically**

```python
# New pipelines get test coverage automatically!
def test_all_registered_pipelines_have_valid_setup():
    """Test that EVERY registered pipeline can be set up."""
    from iris_rag.validation.requirements import PIPELINE_REQUIREMENTS_REGISTRY
    
    for pipeline_type in PIPELINE_REQUIREMENTS_REGISTRY.keys():
        # This test automatically covers new pipelines when they're added!
        requirements = get_pipeline_requirements(pipeline_type)
        
        # Validate requirements are well-formed
        assert requirements.pipeline_name == pipeline_type
        assert len(requirements.required_tables) >= 0
        assert len(requirements.required_embeddings) >= 0
        
        # Test that setup works
        orchestrator._fulfill_requirements(requirements)
```

#### ✅ **Integration Tests Become Simpler**

```python
# BEFORE: Complex integration tests with hardcoded setup
def test_colbert_e2e_integration():
    # Complex setup mimicking _setup_colbert_pipeline
    setup_source_documents_table()
    setup_token_embeddings_table() 
    generate_document_embeddings()
    generate_token_embeddings()
    setup_colbert_specific_configs()
    
    pipeline = create_pipeline("colbert")
    result = pipeline.query("test query")
    assert result["answer"]

# AFTER: Clean integration tests using requirements
def test_colbert_e2e_integration():
    # Requirements-driven setup (works for ANY pipeline!)
    orchestrator.setup_pipeline("colbert", auto_fix=True)
    
    pipeline = create_pipeline("colbert") 
    result = pipeline.query("test query")
    assert result["answer"]
    
# Same pattern works for ALL pipelines - no pipeline-specific test setup!
@pytest.mark.parametrize("pipeline_type", ["basic", "colbert", "crag", "hyde"])  
def test_pipeline_e2e_integration(pipeline_type):
    orchestrator.setup_pipeline(pipeline_type, auto_fix=True)
    pipeline = create_pipeline(pipeline_type)
    result = pipeline.query("test query")
    assert result["answer"]
```

#### ✅ **Mocking Becomes Targeted**

```python
# BEFORE: Must mock entire hardcoded setup methods
@patch('orchestrator._setup_basic_pipeline')
@patch('orchestrator._setup_colbert_pipeline') 
@patch('orchestrator._setup_crag_pipeline')
def test_setup_with_mocks(mock_crag, mock_colbert, mock_basic):
    # Must mock every hardcoded method individually

# AFTER: Mock the generic requirement fulfillment
@patch('orchestrator._fulfill_requirements')
def test_setup_with_generic_mock(mock_fulfill):
    """Test setup logic without actually fulfilling requirements."""
    orchestrator.setup_pipeline("any_pipeline")
    mock_fulfill.assert_called_once()
    
# Or mock specific requirement types
@patch('orchestrator._fulfill_embedding_requirement')
def test_embedding_setup_only(mock_embed):
    # Test just embedding fulfillment logic
```

#### ✅ **Test Failures Are More Informative**

```python
# BEFORE: Vague hardcoded test failures
def test_basic_setup():
    orchestrator._setup_basic_pipeline(req)  
    # If this fails: "AssertionError" - what went wrong? 🤷‍♂️

# AFTER: Requirement-specific failures  
def test_requirements_fulfillment():
    requirements = get_pipeline_requirements("basic")
    
    for table_req in requirements.required_tables:
        assert_table_exists(table_req.name), f"Missing required table: {table_req.name}"
        
    for embed_req in requirements.required_embeddings:
        assert_embeddings_exist(embed_req.table), f"Missing embeddings: {embed_req.name}"
    
    # Clear failure messages: "Missing required table: SourceDocuments"
```

**TDD Summary**: The elegant architecture makes tests more maintainable, more comprehensive, and easier to write. When adding a new pipeline, you get test coverage automatically instead of writing duplicate test code!

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
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        THE single method for all RAG operations - replaces execute()/run().
        
        Override only the parts that need customization while maintaining
        the standard response format for compatibility.
        """
        # Use parent for standard retrieval and generation
        parent_result = super().query(query_text, top_k, **kwargs)
        
        # Add custom processing while preserving response structure
        enhanced_result = self._apply_custom_logic(parent_result)
        
        # Ensure metadata reflects custom processing
        enhanced_result["metadata"].update({
            "pipeline_type": "custom",
            "custom_processing": True
        })
        
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
    
    def execute(self, ...):  # DON'T: Multiple query methods
        # Confusing API with multiple entry points
    
    def run(self, ...):  # DON'T: More method confusion
        # Use single query() method instead

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
    
    # Test the unified query() method
    result = pipeline.query("test query", top_k=3)
    
    # Verify standard response format
    assert "query" in result
    assert "answer" in result
    assert "retrieved_documents" in result
    assert "contexts" in result
    assert "metadata" in result
    assert "execution_time" in result
    assert len(result["retrieved_documents"]) <= 3
```

### 2. Integration Tests
```python
@pytest.mark.integration
def test_pipeline_e2e():
    # Test with real database connection
    pipeline = MyCustomPipeline()
    pipeline.load_documents(["test_doc.txt"])
    
    # Use the unified query() method (not execute())
    result = pipeline.query("What is the main topic?")
    assert result["answer"] is not None
    assert "retrieved_documents" in result
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
    result = pipeline.query("What is machine learning?")  # Use query(), not execute()
"""
```

## Common Mistakes Checklist

Before submitting your pipeline, verify:

- [ ] **Inherits from appropriate base class** (not RAGPipeline directly)
- [ ] **No code duplication** from parent classes
- [ ] **Uses unified query() method** (no execute(), run() methods)
- [ ] **Heavy imports are lazy-loaded** (inside methods, not module-level)
- [ ] **Configuration uses dedicated config section** 
- [ ] **Registration only requires config changes** (no __init__.py modifications)
- [ ] **Maintains standard response format** (query, answer, retrieved_documents, contexts, metadata, execution_time)
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
        """
        THE single method for all RAG operations.
        
        Override to add post-processing while maintaining standard response format.
        """
        # Use parent for standard functionality
        result = super().query(query_text, top_k, **kwargs)
        
        # Add custom post-processing
        if self.post_process:
            result = self._post_process_result(result)
        
        # Update metadata to reflect custom processing
        result["metadata"].update({
            "post_processed": self.post_process,
            "pipeline_type": "example_custom"
        })
        
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