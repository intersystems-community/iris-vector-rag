# Dynamic Pipeline System

This document provides comprehensive documentation for the config-driven dynamic RAG pipeline loading system implemented in the RAG Templates framework.

## Overview

The dynamic pipeline system replaces hard-coded pipeline instantiation with a flexible, configuration-based approach that allows developers to:

- Define pipelines in [`config/pipelines.yaml`](../config/pipelines.yaml)
- Enable/disable pipelines without code changes
- Add custom pipelines from external packages
- Configure pipeline-specific parameters
- Automatically inject framework dependencies

## Architecture

### Core Components

The dynamic pipeline system consists of four main services:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dynamic Pipeline System                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Configuration   │  │ Module Loader   │  │ Pipeline        │ │
│  │ Service         │  │                 │  │ Factory         │ │
│  │                 │  │ - Dynamic       │  │                 │ │
│  │ - YAML Parsing  │  │   Imports       │  │ - Instantiation │ │
│  │ - Validation    │  │ - Class Loading │  │ - Dependency    │ │
│  │ - Schema Check  │  │ - Error Handling│  │   Injection     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                      ┌─────────────────┐ │
│  │ Pipeline        │                      │ Error Handler   │ │
│  │ Registry        │                      │                 │ │
│  │                 │                      │ - Import Errors │ │
│  │ - Name Mapping  │                      │ - Class Errors  │ │
│  │ - Instance Cache│                      │ - Init Errors   │ │
│  │ - Lifecycle     │                      │ - Logging       │ │
│  └─────────────────┘                      └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Service Responsibilities

1. **[`PipelineConfigService`](../iris_rag/config/pipeline_config_service.py)**: Handles YAML parsing and validation
2. **[`ModuleLoader`](../iris_rag/utils/module_loader.py)**: Manages dynamic imports and caching
3. **[`PipelineFactory`](../iris_rag/pipelines/factory.py)**: Creates and configures pipeline instances
4. **[`PipelineRegistry`](../iris_rag/pipelines/registry.py)**: Maintains pipeline inventory and lifecycle

## Configuration Schema

### Pipeline Definition Structure

Each pipeline in [`config/pipelines.yaml`](../config/pipelines.yaml) follows this structure:

```yaml
pipelines:
  - name: "PipelineName"           # Unique identifier
    module: "python.module.path"   # Module containing the pipeline class
    class: "PipelineClassName"     # Name of the pipeline class
    enabled: true                  # Whether to load this pipeline
    params:                        # Pipeline-specific parameters
      param1: value1
      param2: value2
```

### Example Configuration

```yaml
# config/pipelines.yaml
pipelines:
  - name: "BasicRAG"
    module: "iris_rag.pipelines.basic"
    class: "BasicRAGPipeline"
    enabled: true
    params:
      top_k: 5
      chunk_size: 1000
      similarity_threshold: 0.7

  - name: "ColBERTRAG"
    module: "iris_rag.pipelines.colbert"
    class: "ColBERTRAGPipeline"
    enabled: true
    params:
      top_k: 10
      max_query_length: 512
      doc_maxlen: 180

  - name: "CustomExternalRAG"
    module: "external_package.rag_pipelines"
    class: "AdvancedRAGPipeline"
    enabled: false
    params:
      custom_param: "value"
      advanced_feature: true

# Framework dependencies (shared across all pipelines)
framework:
  llm:
    model: "gpt-4o-mini"
    temperature: 0
    max_tokens: 1024
  embeddings:
    model: "text-embedding-3-small"
    dimension: 1536
```

## Pipeline Constructor Interface

### Standard Constructor Signature

All pipelines must follow this constructor pattern for compatibility with the dynamic loading system:

```python
class RAGPipeline:
    def __init__(self, llm_func, embedding_func, vector_store, config_manager, **kwargs):
        """
        Standard constructor interface for all RAG pipelines.
        
        Args:
            llm_func: Framework-provided LLM function for answer generation
            embedding_func: Framework-provided embedding function for vector operations
            vector_store: Framework-provided IRIS vector store instance
            config_manager: Framework-provided configuration manager
            **kwargs: Pipeline-specific parameters from config/pipelines.yaml
        """
        # Framework dependencies (automatically injected)
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self.vector_store = vector_store
        self.config_manager = config_manager
        
        # Pipeline-specific configuration from params section
        self.pipeline_params = kwargs
        
        # Extract specific parameters
        self.top_k = kwargs.get('top_k', 5)
        self.custom_param = kwargs.get('custom_param', 'default_value')
        
        # Initialize pipeline-specific components
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Override in subclasses for pipeline-specific initialization."""
        pass
```

### Framework Dependencies

The following dependencies are automatically injected by the [`PipelineFactory`](../iris_rag/pipelines/factory.py):

- **`llm_func`**: Function for LLM-based answer generation
- **`embedding_func`**: Function for generating text embeddings
- **`vector_store`**: IRIS vector store instance for database operations
- **`config_manager`**: Configuration manager for accessing global settings

## Creating Custom Pipelines

### Step 1: Implement Pipeline Class

Create a new pipeline class that follows the standard constructor interface:

```python
# my_package/pipelines.py
from typing import Dict, List, Any, Optional, Callable

class MyCustomRAGPipeline:
    def __init__(self, llm_func: Callable[[str], str], 
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 vector_store: Any,
                 config_manager: Any,
                 **kwargs):
        """Custom RAG pipeline implementation."""
        
        # Framework dependencies
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self.vector_store = vector_store
        self.config_manager = config_manager
        
        # Pipeline-specific parameters
        self.retrieval_method = kwargs.get('retrieval_method', 'semantic')
        self.rerank_enabled = kwargs.get('rerank_enabled', False)
        self.custom_threshold = kwargs.get('custom_threshold', 0.8)
        
        # Initialize custom components
        self._setup_custom_retriever()
    
    def _setup_custom_retriever(self):
        """Initialize custom retrieval components."""
        # Custom initialization logic here
        pass
    
    def execute(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Execute the custom RAG pipeline."""
        if top_k is None:
            top_k = self.pipeline_params.get('top_k', 5)
        
        # Custom pipeline logic here
        retrieved_docs = self._retrieve_documents(query, top_k)
        answer = self._generate_answer(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }
    
    def _retrieve_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Custom document retrieval logic."""
        # Implementation here
        pass
    
    def _generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM."""
        context = "\n".join([doc.get('content', '') for doc in documents])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        return self.llm_func(prompt)
```

### Step 2: Add Configuration

Add your pipeline to [`config/pipelines.yaml`](../config/pipelines.yaml):

```yaml
pipelines:
  # ... existing pipelines ...
  
  - name: "MyCustomRAG"
    module: "my_package.pipelines"
    class: "MyCustomRAGPipeline"
    enabled: true
    params:
      retrieval_method: "hybrid"
      rerank_enabled: true
      custom_threshold: 0.85
      top_k: 8
```

### Step 3: Use the Pipeline

The pipeline is now available through the dynamic loading system:

```python
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry

# Setup framework dependencies
framework_dependencies = {
    "llm_func": my_llm_function,
    "embedding_func": my_embedding_function,
    "vector_store": my_vector_store,
    "config_manager": my_config_manager
}

# Initialize services
config_service = PipelineConfigService()
module_loader = ModuleLoader()
pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
pipeline_registry = PipelineRegistry(pipeline_factory)

# Register all pipelines
pipeline_registry.register_pipelines()

# Use your custom pipeline
custom_pipeline = pipeline_registry.get_pipeline("MyCustomRAG")
result = custom_pipeline.execute("What is machine learning?")
```

## Usage Examples

### Basic Usage

```python
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry

# Initialize the dynamic pipeline system
config_service = PipelineConfigService()
module_loader = ModuleLoader()

# Setup framework dependencies
framework_dependencies = {
    "llm_func": lambda prompt: "Generated answer",
    "embedding_func": lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    "vector_store": my_vector_store,
    "config_manager": my_config_manager
}

# Create factory and registry
pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
pipeline_registry = PipelineRegistry(pipeline_factory)

# Register all enabled pipelines from config/pipelines.yaml
pipeline_registry.register_pipelines()

# List available pipelines
available_pipelines = pipeline_registry.list_pipeline_names()
print(f"Available pipelines: {available_pipelines}")

# Get a specific pipeline
basic_rag = pipeline_registry.get_pipeline("BasicRAG")
result = basic_rag.execute("What is the capital of France?")
```

### Evaluation Integration

The evaluation script automatically uses the dynamic pipeline system:

```bash
# Evaluate all enabled pipelines
python eval/execute_comprehensive_ragas_evaluation.py --pipelines ALL

# Evaluate specific pipelines
python eval/execute_comprehensive_ragas_evaluation.py --pipelines BasicRAG ColBERTRAG MyCustomRAG
```

### Programmatic Pipeline Selection

```python
def get_pipelines_for_evaluation(target_pipelines: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get pipeline instances for evaluation."""
    
    # Initialize dynamic loading system
    config_service = PipelineConfigService()
    module_loader = ModuleLoader()
    pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
    pipeline_registry = PipelineRegistry(pipeline_factory)
    
    # Register all pipelines
    pipeline_registry.register_pipelines()
    
    # Get pipelines based on selection
    if target_pipelines is None or (len(target_pipelines) == 1 and target_pipelines[0] == "ALL"):
        # Get all registered pipelines
        pipeline_names = pipeline_registry.list_pipeline_names()
        return {name: pipeline_registry.get_pipeline(name) for name in pipeline_names}
    else:
        # Get specific pipelines
        pipelines = {}
        for name in target_pipelines:
            if pipeline_registry.is_pipeline_registered(name):
                pipelines[name] = pipeline_registry.get_pipeline(name)
            else:
                print(f"Warning: Pipeline '{name}' not found in registry")
        return pipelines
```

## Error Handling

### Common Errors and Solutions

#### 1. Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'my_package'`

**Solution**: Ensure the module is installed and accessible in the Python path:

```python
import sys
sys.path.append('/path/to/your/package')
```

#### 2. Class Not Found Errors

**Error**: `AttributeError: module 'my_package.pipelines' has no attribute 'MyCustomRAGPipeline'`

**Solution**: Verify the class name matches exactly in both the module and configuration:

```yaml
# config/pipelines.yaml
- name: "MyCustomRAG"
  module: "my_package.pipelines"
  class: "MyCustomRAGPipeline"  # Must match exactly
```

#### 3. Constructor Signature Errors

**Error**: `TypeError: __init__() missing required positional argument`

**Solution**: Ensure your pipeline constructor follows the standard signature:

```python
def __init__(self, llm_func, embedding_func, vector_store, config_manager, **kwargs):
    # Implementation
```

#### 4. Configuration Validation Errors

**Error**: `PipelineConfigurationError: Configuration must contain a 'pipelines' list`

**Solution**: Verify your YAML structure is correct:

```yaml
pipelines:  # Must be a list
  - name: "PipelineName"
    # ... other fields
```

### Logging and Debugging

Enable debug logging to troubleshoot pipeline loading issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# The dynamic loading system will now output detailed debug information
```

## Performance Considerations

### Module Caching

The [`ModuleLoader`](../iris_rag/utils/module_loader.py) caches imported modules to avoid repeated imports:

```python
# Modules are cached after first import
loader = ModuleLoader()
pipeline_class1 = loader.load_pipeline_class("iris_rag.pipelines.basic", "BasicRAGPipeline")
pipeline_class2 = loader.load_pipeline_class("iris_rag.pipelines.basic", "BasicRAGPipeline")  # Uses cache
```

### Lazy Loading

Pipelines are only loaded when requested, not during system initialization:

```python
# Pipelines are registered but not instantiated
pipeline_registry.register_pipelines()

# Pipeline is instantiated only when first accessed
pipeline = pipeline_registry.get_pipeline("BasicRAG")  # Instantiated here
```

### Configuration Caching

Pipeline definitions are cached after first load to avoid re-parsing YAML:

```python
# Configuration is cached in PipelineFactory
factory = PipelineFactory(config_service, module_loader, framework_dependencies)
# Subsequent pipeline creations use cached configuration
```

## Migration from Hard-coded Pipelines

### Before (Hard-coded)

```python
# Old approach - hard-coded pipeline instantiation
pipeline_classes = {
    'BasicRAG': BasicRAGPipeline,
    'ColBERTRAG': ColBERTRAGPipeline,
    'CRAG': CRAGPipeline
}

pipelines = {}
for name, pipeline_class in pipeline_classes.items():
    pipelines[name] = pipeline_class(
        connection_manager=connection_manager,
        config_manager=config_manager,
        llm_func=llm_func
    )
```

### After (Dynamic Loading)

```python
# New approach - dynamic loading from configuration
config_service = PipelineConfigService()
module_loader = ModuleLoader()
pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
pipeline_registry = PipelineRegistry(pipeline_factory)

# All pipelines loaded from config/pipelines.yaml
pipeline_registry.register_pipelines()
pipelines = {name: pipeline_registry.get_pipeline(name) 
            for name in pipeline_registry.list_pipeline_names()}
```

### Migration Steps

1. **Add pipeline definitions** to [`config/pipelines.yaml`](../config/pipelines.yaml)
2. **Update pipeline constructors** to follow the standard signature
3. **Replace hard-coded instantiation** with dynamic loading
4. **Test pipeline loading** and functionality
5. **Remove hard-coded pipeline imports** and classes

## Testing

### Unit Tests

Test individual components of the dynamic pipeline system:

```python
# tests/test_dynamic_pipeline_system.py
import pytest
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.pipelines.factory import PipelineFactory

def test_pipeline_config_loading():
    """Test loading pipeline configuration from YAML."""
    service = PipelineConfigService()
    definitions = service.load_pipeline_definitions("config/pipelines.yaml")
    assert isinstance(definitions, list)
    assert len(definitions) > 0

def test_module_loading():
    """Test dynamic module loading."""
    loader = ModuleLoader()
    pipeline_class = loader.load_pipeline_class(
        "iris_rag.pipelines.basic", 
        "BasicRAGPipeline"
    )
    assert pipeline_class is not None

def test_pipeline_factory():
    """Test pipeline instantiation through factory."""
    # Setup test dependencies
    framework_deps = {
        "llm_func": lambda x: "test",
        "embedding_func": lambda x: [[0.1, 0.2]],
        "vector_store": None,
        "config_manager": None
    }
    
    config_service = PipelineConfigService()
    module_loader = ModuleLoader()
    factory = PipelineFactory(config_service, module_loader, framework_deps)
    
    pipeline = factory.create_pipeline("BasicRAG")
    assert pipeline is not None
```

### Integration Tests

Test the complete dynamic pipeline system:

```python
def test_end_to_end_dynamic_loading():
    """Test complete pipeline loading and execution."""
    # Initialize system
    config_service = PipelineConfigService()
    module_loader = ModuleLoader()
    pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
    pipeline_registry = PipelineRegistry(pipeline_factory)
    
    # Register pipelines
    pipeline_registry.register_pipelines()
    
    # Test pipeline execution
    pipeline = pipeline_registry.get_pipeline("BasicRAG")
    result = pipeline.execute("Test query")
    
    assert "answer" in result
    assert "retrieved_documents" in result
```

## Best Practices

### Configuration Management

1. **Use descriptive pipeline names** that clearly indicate their purpose
2. **Group related parameters** logically in the params section
3. **Document parameter meanings** in pipeline class docstrings
4. **Use consistent parameter naming** across similar pipelines

### Error Handling

1. **Implement graceful degradation** when pipelines fail to load
2. **Provide clear error messages** for configuration issues
3. **Log detailed information** for debugging pipeline problems
4. **Validate configuration** before attempting to load pipelines

### Performance

1. **Cache expensive operations** in pipeline initialization
2. **Use lazy loading** for optional components
3. **Minimize framework dependency overhead** in pipeline constructors
4. **Profile pipeline loading times** for optimization opportunities

### Security

1. **Validate module paths** to prevent arbitrary code execution
2. **Sanitize configuration inputs** to prevent injection attacks
3. **Use trusted sources** for external pipeline packages
4. **Implement access controls** for pipeline configuration files

## Troubleshooting

### Debug Checklist

When pipelines fail to load, check:

1. **Configuration syntax**: Ensure YAML is valid and properly formatted
2. **Module availability**: Verify the module can be imported manually
3. **Class existence**: Confirm the class exists in the specified module
4. **Constructor signature**: Ensure the constructor matches the standard interface
5. **Dependencies**: Check that all required packages are installed
6. **Permissions**: Verify file system permissions for configuration files

### Common Issues

#### Pipeline Not Found in Registry

```python
# Check if pipeline is enabled in configuration
config_service = PipelineConfigService()
definitions = config_service.load_pipeline_definitions("config/pipelines.yaml")
enabled_pipelines = [d for d in definitions if d.get('enabled', True)]
print(f"Enabled pipelines: {[d['name'] for d in enabled_pipelines]}")
```

#### Import Path Issues

```python
# Test module import manually
try:
    import importlib
    module = importlib.import_module("my_package.pipelines")
    pipeline_class = getattr(module, "MyCustomRAGPipeline")
    print(f"Successfully imported: {pipeline_class}")
except Exception as e:
    print(f"Import failed: {e}")
```

#### Configuration Validation

```python
# Validate configuration structure
import yaml
with open("config/pipelines.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
assert 'pipelines' in config, "Missing 'pipelines' key"
assert isinstance(config['pipelines'], list), "'pipelines' must be a list"

for pipeline in config['pipelines']:
    required_fields = ['name', 'module', 'class']
    for field in required_fields:
        assert field in pipeline, f"Missing required field: {field}"
```

## Related Documentation

- [Dynamic Pipeline Configuration Design](DYNAMIC_PIPELINE_CONFIGURATION_DESIGN.md) - Original design document
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Developer Guide](DEVELOPER_GUIDE.md) - General development guidelines
- [Configuration Guide](USER_GUIDE.md#configuration) - Configuration management