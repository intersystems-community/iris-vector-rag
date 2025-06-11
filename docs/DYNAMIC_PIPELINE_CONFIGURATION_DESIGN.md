# Dynamic Pipeline Configuration System Design

> **✅ IMPLEMENTATION STATUS: COMPLETE**
>
> This design document has been fully implemented. For current usage documentation, see:
> - **[Dynamic Pipeline System Guide](DYNAMIC_PIPELINE_SYSTEM.md)** - Comprehensive usage and developer documentation
> - **[Main README - Dynamic Pipeline Configuration](../README.md#-dynamic-pipeline-configuration)** - Quick start guide
> - **[Pipeline Configuration File](../config/pipelines.yaml)** - Live configuration

## Overview

This document outlines the architectural design for a configuration-driven system that dynamically loads RAG pipelines at runtime. The system replaces hard-coded pipeline instantiation with a flexible, extensible configuration-based approach.

**Implementation Status**: ✅ **COMPLETE** - All components have been implemented and are in active use.

## Current State Analysis

### Hard-coded Pipeline Instantiation

Currently, in [`eval/execute_comprehensive_ragas_evaluation.py`](eval/execute_comprehensive_ragas_evaluation.py:249-257), pipelines are hard-coded:

```python
pipeline_classes = {
    'BasicRAG': BasicRAGPipeline,
    'HyDERAG': HyDERAGPipeline,
    'CRAG': CRAGPipeline,
    'ColBERTRAG': ColBERTRAGPipeline,
    'NodeRAG': NodeRAGPipeline,
    'GraphRAG': GraphRAGPipeline,
    'HybridIFindRAG': HybridIFindRAGPipeline
}
```

### Pipeline Constructor Pattern

Existing pipelines follow a consistent constructor pattern:

```python
def __init__(self, connection_manager: ConnectionManager, 
             config_manager: ConfigurationManager,
             llm_func: Optional[Callable[[str], str]] = None, 
             vector_store=None):
```

## Proposed Architecture

### 1. Configuration Schema Design

#### YAML Configuration Structure

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
    
  - name: "HyDERAG"
    module: "iris_rag.pipelines.hyde"
    class: "HyDERAGPipeline"
    enabled: true
    params:
      top_k: 5
      use_hypothetical_doc: true
      temperature: 0.1
    
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

# Framework dependencies (not per-pipeline)
framework:
  llm:
    model: "gpt-4o-mini"
    temperature: 0
    max_tokens: 1024
  embeddings:
    model: "text-embedding-3-small"
```

#### JSON Schema Alternative

```json
{
  "pipelines": [
    {
      "name": "BasicRAG",
      "module": "iris_rag.pipelines.basic",
      "class": "BasicRAGPipeline",
      "enabled": true,
      "params": {
        "top_k": 5,
        "chunk_size": 1000,
        "similarity_threshold": 0.7
      }
    }
  ],
  "framework": {
    "llm": {
      "model": "gpt-4o-mini",
      "temperature": 0,
      "max_tokens": 1024
    }
  }
}
```

### 2. Dynamic Loading System Architecture

#### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Loader System                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Configuration   │  │ Dynamic Module  │  │ Pipeline        │ │
│  │ Parser          │  │ Loader          │  │ Factory         │ │
│  │                 │  │                 │  │                 │ │
│  │ - YAML/JSON     │  │ - importlib     │  │ - Instantiation │ │
│  │ - Validation    │  │ - Error Handling│  │ - Dependency    │ │
│  │ - Schema Check  │  │ - Module Cache  │  │   Injection     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Pipeline        │  │ Error Handler   │  │ Registry        │ │
│  │ Registry        │  │                 │  │ Manager         │ │
│  │                 │  │ - Import Errors │  │                 │ │
│  │ - Name Mapping  │  │ - Class Errors  │  │ - Available     │ │
│  │ - Instance Cache│  │ - Init Errors   │  │   Pipelines     │ │
│  │ - Lifecycle     │  │ - Logging       │  │ - Filtering     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Service Boundaries

1. **Configuration Service**: Handles YAML/JSON parsing and validation
2. **Module Loading Service**: Manages dynamic imports and caching
3. **Pipeline Factory Service**: Creates and configures pipeline instances
4. **Registry Service**: Maintains pipeline inventory and lifecycle

### 3. Implementation Design

#### Core Classes

```python
# iris_rag/core/pipeline_loader.py

from typing import Dict, List, Any, Optional, Type
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration for a single pipeline."""
    name: str
    module: str
    class_name: str
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

class PipelineConfigurationError(Exception):
    """Raised when pipeline configuration is invalid."""
    pass

class PipelineLoadingError(Exception):
    """Raised when pipeline loading fails."""
    pass

class DynamicPipelineLoader:
    """
    Loads and manages RAG pipelines dynamically from configuration.
    
    Responsibilities:
    - Parse pipeline configuration files
    - Dynamically import pipeline modules
    - Instantiate pipelines with proper dependency injection
    - Handle errors gracefully with detailed logging
    - Maintain pipeline registry
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._pipeline_configs: List[PipelineConfig] = []
        self._pipeline_registry: Dict[str, Any] = {}
        self._module_cache: Dict[str, Any] = {}
        
    def load_configuration(self) -> None:
        """Load and validate pipeline configuration."""
        
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipeline names."""
        
    def load_pipeline(self, name: str, **framework_deps) -> Any:
        """Load a specific pipeline by name."""
        
    def load_all_pipelines(self, **framework_deps) -> Dict[str, Any]:
        """Load all enabled pipelines."""
        
    def _import_pipeline_class(self, config: PipelineConfig) -> Type:
        """Dynamically import pipeline class."""
        
    def _instantiate_pipeline(self, pipeline_class: Type, 
                            config: PipelineConfig, 
                            **framework_deps) -> Any:
        """Instantiate pipeline with dependency injection."""
```

#### Configuration Parser

```python
# iris_rag/core/config_parser.py

import yaml
import json
from typing import Dict, List, Any
from pathlib import Path

class ConfigurationParser:
    """Parses and validates pipeline configuration files."""
    
    @staticmethod
    def parse_yaml(file_path: Path) -> Dict[str, Any]:
        """Parse YAML configuration file."""
        
    @staticmethod
    def parse_json(file_path: Path) -> Dict[str, Any]:
        """Parse JSON configuration file."""
        
    @staticmethod
    def validate_schema(config: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        
    @staticmethod
    def extract_pipeline_configs(config: Dict[str, Any]) -> List[PipelineConfig]:
        """Extract pipeline configurations from parsed config."""
```

### 4. Integration Strategy

#### Modified Evaluation Script

```python
# eval/execute_comprehensive_ragas_evaluation.py (modified)

def execute_pipeline_evaluations(queries: List[Dict[str, Any]],
                                connection_manager: ConnectionManager,
                                config_manager: ConfigurationManager,
                                target_pipelines: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute evaluations on dynamically loaded RAG pipelines.
    """
    # Initialize dynamic pipeline loader
    loader = DynamicPipelineLoader("config/pipelines.yaml")
    loader.load_configuration()
    
    # Prepare framework dependencies
    def create_llm_function():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1024
        )
        return lambda prompt: llm.invoke(prompt).content
    
    framework_deps = {
        'connection_manager': connection_manager,
        'config_manager': config_manager,
        'llm_func': create_llm_function(),
        'vector_store': None  # Optional
    }
    
    # Load pipelines based on target selection
    if target_pipelines:
        pipelines = {}
        for name in target_pipelines:
            try:
                pipelines[name] = loader.load_pipeline(name, **framework_deps)
            except PipelineLoadingError as e:
                logger.error(f"Failed to load pipeline {name}: {e}")
    else:
        pipelines = loader.load_all_pipelines(**framework_deps)
    
    # Execute evaluations
    all_results = {}
    for pipeline_name, pipeline in pipelines.items():
        logger.info(f"🔄 Evaluating {pipeline_name} pipeline...")
        try:
            results = evaluate_single_pipeline(pipeline, queries)
            all_results[pipeline_name] = results
            logger.info(f"✅ {pipeline_name} evaluation completed")
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {pipeline_name}: {e}")
            all_results[pipeline_name] = []
    
    return all_results
```

#### CLI Integration

```python
# Modified argument parsing
def main():
    parser = argparse.ArgumentParser(description='Execute comprehensive RAGAS evaluation')
    parser.add_argument('--num-queries', type=int, help='Number of queries to run')
    parser.add_argument('--pipelines', nargs='+', 
                       help='Specific pipelines to evaluate (names from config)')
    parser.add_argument('--config', default='config/pipelines.yaml',
                       help='Pipeline configuration file path')
    parser.add_argument('--list-pipelines', action='store_true',
                       help='List available pipelines and exit')
    
    args = parser.parse_args()
    
    if args.list_pipelines:
        loader = DynamicPipelineLoader(args.config)
        loader.load_configuration()
        available = loader.get_available_pipelines()
        print("Available pipelines:")
        for name in available:
            print(f"  - {name}")
        return
    
    # Continue with evaluation...
```

### 5. Pipeline Constructor Compatibility

#### Standardized Constructor Interface

All pipelines should follow this constructor pattern for compatibility:

```python
class RAGPipeline:
    def __init__(self, connection_manager: ConnectionManager,
                 config_manager: ConfigurationManager,
                 llm_func: Optional[Callable[[str], str]] = None,
                 vector_store=None,
                 **pipeline_specific_kwargs):
        """
        Standard constructor interface for all RAG pipelines.
        
        Args:
            connection_manager: Framework-provided database connections
            config_manager: Framework-provided configuration management
            llm_func: Framework-provided LLM function
            vector_store: Framework-provided vector store (optional)
            **pipeline_specific_kwargs: Pipeline-specific parameters from config
        """
        # Framework dependencies
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.llm_func = llm_func
        self.vector_store = vector_store
        
        # Pipeline-specific configuration
        self.pipeline_params = pipeline_specific_kwargs
        
        # Initialize pipeline-specific components
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Override in subclasses for pipeline-specific initialization."""
        pass
```

#### Migration Strategy for Existing Pipelines

1. **Backward Compatibility**: Existing constructors remain functional
2. **Gradual Migration**: Add `**kwargs` parameter to accept config-driven params
3. **Parameter Extraction**: Extract pipeline-specific params from kwargs
4. **Configuration Integration**: Use config-driven params alongside existing logic

### 6. Error Handling and Logging

#### Comprehensive Error Management

```python
class PipelineErrorHandler:
    """Centralized error handling for pipeline loading."""
    
    @staticmethod
    def handle_import_error(module_name: str, error: ImportError) -> None:
        """Handle module import failures."""
        
    @staticmethod
    def handle_class_error(module_name: str, class_name: str, error: AttributeError) -> None:
        """Handle class not found errors."""
        
    @staticmethod
    def handle_instantiation_error(pipeline_name: str, error: Exception) -> None:
        """Handle pipeline instantiation failures."""
```

#### Logging Strategy

- **INFO**: Successful pipeline loading and configuration parsing
- **WARNING**: Disabled pipelines, missing optional dependencies
- **ERROR**: Failed imports, instantiation errors, configuration validation failures
- **DEBUG**: Detailed parameter injection, module caching information

### 7. Extensibility Features

#### External Package Support

The system supports pipelines from external packages:

```yaml
pipelines:
  - name: "ExternalAdvancedRAG"
    module: "advanced_rag_package.pipelines"
    class: "AdvancedRAGPipeline"
    enabled: true
    params:
      external_param: "value"
```

#### Plugin Architecture

Future extension could include a plugin discovery mechanism:

```python
class PipelinePluginManager:
    """Manages pipeline plugins and extensions."""
    
    def discover_plugins(self) -> List[PipelineConfig]:
        """Auto-discover pipeline plugins."""
        
    def register_plugin(self, plugin_config: PipelineConfig) -> None:
        """Register a new pipeline plugin."""
```

### 8. Testing Strategy

#### Unit Tests

```python
# tests/test_dynamic_pipeline_loader.py

class TestDynamicPipelineLoader:
    def test_load_configuration_valid_yaml(self):
        """Test loading valid YAML configuration."""
        
    def test_load_configuration_invalid_yaml(self):
        """Test handling of invalid YAML."""
        
    def test_import_existing_pipeline(self):
        """Test importing existing pipeline class."""
        
    def test_import_nonexistent_pipeline(self):
        """Test handling of non-existent pipeline."""
        
    def test_instantiate_pipeline_with_params(self):
        """Test pipeline instantiation with parameters."""
        
    def test_filter_target_pipelines(self):
        """Test filtering pipelines by target list."""
```

#### Integration Tests

```python
# tests/test_integration/test_dynamic_evaluation.py

class TestDynamicEvaluation:
    def test_end_to_end_evaluation_with_config(self):
        """Test complete evaluation using configuration file."""
        
    def test_cli_pipeline_selection(self):
        """Test CLI pipeline selection functionality."""
```

### 9. Performance Considerations

#### Optimization Strategies

1. **Module Caching**: Cache imported modules to avoid repeated imports
2. **Lazy Loading**: Load pipelines only when needed
3. **Configuration Caching**: Cache parsed configuration to avoid re-parsing
4. **Instance Reuse**: Optionally reuse pipeline instances for multiple queries

#### Memory Management

```python
class PipelineInstanceManager:
    """Manages pipeline instance lifecycle and memory usage."""
    
    def __init__(self, max_instances: int = 10):
        self.max_instances = max_instances
        self._instances: Dict[str, Any] = {}
        
    def get_or_create_instance(self, name: str, **deps) -> Any:
        """Get existing instance or create new one."""
        
    def cleanup_instances(self) -> None:
        """Clean up unused instances."""
```

### 10. Migration Plan

#### Phase 1: Core Infrastructure
- Implement `DynamicPipelineLoader` class
- Create configuration parser and validator
- Add basic error handling and logging

#### Phase 2: Integration
- Modify [`eval/execute_comprehensive_ragas_evaluation.py`](eval/execute_comprehensive_ragas_evaluation.py) to use dynamic loader
- Update CLI argument parsing
- Create default configuration file

#### Phase 3: Pipeline Migration
- Update existing pipeline constructors for compatibility
- Create migration guide for external pipelines
- Add comprehensive test coverage

#### Phase 4: Advanced Features
- Implement plugin discovery
- Add performance optimizations
- Create documentation and examples

## Benefits

### Immediate Benefits
1. **Flexibility**: Easy addition/removal of pipelines without code changes
2. **Configuration Management**: Centralized pipeline parameter management
3. **Extensibility**: Support for external pipeline packages
4. **Maintainability**: Cleaner separation of concerns

### Long-term Benefits
1. **Scalability**: Easy scaling to hundreds of pipeline variants
2. **Testing**: Simplified A/B testing of pipeline configurations
3. **Deployment**: Environment-specific pipeline configurations
4. **Integration**: Seamless integration with external systems

## Conclusion

This dynamic pipeline configuration system provides a robust, extensible foundation for managing RAG pipelines. The modular architecture ensures clean separation of concerns while maintaining backward compatibility and supporting future growth.

The system transforms hard-coded pipeline management into a flexible, configuration-driven approach that supports the project's goals of modularity, testability, and extensibility.