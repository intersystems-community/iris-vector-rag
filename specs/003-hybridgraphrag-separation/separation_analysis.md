# HybridGraphRAG Separation Architecture Analysis

## 🎯 Strategic Goals

1. **Dependency Isolation**: Remove iris-vector-graph dependency from core rag-templates
2. **Modular Architecture**: Enable external pipeline plugins via declarative configuration
3. **Plugin Ecosystem**: Create foundation for community-contributed pipeline extensions
4. **Maintainability**: Separate concerns between core framework and specialized implementations

## 🔍 Current Dependency Analysis

### HybridGraphRAG Dependencies
```
HybridGraphRAG Pipeline Dependencies:
├── iris-vector-graph (EXTERNAL - Heavy dependency)
│   ├── HNSW optimization libraries
│   ├── RRF fusion algorithms
│   ├── Specialized IRIS graph operations
│   └── Advanced vector operations
├── rag-templates (CORE)
│   ├── GraphRAGPipeline (base class)
│   ├── EntityExtractionService
│   ├── ConnectionManager
│   ├── ConfigurationManager
│   └── SchemaManager ecosystem
└── HybridGraphRAGSchemaManager (SPECIALIZED)
    ├── iris_graph_core table management
    ├── HNSW index creation
    └── Advanced schema operations
```

### Coupling Points to Break
1. **Direct Import Dependencies**: `from iris_graph_core import ...`
2. **Schema Manager Coupling**: HybridGraphRAGSchemaManager embedded in core
3. **Pipeline Factory Integration**: Hard-coded in ValidatedPipelineFactory
4. **Configuration Coupling**: HybridGraphRAG config embedded in pipelines.yaml

## 🏗️ Proposed Separation Architecture

### Repository Structure
```
Ecosystem Architecture:
├── rag-templates/ (CORE FRAMEWORK)
│   ├── Plugin System Infrastructure
│   ├── Pipeline Base Classes
│   ├── Configuration Framework
│   ├── Schema Management Framework
│   └── Plugin Discovery & Loading
├── hybridgraphrag-plugin/ (EXTERNAL PLUGIN)
│   ├── HybridGraphRAG Implementation
│   ├── iris-vector-graph Integration
│   ├── Specialized Schema Managers
│   └── Plugin Manifest & Config
└── [other-plugins]/ (FUTURE EXTENSIONS)
    ├── Advanced RAG techniques
    ├── Domain-specific pipelines
    └── Vendor-specific integrations
```

### Plugin Architecture Requirements

#### 1. Plugin Manifest System
```yaml
# hybridgraphrag-plugin/plugin.yaml
plugin:
  name: "HybridGraphRAG"
  version: "1.0.0"
  description: "Advanced hybrid search with iris-vector-graph integration"

  dependencies:
    external:
      - "iris-vector-graph>=2.0.0"
    rag_templates:
      minimum_version: "1.0.0"
      required_features: ["schema_manager_factory", "pipeline_base"]

  provides:
    pipelines:
      - name: "HybridGraphRAG"
        class: "hybridgraphrag.HybridGraphRAGPipeline"
        schema_manager: "hybridgraphrag.HybridGraphRAGSchemaManager"

    schema_requirements:
      tables:
        - "KG_NODEEMBEDDINGS_OPTIMIZED"
        - "RDF_EDGES"
        - "RDF_LABELS"
        - "RDF_PROPS"
      indexes:
        - name: "idx_kg_nodeembeddings_vector"
          type: "HNSW"
          required: false

    configuration:
      schema: "hybridgraphrag/config_schema.json"
      defaults: "hybridgraphrag/defaults.yaml"
```

#### 2. Plugin Discovery & Loading
```python
# rag-templates plugin system
class PluginManager:
    def discover_plugins(self) -> List[PluginManifest]:
        """Discover available plugins from:
        - Python packages with rag_templates_plugin entry points
        - Local plugin directories
        - Configuration-specified plugin paths
        """

    def load_plugin(self, plugin_name: str) -> Plugin:
        """Dynamically load plugin with dependency validation"""

    def validate_plugin_compatibility(self, plugin: Plugin) -> ValidationResult:
        """Ensure plugin is compatible with current rag-templates version"""
```

#### 3. Plugin Interface Contract
```python
# rag-templates/iris_rag/plugins/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class RAGPlugin(ABC):
    """Interface that all RAG plugins must implement."""

    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """Return plugin manifest with capabilities and requirements."""

    @abstractmethod
    def get_pipeline_classes(self) -> Dict[str, Type[RAGPipeline]]:
        """Return mapping of pipeline names to implementation classes."""

    @abstractmethod
    def get_schema_managers(self) -> Dict[str, Type[SchemaManager]]:
        """Return specialized schema managers provided by plugin."""

    @abstractmethod
    def validate_environment(self) -> ValidationResult:
        """Validate that plugin dependencies and environment are satisfied."""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure plugin with user-provided settings."""
```

## 🔧 Required rag-templates Modifications

### 1. Plugin System Infrastructure

#### Plugin Manager Integration
```python
# iris_rag/plugins/manager.py
class PluginManager:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.loaded_plugins: Dict[str, RAGPlugin] = {}
        self.plugin_manifests: Dict[str, PluginManifest] = {}

    def initialize_plugins(self):
        """Load and initialize all configured plugins."""
        enabled_plugins = self.config_manager.get("plugins:enabled", [])

        for plugin_name in enabled_plugins:
            try:
                plugin = self.load_plugin(plugin_name)
                if plugin.validate_environment().is_valid:
                    self.loaded_plugins[plugin_name] = plugin
                    self._register_plugin_capabilities(plugin)
            except Exception as e:
                logger.warning(f"Failed to load plugin {plugin_name}: {e}")
```

#### Enhanced Pipeline Factory
```python
# iris_rag/pipelines/factory.py - Enhanced for plugins
class EnhancedPipelineFactory:
    def __init__(self, config_manager, connection_manager, plugin_manager):
        self.plugin_manager = plugin_manager
        # ... existing initialization

    def create_pipeline(self, pipeline_type: str, **kwargs) -> RAGPipeline:
        # First check core pipelines
        if pipeline_type in self.core_pipelines:
            return self._create_core_pipeline(pipeline_type, **kwargs)

        # Then check plugin-provided pipelines
        for plugin in self.plugin_manager.loaded_plugins.values():
            if pipeline_type in plugin.get_pipeline_classes():
                return self._create_plugin_pipeline(plugin, pipeline_type, **kwargs)

        raise ValueError(f"Pipeline type {pipeline_type} not found in core or plugins")
```

### 2. Configuration System Extensions

#### Plugin Configuration Schema
```yaml
# iris_rag/config/default_config.yaml - Plugin section
plugins:
  enabled:
    - "hybridgraphrag"  # Enable external plugins

  discovery:
    search_paths:
      - "./plugins"
      - "~/.rag-templates/plugins"
    entry_points: true  # Use Python entry points for discovery

  configuration:
    hybridgraphrag:
      iris_graph_core:
        auto_setup: true
        fallback_to_basic: true
      fusion_weights: [0.4, 0.3, 0.3]
```

#### Dynamic Pipeline Registration
```yaml
# config/pipelines.yaml - Plugin-aware configuration
pipelines:
  # Core pipelines
  - name: "BasicRAG"
    module: "iris_rag.pipelines.basic"
    class: "BasicRAGPipeline"
    type: "core"

  # Plugin-provided pipelines (dynamically loaded)
  - name: "HybridGraphRAG"
    plugin: "hybridgraphrag"  # Provided by plugin instead of module
    type: "plugin"
    enabled: true
    config_override:  # Plugin-specific overrides
      top_k: 10
      max_depth: 3
```

### 3. Schema Manager Plugin Support

#### Plugin Schema Manager Registry
```python
# iris_rag/storage/schema_manager.py - Enhanced factory
@classmethod
def create_schema_manager(cls, pipeline_type: str, connection_manager, config_manager):
    # Check if pipeline is provided by plugin
    plugin_manager = getattr(config_manager, 'plugin_manager', None)
    if plugin_manager:
        for plugin in plugin_manager.loaded_plugins.values():
            schema_managers = plugin.get_schema_managers()
            if pipeline_type in schema_managers:
                return schema_managers[pipeline_type](connection_manager, config_manager)

    # Fall back to core schema manager logic
    return cls._create_core_schema_manager(pipeline_type, connection_manager, config_manager)
```

## 🚀 HybridGraphRAG Plugin Implementation

### Plugin Package Structure
```
hybridgraphrag-plugin/
├── setup.py  # Entry point registration
├── plugin.yaml  # Plugin manifest
├── hybridgraphrag/
│   ├── __init__.py
│   ├── pipeline.py  # HybridGraphRAGPipeline
│   ├── schema_manager.py  # HybridGraphRAGSchemaManager
│   ├── discovery.py  # iris-vector-graph integration
│   └── config/
│       ├── defaults.yaml
│       └── schema.json
└── tests/
    ├── test_plugin.py
    └── integration/
```

### Entry Point Registration
```python
# setup.py
setup(
    name="hybridgraphrag-plugin",
    version="1.0.0",
    packages=["hybridgraphrag"],
    entry_points={
        "rag_templates_plugins": [
            "hybridgraphrag = hybridgraphrag:HybridGraphRAGPlugin"
        ]
    },
    install_requires=[
        "rag-templates>=1.0.0",
        "iris-vector-graph>=2.0.0"
    ]
)
```

### Plugin Implementation
```python
# hybridgraphrag/__init__.py
from iris_rag.plugins.interface import RAGPlugin
from .pipeline import HybridGraphRAGPipeline
from .schema_manager import HybridGraphRAGSchemaManager

class HybridGraphRAGPlugin(RAGPlugin):
    def get_pipeline_classes(self):
        return {"HybridGraphRAG": HybridGraphRAGPipeline}

    def get_schema_managers(self):
        return {"HybridGraphRAG": HybridGraphRAGSchemaManager}

    def validate_environment(self):
        try:
            import iris_graph_core
            return ValidationResult(True, "iris-vector-graph available")
        except ImportError:
            return ValidationResult(False, "iris-vector-graph not installed")
```

## 📊 Benefits Analysis

### Advantages of Separation
1. **Dependency Isolation**: Core framework remains lightweight
2. **Optional Features**: Users only install what they need
3. **Independent Versioning**: Plugin can evolve independently
4. **Community Ecosystem**: Enables third-party pipeline contributions
5. **Easier Maintenance**: Clear separation of concerns
6. **Testing Isolation**: Plugin tests don't affect core CI/CD

### Challenges to Address
1. **Discovery Complexity**: Plugin discovery and loading infrastructure
2. **Version Compatibility**: Managing rag-templates API compatibility
3. **Configuration Complexity**: Plugin-specific configuration integration
4. **Documentation**: Separate docs for plugin ecosystem
5. **Testing**: Integration testing across repositories

## 🎯 Implementation Priority

1. **Phase 1**: Core plugin infrastructure in rag-templates
2. **Phase 2**: Extract HybridGraphRAG to separate repository
3. **Phase 3**: Plugin discovery and configuration integration
4. **Phase 4**: Documentation and community guidelines

This architecture provides a clean separation while maintaining the declarative configuration approach you requested, and sets up rag-templates as a true platform for RAG pipeline development.