# Existing Plugin Infrastructure Analysis

## 🎯 Summary

You've already implemented a sophisticated plugin architecture! The foundation for pipeline plugins is already in place. Here's what exists and what needs to be adapted for HybridGraphRAG separation:

## ✅ Already Implemented

### 1. Dynamic Module Loading (`iris_rag/utils/module_loader.py`)
- ✅ Dynamic import with validation
- ✅ RAGPipeline subclass validation
- ✅ Module caching for performance
- ✅ Comprehensive error handling

### 2. Pipeline Factory System (`iris_rag/pipelines/factory.py`)
- ✅ Configuration-driven pipeline creation
- ✅ Dependency injection framework
- ✅ Pipeline lifecycle management
- ✅ Framework dependencies support

### 3. Configuration Service (`iris_rag/config/pipeline_config_service.py`)
- ✅ YAML-based pipeline definitions
- ✅ Schema validation
- ✅ Project-relative path resolution
- ✅ Comprehensive validation framework

### 4. Ontology Plugin System (`iris_rag/ontology/plugins/`)
- ✅ General-purpose plugin architecture
- ✅ Dynamic plugin discovery
- ✅ Configuration-based plugin creation
- ✅ Domain-agnostic design patterns

### 5. Declarative Configuration (`config/pipelines.yaml`)
- ✅ Pipeline declarations with module/class specification
- ✅ Parameter injection
- ✅ Enable/disable flags
- ✅ Schema requirements support (partially)

## 🔧 What Needs Extension for HybridGraphRAG Separation

### 1. External Package Plugin Support

Currently, the system loads modules from within the `iris_rag` package. For external plugin packages, we need:

```python
# Enhanced ModuleLoader to support external packages
class ModuleLoader:
    def load_external_pipeline_class(self, package_name: str, module_path: str, class_name: str) -> Type:
        """Load pipeline class from external package."""
        full_module_path = f"{package_name}.{module_path}"
        return self.load_pipeline_class(full_module_path, class_name)

    def discover_plugin_packages(self) -> List[str]:
        """Discover packages with rag_templates_plugin entry points."""
        import pkg_resources
        plugins = []
        for entry_point in pkg_resources.iter_entry_points('rag_templates_plugins'):
            plugins.append(entry_point.module_name)
        return plugins
```

### 2. Plugin Manifest Integration

Extend the pipeline configuration to support plugin manifests:

```yaml
# config/pipelines.yaml - Plugin support
pipelines:
  - name: "HybridGraphRAG"
    type: "plugin"  # NEW: Indicate this is a plugin-provided pipeline
    plugin: "hybridgraphrag-plugin"  # NEW: Plugin package name
    class: "HybridGraphRAGPipeline"  # Class within plugin
    enabled: true
    schema_requirements:
      external_dependencies:  # NEW: External dependency declarations
        - "iris-vector-graph>=2.0.0"
      schema_manager: "HybridGraphRAGSchemaManager"
      tables:
        - "KG_NODEEMBEDDINGS_OPTIMIZED"
        - "RDF_EDGES"
        - "RDF_LABELS"
        - "RDF_PROPS"
```

### 3. Plugin Discovery and Validation

Add plugin discovery to the configuration service:

```python
# Enhanced PipelineConfigService
class PipelineConfigService:
    def discover_plugin_pipelines(self) -> List[Dict]:
        """Discover pipelines from installed plugin packages."""
        plugin_pipelines = []

        for entry_point in pkg_resources.iter_entry_points('rag_templates_plugins'):
            try:
                plugin_class = entry_point.load()
                plugin = plugin_class()
                manifest = plugin.get_manifest()

                # Convert plugin manifest to pipeline definitions
                for pipeline_info in manifest.provides.get('pipelines', []):
                    plugin_pipelines.append({
                        'name': pipeline_info['name'],
                        'type': 'plugin',
                        'plugin': entry_point.name,
                        'class': pipeline_info['class'],
                        'enabled': True,
                        'schema_requirements': manifest.schema_requirements
                    })
            except Exception as e:
                logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

        return plugin_pipelines
```

## 🚀 Proposed HybridGraphRAG Plugin Implementation

### 1. Plugin Package Structure
```
hybridgraphrag-plugin/
├── setup.py                    # Entry point registration
├── pyproject.toml              # Modern Python packaging
├── hybridgraphrag/
│   ├── __init__.py             # Plugin class
│   ├── pipeline.py             # HybridGraphRAGPipeline (moved from rag-templates)
│   ├── schema_manager.py       # HybridGraphRAGSchemaManager (moved)
│   ├── discovery.py            # iris-vector-graph integration (moved)
│   └── retrieval.py            # Retrieval methods (moved)
└── tests/
```

### 2. Entry Point Registration
```python
# setup.py
setup(
    name="hybridgraphrag-plugin",
    entry_points={
        "rag_templates_plugins": [
            "hybridgraphrag = hybridgraphrag:HybridGraphRAGPlugin"
        ]
    },
    install_requires=[
        "rag-templates>=1.0.0",
        "iris-vector-graph>=2.0.0"  # Heavy dependency isolated to plugin
    ]
)
```

### 3. Plugin Implementation (Adapting Existing Ontology Pattern)
```python
# hybridgraphrag/__init__.py
from typing import Dict, Any, Type, Optional
from iris_rag.plugins.interface import RAGPlugin  # Adapt ontology plugin interface
from .pipeline import HybridGraphRAGPipeline
from .schema_manager import HybridGraphRAGSchemaManager

class HybridGraphRAGPlugin(RAGPlugin):
    """Plugin providing HybridGraphRAG capabilities."""

    def get_pipeline_classes(self) -> Dict[str, Type]:
        return {"HybridGraphRAG": HybridGraphRAGPipeline}

    def get_schema_managers(self) -> Dict[str, Type]:
        return {"HybridGraphRAGSchemaManager": HybridGraphRAGSchemaManager}

    def validate_environment(self) -> bool:
        try:
            import iris_graph_core
            return True
        except ImportError:
            return False
```

## 📋 Migration Strategy

### Phase 1: Extend Existing Infrastructure (Minimal Changes)
1. **Add plugin type support** to `PipelineConfigService`
2. **Extend ModuleLoader** for external package loading
3. **Add plugin discovery** to configuration loading

### Phase 2: Extract HybridGraphRAG
1. **Create hybridgraphrag-plugin package**
2. **Move HybridGraphRAG components** to plugin
3. **Update rag-templates** to remove iris-vector-graph dependency

### Phase 3: Clean Integration
1. **Test plugin loading** with existing test suite
2. **Update documentation** for plugin usage
3. **Create plugin development guide**

## 🎯 Key Benefits of Your Existing Architecture

1. **Minimal Breaking Changes**: Your existing infrastructure can be extended, not replaced
2. **Proven Patterns**: The ontology plugin system shows this approach works
3. **Configuration-Driven**: Already supports declarative pipeline configuration
4. **Robust Error Handling**: Comprehensive validation and error reporting
5. **Performance Optimized**: Module caching and efficient loading

## 🔧 Required Changes Summary

The existing infrastructure is excellent! You only need:

1. **3 small extensions** to support external plugin packages
2. **1 new plugin package** for HybridGraphRAG
3. **Remove iris-vector-graph** from core rag-templates requirements

This is much simpler than building a new plugin system from scratch. Your existing architecture already handles the complex parts beautifully!