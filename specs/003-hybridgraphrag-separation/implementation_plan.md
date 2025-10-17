# HybridGraphRAG Separation Implementation Plan

## 🎯 Overview

This plan leverages your existing sophisticated plugin infrastructure to separate HybridGraphRAG with minimal changes to the core framework.

## 🔧 Required Extensions (Minimal Changes)

### 1. Extend PipelineConfigService for Plugin Support

**File**: `iris_rag/config/pipeline_config_service.py`

Add plugin discovery capability:

```python
def load_pipeline_definitions(self, config_file_path: str) -> List[Dict]:
    """Enhanced to support plugin-provided pipelines."""

    # Load static definitions from YAML (existing code)
    static_pipelines = self._load_static_pipeline_definitions(config_file_path)

    # Discover plugin-provided pipelines
    plugin_pipelines = self._discover_plugin_pipelines()

    # Merge and return
    all_pipelines = static_pipelines + plugin_pipelines
    return all_pipelines

def _discover_plugin_pipelines(self) -> List[Dict]:
    """Discover pipelines from installed plugin packages."""
    plugin_pipelines = []

    try:
        import pkg_resources
        for entry_point in pkg_resources.iter_entry_points('rag_templates_plugins'):
            try:
                # Load plugin class
                plugin_class = entry_point.load()
                plugin = plugin_class()

                # Get pipeline definitions from plugin
                pipeline_classes = plugin.get_pipeline_classes()
                schema_managers = plugin.get_schema_managers()

                for pipeline_name, pipeline_class in pipeline_classes.items():
                    plugin_def = {
                        'name': pipeline_name,
                        'type': 'plugin',
                        'plugin_package': entry_point.name,
                        'module': pipeline_class.__module__,
                        'class': pipeline_class.__name__,
                        'enabled': True,
                        'params': plugin.get_default_configuration()
                    }

                    # Add schema requirements if plugin provides schema manager
                    if pipeline_name in schema_managers:
                        plugin_def['schema_requirements'] = {
                            'schema_manager': schema_managers[pipeline_name].__name__,
                            'tables': getattr(plugin, 'REQUIRED_TABLES', []),
                            'dependencies': getattr(plugin, 'EXTERNAL_DEPENDENCIES', [])
                        }

                    plugin_pipelines.append(plugin_def)

            except Exception as e:
                self.logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

    except ImportError:
        # pkg_resources not available - no plugins
        pass

    return plugin_pipelines
```

### 2. Extend ModuleLoader for External Packages

**File**: `iris_rag/utils/module_loader.py`

Add external package support:

```python
def load_pipeline_class(self, module_path: str, class_name: str, package_type: str = "core") -> Type:
    """Enhanced to support plugin packages."""

    if package_type == "plugin":
        return self._load_plugin_pipeline_class(module_path, class_name)
    else:
        # Existing core pipeline loading
        return self._load_core_pipeline_class(module_path, class_name)

def _load_plugin_pipeline_class(self, module_path: str, class_name: str) -> Type:
    """Load pipeline class from external plugin package."""

    # For plugin modules, the module_path is already the full import path
    # e.g., "hybridgraphrag.pipeline" instead of "iris_rag.pipelines.basic"

    try:
        # Import the plugin module directly
        self.logger.debug(f"Importing plugin module: {module_path}")
        module = importlib.import_module(module_path)

        # Rest of the validation is the same as existing code
        if not hasattr(module, class_name):
            raise ModuleLoadingError(
                f"Class '{class_name}' not found in plugin module '{module_path}'"
            )

        pipeline_class = getattr(module, class_name)

        if not isinstance(pipeline_class, type):
            raise ModuleLoadingError(
                f"'{class_name}' in plugin module '{module_path}' is not a class"
            )

        if not issubclass(pipeline_class, RAGPipeline):
            raise ModuleLoadingError(
                f"Class '{class_name}' in plugin module '{module_path}' is not a subclass of RAGPipeline"
            )

        self.logger.info(f"Successfully loaded plugin pipeline class: {class_name}")
        return pipeline_class

    except ImportError as e:
        error_msg = f"Failed to import plugin module '{module_path}': {str(e)}"
        self.logger.error(error_msg)
        raise ModuleLoadingError(error_msg)
```

### 3. Enhance PipelineFactory for Plugin Types

**File**: `iris_rag/pipelines/factory.py`

Update pipeline creation to handle plugin types:

```python
def create_pipeline(self, pipeline_name: str) -> Optional[RAGPipeline]:
    """Enhanced to support plugin-provided pipelines."""

    # Existing code for loading definitions...

    pipeline_def = self._pipeline_definitions[pipeline_name]
    pipeline_type = pipeline_def.get("type", "core")

    try:
        # Determine how to load the pipeline class
        if pipeline_type == "plugin":
            # Load from external plugin package
            pipeline_class = self.module_loader.load_pipeline_class(
                pipeline_def["module"],
                pipeline_def["class"],
                package_type="plugin"
            )
        else:
            # Existing core pipeline loading
            pipeline_class = self.module_loader.load_pipeline_class(
                pipeline_def["module"],
                pipeline_def["class"]
            )

        # Rest of the pipeline creation logic remains the same...
```

## 📦 HybridGraphRAG Plugin Package

### Package Structure
```
hybridgraphrag-plugin/
├── setup.py
├── pyproject.toml
├── README.md
├── hybridgraphrag/
│   ├── __init__.py           # Plugin class
│   ├── pipeline.py           # HybridGraphRAGPipeline (moved)
│   ├── schema_manager.py     # HybridGraphRAGSchemaManager (moved)
│   ├── discovery.py          # GraphCoreDiscovery (moved)
│   ├── retrieval.py          # HybridRetrievalMethods (moved)
│   └── _hybrid_utils.py      # Utility functions (moved)
└── tests/
    ├── __init__.py
    ├── test_plugin.py
    └── test_integration.py
```

### Entry Point Setup
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="hybridgraphrag-plugin",
    version="1.0.0",
    description="HybridGraphRAG plugin for rag-templates",
    packages=find_packages(),
    install_requires=[
        "rag-templates>=1.0.0",
        "iris-vector-graph>=2.0.0",  # Heavy dependency isolated here
    ],
    entry_points={
        "rag_templates_plugins": [
            "hybridgraphrag = hybridgraphrag:HybridGraphRAGPlugin"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
```

### Plugin Implementation (Adapting Ontology Pattern)
```python
# hybridgraphrag/__init__.py
from typing import Dict, Any, Type, List
from iris_rag.core.base import RAGPipeline
from iris_rag.storage.schema_manager import SchemaManager

class HybridGraphRAGPlugin:
    """Plugin providing HybridGraphRAG capabilities."""

    REQUIRED_TABLES = [
        "KG_NODEEMBEDDINGS_OPTIMIZED",
        "RDF_EDGES",
        "RDF_LABELS",
        "RDF_PROPS"
    ]

    EXTERNAL_DEPENDENCIES = [
        "iris-vector-graph>=2.0.0"
    ]

    def get_pipeline_classes(self) -> Dict[str, Type[RAGPipeline]]:
        """Return pipeline classes provided by this plugin."""
        from .pipeline import HybridGraphRAGPipeline
        return {"HybridGraphRAG": HybridGraphRAGPipeline}

    def get_schema_managers(self) -> Dict[str, Type[SchemaManager]]:
        """Return schema manager classes provided by this plugin."""
        from .schema_manager import HybridGraphRAGSchemaManager
        return {"HybridGraphRAGSchemaManager": HybridGraphRAGSchemaManager}

    def get_default_configuration(self) -> Dict[str, Any]:
        """Return default configuration for HybridGraphRAG."""
        return {
            "top_k": 10,
            "max_depth": 2,
            "max_entities": 50,
            "fusion_weights": [0.4, 0.3, 0.3],
            "fallback_to_graphrag": True
        }

    def validate_environment(self) -> bool:
        """Validate that required dependencies are available."""
        try:
            import iris_graph_core
            return True
        except ImportError:
            return False
```

## 🗂️ File Movement Plan

### Files to Move to Plugin Package

1. **`iris_rag/pipelines/hybrid_graphrag.py`** → **`hybridgraphrag/pipeline.py`**
2. **`iris_rag/storage/hybrid_schema_manager.py`** → **`hybridgraphrag/schema_manager.py`**
3. **`iris_rag/pipelines/hybrid_graphrag_discovery.py`** → **`hybridgraphrag/discovery.py`**
4. **`iris_rag/pipelines/hybrid_graphrag_retrieval.py`** → **`hybridgraphrag/retrieval.py`**
5. **`iris_rag/pipelines/_hybrid_utils.py`** → **`hybridgraphrag/_hybrid_utils.py`**

### Files to Update in Core rag-templates

1. **`config/pipelines.yaml`** - Remove HybridGraphRAG static definition
2. **`iris_rag/__init__.py`** - Remove HybridGraphRAG imports
3. **`iris_rag/validation/factory.py`** - Remove HybridGraphRAG imports
4. **`requirements.txt`** - Remove `iris-vector-graph` dependency

## 🧪 Testing Strategy

### 1. Plugin Loading Tests
```python
# tests/test_plugin_loading.py
def test_plugin_discovery():
    """Test that HybridGraphRAG plugin is discovered."""
    service = PipelineConfigService()
    plugins = service._discover_plugin_pipelines()

    plugin_names = [p['name'] for p in plugins]
    assert 'HybridGraphRAG' in plugin_names

def test_plugin_pipeline_creation():
    """Test creating HybridGraphRAG pipeline from plugin."""
    pipeline = create_pipeline("HybridGraphRAG")
    assert isinstance(pipeline, HybridGraphRAGPipeline)
```

### 2. Integration Tests
```python
# tests/test_integration.py
def test_hybrid_graphrag_full_workflow():
    """Test complete HybridGraphRAG workflow through plugin."""
    # Test with iris-vector-graph dependency isolated to plugin
    pass
```

## 📋 Migration Steps

### Step 1: Implement Core Extensions (1-2 hours)
1. Add plugin discovery to `PipelineConfigService`
2. Extend `ModuleLoader` for external packages
3. Update `PipelineFactory` for plugin types
4. Test with existing HybridGraphRAG (before moving)

### Step 2: Create Plugin Package (2-3 hours)
1. Create `hybridgraphrag-plugin` package structure
2. Move HybridGraphRAG files to plugin package
3. Implement `HybridGraphRAGPlugin` class
4. Setup entry points and dependencies

### Step 3: Clean Core Framework (1 hour)
1. Remove HybridGraphRAG from core `iris_rag`
2. Remove `iris-vector-graph` from core requirements
3. Update imports and references
4. Clean up configuration files

### Step 4: Integration Testing (1-2 hours)
1. Test plugin installation: `pip install hybridgraphrag-plugin`
2. Test plugin discovery and loading
3. Test HybridGraphRAG functionality through plugin
4. Verify iris-vector-graph isolation

## 🎯 Expected Benefits

1. **Dependency Isolation**: `iris-vector-graph` only needed if using HybridGraphRAG
2. **Modular Installation**: Users install `rag-templates` + specific plugins
3. **Independent Evolution**: Plugin can evolve independently of core framework
4. **Community Ecosystem**: Pattern established for future plugin contributions
5. **Reduced Core Complexity**: Core framework remains focused and lightweight

## 📈 Success Metrics

- ✅ Core `rag-templates` installs without `iris-vector-graph`
- ✅ HybridGraphRAG works identically through plugin
- ✅ Plugin discovery and loading works seamlessly
- ✅ Configuration remains declarative
- ✅ No breaking changes to existing GraphRAG or basic pipelines

This implementation leverages your excellent existing architecture while achieving the dependency separation you want!