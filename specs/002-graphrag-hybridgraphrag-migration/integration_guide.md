# HybridGraphRAG Integration Guide: Configuration → Schema → Pipeline Requirements

## Overview

This document outlines the proper integration of HybridGraphRAG into the existing configuration manager → schema manager → pipeline requirements flow.

## 🔧 Integration Points

### 1. Configuration Manager Integration

The HybridGraphRAG configuration needs to be properly integrated into the ConfigurationManager to ensure schema requirements are automatically detected and fulfilled.

#### 1.1 Pipeline Configuration Updates

```yaml
# config/pipelines.yaml - Add HybridGraphRAG configuration
pipelines:
  hybrid_graphrag:
    schema_requirements:
      tables:
        - "SourceDocuments"
        - "Entities"
        - "EntityRelationships"
        - "KG_NODEEMBEDDINGS_OPTIMIZED"  # iris_graph_core tables
        - "RDF_EDGES"
        - "RDF_LABELS"
        - "RDF_PROPS"
      indexes:
        - name: "idx_kg_nodeembeddings_vector"
          table: "KG_NODEEMBEDDINGS_OPTIMIZED"
          type: "HNSW"
          required: false  # Optional for Community Edition
      schema_manager: "HybridGraphRAGSchemaManager"

    retrieval_methods:
      - "kg"           # Knowledge graph traversal
      - "vector"       # Vector similarity search
      - "text"         # Text search
      - "hybrid"       # Multi-modal fusion

    dependencies:
      - "iris_graph_core"
      - "vector_embeddings"
      - "knowledge_graph"
```

#### 1.2 Configuration Manager Enhancement

```python
# iris_rag/config/manager.py - Add HybridGraphRAG support

def get_pipeline_requirements(self, pipeline_type: str) -> Dict[str, Any]:
    """Get pipeline-specific requirements including schema needs."""
    pipeline_config = self.get(f"pipelines:{pipeline_type}", {})

    return {
        "schema_requirements": pipeline_config.get("schema_requirements", {}),
        "schema_manager": pipeline_config.get("schema_manager", "SchemaManager"),
        "dependencies": pipeline_config.get("dependencies", []),
        "retrieval_methods": pipeline_config.get("retrieval_methods", [])
    }

def get_hybrid_graphrag_config(self) -> Dict[str, Any]:
    """Get HybridGraphRAG-specific configuration."""
    default_config = {
        "enabled": True,
        "schema_auto_setup": True,
        "fallback_to_graphrag": True,
        "iris_graph_core": {
            "enabled": True,
            "auto_create_tables": True,
            "community_edition_compatible": True
        },
        "fusion_weights": [0.4, 0.3, 0.3],  # [vector, text, graph]
        "retrieval_methods": ["kg", "vector", "text", "hybrid"]
    }

    user_config = self.get("hybrid_graphrag", {})
    if isinstance(user_config, dict):
        default_config.update(user_config)

    return default_config
```

### 2. Schema Manager Integration

#### 2.1 Schema Manager Factory Pattern

```python
# iris_rag/storage/schema_manager.py - Add schema manager factory

@classmethod
def create_schema_manager(cls, pipeline_type: str, connection_manager, config_manager):
    """Factory method to create appropriate schema manager for pipeline type."""

    # Get pipeline requirements
    requirements = config_manager.get_pipeline_requirements(pipeline_type)
    schema_manager_type = requirements.get("schema_manager", "SchemaManager")

    if schema_manager_type == "HybridGraphRAGSchemaManager":
        from .hybrid_schema_manager import HybridGraphRAGSchemaManager
        return HybridGraphRAGSchemaManager(connection_manager, config_manager)
    elif pipeline_type in ["graphrag", "hybrid_graphrag"]:
        # Enhanced schema manager for graph-based pipelines
        manager = cls(connection_manager, config_manager)
        manager.pipeline_type = pipeline_type
        return manager
    else:
        return cls(connection_manager, config_manager)

def ensure_pipeline_schema(self, pipeline_type: str) -> bool:
    """Ensure schema for specific pipeline type based on requirements."""
    try:
        requirements = self.config_manager.get_pipeline_requirements(pipeline_type)
        schema_requirements = requirements.get("schema_requirements", {})

        # Ensure required tables
        required_tables = schema_requirements.get("tables", [])
        for table_name in required_tables:
            if not self.ensure_table_schema(table_name, pipeline_type):
                logger.error(f"Failed to ensure required table: {table_name}")
                return False

        # Create required indexes
        required_indexes = schema_requirements.get("indexes", [])
        for index_config in required_indexes:
            if not self._ensure_index(index_config):
                if index_config.get("required", True):
                    logger.error(f"Failed to create required index: {index_config['name']}")
                    return False
                else:
                    logger.info(f"Optional index not created: {index_config['name']}")

        return True

    except Exception as e:
        logger.error(f"Failed to ensure schema for {pipeline_type}: {e}")
        return False
```

#### 2.2 Pipeline Requirements Integration

```python
# iris_rag/pipelines/__init__.py - Update pipeline factory

def create_pipeline(pipeline_type: str, **kwargs) -> RAGPipeline:
    """Enhanced pipeline factory with automatic schema management."""

    # Get configuration
    config_manager = kwargs.get('config_manager') or ConfigurationManager()
    connection_manager = kwargs.get('connection_manager') or ConnectionManager(config_manager)

    # Create appropriate schema manager
    schema_manager = SchemaManager.create_schema_manager(
        pipeline_type, connection_manager, config_manager
    )

    # Auto-setup schema if requested
    auto_setup = kwargs.get('auto_setup', True)
    if auto_setup:
        if not schema_manager.ensure_pipeline_schema(pipeline_type):
            if not kwargs.get('ignore_schema_errors', False):
                raise RuntimeError(f"Failed to setup schema for {pipeline_type}")

    # Create pipeline with proper schema manager
    if pipeline_type == "hybrid_graphrag":
        from .hybrid_graphrag import HybridGraphRAGPipeline
        return HybridGraphRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            schema_manager=schema_manager,
            **kwargs
        )
    elif pipeline_type == "graphrag":
        from .graphrag import GraphRAGPipeline
        return GraphRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            schema_manager=schema_manager,
            **kwargs
        )
    # ... other pipeline types
```

### 3. Pipeline Requirements Declaration

#### 3.1 HybridGraphRAG Pipeline Requirements

```python
# iris_rag/pipelines/hybrid_graphrag.py - Declare requirements

class HybridGraphRAGPipeline(RAGPipeline):
    """HybridGraphRAG pipeline with proper requirements declaration."""

    REQUIRED_TABLES = [
        "SourceDocuments",
        "Entities",
        "EntityRelationships",
        "KG_NODEEMBEDDINGS_OPTIMIZED",
        "RDF_EDGES",
        "RDF_LABELS",
        "RDF_PROPS"
    ]

    REQUIRED_DEPENDENCIES = [
        "iris_graph_core",
        "vector_embeddings",
        "knowledge_graph"
    ]

    SCHEMA_MANAGER_CLASS = "HybridGraphRAGSchemaManager"

    def __init__(self, connection_manager, config_manager, schema_manager=None, **kwargs):
        """Initialize with proper schema manager."""

        # Use provided schema manager or create appropriate one
        if schema_manager is None:
            from iris_rag.storage.hybrid_schema_manager import HybridGraphRAGSchemaManager
            schema_manager = HybridGraphRAGSchemaManager(connection_manager, config_manager)

        super().__init__(connection_manager, config_manager, **kwargs)
        self.schema_manager = schema_manager

        # Validate requirements
        self._validate_requirements()

    def _validate_requirements(self) -> bool:
        """Validate that all requirements are met."""
        # Check schema requirements
        validation = self.schema_manager.validate_hybrid_schema()
        missing_tables = [table for table, exists in validation.items() if not exists]

        if missing_tables:
            logger.warning(f"Missing iris_graph_core tables: {missing_tables}")
            # Attempt auto-repair if enabled
            if self.config_manager.get("hybrid_graphrag:schema_auto_setup", True):
                logger.info("Attempting to create missing tables...")
                if not self.schema_manager.ensure_hybrid_graphrag_schema():
                    logger.error("Failed to auto-create missing tables")
                    return False
            else:
                return False

        return True

    @classmethod
    def get_requirements(cls) -> Dict[str, Any]:
        """Return pipeline requirements for validation framework."""
        return {
            "tables": cls.REQUIRED_TABLES,
            "dependencies": cls.REQUIRED_DEPENDENCIES,
            "schema_manager": cls.SCHEMA_MANAGER_CLASS,
            "optional_features": ["HNSW_indexes", "vector_optimization"]
        }
```

### 4. Validation Framework Integration

#### 4.1 Requirements Validation

```python
# iris_rag/validation/validator.py - Enhanced requirements validation

class PipelineValidator:
    """Enhanced pipeline validator with schema requirements support."""

    def validate_pipeline_requirements(self, pipeline_type: str) -> Dict[str, bool]:
        """Validate all requirements for a pipeline type."""

        validation_results = {}

        try:
            # Get pipeline class and requirements
            pipeline_class = self._get_pipeline_class(pipeline_type)
            requirements = pipeline_class.get_requirements()

            # Validate schema requirements
            schema_validation = self._validate_schema_requirements(
                pipeline_type, requirements
            )
            validation_results.update(schema_validation)

            # Validate dependencies
            dependency_validation = self._validate_dependencies(
                requirements.get("dependencies", [])
            )
            validation_results.update(dependency_validation)

            return validation_results

        except Exception as e:
            logger.error(f"Requirements validation failed for {pipeline_type}: {e}")
            return {"validation_error": False}

    def _validate_schema_requirements(self, pipeline_type: str, requirements: Dict) -> Dict[str, bool]:
        """Validate schema requirements."""
        results = {}

        # Create appropriate schema manager
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager.create_schema_manager(
            pipeline_type, connection_manager, config_manager
        )

        # Validate required tables
        required_tables = requirements.get("tables", [])
        for table in required_tables:
            exists = schema_manager._table_exists(table)
            results[f"table_{table}"] = exists

        # Validate schema manager compatibility
        schema_manager_class = requirements.get("schema_manager")
        if schema_manager_class:
            compatible = isinstance(schema_manager,
                getattr(__import__(schema_manager.__module__), schema_manager_class, type(None))
            )
            results["schema_manager_compatible"] = compatible

        return results
```

## 🚀 Implementation Summary

This integration ensures that:

1. **Configuration-Driven Schema Management**: Pipeline requirements are declared in configuration files and automatically enforced

2. **Automatic Schema Setup**: When creating HybridGraphRAG pipelines, the required iris_graph_core tables are automatically created

3. **Graceful Fallbacks**: If advanced features aren't available (e.g., HNSW indexes), the system gracefully falls back to basic functionality

4. **Validation Framework**: Requirements are validated both at pipeline creation and through the validation framework

5. **Consistent Architecture**: All pipelines follow the same configuration → schema → validation pattern

This ensures HybridGraphRAG integrates seamlessly into the existing infrastructure while maintaining backward compatibility and proper error handling.