# RAG Templates Refactoring Specification

## Executive Summary

This specification defines the transformation of the existing `/Users/tdyar/ws/rag-templates` repository into a clean, installable Python package that can be seamlessly integrated with the personal assistant project. The refactored module will provide unified RAG pipeline interfaces, standardized connection management, and modular components following SPARC principles.

**Target Integration**:
```python
from rag_templates import create_pipeline, ConnectionManager
pipeline = create_pipeline("basic", config=my_config)
```

## 1. Package Structure Reorganization

### 1.1 Current State Analysis

**Existing Structure Issues**:
- Scattered directories: [`common/`](../rag-templates/common), [`basic_rag/`](../rag-templates/basic_rag), [`colbert/`](../rag-templates/colbert), [`crag/`](../rag-templates/crag)
- Multiple pipeline implementations with inconsistent interfaces
- Mixed concerns: connection management, embedding utilities, and pipeline logic
- No clear entry points or standardized configuration

### 1.2 Target Package Structure

```
rag_templates/
├── __init__.py                    # Main package interface
├── core/
│   ├── __init__.py
│   ├── base_pipeline.py          # Abstract base classes
│   ├── connection_manager.py     # Unified connection handling
│   ├── document.py               # Document data models
│   └── exceptions.py             # Custom exceptions
├── pipelines/
│   ├── __init__.py
│   ├── basic_rag.py              # Basic RAG implementation
│   ├── colbert.py                # ColBERT implementation
│   ├── crag.py                   # CRAG implementation
│   └── factory.py                # Pipeline factory
├── storage/
│   ├── __init__.py
│   ├── iris_adapter.py           # IRIS-specific operations
│   ├── vector_store.py           # Vector storage interface
│   └── schema_manager.py         # Database schema management
├── embeddings/
│   ├── __init__.py
│   ├── huggingface.py            # HuggingFace embeddings
│   ├── sentence_transformers.py  # Sentence transformers
│   └── base.py                   # Embedding interface
├── config/
│   ├── __init__.py
│   ├── loader.py                 # Configuration loading
│   ├── validator.py              # Configuration validation
│   └── defaults.py               # Default configurations
└── utils/
    ├── __init__.py
    ├── chunking.py               # Text chunking utilities
    ├── preprocessing.py          # Text preprocessing
    └── metrics.py                # Evaluation metrics
```

### 1.3 Migration Mapping

**TDD Anchor**: [`test_package_structure_migration()`](rag_templates_refactoring_specification.md:1.3)

```python
# PSEUDOCODE: Package Structure Migration

class PackageMigrator:
    def __init__(self, source_path: str, target_path: str):
        self.source_path = source_path
        self.target_path = target_path
        self.migration_map = self._build_migration_map()
    
    def _build_migration_map(self) -> Dict[str, str]:
        """
        Map existing files to new package structure
        
        TDD Anchor: test_migration_mapping()
        """
        return {
            # Core components
            "common/base_pipeline.py": "rag_templates/core/base_pipeline.py",
            "common/connection_manager.py": "rag_templates/core/connection_manager.py",
            "common/utils.py": "rag_templates/core/document.py",
            
            # Pipeline implementations
            "basic_rag/pipeline_refactored.py": "rag_templates/pipelines/basic_rag.py",
            "colbert/pipeline.py": "rag_templates/pipelines/colbert.py",
            "crag/pipeline.py": "rag_templates/pipelines/crag.py",
            
            # Storage components
            "common/iris_connector.py": "rag_templates/storage/iris_adapter.py",
            "common/vector_store.py": "rag_templates/storage/vector_store.py",
            
            # Embedding utilities
            "common/embedding_utils.py": "rag_templates/embeddings/base.py",
            
            # Configuration
            "config/config.yaml": "rag_templates/config/defaults.py",
        }
    
    def migrate_files(self) -> MigrationResult:
        """
        Execute file migration with content transformation
        
        TDD Anchor: test_file_migration()
        """
        results = []
        
        for source_file, target_file in self.migration_map.items():
            try:
                # Read source content
                content = self._read_source_file(source_file)
                
                # Transform content for new structure
                transformed_content = self._transform_content(content, target_file)
                
                # Write to target location
                self._write_target_file(target_file, transformed_content)
                
                results.append(MigrationItem(
                    source=source_file,
                    target=target_file,
                    status="success"
                ))
                
            except Exception as e:
                results.append(MigrationItem(
                    source=source_file,
                    target=target_file,
                    status="failed",
                    error=str(e)
                ))
        
        return MigrationResult(items=results)
    
    def _transform_content(self, content: str, target_file: str) -> str:
        """
        Transform file content for new package structure
        
        TDD Anchor: test_content_transformation()
        """
        transformations = [
            self._update_imports,
            self._standardize_interfaces,
            self._add_type_hints,
            self._update_docstrings
        ]
        
        for transform in transformations:
            content = transform(content, target_file)
        
        return content
```

## 2. API Standardization

### 2.1 Unified Pipeline Interface

**TDD Anchor**: [`test_pipeline_interface_standardization()`](rag_templates_refactoring_specification.md:2.1)

```python
# PSEUDOCODE: Standardized Pipeline Interface

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from rag_templates.core.document import Document
from rag_templates.core.connection_manager import ConnectionManager

class RAGPipeline(ABC):
    """
    Standardized interface for all RAG pipeline implementations
    
    TDD Anchor: test_rag_pipeline_interface()
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        config: Dict[str, Any],
        embedding_func: Optional[Callable] = None,
        llm_func: Optional[Callable] = None
    ):
        self.connection_manager = connection_manager
        self.config = config
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self._validate_config()
    
    @abstractmethod
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: User query string
            top_k: Maximum number of documents to retrieve
            threshold: Minimum similarity threshold
            **kwargs: Pipeline-specific parameters
            
        Returns:
            List of relevant documents with similarity scores
            
        TDD Anchor: test_document_retrieval()
        """
        pass
    
    @abstractmethod
    def store_document(self, document: Document) -> str:
        """
        Store document in vector database
        
        Args:
            document: Document to store with content and metadata
            
        Returns:
            Document ID of stored document
            
        TDD Anchor: test_document_storage()
        """
        pass
    
    def get_document_count(self) -> int:
        """
        Get total number of stored documents
        
        TDD Anchor: test_document_count()
        """
        cursor = self.connection_manager.get_connection().cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.config['table_name']}")
        result = cursor.fetchone()
        cursor.close()
        return int(result[0]) if result else 0
    
    def health_check(self) -> bool:
        """
        Check pipeline and database health
        
        TDD Anchor: test_health_check()
        """
        try:
            cursor = self.connection_manager.get_connection().cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception:
            return False
    
    def _validate_config(self) -> None:
        """
        Validate pipeline configuration
        
        TDD Anchor: test_config_validation()
        """
        required_keys = ['table_name', 'schema', 'embedding_dimension']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config['embedding_dimension'] <= 0:
            raise ValueError("embedding_dimension must be positive")

class BasicRAGPipeline(RAGPipeline):
    """
    Basic RAG implementation using vector similarity search
    
    TDD Anchor: test_basic_rag_implementation()
    """
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents using vector similarity search
        
        TDD Anchor: test_basic_rag_retrieval()
        """
        # Generate query embedding
        if not self.embedding_func:
            raise ValueError("Embedding function not configured")
        
        query_embedding = self.embedding_func([query])[0]
        embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        # Execute vector search
        cursor = self.connection_manager.get_connection().cursor()
        
        sql = f"""
            SELECT TOP {top_k}
                doc_id,
                title,
                text_content,
                VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', {self.config['embedding_dimension']})) as similarity_score,
                metadata,
                source_file
            FROM {self.config['schema']}.{self.config['table_name']}
            WHERE embedding IS NOT NULL
            AND VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', {self.config['embedding_dimension']})) >= ?
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(sql, [embedding_str, embedding_str, threshold])
        results = cursor.fetchall()
        cursor.close()
        
        # Convert to Document objects
        documents = []
        for row in results:
            doc = Document(
                id=str(row[0]),
                content=str(row[2]) if row[2] else "",
                score=float(row[3]) if row[3] else 0.0,
                metadata=json.loads(row[4]) if row[4] else {}
            )
            documents.append(doc)
        
        return documents
    
    def store_document(self, document: Document) -> str:
        """
        Store document with vector embedding
        
        TDD Anchor: test_basic_rag_storage()
        """
        if not document.embedding:
            if not self.embedding_func:
                raise ValueError("Document has no embedding and no embedding function configured")
            document.embedding = self.embedding_func([document.content])[0]
        
        embedding_str = ','.join([f'{x:.10f}' for x in document.embedding])
        metadata_json = json.dumps(document.metadata) if document.metadata else None
        title = document.metadata.get('title', '') if document.metadata else ''
        
        cursor = self.connection_manager.get_connection().cursor()
        
        sql = f"""
            INSERT INTO {self.config['schema']}.{self.config['table_name']}
            (doc_id, title, text_content, embedding, metadata, created_at)
            VALUES (?, ?, ?, TO_VECTOR(?, 'FLOAT', {self.config['embedding_dimension']}), ?, CURRENT_TIMESTAMP)
        """
        
        cursor.execute(sql, [
            document.id,
            title,
            document.content,
            embedding_str,
            metadata_json
        ])
        
        self.connection_manager.get_connection().commit()
        cursor.close()
        
        return document.id
```

### 2.2 Pipeline Factory

**TDD Anchor**: [`test_pipeline_factory()`](rag_templates_refactoring_specification.md:2.2)

```python
# PSEUDOCODE: Pipeline Factory

class PipelineFactory:
    """
    Factory for creating RAG pipeline instances
    
    TDD Anchor: test_pipeline_factory_creation()
    """
    
    _pipeline_registry = {
        'basic': BasicRAGPipeline,
        'colbert': ColBERTRAGPipeline,
        'crag': CRAGPipeline,
    }
    
    @classmethod
    def create_pipeline(
        cls,
        pipeline_type: str,
        config: Dict[str, Any],
        connection_manager: Optional[ConnectionManager] = None,
        embedding_func: Optional[Callable] = None,
        llm_func: Optional[Callable] = None
    ) -> RAGPipeline:
        """
        Create pipeline instance of specified type
        
        Args:
            pipeline_type: Type of pipeline ('basic', 'colbert', 'crag')
            config: Pipeline configuration
            connection_manager: Optional connection manager
            embedding_func: Optional embedding function
            llm_func: Optional LLM function
            
        Returns:
            Configured pipeline instance
            
        TDD Anchor: test_create_pipeline()
        """
        if pipeline_type not in cls._pipeline_registry:
            available = ', '.join(cls._pipeline_registry.keys())
            raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {available}")
        
        # Create connection manager if not provided
        if connection_manager is None:
            connection_manager = ConnectionManager(
                connection_type=config.get('connection_type', 'odbc')
            )
        
        # Get pipeline class and instantiate
        pipeline_class = cls._pipeline_registry[pipeline_type]
        
        return pipeline_class(
            connection_manager=connection_manager,
            config=config,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
    
    @classmethod
    def register_pipeline(cls, name: str, pipeline_class: type) -> None:
        """
        Register custom pipeline implementation
        
        TDD Anchor: test_register_custom_pipeline()
        """
        if not issubclass(pipeline_class, RAGPipeline):
            raise ValueError("Pipeline class must inherit from RAGPipeline")
        
        cls._pipeline_registry[name] = pipeline_class
    
    @classmethod
    def list_pipelines(cls) -> List[str]:
        """
        List available pipeline types
        
        TDD Anchor: test_list_pipelines()
        """
        return list(cls._pipeline_registry.keys())

# Main package interface function
def create_pipeline(
    pipeline_type: str,
    config: Dict[str, Any],
    **kwargs
) -> RAGPipeline:
    """
    Main entry point for creating RAG pipelines
    
    TDD Anchor: test_main_create_pipeline()
    """
    return PipelineFactory.create_pipeline(pipeline_type, config, **kwargs)
```

## 3. Dependency Management

### 3.1 IRIS/JDBC Dependencies

**TDD Anchor**: [`test_dependency_management()`](rag_templates_refactoring_specification.md:3.1)

```python
# PSEUDOCODE: Dependency Management

class DependencyManager:
    """
    Manages optional dependencies and graceful fallbacks
    
    TDD Anchor: test_dependency_manager()
    """
    
    def __init__(self):
        self._available_drivers = {}
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """
        Check availability of optional dependencies
        
        TDD Anchor: test_check_dependencies()
        """
        # Check JDBC driver
        try:
            import jaydebeapi
            self._available_drivers['jdbc'] = True
        except ImportError:
            self._available_drivers['jdbc'] = False
        
        # Check ODBC driver
        try:
            import pyodbc
            self._available_drivers['odbc'] = True
        except ImportError:
            self._available_drivers['odbc'] = False
        
        # Check IRIS DBAPI
        try:
            import iris
            self._available_drivers['dbapi'] = True
        except ImportError:
            self._available_drivers['dbapi'] = False
    
    def get_available_drivers(self) -> List[str]:
        """
        Get list of available database drivers
        
        TDD Anchor: test_get_available_drivers()
        """
        return [driver for driver, available in self._available_drivers.items() if available]
    
    def get_recommended_driver(self) -> str:
        """
        Get recommended driver based on availability
        
        TDD Anchor: test_get_recommended_driver()
        """
        # Priority order: odbc (most stable) -> dbapi -> jdbc
        priority = ['odbc', 'dbapi', 'jdbc']
        
        for driver in priority:
            if self._available_drivers.get(driver, False):
                return driver
        
        raise RuntimeError("No compatible database drivers found")
    
    def validate_driver_requirements(self, driver: str) -> bool:
        """
        Validate that required dependencies are available for driver
        
        TDD Anchor: test_validate_driver_requirements()
        """
        if driver not in self._available_drivers:
            return False
        
        return self._available_drivers[driver]

# Package-level dependency checking
def check_dependencies() -> DependencyReport:
    """
    Check all package dependencies and return report
    
    TDD Anchor: test_package_dependency_check()
    """
    manager = DependencyManager()
    
    return DependencyReport(
        available_drivers=manager.get_available_drivers(),
        recommended_driver=manager.get_recommended_driver(),
        missing_dependencies=_get_missing_dependencies(),
        installation_instructions=_get_installation_instructions()
    )

def _get_missing_dependencies() -> List[str]:
    """Get list of missing optional dependencies"""
    missing = []
    
    optional_deps = {
        'jaydebeapi': 'JDBC support',
        'pyodbc': 'ODBC support', 
        'iris': 'IRIS DBAPI support',
        'sentence-transformers': 'Sentence transformer embeddings',
        'transformers': 'HuggingFace transformers'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{dep} ({description})")
    
    return missing
```

### 3.2 Embedding Model Dependencies

**TDD Anchor**: [`test_embedding_dependencies()`](rag_templates_refactoring_specification.md:3.2)

```python
# PSEUDOCODE: Embedding Model Management

class EmbeddingManager:
    """
    Manages embedding model dependencies and initialization
    
    TDD Anchor: test_embedding_manager()
    """
    
    def __init__(self):
        self._model_cache = {}
        self._available_backends = self._check_backends()
    
    def _check_backends(self) -> Dict[str, bool]:
        """
        Check availability of embedding backends
        
        TDD Anchor: test_check_embedding_backends()
        """
        backends = {}
        
        # Check sentence-transformers
        try:
            import sentence_transformers
            backends['sentence_transformers'] = True
        except ImportError:
            backends['sentence_transformers'] = False
        
        # Check transformers + torch
        try:
            import transformers
            import torch
            backends['transformers'] = True
        except ImportError:
            backends['transformers'] = False
        
        return backends
    
    def get_embedding_function(
        self,
        model_name: str,
        backend: Optional[str] = None
    ) -> Callable[[List[str]], List[List[float]]]:
        """
        Get embedding function for specified model
        
        TDD Anchor: test_get_embedding_function()
        """
        cache_key = f"{model_name}:{backend}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Auto-select backend if not specified
        if backend is None:
            backend = self._select_best_backend(model_name)
        
        # Create embedding function
        if backend == 'sentence_transformers':
            func = self._create_sentence_transformer_func(model_name)
        elif backend == 'transformers':
            func = self._create_transformers_func(model_name)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self._model_cache[cache_key] = func
        return func
    
    def _select_best_backend(self, model_name: str) -> str:
        """
        Select best available backend for model
        
        TDD Anchor: test_select_best_backend()
        """
        # Prefer sentence-transformers for simplicity
        if self._available_backends.get('sentence_transformers', False):
            return 'sentence_transformers'
        elif self._available_backends.get('transformers', False):
            return 'transformers'
        else:
            raise RuntimeError("No embedding backends available")
    
    def _create_sentence_transformer_func(self, model_name: str) -> Callable:
        """Create sentence-transformers embedding function"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        
        def embed_texts(texts: List[str]) -> List[List[float]]:
            embeddings = model.encode(texts)
            return embeddings.tolist()
        
        return embed_texts
    
    def _create_transformers_func(self, model_name: str) -> Callable:
        """Create transformers embedding function"""
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        def embed_texts(texts: List[str]) -> List[List[float]]:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.tolist()
        
        return embed_texts
```

## 4. Configuration System

### 4.1 Environment-Based Configuration

**TDD Anchor**: [`test_configuration_system()`](rag_templates_refactoring_specification.md:4.1)

```python
# PSEUDOCODE: Configuration System

class ConfigurationManager:
    """
    Manages configuration loading, validation, and environment overrides
    
    TDD Anchor: test_configuration_manager()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config = None
        self._env_prefix = "RAG_TEMPLATES_"
    
    def load_config(self, config_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load configuration from file, dict, or environment
        
        TDD Anchor: test_load_config()
        """
        if config_dict:
            base_config = config_dict
        elif self.config_path:
            base_config = self._load_from_file(self.config_path)
        else:
            base_config = self._get_default_config()
        
        # Apply environment overrides
        config = self._apply_env_overrides(base_config)
        
        # Validate configuration
        self._validate_config(config)
        
        self._config = config
        return config
    
    def _load_from_file(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        TDD Anchor: test_load_from_file()
        """
        import yaml
        
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        TDD Anchor: test_get_default_config()
        """
        return {
            'database': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'schema': 'RAG',
                'connection_type': 'odbc'
            },
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384,
                'backend': 'auto'
            },
            'storage': {
                'table_name': 'Documents',
                'batch_size': 100
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            }
        }
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides
        
        TDD Anchor: test_apply_env_overrides()
        """
        import os
        
        # Flatten config for environment mapping
        flat_config = self._flatten_dict(config)
        
        # Check for environment overrides
        for key, value in flat_config.items():
            env_key = f"{self._env_prefix}{key.upper().replace('.', '_')}"
            env_value = os.environ.get(env_key)
            
            if env_value is not None:
                # Convert environment string to appropriate type
                converted_value = self._convert_env_value(env_value, type(value))
                self._set_nested_value(config, key, converted_value)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration values
        
        TDD Anchor: test_validate_config()
        """
        validators = [
            self._validate_database_config,
            self._validate_embedding_config,
            self._validate_storage_config,
            self._validate_retrieval_config
        ]
        
        for validator in validators:
            validator(config)
    
    def _validate_database_config(self, config: Dict[str, Any]) -> None:
        """Validate database configuration"""
        db_config = config.get('database', {})
        
        required_fields = ['host', 'port', 'namespace', 'schema']
        for field in required_fields:
            if field not in db_config:
                raise ConfigurationError(f"Missing required database config: {field}")
        
        if not isinstance(db_config['port'], int) or db_config['port'] <= 0:
            raise ConfigurationError("Database port must be a positive integer")
    
    def _validate_embedding_config(self, config: Dict[str, Any]) -> None:
        """Validate embedding configuration"""
        emb_config = config.get('embedding', {})
        
        if 'dimension' in emb_config:
            if not isinstance(emb_config['dimension'], int) or emb_config['dimension'] <= 0:
                raise ConfigurationError("Embedding dimension must be a positive integer")
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        TDD Anchor: test_get_config_value()
        """
        if self._config is None:
            self.load_config()
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass

# Package-level configuration functions
def load_config(config_path: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """
    Load and return configuration
    
    TDD Anchor: test_package_load_config()
    """
    manager = ConfigurationManager(config_path)
    config = manager.load_config()
    
    # Apply any direct overrides
    for key, value in overrides.items():
        manager._set_nested_value(config, key, value)
    
    return config

def get_config_value(key_path: str, default: Any = None, config: Optional[Dict] = None) -> Any:
    """
    Get configuration value by path
    
    TDD Anchor: test_package_get_config_value()
    """
    if config is None:
        config = load_config()
    
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
```

### 4.2 Configuration Validation

**TDD Anchor**: [`test_config_validation()`](rag_templates_refactoring_specification.md:4.2)

```python
# PSEUDOCODE: Configuration Validation

class ConfigValidator:
    """
    Validates configuration schemas and values
    
    TDD Anchor: test_config_validator()
    """
    
    def __init__(self):
        self.schema = self._get_config_schema()
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """
        Define configuration schema for validation
        
        TDD Anchor: test_get_config_schema()
        """
        return {
            'database': {
                'type': 'object',
                'required': ['host', 'port', 'namespace', 'schema'],
                'properties': {
                    'host': {'type': 'string', 'minLength': 1},
                    'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535},
                    'namespace': {'type': 'string', 'minLength': 1},
                    'schema': {'type': 'string', 'minLength': 1},
                    'connection_type': {
                        'type': 'string',
                        'enum': ['odbc', 'jdbc', 'dbapi']
                    },
                    'username': {'type': 'string'},
                    'password': {'type': 'string'}
                }
            },
            'embedding': {
                'type': 'object',
                'required': ['model_name', 'dimension'],
                'properties': {
                    'model_name': {'type': 'string',