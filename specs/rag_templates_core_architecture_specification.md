# RAG Templates Core Architecture Specification

## 1. Package Structure and Organization

### 1.1 Hierarchical Package Design

**TDD Anchor**: [`test_package_structure_compliance()`](rag_templates_core_architecture_specification.md:1.1)

```python
# PSEUDOCODE: Package Structure Validation

class PackageStructureValidator:
    """
    Validates package structure compliance with architecture requirements
    
    TDD Anchor: test_package_structure_validator()
    """
    
    def __init__(self, package_root: str):
        self.package_root = package_root
        self.required_structure = self._define_required_structure()
    
    def _define_required_structure(self) -> Dict[str, PackageModule]:
        """
        Define required package structure with constraints
        
        TDD Anchor: test_define_required_structure()
        """
        return {
            'rag_templates/__init__.py': PackageModule(
                path='rag_templates/__init__.py',
                max_lines=50,
                required_exports=['create_pipeline', 'ConnectionManager', 'RAGPipeline'],
                dependencies=[]
            ),
            'rag_templates/core/__init__.py': PackageModule(
                path='rag_templates/core/__init__.py',
                max_lines=30,
                required_exports=['RAGPipeline', 'ConnectionManager', 'Document'],
                dependencies=[]
            ),
            'rag_templates/core/base_pipeline.py': PackageModule(
                path='rag_templates/core/base_pipeline.py',
                max_lines=200,
                required_exports=['RAGPipeline'],
                dependencies=['typing', 'abc', 'time']
            ),
            'rag_templates/core/connection_manager.py': PackageModule(
                path='rag_templates/core/connection_manager.py',
                max_lines=300,
                required_exports=['ConnectionManager'],
                dependencies=['typing', 'contextlib']
            ),
            'rag_templates/core/document.py': PackageModule(
                path='rag_templates/core/document.py',
                max_lines=150,
                required_exports=['Document'],
                dependencies=['dataclasses', 'typing']
            ),
            'rag_templates/core/exceptions.py': PackageModule(
                path='rag_templates/core/exceptions.py',
                max_lines=100,
                required_exports=['RAGException', 'ConfigurationError', 'ConnectionError'],
                dependencies=[]
            ),
            'rag_templates/pipelines/__init__.py': PackageModule(
                path='rag_templates/pipelines/__init__.py',
                max_lines=50,
                required_exports=['create_pipeline', 'PipelineFactory'],
                dependencies=[]
            ),
            'rag_templates/pipelines/factory.py': PackageModule(
                path='rag_templates/pipelines/factory.py',
                max_lines=200,
                required_exports=['PipelineFactory'],
                dependencies=['typing']
            ),
            'rag_templates/config/__init__.py': PackageModule(
                path='rag_templates/config/__init__.py',
                max_lines=30,
                required_exports=['load_config', 'ConfigurationManager'],
                dependencies=[]
            ),
            'rag_templates/config/loader.py': PackageModule(
                path='rag_templates/config/loader.py',
                max_lines=250,
                required_exports=['ConfigurationManager'],
                dependencies=['yaml', 'os', 'typing']
            )
        }
    
    def validate_structure(self) -> StructureValidationResult:
        """
        Validate complete package structure
        
        TDD Anchor: test_validate_structure()
        """
        results = []
        
        for module_path, module_spec in self.required_structure.items():
            full_path = os.path.join(self.package_root, module_path)
            
            # Check file exists
            if not os.path.exists(full_path):
                results.append(ModuleValidationResult(
                    module_path=module_path,
                    passed=False,
                    issues=['File does not exist']
                ))
                continue
            
            # Validate module compliance
            module_result = self._validate_module(full_path, module_spec)
            results.append(module_result)
        
        overall_passed = all(result.passed for result in results)
        
        return StructureValidationResult(
            overall_passed=overall_passed,
            module_results=results
        )
    
    def _validate_module(self, file_path: str, spec: PackageModule) -> ModuleValidationResult:
        """Validate individual module compliance"""
        issues = []
        
        # Check line count
        with open(file_path, 'r') as f:
            line_count = sum(1 for _ in f)
        
        if line_count > spec.max_lines:
            issues.append(f"Module exceeds {spec.max_lines} lines: {line_count}")
        
        # Check required exports (simplified - would use AST in real implementation)
        with open(file_path, 'r') as f:
            content = f.read()
        
        for export in spec.required_exports:
            if f"class {export}" not in content and f"def {export}" not in content:
                issues.append(f"Missing required export: {export}")
        
        return ModuleValidationResult(
            module_path=spec.path,
            passed=len(issues) == 0,
            issues=issues,
            line_count=line_count
        )

@dataclass
class PackageModule:
    path: str
    max_lines: int
    required_exports: List[str]
    dependencies: List[str]

@dataclass
class ModuleValidationResult:
    module_path: str
    passed: bool
    issues: List[str]
    line_count: Optional[int] = None

@dataclass
class StructureValidationResult:
    overall_passed: bool
    module_results: List[ModuleValidationResult]
```

### 1.2 Dependency Management Strategy

**TDD Anchor**: [`test_dependency_isolation()`](rag_templates_core_architecture_specification.md:1.2)

```python
# PSEUDOCODE: Dependency Management

class DependencyManager:
    """
    Manages optional dependencies with graceful fallbacks
    
    TDD Anchor: test_dependency_manager()
    """
    
    def __init__(self):
        self.available_backends = {}
        self.fallback_strategies = {}
        self._discover_available_backends()
    
    def _discover_available_backends(self):
        """
        Discover available optional dependencies
        
        TDD Anchor: test_discover_available_backends()
        """
        # Database backends
        self.available_backends['database'] = []
        
        try:
            import pyodbc
            self.available_backends['database'].append('odbc')
        except ImportError:
            pass
        
        try:
            import jaydebeapi
            self.available_backends['database'].append('jdbc')
        except ImportError:
            pass
        
        try:
            import iris
            self.available_backends['database'].append('dbapi')
        except ImportError:
            pass
        
        # Embedding backends
        self.available_backends['embedding'] = []
        
        try:
            import sentence_transformers
            self.available_backends['embedding'].append('sentence_transformers')
        except ImportError:
            pass
        
        try:
            import transformers
            self.available_backends['embedding'].append('transformers')
        except ImportError:
            pass
    
    def get_best_backend(self, category: str, preferred: Optional[str] = None) -> str:
        """
        Get best available backend for category
        
        TDD Anchor: test_get_best_backend()
        """
        available = self.available_backends.get(category, [])
        
        if not available:
            raise DependencyError(f"No backends available for {category}")
        
        if preferred and preferred in available:
            return preferred
        
        # Return first available (could implement priority logic)
        return available[0]
    
    def require_backend(self, category: str, backend: str):
        """
        Require specific backend or raise error
        
        TDD Anchor: test_require_backend()
        """
        available = self.available_backends.get(category, [])
        
        if backend not in available:
            raise DependencyError(
                f"Required backend '{backend}' not available for {category}. "
                f"Available: {available}"
            )
    
    def get_import_with_fallback(self, imports: List[ImportSpec]) -> Any:
        """
        Import with fallback chain
        
        TDD Anchor: test_get_import_with_fallback()
        """
        for import_spec in imports:
            try:
                module = __import__(import_spec.module, fromlist=[import_spec.name])
                return getattr(module, import_spec.name)
            except ImportError:
                continue
        
        raise DependencyError(f"None of the fallback imports succeeded: {imports}")

@dataclass
class ImportSpec:
    module: str
    name: str
    description: str

class DependencyError(Exception):
    """Dependency management errors"""
    pass
```

## 2. API Design and Interface Contracts

### 2.1 Core Interface Definitions

**TDD Anchor**: [`test_core_interface_contracts()`](rag_templates_core_architecture_specification.md:2.1)

```python
# PSEUDOCODE: Core Interface Contracts

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions"""
    
    def __call__(self, text: str) -> List[float]:
        """Generate embedding for text"""
        ...

@runtime_checkable
class LLMFunction(Protocol):
    """Protocol for LLM functions"""
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate response for prompt"""
        ...

class RAGPipeline(ABC):
    """
    Core interface contract for all RAG pipelines
    
    TDD Anchor: test_rag_pipeline_contract()
    """
    
    # Required interface methods
    @abstractmethod
    def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents"""
        pass
    
    @abstractmethod
    def store_document(self, document: Document, **kwargs) -> str:
        """Store document and return ID"""
        pass
    
    @abstractmethod
    def generate_answer(
        self,
        query: str,
        documents: List[Document],
        **kwargs
    ) -> str:
        """Generate answer from documents"""
        pass
    
    # Standard interface methods (implemented in base class)
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute complete pipeline"""
        documents = self.retrieve_documents(query, **kwargs)
        answer = self.generate_answer(query, documents, **kwargs)
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_documents': [doc.to_dict() for doc in documents],
            'metadata': self._get_execution_metadata(**kwargs)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check pipeline health"""
        return {
            'status': 'healthy',
            'components': self._check_components(),
            'timestamp': time.time()
        }
    
    def get_document_count(self) -> int:
        """Get total document count"""
        raise NotImplementedError("Subclasses must implement")
    
    # Internal methods
    @abstractmethod
    def _get_execution_metadata(self, **kwargs) -> Dict[str, Any]:
        """Get pipeline-specific metadata"""
        pass
    
    @abstractmethod
    def _check_components(self) -> Dict[str, Any]:
        """Check component health"""
        pass

class ConnectionManager(ABC):
    """
    Core interface for database connection management
    
    TDD Anchor: test_connection_manager_contract()
    """
    
    @abstractmethod
    def connect(self) -> Any:
        """Establish database connection"""
        pass
    
    @abstractmethod
    def execute(self, query: str, params: Optional[List] = None) -> Any:
        """Execute query with optional parameters"""
        pass
    
    @abstractmethod
    def execute_many(self, query: str, params_list: List[List]) -> Any:
        """Execute query with multiple parameter sets"""
        pass
    
    @abstractmethod
    def close(self):
        """Close database connection"""
        pass
    
    @contextmanager
    def transaction(self):
        """Transaction context manager"""
        try:
            yield
            self.commit()
        except Exception:
            self.rollback()
            raise
    
    @abstractmethod
    def commit(self):
        """Commit transaction"""
        pass
    
    @abstractmethod
    def rollback(self):
        """Rollback transaction"""
        pass
```

### 2.2 Configuration Interface

**TDD Anchor**: [`test_configuration_interface()`](rag_templates_core_architecture_specification.md:2.2)

```python
# PSEUDOCODE: Configuration Interface

class ConfigurationManager:
    """
    Manages configuration loading and validation
    
    TDD Anchor: test_configuration_manager()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.schema_validator = ConfigValidator()
        self._loaded_config = None
    
    def load_config(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        environment_override: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from multiple sources
        
        TDD Anchor: test_load_config()
        """
        # Priority order: config_dict > config_file > default_config > environment
        config = self._get_default_config()
        
        # Load from file if specified
        if config_file or self.config_path:
            file_config = self._load_from_file(config_file or self.config_path)
            config = self._merge_configs(config, file_config)
        
        # Override with provided dictionary
        if config_dict:
            config = self._merge_configs(config, config_dict)
        
        # Apply environment variable overrides
        if environment_override:
            config = self._apply_environment_overrides(config)
        
        # Validate final configuration
        validation_result = self.schema_validator.validate(config)
        if not validation_result.is_valid:
            raise ConfigurationError(f"Invalid configuration: {validation_result.errors}")
        
        self._loaded_config = config
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'database': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'schema': 'RAG',
                'connection_type': 'odbc'
            },
            'embedding': {
                'backend': 'auto',
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384
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
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides
        
        TDD Anchor: test_apply_environment_overrides()
        """
        env_mappings = {
            'RAG_DB_HOST': ['database', 'host'],
            'RAG_DB_PORT': ['database', 'port'],
            'RAG_DB_NAMESPACE': ['database', 'namespace'],
            'RAG_DB_SCHEMA': ['database', 'schema'],
            'RAG_CONNECTION_TYPE': ['database', 'connection_type'],
            'RAG_EMBEDDING_MODEL': ['embedding', 'model_name'],
            'RAG_EMBEDDING_BACKEND': ['embedding', 'backend'],
            'RAG_TABLE_NAME': ['storage', 'table_name']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Navigate to nested config location
                current = config
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                
                # Set value with type conversion
                current[config_path[-1]] = self._convert_env_value(env_value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Return as string
        return value

class ConfigurationError(Exception):
    """Configuration validation errors"""
    pass
```

## 3. Performance and Quality Requirements

### 3.1 Performance Specifications

**TDD Anchor**: [`test_performance_requirements()`](rag_templates_core_architecture_specification.md:3.1)

```python
# PSEUDOCODE: Performance Requirements

class PerformanceRequirements:
    """
    Defines performance requirements and validation
    
    TDD Anchor: test_performance_requirements()
    """
    
    # Response time requirements (milliseconds)
    RESPONSE_TIME_LIMITS = {
        'document_retrieval': 500,  # Document retrieval < 500ms
        'document_storage': 200,    # Document storage < 200ms
        'embedding_generation': 1000,  # Embedding generation < 1s
        'answer_generation': 5000,  # Answer generation < 5s
        'health_check': 100,        # Health check < 100ms
        'pipeline_run': 10000       # Complete pipeline < 10s
    }
    
    # Throughput requirements
    THROUGHPUT_LIMITS = {
        'documents_per_second': 10,     # Store 10 docs/sec minimum
        'queries_per_second': 5,        # Process 5 queries/sec minimum
        'concurrent_connections': 50    # Support 50 concurrent connections
    }
    
    # Memory requirements
    MEMORY_LIMITS = {
        'max_memory_per_pipeline': 1024 * 1024 * 1024,  # 1GB per pipeline
        'max_document_size': 10 * 1024 * 1024,          # 10MB per document
        'embedding_cache_size': 100 * 1024 * 1024       # 100MB embedding cache
    }
    
    @classmethod
    def validate_performance(cls, metrics: Dict[str, float]) -> PerformanceReport:
        """
        Validate performance metrics against requirements
        
        TDD Anchor: test_validate_performance()
        """
        results = []
        
        for metric_name, measured_value in metrics.items():
            if metric_name in cls.RESPONSE_TIME_LIMITS:
                limit = cls.RESPONSE_TIME_LIMITS[metric_name]
                passed = measured_value <= limit
                results.append(PerformanceResult(
                    metric=metric_name,
                    measured=measured_value,
                    limit=limit,
                    passed=passed,
                    category='response_time'
                ))
            
            elif metric_name in cls.THROUGHPUT_LIMITS:
                limit = cls.THROUGHPUT_LIMITS[metric_name]
                passed = measured_value >= limit
                results.append(PerformanceResult(
                    metric=metric_name,
                    measured=measured_value,
                    limit=limit,
                    passed=passed,
                    category='throughput'
                ))
        
        overall_passed = all(result.passed for result in results)
        
        return PerformanceReport(
            overall_passed=overall_passed,
            results=results
        )

@dataclass
class PerformanceResult:
    metric: str
    measured: float
    limit: float
    passed: bool
    category: str

@dataclass
class PerformanceReport:
    overall_passed: bool
    results: List[PerformanceResult]
```

This core architecture specification provides the foundation for implementing the RAG templates refactoring with proper structure validation, dependency management, interface contracts, and performance requirements. Each component includes comprehensive TDD anchors for test-driven development.