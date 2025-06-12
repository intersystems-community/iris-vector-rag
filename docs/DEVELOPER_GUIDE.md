# Developer Guide

Complete guide for developing, extending, and contributing to the RAG Templates project.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Environment Setup](#development-environment-setup)
- [Code Organization](#code-organization)
- [Design Patterns](#design-patterns)
- [Extension Patterns](#extension-patterns)
- [Testing Strategy](#testing-strategy)
- [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

### System Architecture

The RAG Templates framework follows a modular, layered architecture designed for extensibility and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Factory  │  Configuration  │  Personal Assistant │
│                    │     Manager     │      Adapter        │
├─────────────────────────────────────────────────────────────┤
│                     Pipeline Layer                          │
├─────────────────────────────────────────────────────────────┤
│ BasicRAG │ ColBERT │ CRAG │ GraphRAG │ HyDE │ NodeRAG │ ... │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                             │
├─────────────────────────────────────────────────────────────┤
│ RAGPipeline │ ConnectionManager │ Document │ Exceptions     │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                      │
├─────────────────────────────────────────────────────────────┤
│ Storage Layer │ Embedding Manager │ Config Loader │ Utils  │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
├─────────────────────────────────────────────────────────────┤
│              InterSystems IRIS Backend                      │
└─────────────────────────────────────────────────────────────┘
```

### Component Relationships

#### Core Components

1. **[`RAGPipeline`](../rag_templates/core/base.py:3)** - Abstract base class defining the pipeline interface
2. **[`ConnectionManager`](../rag_templates/core/connection.py:19)** - Database connection management with caching
3. **[`ConfigurationManager`](../rag_templates/config/manager.py:10)** - Configuration loading from YAML and environment
4. **[`Document`](../rag_templates/core/models.py:10)** - Immutable document representation

#### Pipeline Implementations

Each RAG technique implements the [`RAGPipeline`](../rag_templates/core/base.py:3) interface:

- **BasicRAG**: Standard vector similarity search
- **ColBERT**: Token-level retrieval with late interaction
- **CRAG**: Corrective RAG with retrieval evaluation
- **GraphRAG**: Knowledge graph-enhanced retrieval
- **HyDE**: Hypothetical document embeddings
- **NodeRAG**: Node-based document representation
- **HybridIFindRAG**: Native IRIS iFind integration

### Data Flow

```
Query Input → Pipeline Selection → Document Retrieval → 
Context Augmentation → Answer Generation → Response Output
```

1. **Query Processing**: Input validation and preprocessing
2. **Retrieval**: Vector search or technique-specific retrieval
3. **Augmentation**: Context preparation and prompt engineering
4. **Generation**: LLM-based answer generation
## Schema Management for Developers

### SchemaManager Overview

The [`SchemaManager`](../iris_rag/storage/schema_manager.py:16) is a critical component for maintaining database schema integrity. Developers working with vector embeddings and database schemas should understand its capabilities and integration patterns.

### Key Developer Considerations

#### 1. Automatic Schema Validation

When creating new pipelines that store vector data, always ensure schema validation:

```python
from iris_rag.storage.schema_manager import SchemaManager

class MyCustomPipeline(RAGPipeline):
    def __init__(self, connection_manager, config_manager):
        self.schema_manager = SchemaManager(connection_manager, config_manager)
        
    def store_vectors(self, table_name: str, data: List[Dict]):
        # Always validate schema before storing vector data
        if not self.schema_manager.ensure_table_schema(table_name):
            raise RuntimeError(f"Schema validation failed for {table_name}")
        
        # Proceed with data storage...
```

#### 2. Extending SchemaManager for New Tables

To add support for new tables, extend the `_get_expected_schema_config` method:

```python
def _get_expected_schema_config(self, table_name: str) -> Dict[str, Any]:
    if table_name == "MyCustomTable":
        embedding_config = self.config_manager.get_embedding_config()
        model_name = embedding_config.get("model", "all-MiniLM-L6-v2")
        
        return {
            "schema_version": self.schema_version,
            "vector_dimension": self._get_model_dimension(model_name),
            "embedding_model": model_name,
            "configuration": {
                "table_type": "custom_storage",
                "supports_vector_search": True,
                "created_by": "MyCustomPipeline"
            }
        }
    
    # Call parent method for existing tables
    return super()._get_expected_schema_config(table_name)
```

#### 3. Migration Strategy Development

When developing new migration strategies, consider:

- **Data Preservation**: Implement backup/restore mechanisms
- **Rollback Capability**: Ensure migrations can be reversed
- **Performance**: Minimize downtime during migrations
- **Validation**: Verify schema integrity after migration

#### 4. Configuration Integration

Ensure your pipeline configurations work with SchemaManager:

```yaml
# config.yaml
embeddings:
  model: "all-mpnet-base-v2"  # 768 dimensions
  
storage:
  iris:
    vector_data_type: "FLOAT"  # or "DOUBLE"
    
pipelines:
  my_custom_pipeline:
    auto_migrate: true
    preserve_data: false  # Set to true for production
```

### Best Practices for Schema Management

1. **Always Call Schema Validation**: Before any vector storage operations
2. **Handle Migration Failures**: Implement proper error handling and rollback
3. **Test Schema Changes**: Use development environments to test migrations
4. **Monitor Schema Status**: Regularly check schema health in production
5. **Document Schema Changes**: Maintain clear documentation of schema evolution

### Testing Schema Management

```python
def test_schema_migration():
    # Test schema migration with different embedding models
    config_manager.update_config("embeddings.model", "all-mpnet-base-v2")
    
    # Should trigger migration from 384 to 768 dimensions
    assert schema_manager.needs_migration("DocumentEntities")
    
    # Perform migration
    success = schema_manager.migrate_table("DocumentEntities")
    assert success
    
    # Verify new schema
    config = schema_manager.get_current_schema_config("DocumentEntities")
    assert config["vector_dimension"] == 768
```
5. **Post-processing**: Response formatting and metadata

## Development Environment Setup

### Prerequisites

- **Python**: 3.11 or higher
- **InterSystems IRIS**: 2025.1 or higher (Community or Licensed)
- **Git**: For version control
- **Docker**: For containerized development (optional)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/your-org/intersystems-iris-rag.git
cd intersystems-iris-rag
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Development installation with all extras
pip install -e ".[dev,test,docs]"

# Or install specific dependency groups
pip install -e ".[dev]"      # Development tools
pip install -e ".[test]"     # Testing dependencies
pip install -e ".[docs]"     # Documentation tools
```

#### 4. Set Up IRIS Database

**Option A: Docker (Recommended)**
```bash
# Start IRIS container
docker run -d \
  --name iris-rag-dev \
  -p 1972:1972 \
  -p 52773:52773 \
  -e IRIS_PASSWORD=SYS \
  intersystemsdc/iris-community:latest

# Verify connection
docker exec iris-rag-dev iris session iris -U USER
```

**Option B: Local Installation**
Download from [InterSystems Developer Community](https://community.intersystems.com/)

#### 5. Configure Environment

Create `.env` file:
```bash
# Database configuration
RAG_DATABASE__IRIS__HOST=localhost
RAG_DATABASE__IRIS__PORT=1972
RAG_DATABASE__IRIS__USERNAME=demo
RAG_DATABASE__IRIS__PASSWORD=demo
RAG_DATABASE__IRIS__NAMESPACE=USER

# Development settings
RAG_LOG_LEVEL=DEBUG
RAG_ENABLE_PROFILING=true
```

#### 6. Initialize Database Schema

```bash
python -c "
from iris_rag.core import ConnectionManager, ConfigurationManager
from iris_rag.storage.iris import IRISStorage

config = ConfigurationManager()
conn_mgr = ConnectionManager(config)
storage = IRISStorage(conn_mgr, config)
storage.initialize_schema()
print('Database schema initialized successfully')
"
```

#### 7. Run Tests

```bash
# Run basic tests
pytest tests/

# Run with coverage
pytest --cov=iris_rag tests/

# Run specific test categories
pytest tests/test_core/          # Core functionality
pytest tests/test_pipelines/     # Pipeline implementations
pytest tests/test_integration/   # Integration tests
```

### Development Tools

#### Code Quality Tools

```bash
# Code formatting
black iris_rag/ tests/
isort iris_rag/ tests/

# Linting
flake8 iris_rag/ tests/
pylint iris_rag/

# Type checking
mypy iris_rag/
```

#### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Code Organization

### Package Structure

```
iris_rag/
├── __init__.py              # Main package exports
├── core/                    # Core abstractions and interfaces
│   ├── __init__.py
│   ├── base.py             # RAGPipeline abstract base class
│   ├── connection.py       # ConnectionManager implementation
│   ├── models.py           # Document and data models
│   └── exceptions.py       # Custom exceptions
├── config/                  # Configuration management
│   ├── __init__.py
│   └── manager.py          # ConfigurationManager implementation
├── pipelines/              # RAG pipeline implementations
│   ├── __init__.py
│   ├── basic.py           # BasicRAG implementation
│   ├── colbert.py         # ColBERT implementation
│   ├── crag.py            # CRAG implementation
│   ├── graphrag.py        # GraphRAG implementation
│   ├── hyde.py            # HyDE implementation
│   ├── noderag.py         # NodeRAG implementation
│   └── hybrid_ifind.py    # HybridIFindRAG implementation
├── storage/                # Storage layer implementations
│   ├── __init__.py
│   └── iris.py            # IRIS storage implementation
├── embeddings/             # Embedding management
│   ├── __init__.py
│   └── manager.py         # EmbeddingManager implementation
├── adapters/               # External system adapters
│   ├── __init__.py
│   └── personal_assistant.py  # Personal Assistant adapter
└── utils/                  # Utility functions
    ├── __init__.py
    ├── migration.py       # Migration utilities
    └── validation.py      # Validation helpers
```

### Module Guidelines

#### File Size Limits

- **Core modules**: Maximum 300 lines
- **Pipeline implementations**: Maximum 500 lines
- **Utility modules**: Maximum 200 lines
- **Test files**: Maximum 1000 lines

#### Import Organization

```python
# Standard library imports
import os
import time
from typing import Dict, List, Optional

# Third-party imports
import yaml
import numpy as np

# Local imports
from iris_rag.core.base import RAGPipeline
from iris_rag.core.models import Document
from iris_rag.config.manager import ConfigurationManager
```

#### Naming Conventions

- **Classes**: PascalCase (`RAGPipeline`, `ConnectionManager`)
- **Functions/Methods**: snake_case (`execute()`, `load_documents()`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TOP_K`, `MAX_RETRIES`)
- **Private members**: Leading underscore (`_internal_method()`)

## Design Patterns

### 1. Abstract Factory Pattern

Used for pipeline creation with automatic configuration:

```python
# Factory implementation
class PipelineFactory:
    _registry = {}
    
    @classmethod
    def register(cls, name: str, pipeline_class: type):
        cls._registry[name] = pipeline_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> RAGPipeline:
        if name not in cls._registry:
            raise ValueError(f"Unknown pipeline: {name}")
        return cls._registry[name](**kwargs)

# Usage
pipeline = PipelineFactory.create("basic", config_manager=config)
```

### 2. Strategy Pattern

Used for different embedding backends:

```python
class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

class SentenceTransformersStrategy(EmbeddingStrategy):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Implementation
        pass

class OpenAIStrategy(EmbeddingStrategy):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Implementation
        pass
```

### 3. Dependency Injection

Used throughout for testability and flexibility:

```python
class BasicRAGPipeline(RAGPipeline):
    def __init__(
        self,
        connection_manager: ConnectionManager,
        config_manager: ConfigurationManager,
        embedding_manager: Optional[EmbeddingManager] = None,
        llm_func: Optional[Callable] = None
    ):
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.embedding_manager = embedding_manager or EmbeddingManager(config_manager)
        self.llm_func = llm_func or self._default_llm_func
```

### 4. Template Method Pattern

Used in the base [`RAGPipeline`](../rag_templates/core/base.py:3) class:

```python
class RAGPipeline(ABC):
    def execute(self, query_text: str, **kwargs) -> dict:
        # Template method defining the algorithm structure
        documents = self.retrieve_documents(query_text, **kwargs)
        answer = self.generate_answer(query_text, documents, **kwargs)
        
        return {
            'query': query_text,
            'answer': answer,
            'retrieved_documents': documents,
            'metadata': self._get_execution_metadata(**kwargs)
        }
    
    @abstractmethod
    def retrieve_documents(self, query_text: str, **kwargs) -> List[Document]:
        pass
    
    @abstractmethod
    def generate_answer(self, query_text: str, documents: List[Document], **kwargs) -> str:
        pass
```

## Extension Patterns

### Adding New RAG Techniques

#### 1. Create Pipeline Implementation

```python
# iris_rag/pipelines/my_technique.py
from typing import List, Dict, Any
from iris_rag.core.base import RAGPipeline
from iris_rag.core.models import Document

class MyTechniqueRAGPipeline(RAGPipeline):
    """
    Implementation of My Technique RAG approach.
    
    This technique implements [describe the approach].
    """
    
    def __init__(self, connection_manager, config_manager, **kwargs):
        super().__init__()
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        # Initialize technique-specific components
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Load and process documents for My Technique."""
        # Implementation specific to your technique
        pass
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Document]:
        """Retrieve documents using My Technique approach."""
        # Implementation specific to your technique
        pass
    
    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """Execute the complete My Technique pipeline."""
        # Use the template method or override for custom flow
        return super().execute(query_text, **kwargs)
```

#### 2. Register Pipeline

```python
# iris_rag/pipelines/__init__.py
from .my_technique import MyTechniqueRAGPipeline

# Register with factory
PipelineFactory.register("my_technique", MyTechniqueRAGPipeline)
```

#### 3. Add Configuration Schema

```python
# Add to default configuration
DEFAULT_CONFIG = {
    'pipelines': {
        'my_technique': {
            'parameter1': 'default_value',
            'parameter2': 100,
            'enable_feature': True
        }
    }
}
```

#### 4. Write Tests

```python
# tests/test_pipelines/test_my_technique.py
import pytest
from iris_rag.pipelines.my_technique import MyTechniqueRAGPipeline

class TestMyTechniqueRAGPipeline:
    def test_initialization(self, mock_connection_manager, mock_config_manager):
        pipeline = MyTechniqueRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager
        )
        assert pipeline is not None
    
    def test_execute_returns_expected_format(self, pipeline, sample_query):
        result = pipeline.execute(sample_query)
        
        assert 'query' in result
        assert 'answer' in result
        assert 'retrieved_documents' in result
        assert result['query'] == sample_query
```

### Adding New Storage Backends

#### 1. Implement Storage Interface

```python
# iris_rag/storage/my_backend.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from iris_rag.core.models import Document

class StorageBackend(ABC):
    @abstractmethod
    def store_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def vector_search(self, query_vector: List[float], top_k: int) -> List[Document]:
        pass

class MyBackendStorage(StorageBackend):
    def __init__(self, connection_manager, config_manager):
        self.connection_manager = connection_manager
        self.config_manager = config_manager
    
    def store_documents(self, documents: List[Document]) -> None:
        # Implementation for your backend
        pass
    
    def vector_search(self, query_vector: List[float], top_k: int) -> List[Document]:
        # Implementation for your backend
        pass
```

### Adding New Embedding Backends

#### 1. Implement Embedding Strategy

```python
# iris_rag/embeddings/my_backend.py
from typing import List
from iris_rag.embeddings.base import EmbeddingStrategy

class MyEmbeddingBackend(EmbeddingStrategy):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize your embedding model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Implementation for your embedding backend
        pass
    
    def embed_query(self, query: str) -> List[float]:
        # Implementation for query embedding
        pass
```

## Testing Strategy

### Test-Driven Development (TDD)

The project follows TDD principles as defined in [`.clinerules`](../.clinerules):

1. **Red**: Write failing tests first
2. **Green**: Implement minimum code to pass
3. **Refactor**: Clean up while keeping tests passing

### Test Categories

#### 1. Unit Tests

Test individual components in isolation:

```python
# tests/test_core/test_base.py
def test_rag_pipeline_abstract_methods():
    """Test that RAGPipeline cannot be instantiated directly."""
    with pytest.raises(TypeError):
        RAGPipeline()

def test_rag_pipeline_subclass_must_implement_abstract_methods():
    """Test that subclasses must implement abstract methods."""
    class IncompleteRAGPipeline(RAGPipeline):
        pass
    
    with pytest.raises(TypeError):
        IncompleteRAGPipeline()
```

#### 2. Integration Tests

Test component interactions:

```python
# tests/test_integration/test_pipeline_integration.py
def test_basic_rag_end_to_end(iris_connection, sample_documents):
    """Test complete BasicRAG pipeline execution."""
    config = ConfigurationManager("test_config.yaml")
    conn_mgr = ConnectionManager(config)
    
    pipeline = BasicRAGPipeline(conn_mgr, config)
    pipeline.load_documents(sample_documents)
    
    result = pipeline.execute("What is machine learning?")
    
    assert 'answer' in result
    assert len(result['retrieved_documents']) > 0
```

#### 3. Real Data Tests

Test with actual PMC documents (1000+ docs):

```python
# tests/test_real_data/test_1000_docs.py
@pytest.mark.real_data
def test_all_techniques_with_1000_docs(pmc_1000_docs_fixture):
    """Test all RAG techniques with 1000+ real documents."""
    techniques = ['basic', 'colbert', 'crag', 'graphrag', 'hyde', 'noderag']
    
    for technique in techniques:
        pipeline = create_pipeline(technique, config_path="test_config.yaml")
        pipeline.load_documents(pmc_1000_docs_fixture)
        
        result = pipeline.execute("What are the effects of diabetes?")
        
        assert result['answer']
        assert len(result['retrieved_documents']) > 0
```

### Test Configuration

#### pytest Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    real_data: Tests requiring real PMC data
    slow: Slow tests (>5 seconds)
    performance: Performance benchmarks
```

#### Test Fixtures

```python
# tests/conftest.py
import pytest
from iris_rag.core import ConnectionManager, ConfigurationManager

@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for testing."""
    config = {
        'database': {
            'iris': {
                'host': 'localhost',
                'port': 1972,
                'username': 'test',
                'password': 'test'
            }
        }
    }
    return ConfigurationManager(config_dict=config)

@pytest.fixture
def iris_connection(mock_config_manager):
    """Real IRIS connection for integration tests."""
    conn_mgr = ConnectionManager(mock_config_manager)
    return conn_mgr.get_connection('iris')
```

### Performance Testing

#### Benchmark Tests

```python
# tests/test_performance/test_benchmarks.py
import time
import pytest

@pytest.mark.performance
def test_basic_rag_performance(pipeline, sample_queries):
    """Test BasicRAG performance with multiple queries."""
    response_times = []
    
    for query in sample_queries:
        start_time = time.time()
        result = pipeline.execute(query)
        end_time = time.time()
        
        response_times.append(end_time - start_time)
        assert result['answer']  # Ensure valid response
    
    avg_response_time = sum(response_times) / len(response_times)
    assert avg_response_time < 1.0  # Should respond within 1 second
```

## Contributing Guidelines

### Code Standards

#### 1. Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 88 characters

#### 2. Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints for all function signatures
- Update relevant documentation files

```python
def execute(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Execute the RAG pipeline for a given query.
    
    Args:
        query_text: The input query string.
        top_k: Number of documents to retrieve.
        **kwargs: Additional pipeline-specific arguments.
    
    Returns:
        Dictionary containing query, answer, retrieved documents, and metadata.
    
    Raises:
        ValueError: If query_text is empty or invalid.
        ConnectionError: If database connection fails.
    """
```

#### 3. Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately

```python
try:
    result = self.embedding_manager.embed_texts([query_text])
except EmbeddingError as e:
    logger.error(f"Failed to generate embeddings for query: {e}")
    raise RAGPipelineError(f"Embedding generation failed: {e}") from e
```

### Contribution Process

#### 1. Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes and commit
git add .
git commit -m "feat: add new RAG technique implementation"

# 3. Run tests
pytest tests/

# 4. Run code quality checks
black iris_rag/ tests/
flake8 iris_rag/ tests/
mypy iris_rag/

# 5. Push and create pull request
git push origin feature/my-new-feature
```

#### 2. Commit Message Format

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

#### 3. Pull Request Requirements

- [ ] All tests pass
- [ ] Code coverage maintained (>90%)
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Performance impact assessed
- [ ] Security implications reviewed

#### 4. Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Reviewer tests the changes locally
4. **Documentation**: Ensure documentation is complete and accurate
5. **Merge**: Approved changes are merged to main branch

### Release Process

#### 1. Version Management

- Follow semantic versioning (SemVer)
- Update version in `pyproject.toml`
- Create release notes in [`CHANGELOG.md`](CHANGELOG.md)

#### 2. Release Checklist

- [ ] All tests pass on main branch
- [ ] Documentation is up to date
- [ ] Version number updated
- [ ] Release notes prepared
- [ ] Security scan completed
- [ ] Performance benchmarks run

#### 3. Deployment

```bash
# 1. Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"

# 2. Build package
python -m build

# 3. Upload to PyPI
python -m twine upload dist/*
```

---

For additional information, see:
- [API Reference](API_REFERENCE.md)
- [User Guide](USER_GUIDE.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Performance Guide](PERFORMANCE_GUIDE.md)