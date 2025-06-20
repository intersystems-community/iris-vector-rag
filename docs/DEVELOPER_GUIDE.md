# Developer Guide

Complete guide for developing, extending, and contributing to the RAG Templates project.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Environment Setup](#development-environment-setup)
- [Code Organization](#code-organization)
- [Design Patterns](#design-patterns)
- [Extension Patterns](#extension-patterns)
- [Testing Strategy](#testing-strategy)
- [CLI Development](#cli-development)
- [Database Integration](#database-integration)
- [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

### System Architecture

The RAG Templates framework follows a modular, layered architecture designed for extensibility and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  CLI (ragctl)     │  Configuration  │  Controllers         │
│                   │     Manager     │                      │
├─────────────────────────────────────────────────────────────┤
│                     Pipeline Layer                          │
├─────────────────────────────────────────────────────────────┤
│ BasicRAG │ ColBERT │ CRAG │ GraphRAG │ HyDE │ HybridIFind   │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                             │
├─────────────────────────────────────────────────────────────┤
│ RAGPipeline │ ConnectionManager │ Document │ Exceptions     │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                      │
├─────────────────────────────────────────────────────────────┤
│ Storage Layer │ Embedding Manager │ Schema Manager │ Utils  │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
├─────────────────────────────────────────────────────────────┤
│              InterSystems IRIS Backend                      │
└─────────────────────────────────────────────────────────────┘
```

### Component Relationships

#### Core Components

1. **[`ConnectionManager`](iris_rag/core/connection.py:23)** - Database connection management with caching
2. **[`ConfigurationManager`](iris_rag/config/manager.py:10)** - Configuration loading from YAML and environment
3. **[`EmbeddingManager`](iris_rag/embeddings/manager.py:15)** - Unified embedding generation with fallback support
4. **[`SchemaManager`](iris_rag/storage/schema_manager.py:16)** - Database schema versioning and migration

#### Pipeline Implementations

Each RAG technique implements a common pipeline interface:

- **BasicRAG**: Standard vector similarity search
- **ColBERT**: Token-level retrieval with late interaction
- **CRAG**: Corrective RAG with retrieval evaluation
- **GraphRAG**: Knowledge graph-enhanced retrieval
- **HyDE**: Hypothetical document embeddings
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
5. **Post-processing**: Response formatting and metadata

## Development Environment Setup

### Prerequisites

- **Python**: 3.11 or higher
- **InterSystems IRIS**: 2025.1 or higher (Community or Licensed)
- **Git**: For version control
- **Docker**: For containerized development (recommended)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/your-org/rag-templates.git
cd rag-templates
```

#### 2. Set Up Python Virtual Environment

```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Install dependencies using pip within the activated virtual environment
pip install -r requirements.txt

# For editable mode (recommended for development)
pip install -e .
```

#### 4. Set Up IRIS Database

**Option A: Docker (Recommended)**
```bash
# Start IRIS container
docker-compose up -d

# Verify connection
docker exec iris_db_rag_standalone iris session iris -U USER
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
# Using Makefile
make setup-db

# Or manually
python common/db_init_with_indexes.py
```

#### 7. Load Sample Data

```bash
# Load sample documents
make load-data

# Load 1000+ documents for comprehensive testing
make load-1000
```

#### 8. Run Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-1000

# Run with coverage
pytest --cov=iris_rag tests/
```

### Development Tools

#### Code Quality Tools

```bash
# Code formatting
black iris_rag/ tests/
ruff format iris_rag/ tests/

# Linting
ruff check iris_rag/ tests/
mypy iris_rag/

# Using Makefile
make format
make lint
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
│   └── hybrid_ifind.py    # HybridIFindRAG implementation
├── storage/                # Storage layer implementations
│   ├── __init__.py
│   ├── schema_manager.py  # Schema management and migration
│   └── vector_store_iris.py # IRIS vector store implementation
├── embeddings/             # Embedding management
│   ├── __init__.py
│   └── manager.py         # EmbeddingManager implementation
├── cli/                    # Command-line interface
│   ├── __init__.py
│   ├── __main__.py        # CLI entry point
│   └── reconcile_cli.py   # Reconciliation CLI commands
├── controllers/            # High-level orchestration
│   └── __init__.py
└── utils/                  # Utility functions
    ├── __init__.py
    ├── migration.py       # Migration utilities
    └── validation.py      # Validation helpers

common/                     # Shared utilities
├── db_vector_utils.py     # Vector insertion utilities
├── iris_connection_manager.py # Connection management
└── utils.py               # Common utilities

tests/                      # Test suite
├── conftest.py            # Test fixtures
├── test_core/             # Core component tests
├── test_pipelines/        # Pipeline tests
├── test_integration/      # Integration tests
├── test_storage/          # Storage tests
├── fixtures/              # Test fixtures
└── mocks/                 # Mock objects
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
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.config.manager import ConfigurationManager
```

#### Naming Conventions

- **Classes**: PascalCase (`RAGPipeline`, `ConnectionManager`)
- **Functions/Methods**: snake_case (`execute()`, `load_documents()`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TOP_K`, `MAX_RETRIES`)
- **Private members**: Leading underscore (`_internal_method()`)

## Design Patterns

### 1. Dependency Injection

Used throughout for testability and flexibility:

```python
class BasicRAGPipeline:
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

### 2. Strategy Pattern

Used for different embedding backends:

```python
class EmbeddingManager:
    def __init__(self, config_manager: ConfigurationManager):
        self.primary_backend = config_manager.get("embeddings.primary_backend", "sentence_transformers")
        self.fallback_backends = config_manager.get("embeddings.fallback_backends", ["openai"])
        self._initialize_backend(self.primary_backend)
```

### 3. Factory Pattern

Used for pipeline creation:

```python
def create_pipeline(pipeline_type: str, **kwargs):
    """Factory function for creating pipeline instances."""
    pipeline_classes = {
        "basic": BasicRAGPipeline,
        "colbert": ColBERTRAGPipeline,
        "crag": CRAGPipeline,
        "hyde": HyDERAGPipeline,
        "graphrag": GraphRAGPipeline,
        "hybrid_ifind": HybridIFindRAGPipeline
    }
    
    if pipeline_type not in pipeline_classes:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    return pipeline_classes[pipeline_type](**kwargs)
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

__all__ = [
    "BasicRAGPipeline",
    "ColBERTRAGPipeline",
    "CRAGPipeline",
    "HyDERAGPipeline",
    "GraphRAGPipeline",
    "HybridIFindRAGPipeline",
    "MyTechniqueRAGPipeline"
]
```

#### 3. Add Configuration Schema

```yaml
# config/config.yaml
pipelines:
  my_technique:
    parameter1: 'default_value'
    parameter2: 100
    enable_feature: true
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

## Testing Strategy

### Test-Driven Development (TDD)

The project follows TDD principles as defined in [`.clinerules`](.clinerules):

1. **Red**: Write failing tests first
2. **Green**: Implement minimum code to pass
3. **Refactor**: Clean up while keeping tests passing

### Test Categories

#### 1. Unit Tests

Test individual components in isolation:

```python
# tests/test_core/test_connection.py
def test_connection_manager_initialization():
    """Test that ConnectionManager initializes correctly."""
    config_manager = ConfigurationManager()
    conn_mgr = ConnectionManager(config_manager)
    assert conn_mgr.config_manager is config_manager
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
# tests/test_comprehensive_e2e_iris_rag_1000_docs.py
@pytest.mark.real_data
def test_all_techniques_with_1000_docs():
    """Test all RAG techniques with 1000+ real documents."""
    techniques = ['basic', 'colbert', 'crag', 'graphrag', 'hyde', 'hybrid_ifind']
    
    for technique in techniques:
        pipeline = create_pipeline(technique)
        result = pipeline.execute("What are the effects of diabetes?")
        
        assert result['answer']
        assert len(result['retrieved_documents']) > 0
```

### Test Configuration

#### pytest Configuration

The project uses [`pytest.ini`](pytest.ini) for test configuration:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    requires_1000_docs: mark tests that require at least 1000 documents
    e2e_metrics: mark tests that measure end-to-end performance
    real_pmc: mark tests that require real PMC documents
    real_iris: mark tests that require a real IRIS connection
```

#### Test Fixtures

Key fixtures are defined in [`tests/conftest.py`](tests/conftest.py):

```python
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

### Running Tests

#### Using Makefile

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run comprehensive test with 1000 docs
make test-1000

# Run RAGAs evaluation
make test-ragas-1000-enhanced
```

#### Using pytest directly

```bash
# Run specific test categories
pytest tests/test_core/          # Core functionality
pytest tests/test_pipelines/     # Pipeline implementations
pytest tests/test_integration/   # Integration tests

# Run with markers
pytest -m "real_data"           # Tests requiring real data
pytest -m "requires_1000_docs"  # Tests requiring 1000+ docs

# Run with coverage
pytest --cov=iris_rag tests/
```

## CLI Development

### CLI Architecture

The project includes a comprehensive CLI tool accessible via:

- **Standalone**: [`./ragctl`](ragctl) 
- **Module**: `python -m iris_rag.cli`

### CLI Commands

```bash
# Pipeline management
./ragctl run --pipeline colbert
./ragctl status --pipeline noderag

# Daemon mode for continuous reconciliation
./ragctl daemon --interval 1800

# Configuration management
./ragctl config --validate
./ragctl config --show
```

### Adding New CLI Commands

1. **Extend the CLI module** in [`iris_rag/cli/reconcile_cli.py`](iris_rag/cli/reconcile_cli.py)
2. **Add command handlers** following the existing pattern
3. **Update help documentation** and examples
4. **Write tests** for new commands

## Database Integration

### Schema Management

The [`SchemaManager`](iris_rag/storage/schema_manager.py:16) handles database schema versioning and migrations:

```python
from iris_rag.storage.schema_manager import SchemaManager

class MyCustomPipeline:
    def __init__(self, connection_manager, config_manager):
        self.schema_manager = SchemaManager(connection_manager, config_manager)
        
    def store_vectors(self, table_name: str, data: List[Dict]):
        # Always validate schema before storing vector data
        if not self.schema_manager.ensure_table_schema(table_name):
            raise RuntimeError(f"Schema validation failed for {table_name}")
        
        # Proceed with data storage...
```

### Vector Operations

**Always use the [`common.db_vector_utils.insert_vector()`](common/db_vector_utils.py:6) utility** for vector insertions:

```python
from common.db_vector_utils import insert_vector

# Correct way to insert vectors
success = insert_vector(
    cursor=cursor,
    table_name="RAG.DocumentChunks",
    vector_column_name="embedding",
    vector_data=embedding_vector,
    target_dimension=384,
    key_columns={"chunk_id": chunk_id},
    additional_data={"content": text_content}
)
```

### SQL Guidelines

- **Use `TOP` instead of `LIMIT`**: IRIS SQL uses `SELECT TOP n` syntax
- **Use prepared statements**: Always use parameterized queries for safety
- **Handle CLOB data**: Use proper CLOB handling for large text content

## Contributing Guidelines

### Code Standards

#### 1. Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88 characters)
- Use Ruff for linting and import sorting
- Include type hints for all function signatures

#### 2. Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Update relevant documentation files
- Include code examples where appropriate

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

### Development Workflow

#### 1. Branch Strategy

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add .
git commit -m "feat: add new RAG technique implementation"

# Run tests and quality checks
make test
make format
make lint

# Push and create pull request
git push origin feature/my-new-feature
```

#### 2. Commit Message Format

Follow conventional commits as documented in [`docs/guides/COMMIT_MESSAGE.md`](docs/guides/COMMIT_MESSAGE.md):

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
- Update version in [`pyproject.toml`](pyproject.toml)
- Create release notes in `CHANGELOG.md`

#### 2. Release Checklist

- [ ] All tests pass on main branch
- [ ] Documentation is up to date
- [ ] Version number updated
- [ ] Release notes prepared
- [ ] Security scan completed
- [ ] Performance benchmarks run

#### 3. Deployment

```bash
# Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

For additional information, see:
- [Configuration Guide](CONFIGURATION.md)
- [User Guide](USER_GUIDE.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Performance Guide](PERFORMANCE_GUIDE.md)
- [CLI Usage Guide](CLI_RECONCILIATION_USAGE.md)