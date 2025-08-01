# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dead-simple library for building Retrieval Augmented Generation (RAG) applications using InterSystems IRIS vector database. The project provides zero-configuration APIs in both Python and JavaScript with enterprise-grade capabilities.

## Common Commands

### Essential Development Commands

```bash
# Environment Setup (using uv)
make setup-env          # Verify uv is installed
make install            # Install dependencies with uv
make setup-db          # Initialize IRIS database schema

# Testing (all commands use uv run)
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests
make test-e2e          # Run end-to-end tests
make test-ragas-1000-enhanced  # Full RAGAS evaluation on all pipelines

# Running a single test
uv run pytest tests/test_simple_api_phase1.py::test_basic_functionality -v

# Data Management
make load-data         # Load sample PMC documents
make clear-rag-data    # Clear RAG document tables

# Docker/IRIS Management
make docker-up         # Start IRIS container
make docker-down       # Stop IRIS container
make docker-logs       # View container logs

# JavaScript/Node.js (in nodejs/ directory)
npm install            # Install dependencies
npm test              # Run tests
npm run lint          # Run ESLint
npm run format        # Format with Prettier
```

### Linting and Type Checking

```bash
# Python (using uv)
uv run pylint iris_rag/
uv run mypy iris_rag/
uv run flake8 iris_rag/ tests/ --max-line-length=120
uv run black iris_rag/ tests/ --line-length=120

# JavaScript (in nodejs/)
npm run lint
```

## Architecture Overview

### Three-Tier API Design

1. **Simple API** (`rag_templates/simple.py`): Zero-configuration, works immediately
2. **Standard API** (`rag_templates/standard.py`): Basic configuration options
3. **Enterprise API** (`iris_rag/`): Full control, all techniques available

### Core Components

**RAG Pipeline Implementations** (`iris_rag/pipelines/`):
- `basic.py` - Standard RAG with semantic search
- `colbert.py` - ColBERT token-level embeddings
- `crag.py` - Corrective RAG with validation
- `hyde.py` - Hypothetical Document Embeddings
- `graphrag.py` - Graph-based RAG
- `noderag.py` - Node.js-based chunking
- `hybrid_ifind.py` - Hybrid semantic/keyword search

**Storage Layer** (`iris_rag/storage/`):
- IRIS vector database integration
- Document storage and retrieval
- Vector similarity search operations

**Configuration System** (`iris_rag/config/`):
- Hierarchical configuration management
- Pipeline-specific settings
- Environment variable support

**Monitoring** (`iris_rag/monitoring/`):
- Real-time health checking
- Performance metrics collection
- System resource monitoring

### Key Design Patterns

1. **Factory Pattern**: Pipeline creation through `PipelineFactory`
2. **Strategy Pattern**: Different RAG techniques as interchangeable strategies
3. **Builder Pattern**: Configuration builders for complex setups
4. **Observer Pattern**: Event-driven monitoring system

### Database Schema & Connections

The system uses InterSystems IRIS with custom vector operations:
- `rag_schemas.document` - Document storage with embeddings  
- `rag_schemas.chunk` - Document chunks with vector embeddings
- Custom vector similarity functions in ObjectScript

**IRIS Connection Architecture:**
- **DBAPI System** (`iris_dbapi_connector`) - For RAG queries and data operations
- **JDBC System** (`iris_connection_manager`) - For schema management with DBAPI→JDBC fallback
- See [IRIS Connection Architecture Guide](docs/IRIS_CONNECTION_ARCHITECTURE.md) for detailed usage patterns

### Testing Strategy

Tests are organized in phases:
- Phase 1: Simple API validation
- Phase 2: Standard API features
- Phase 3: JavaScript implementation
- Phase 4: Enterprise features
- Phase 5: ObjectScript integration

Use `@pytest.mark.phase1` through `@pytest.mark.phase5` to run specific test phases.

## Important Considerations

### Environment Management
- This project uses `uv` for Python dependency management
- All Python commands should be prefixed with `uv run`
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Dependencies are locked in pyproject.toml (uv.lock is gitignored)

### Environment Variables
- `PYTHONPATH` is automatically managed by uv
- `IRIS_USERNAME` / `IRIS_PASSWORD` for database access
- `OPENAI_API_KEY` for embeddings and LLM calls

### Declarative State Management
The project uses a declarative reconciliation system for managing database state:

```python
# Declare desired state
from iris_rag.controllers.declarative_state import ensure_documents
ensure_documents(count=1000, pipeline="colbert")

# Or use YAML specification
from iris_rag.controllers.declarative_state import DeclarativeStateManager
manager = DeclarativeStateManager()
manager.declare_state("config/example_states/production.yaml")
manager.sync()
```

Key features:
- **Drift Detection**: Automatically detects when actual state differs from desired
- **Auto-Remediation**: Reconciles differences to achieve desired state
- **Test Isolation**: Each test gets isolated tables to prevent contamination
- **Quality Enforcement**: Validates embedding diversity and mock contamination

### Performance Notes
- ColBERT pipeline requires significant memory for token embeddings
- Use caching system for repeated queries
- Monitor vector dimension consistency (1536 for OpenAI embeddings)

### Common Pitfalls
- Ensure IRIS container is running before tests
- Vector dimensions must match across documents
- JavaScript and Python APIs maintain feature parity
- MCP server requires explicit feature flag installation