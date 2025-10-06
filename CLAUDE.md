# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Initial setup (using uv for modern Python package management)
make setup-env    # Create .venv using uv
make install      # Install all dependencies via uv sync
source .venv/bin/activate  # Activate virtual environment

# Database setup
docker-compose up -d  # Start IRIS database
make setup-db         # Initialize database
make load-data        # Load sample data
```

### Testing
```bash
# Run tests using the script runner
./scripts/ci/run-tests.sh           # Run all tests
./scripts/ci/run-tests.sh -t unit   # Run only unit tests
./scripts/ci/run-tests.sh -t integration -v  # Integration tests with verbose output
./scripts/ci/run-tests.sh -p -c     # Parallel execution without coverage

# Direct pytest execution
pytest tests/                       # All tests
pytest tests/unit/                  # Unit tests only
pytest tests/integration/           # Integration tests only
pytest tests/e2e/                   # End-to-end tests only
pytest --cov=iris_rag --cov=rag_templates  # With coverage
```

### Linting and Formatting
```bash
# Format code (apply isort and black per pyproject.toml configuration)
black .
isort .

# Lint code
flake8 .
mypy iris_rag/
```

### Docker Operations
```bash
# Core services
make docker-up        # Start core services (IRIS, Redis, API, Streamlit)
make docker-down      # Stop all services
make docker-logs      # View logs from all services

# Development environment
make docker-up-dev    # Start with Jupyter notebook
make docker-shell     # Open shell in API container
make docker-iris-shell  # Open IRIS database shell

# Full development setup
make docker-dev       # Start dev environment, wait for health, init data
```

### RAGAS Evaluation
```bash
# Quick evaluation on sample data
make test-ragas-sample

# Full evaluation on 1000 PMC documents
make test-ragas-1000

# Dockerized evaluation
make test-ragas-sample-docker
make test-ragas-1000-docker
```

## Architecture Overview

### Core Framework Structure
- **iris_rag/**: Main RAG framework package
  - **core/**: Abstract base classes (RAGPipeline, VectorStore) and models
  - **pipelines/**: RAG pipeline implementations (BasicRAG, CRAG, GraphRAG, HybridGraphRAG)
  - **storage/**: Vector store implementations, primarily IRISVectorStore
  - **services/**: Business logic services (entity extraction, storage management)
  - **config/**: Configuration management and pipeline-specific configs
  - **validation/**: Pipeline validation and requirements checking
  - **memory/**: Memory management and incremental indexing components

### Available RAG Pipelines
1. **BasicRAGPipeline**: Standard vector similarity search with LLM generation
2. **BasicRAGRerankingPipeline**: Basic RAG with reranking for improved relevance
3. **CRAGPipeline**: Corrective RAG with relevance evaluation and correction
4. **GraphRAGPipeline**: Graph-based RAG using entity relationships and communities
5. **HybridGraphRAGPipeline**: Advanced hybrid search combining vector, text, and graph retrieval
6. **PyLateColBERTPipeline**: ColBERT-based dense retrieval pipeline

### Key Integration Points
- **Vector Database**: InterSystems IRIS with native vector search capabilities
- **LLM Integration**: OpenAI and Anthropic APIs via common.utils.get_llm_func
- **Bridge Adapters**: Generic memory components for external system integration
- **Validation Framework**: Automated pipeline requirement validation and setup

### Pipeline Factory Pattern
```python
from iris_rag import create_pipeline

# Create with validation
pipeline = create_pipeline(
    pipeline_type="basic",
    validate_requirements=True,
    auto_setup=True
)

# Create specific pipeline types
basic_pipeline = create_pipeline("basic")
graphrag_pipeline = create_pipeline("graphrag")
hybrid_pipeline = create_pipeline("hybrid_graphrag")  # Requires graph-ai adjacency
```

### Testing Architecture
- **Unit Tests**: `tests/unit/` - Component-level testing
- **Integration Tests**: `tests/integration/` - Cross-component functionality
- **E2E Tests**: `tests/e2e/` - Full pipeline workflows
- **Enterprise Scale Tests**: 10K document testing with mocking support

### Configuration Management
- **Default Config**: `iris_rag/config/default_config.yaml`
- **Pipeline Configs**: `config/pipelines.yaml`
- **Environment**: `.env` file for API keys and database connections
- **Docker Compose**: Multiple compose files for different deployment scenarios

### HybridGraphRAG Optional Dependencies
The HybridGraphRAG pipeline supports optional dependencies for enhanced performance:

**Production Installation (Recommended):**
```bash
pip install rag-templates[hybrid-graphrag]
```
This installs the `iris-vector-graph` package providing iris_graph_core integration for 50x performance improvements.

**Development Alternative:**
Place the graph-ai project adjacent to rag-templates:
```
/parent-directory/
  ├── rag-templates/
  └── graph-ai/
```

**Graceful Fallback:**
HybridGraphRAG automatically falls back to standard GraphRAG when iris-vector-graph is not available, with helpful installation messages.

### Data Flow
1. **Document Ingestion**: Load documents via pipeline.load_documents()
2. **Chunking & Embedding**: Automatic text segmentation and vector generation
3. **Storage**: Vectors and metadata stored in IRIS vector tables
4. **Query Processing**: Multi-modal retrieval (vector, text, graph) depending on pipeline
5. **Generation**: LLM synthesis with retrieved context
6. **Response**: Standardized response format with sources and metadata
