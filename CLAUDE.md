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

# Backend mode testing (Feature 035)
make test-community                 # Test with Community Edition mode (1 connection)
make test-enterprise                # Test with Enterprise Edition mode (999 connections)
make test-backend-contracts         # Run backend mode contract tests
IRIS_BACKEND_MODE=community pytest tests/  # Manual backend mode override
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

### Available RAG Pipelines (via `create_pipeline()`)
1. **`basic`** → BasicRAGPipeline - Standard vector similarity search
2. **`basic_rerank`** → BasicRAGRerankingPipeline - Vector search + cross-encoder reranking
3. **`crag`** → CRAGPipeline - Corrective RAG with self-evaluation
4. **`graphrag`** → HybridGraphRAGPipeline - Hybrid search (vector + text + graph + RRF)
5. **`pylate_colbert`** → PyLateColBERTPipeline - ColBERT late interaction retrieval

**Additional Pipeline (Direct Import):**
6. **IRIS-Global-GraphRAG** - Academic papers with 3D visualization and global communities

### Key Integration Points
- **Vector Database**: InterSystems IRIS with native vector search capabilities
- **LLM Integration**: OpenAI and Anthropic APIs via common.utils.get_llm_func
- **Bridge Adapters**: Generic memory components for external system integration
- **Validation Framework**: Automated pipeline requirement validation and setup

### Pipeline Factory Pattern
```python
from iris_rag import create_pipeline

# Create with validation (recommended)
pipeline = create_pipeline(
    pipeline_type="basic",           # basic, basic_rerank, crag, graphrag, pylate_colbert
    validate_requirements=True,       # Auto-validate DB setup
    auto_setup=False                 # Auto-fix issues if True
)

# All pipelines share the same standardized API
result = pipeline.query(query="What is diabetes?", top_k=5)

# Standardized response format (100% LangChain & RAGAS compatible):
# - result["answer"]: LLM-generated answer
# - result["retrieved_documents"]: List[Document] with full metadata
# - result["contexts"]: List[str] for RAGAS evaluation
# - result["sources"]: Source references in metadata
# - result["metadata"]: Pipeline-specific metadata fields
```

### Testing Architecture
- **Unit Tests**: `tests/unit/` - Component-level testing
- **Integration Tests**: `tests/integration/` - Cross-component functionality
- **E2E Tests**: `tests/e2e/` - Full pipeline workflows
- **Contract Tests**: `tests/contract/` - API contract validation (TDD approach)
- **Enterprise Scale Tests**: 10K document testing with mocking support

### Test Fixture Strategy (.DAT Fixture-First Principle)

**Constitutional Requirement**: All integration and E2E tests with ≥10 entities MUST use .DAT fixtures loaded via iris-devtools. See `.specify/memory/constitution.md` for complete IRIS testing principles.

**Performance Benefits**:
- **.DAT fixtures**: 0.5-2 seconds for 100 entities (binary IRIS format)
- **JSON fixtures**: 39-75 seconds for same data
- **Speedup**: 100-200x faster test execution

**When to Use What**:
```
Need test data?
├─ Unit test (mocked components)?
│  └─ Use programmatic fixtures (Python code)
│
├─ Integration test (real IRIS database)?
│  ├─ < 10 entities or simple data?
│  │  └─ Use programmatic fixtures
│  │
│  └─ ≥ 10 entities or complex relationships?
│     └─ Use .DAT fixtures (REQUIRED)
│
└─ E2E test (full pipeline)?
   └─ Use .DAT fixtures (REQUIRED)
```

**Fixture Management Commands**:
```bash
# List available fixtures
make fixture-list

# Get fixture details
make fixture-info FIXTURE=medical-graphrag-20

# Load fixture into IRIS
make fixture-load FIXTURE=medical-graphrag-20

# Create new fixture from current database
make fixture-create FIXTURE=my-test-data

# Validate fixture integrity
make fixture-validate FIXTURE=medical-graphrag-20
```

**Using Fixtures in Tests**:
```python
# Automatic fixture loading via pytest marker
@pytest.mark.dat_fixture("medical-graphrag-20")
def test_with_fixture():
    # Fixture automatically loaded before test
    # Database contains 21 entities, 15 relationships
    pass

# Manual fixture loading via FixtureManager
from tests.fixtures.manager import FixtureManager

def test_manual_load():
    manager = FixtureManager()
    result = manager.load_fixture(
        fixture_name="medical-graphrag-20",
        cleanup_first=True,
        validate_checksum=True,
    )
    assert result.success
```

**Fixture Infrastructure** (✅ Production Ready):
The unified fixture infrastructure provides:
- **Fast .DAT Loading**: 100-200x faster than JSON (via iris-devtools)
- **Checksum Validation**: SHA256 integrity checking for data consistency
- **Version Management**: Semantic versioning with migration history tracking
- **State Tracking**: Session-wide fixture state to prevent schema loops
- **pytest Integration**: Automatic cleanup via `@pytest.mark.dat_fixture` decorator

**Fixture Documentation**:
- **Complete Status**: `FIXTURE_INFRASTRUCTURE_COMPLETE.md` (implementation overview)
- **CLI Reference**: `python -m tests.fixtures.cli --help`
- **API Documentation**: `tests/fixtures/manager.py` (FixtureManager class)
- **Constitution**: `.specify/memory/constitution.md` (Principle II)

### Backend Mode Configuration (Feature 035)
**Purpose**: Prevent license pool exhaustion in IRIS Community Edition while allowing parallel execution in Enterprise Edition.

**Modes**:
- **Community**: Single connection limit, sequential test execution
- **Enterprise**: 999 connections, parallel test execution

**Configuration Precedence** (highest to lowest):
1. `IRIS_BACKEND_MODE` environment variable
2. `.specify/config/backend_modes.yaml` file
3. Default (community mode)

**Usage Examples**:
```python
# Pytest fixtures (auto-configured)
def test_example(iris_connection, backend_configuration):
    assert backend_configuration.max_connections == 1  # community mode

# Manual configuration
from iris_rag.testing import load_configuration, ConnectionPool

config = load_configuration()
pool = ConnectionPool(mode=config.mode)
with pool.acquire() as conn:
    # Use connection
    pass
```

**Troubleshooting**:
- **License pool exhaustion**: Switch to `IRIS_BACKEND_MODE=community`
- **Tests timing out**: Check connection pool limits with `config.max_connections`
- **Edition mismatch error**: Set `IRIS_BACKEND_MODE` to match your IRIS edition

### Configuration Management
- **Default Config**: `iris_rag/config/default_config.yaml`
- **Pipeline Configs**: `config/pipelines.yaml`
- **Environment**: `.env` file for API keys and database connections
- **Docker Compose**: Multiple compose files for different deployment scenarios

### HybridGraphRAG Required Dependencies
The HybridGraphRAG pipeline requires iris-vector-graph for operation:

**Installation:**
```bash
pip install rag-templates[hybrid-graphrag]
```
This installs the `iris-vector-graph` package providing iris_graph_core integration for 50x performance improvements.

**Requirements:**
- iris-vector-graph>=2.0.0 is now a mandatory dependency
- No fallback mechanisms - the pipeline will fail fast with clear error messages if the package is missing
- All retrieval methods (hybrid, rrf, text, vector, kg) require iris-vector-graph

### Testing GraphRAG Pipelines

**Important**: HybridGraphRAG integration tests are **intentionally skipped** in CI because they require:
1. Configured LLM for entity extraction from documents
2. iris-vector-graph tables populated with embeddings and optimized indexes
3. Full knowledge graph (entities + relationships) extracted from documents

**Test fixtures cannot provide this setup** because:
- Entity extraction requires LLM API calls (not available/practical in test environment)
- iris-vector-graph requires optimized HNSW tables with real embeddings
- Simple 3-document fixtures cannot replicate the complexity of real knowledge graphs

**Three-Tier Testing Strategy:**

GraphRAG testing uses a pragmatic three-tier approach:

**Tier 1: Contract Tests** (Automated CI) ✅
```bash
pytest tests/contract/test_graphrag_fixtures.py  # 13/13 passing
```
- Purpose: Validate API interfaces and fixture loading
- Coverage: Data structures, fixture service, validation logic
- Run in CI: Yes - fast (< 1s), reliable, no dependencies
- When to run: Always (part of standard test suite)

**Tier 2: Realistic Integration Tests** (Manual, Development) ℹ️
```bash
# Run against real database with 221K+ entities
IRIS_PORT=21972 pytest tests/integration/test_graphrag_realistic.py -v
IRIS_PORT=21972 pytest tests/integration/test_graphrag_with_real_data.py -v
```
- Purpose: Validate GraphRAG against production-like data
- Coverage: KG traversal, vector fallback, metadata completeness
- Run in CI: No - requires IRIS_PORT environment configuration
- When to run: During development, before major releases
- Database requirement: 100+ entities, 50+ relationships

**Tier 3: E2E HybridGraphRAG Tests** (Skipped) ⏭️
```bash
pytest tests/integration/test_hybridgraphrag_e2e.py  # All skipped with clear reasons
```
- Purpose: End-to-end validation of all 5 query methods
- Status: Intentionally skipped - requires LLM + iris-vector-graph setup
- Alternative: Manual testing with real data (see below)

**Why Integration Tests are Skipped:**
- Previous "passing" integration tests were **false positives** - they used 2,376 pre-existing documents in the database, not the 3-document test fixtures
- Maintaining complex LLM mocking + iris-vector-graph setup is brittle and provides little value
- Contract tests + manual validation with real data provides better signal

### Data Flow
1. **Document Ingestion**: Load documents via pipeline.load_documents()
2. **Chunking & Embedding**: Automatic text segmentation and vector generation
3. **Storage**: Vectors and metadata stored in IRIS vector tables
4. **Query Processing**: Multi-modal retrieval (vector, text, graph) depending on pipeline
5. **Generation**: LLM synthesis with retrieved context
6. **Response**: Standardized response format with sources and metadata
