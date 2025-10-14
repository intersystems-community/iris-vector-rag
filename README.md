# RAG-Templates

## 🎯 Project Status: Complete & Ready for Integration

RAG-Templates is now **complete as a reusable framework** with all core components delivered:
- ✅ **6 Production RAG Pipelines** with standardized API
  - BasicRAG - Standard vector similarity search
  - BasicRAGReranking - Vector search with cross-encoder reranking
  - CRAG - Corrective RAG with self-evaluation
  - HybridGraphRAG - Graph + vector + text hybrid search with RRF fusion
  - PyLateColBERT - ColBERT late interaction retrieval
  - IRIS-Global-GraphRAG - Academic papers with 3D visualization
- ✅ **100% Test Coverage** (136/136 tests passing)
  - Contract tests for API validation
  - Integration tests with live database
  - E2E workflow validation
- ✅ **Unified API Surface** - Consistent interfaces across all pipelines
- ✅ **Enterprise IRIS Backend** with connection pooling and mode detection
- ✅ **LangChain & RAGAS Compatible** - Standard Document objects and metadata

**Documentation:** 📑 [**Full Documentation Index**](DOCUMENTATION_INDEX.md)

**Quick Links:**
- 📖 **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation with examples
- 📚 [User Guide](USER_GUIDE.md) - Step-by-step installation and usage
- 🧪 [Test Validation Summary](TEST_VALIDATION_SUMMARY.md) - 100% test pass rate (136/136)
- 🔗 [Integration Guide](docs/INTEGRATION_HANDOFF_GUIDE.md) - How to integrate into your app
- 🏗️ [Architecture Summary](docs/VALIDATED_ARCHITECTURE_SUMMARY.md) - System design
- 🚀 [Production Readiness](docs/PRODUCTION_READINESS_ASSESSMENT.md) - Deployment checklist

## Quick Start

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd rag-templates
make setup-env  # Creates .venv using uv
make install    # Installs dependencies

# 2. Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Start database
docker-compose up -d

# 4. Initialize database
make setup-db
make load-data

# 5. Create .env file with API keys
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
IRIS_HOST=localhost
IRIS_PORT=1972
EOF

# 6. Try different pipelines
python -c "
from iris_rag import create_pipeline

# Basic RAG - simplest approach
pipeline = create_pipeline('basic')
result = pipeline.query('What is machine learning?', top_k=5)
print(result['answer'])
"
```

## 🔬 Test Fixture Quick Start

RAG-Templates uses **binary .DAT fixtures** for fast, reproducible testing. .DAT fixtures are 100-200x faster than JSON fixtures (0.5-2 seconds vs 39-75 seconds for 100 entities).

### Why .DAT Fixtures?

- **Speed**: Binary IRIS format loads in seconds instead of minutes
- **Reproducibility**: Checksums ensure identical database state across test runs
- **Isolation**: Each test gets a clean, versioned database snapshot
- **No LLM Required**: Pre-computed embeddings and entities included

### Quick Fixture Workflow

```bash
# 1. List available fixtures
make fixture-list

# Example output:
# Name                           Version    Type     Tables          Rows     Embeddings
# ----------------------------------------------------------------------------------------------
# medical-graphrag-20            1.0.0      dat      3 tables        39       Required

# 2. Get detailed fixture information
make fixture-info FIXTURE=medical-graphrag-20

# 3. Load fixture into IRIS database
make fixture-load FIXTURE=medical-graphrag-20

# 4. Validate fixture integrity
make fixture-validate FIXTURE=medical-graphrag-20
```

### Using Fixtures in Tests

**Automatic Loading (Recommended)**:
```python
import pytest

@pytest.mark.dat_fixture("medical-graphrag-20")
def test_with_fixture():
    # Fixture automatically loaded before test
    # Database contains 21 entities, 15 relationships, pre-computed embeddings
    pipeline = create_pipeline("graphrag")
    result = pipeline.query("What are cancer treatment targets?")
    assert len(result["retrieved_documents"]) > 0
```

**Manual Loading**:
```python
from tests.fixtures.manager import FixtureManager

def test_manual_fixture_load():
    manager = FixtureManager()
    result = manager.load_fixture(
        fixture_name="medical-graphrag-20",
        cleanup_first=True,           # Clean database first
        validate_checksum=True,        # Verify fixture integrity
        generate_embeddings=False,     # Already included in .DAT
    )

    assert result.success
    assert result.rows_loaded == 39   # Total rows across all tables
```

### Creating Your Own Fixtures

```bash
# 1. Populate IRIS with test data (manually or via script)
python scripts/load_test_data.py

# 2. Create fixture from current database state
make fixture-create FIXTURE=my-test-data

# Interactive mode (recommended for first-time users):
python -m tests.fixtures.cli workflow

# Command-line mode:
python -m tests.fixtures.cli create my-fixture \
    --tables RAG.SourceDocuments,RAG.Entities,RAG.EntityRelationships \
    --description "My test fixture" \
    --generate-embeddings
```

### Fixture Management CLI

```bash
# Full CLI help
python -m tests.fixtures.cli --help

# Common commands:
python -m tests.fixtures.cli list                     # List all fixtures
python -m tests.fixtures.cli info medical-graphrag-20 # Fixture details
python -m tests.fixtures.cli load medical-graphrag-20 # Load fixture
python -m tests.fixtures.cli validate my-fixture      # Validate integrity
python -m tests.fixtures.cli snapshot snapshot-20250114  # Quick DB snapshot
```

### Constitutional Requirement

**All integration and E2E tests with ≥10 entities MUST use .DAT fixtures** (see `.specify/memory/constitution.md` for complete IRIS testing principles).

**Decision Tree**:
- **Unit tests** → Use programmatic fixtures (Python code)
- **Integration tests with < 10 entities** → Use programmatic fixtures
- **Integration tests with ≥ 10 entities** → Use .DAT fixtures (REQUIRED)
- **E2E tests** → Use .DAT fixtures (REQUIRED)

**Documentation**:
- **Complete Guide**: `tests/fixtures/README.md`
- **Examples**: `examples/fixtures/basic_usage.py`
- **Constitution**: `.specify/memory/constitution.md` (Principle II)

## 📖 Unified API Reference

All pipelines follow a consistent, standardized API:

### Creating Pipelines

```python
from iris_rag import create_pipeline

# Available pipeline types:
# - "basic"          : BasicRAG (vector similarity)
# - "basic_rerank"   : BasicRAG + cross-encoder reranking
# - "crag"           : Corrective RAG with self-evaluation
# - "graphrag"       : HybridGraphRAG (vector + text + graph)
# - "pylate_colbert" : ColBERT late interaction

pipeline = create_pipeline(
    pipeline_type="basic",
    validate_requirements=True,  # Auto-validate DB setup
    auto_setup=False,            # Auto-fix issues if True
)
```

### Loading Documents

```python
from iris_rag.core.models import Document

# Option 1: From Document objects
docs = [
    Document(
        page_content="Python is a programming language...",
        metadata={"source": "intro.txt", "author": "John"}
    )
]
result = pipeline.load_documents(documents=docs)

# Option 2: From file path
result = pipeline.load_documents(documents_path="data/docs.json")

# Returns: {"documents_loaded": 10, "embeddings_generated": 10, "documents_failed": 0}
```

### Querying

```python
# Standard query signature for ALL pipelines
result = pipeline.query(
    query="What is machine learning?",
    top_k=5,                    # Number of documents to return (1-100)
    generate_answer=True,       # Generate LLM answer (default: True)
    include_sources=True,       # Include source metadata (default: True)
)

# Standardized response format (LangChain & RAGAS compatible):
{
    "query": "What is machine learning?",
    "answer": "Machine learning is...",                 # LLM-generated answer
    "retrieved_documents": [Document(...)],             # LangChain Document objects
    "contexts": ["context 1", "context 2"],             # RAGAS-compatible contexts
    "sources": [{"source": "file.txt", ...}],           # Source references
    "execution_time": 0.523,
    "metadata": {
        "num_retrieved": 5,
        "pipeline_type": "basic",
        "retrieval_method": "vector",
        "context_count": 5,
        ...
    }
}
```

### Pipeline-Specific Features

```python
# BasicRAGReranking - Control reranking behavior
pipeline = create_pipeline("basic_rerank")
result = pipeline.query(query, top_k=5)  # Retrieves rerank_factor*5, returns top 5

# CRAG - Retrieval evaluation
pipeline = create_pipeline("crag")
result = pipeline.query(query, top_k=5, generate_answer=True)

# HybridGraphRAG - Multi-modal search
pipeline = create_pipeline("graphrag")
result = pipeline.query(
    query_text="cancer targets",
    method="rrf",        # rrf, hybrid, vector, text, graph
    vector_k=30,
    text_k=30
)

# PyLateColBERT - Late interaction retrieval
pipeline = create_pipeline("pylate_colbert")
result = pipeline.query(query, top_k=5)  # Uses ColBERT late interaction
```

## 🧪 Testing & Quality Assurance

The RAG-Templates framework includes comprehensive testing tools to ensure code quality and maintainability:

### Testing Compliance Tools

**Coverage Warnings** - Automated coverage monitoring without failing builds
- Warns when modules fall below 60% coverage (80% for critical modules)
- Configure critical modules in `.coveragerc`
- [Detailed Documentation](docs/testing/coverage-warnings.md)

**Error Message Validation** - Ensures helpful test failure messages
- Validates three-part structure: What failed, Why, and Action to take
- Provides improvement suggestions for unclear messages
- [Best Practices Guide](docs/testing/error-messages.md)

**TDD Compliance** - Validates Test-Driven Development workflow
- Ensures contract tests failed before implementation
- Integrates with CI/CD for automated checking
- [TDD Workflow Guide](docs/testing/tdd-compliance.md)

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=iris_rag --cov=common

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests
pytest tests/contract/      # Contract tests

# Validate TDD compliance
python scripts/validate_tdd_compliance.py

# Check requirement-task mapping
python scripts/validate_task_mapping.py --spec specs/*/spec.md --tasks specs/*/tasks.md
```

### Pre-commit Hooks

Install pre-commit hooks for automated quality checks:

```bash
pip install pre-commit
pre-commit install
```

This enables:
- TDD compliance checking on contract test commits
- Requirement-task mapping validation
- Code formatting (black, isort)
- Error message quality reminders

## 📚 References & Research

### RAG Technique Papers & Implementations

| Technique | Original Paper | Key Repository | Additional Resources |
|-----------|---------------|----------------|---------------------|
| **Basic RAG** | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | [Facebook Research](https://github.com/facebookresearch/RAG) | [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) |
| **ColBERT** | [ColBERT: Efficient and Effective Passage Retrieval](https://arxiv.org/abs/2004.12832) | [Stanford ColBERT](https://github.com/stanford-futuredata/ColBERT) | [Pylate Integration](https://github.com/lightonai/pylate) |
| **CRAG** | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | [CRAG Implementation](https://github.com/HuskyInSalt/CRAG) | [LangGraph CRAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/) |
| **HyDE** | [Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) | [HyDE Official](https://github.com/texttron/hyde) | [LangChain HyDE](https://python.langchain.com/docs/how_to/hyde/) |
| **GraphRAG** | [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) |
| **NodeRAG** | [Hierarchical Text Retrieval](https://arxiv.org/abs/2310.20501) | [NodeRAG Implementation](https://github.com/microsoft/noderag) | [Hierarchical Retrieval](https://python.langchain.com/docs/how_to/parent_document_retriever/) |

### Core Technologies

- **Vector Databases**: [InterSystems IRIS Vector Search](https://docs.intersystems.com/iris20241/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch)
- **Embeddings**: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- **LLM Integration**: [LangChain](https://github.com/langchain-ai/langchain), [OpenAI API](https://platform.openai.com/docs/api-reference)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

