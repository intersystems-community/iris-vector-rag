# RAG-Templates

## 🎯 Project Status: Complete & Ready for Integration

RAG-Templates is now **complete as a reusable framework** with all core components delivered:
- ✅ 4 Production RAG Pipelines (BasicRAG, CRAG, GraphRAG, BasicRAGReranking)
- ✅ **NEW: HybridGraphRAG with IRIS Graph Core Integration** (50x performance, RRF fusion)
- ✅ **NEW: IRIS-Global-GraphRAG Integration** (Academic papers with 3D visualization)
- ✅ **NEW: Multi-Pipeline Comparison System** (TDD-validated, side-by-side evaluation)
- ✅ Enterprise IRIS Backend Integration
- ✅ Generic Memory Components Architecture
- ✅ Unified Bridge Adapter Interface
- ✅ Incremental Indexing Foundation

**Integration Documentation:**
- 📋 [Project Completion Report](docs/PROJECT_COMPLETION_REPORT_VALIDATED.md) - Validated achievements and honest metrics
- 🔗 [Integration Handoff Guide](docs/INTEGRATION_HANDOFF_GUIDE.md) - How to consume rag-templates in your application
- ⚡ **[IRIS Graph Core Integration](docs/IRIS_GRAPH_CORE_INTEGRATION.md)** - Advanced hybrid search with RRF fusion
- 🔬 **[IRIS Global GraphRAG Integration](docs/IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md)** - Academic papers with 3D visualization
- 📊 **Multi-Pipeline Comparison** - TDD-validated side-by-side evaluation system
- 🏗️ [Architecture Summary](docs/VALIDATED_ARCHITECTURE_SUMMARY.md) - Service boundaries and performance characteristics
- 🚀 [Production Readiness](docs/PRODUCTION_READINESS_ASSESSMENT.md) - Infrastructure requirements and deployment guidance

**Scope Transition:**
Application-specific features (PRefLexOR bridge, production monitoring, CI/CD) have been re-scoped to the [kg-ticket-resolver](https://github.com/your-org/kg-ticket-resolver) application project.

## Set up

```bash
# 1. Clone the repository
git clone <repository-url>
cd rag-templates

# 2. Set up Python environment using uv (fast, modern package manager)
make setup-env  # Creates .venv using uv
make install    # Installs all dependencies via uv sync

# 3. Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Optional: Install HybridGraphRAG dependencies for advanced features
pip install rag-templates[hybrid-graphrag]  # Installs iris-vector-graph for 50x performance

# 5. Start the database (uses ports 11972/152773 to avoid conflicts)
docker-compose up -d

# 6. Initialize and load sample data
make setup-db
make load-data

# 7. Test that you can access the iris_rag package
make validate-iris-rag

# 8. Create your .env file and add your API keys in

# 9. Try the various pipeline scripts!
# Note: these scripts currently use OpenAI's ChatGPT as the LLM, so having an API key is essential.
# The first time you run any script after loading new data will take a long time due to overhead of initial chunking, embedding, and storage
cd scripts/basic
python try_basic_rag_pipeline.py

# 10. Try the NEW HybridGraphRAG with advanced search capabilities
# Install with: pip install rag-templates[hybrid-graphrag]
python try_hybrid_graphrag_pipeline.py
```

## 🚀 Quick Start: HybridGraphRAG

**Installation:** `pip install rag-templates[hybrid-graphrag]`

For advanced hybrid search with 50x performance improvements:

```python
from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

# Initialize with hybrid capabilities
pipeline = HybridGraphRAGPipeline()

# Multi-modal hybrid search (vector + text + graph)
results = pipeline.query(
    query_text="cancer drug targets",
    method="hybrid",
    top_k=15
)

# RRF fusion search
results = pipeline.query(
    query_text="protein interactions",
    method="rrf",
    vector_k=30,
    text_k=30
)
```

**Requirements:** Install with `pip install rag-templates[hybrid-graphrag]` to enable iris-vector-graph dependencies.
For development, you can also place the [graph-ai](../graph-ai) project adjacent to rag-templates.
See [IRIS Graph Core Integration](docs/IRIS_GRAPH_CORE_INTEGRATION.md) for detailed setup and configuration.

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

