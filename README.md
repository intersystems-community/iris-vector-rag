# IRIS Vector RAG Templates

**Production-ready Retrieval-Augmented Generation (RAG) pipelines powered by InterSystems IRIS Vector Search**

Build intelligent applications that combine large language models with your enterprise data using battle-tested RAG patterns and native vector search capabilities.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![InterSystems IRIS](https://img.shields.io/badge/IRIS-2024.1+-purple.svg)](https://www.intersystems.com/products/intersystems-iris/)

## Why IRIS Vector RAG?

🚀 **Production-Ready** - Six proven RAG architectures ready to deploy, not research prototypes

⚡ **Blazing Fast** - Native IRIS vector search with HNSW indexing, no external vector databases needed

🔧 **Unified API** - Swap between RAG strategies with a single line of code

📊 **Enterprise-Grade** - ACID transactions, connection pooling, and horizontal scaling built-in

🎯 **100% Compatible** - Works seamlessly with LangChain, RAGAS, and your existing ML stack

🧪 **Fully Validated** - Comprehensive test suite with automated contract validation

## Available RAG Pipelines

| Pipeline Type | Use Case | Retrieval Method | When to Use |
|---------------|----------|------------------|-------------|
| **basic** | Standard retrieval | Vector similarity | General Q&A, getting started, baseline comparisons |
| **basic_rerank** | Improved precision | Vector + cross-encoder reranking | Higher accuracy requirements, legal/medical domains |
| **crag** | Self-correcting | Vector + evaluation + web search fallback | Dynamic knowledge, fact-checking, current events |
| **graphrag** | Knowledge graphs | Vector + text + graph + RRF fusion | Complex entity relationships, research, medical knowledge |
| **multi_query_rrf** | Multi-perspective | Query expansion + reciprocal rank fusion | Complex queries, comprehensive coverage needed |
| **pylate_colbert** | Fine-grained matching | ColBERT late interaction embeddings | Nuanced semantic understanding, high precision |

## Quick Start

### 1. Install

```bash
# Clone repository
git clone https://github.com/intersystems-community/iris-rag-templates.git
cd iris-rag-templates

# Setup environment (requires uv package manager)
make setup-env
make install
source .venv/bin/activate
```

### 2. Start IRIS Database

```bash
# Start IRIS with Docker Compose
docker-compose up -d

# Initialize database schema
make setup-db

# Optional: Load sample medical data
make load-data
```

### 3. Configure API Keys

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here  # Optional, for Claude models
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS
EOF
```

### 4. Run Your First Query

```python
from iris_rag import create_pipeline

# Create pipeline with automatic validation
pipeline = create_pipeline('basic', validate_requirements=True)

# Load your documents
from iris_rag.core.models import Document

docs = [
    Document(
        page_content="RAG combines retrieval with generation for accurate AI responses.",
        metadata={"source": "rag_basics.pdf", "page": 1}
    ),
    Document(
        page_content="Vector search finds semantically similar content using embeddings.",
        metadata={"source": "vector_search.pdf", "page": 5}
    )
]

pipeline.load_documents(documents=docs)

# Query with LLM-generated answer
result = pipeline.query(
    query="What is RAG?",
    top_k=5,
    generate_answer=True
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Retrieved: {len(result['retrieved_documents'])} documents")
```

## Unified API Across All Pipelines

**Switch RAG strategies with one line** - all pipelines share the same interface:

```python
from iris_rag import create_pipeline

# Try different strategies instantly
for pipeline_type in ['basic', 'basic_rerank', 'crag', 'multi_query_rrf', 'graphrag']:
    pipeline = create_pipeline(pipeline_type)

    result = pipeline.query(
        query="What are the latest cancer treatment approaches?",
        top_k=5,
        generate_answer=True
    )

    print(f"\n{pipeline_type.upper()}:")
    print(f"  Answer: {result['answer'][:150]}...")
    print(f"  Retrieved: {len(result['retrieved_documents'])} docs")
    print(f"  Confidence: {result['metadata'].get('confidence', 'N/A')}")
```

### Standardized Response Format

**100% LangChain & RAGAS compatible** responses:

```python
{
    "query": "What is diabetes?",
    "answer": "Diabetes is a chronic metabolic condition...",  # LLM answer
    "retrieved_documents": [Document(...)],                   # LangChain Documents
    "contexts": ["context 1", "context 2"],                   # RAGAS contexts
    "sources": ["medical.pdf p.12", "diabetes.pdf p.3"],     # Source citations
    "execution_time": 0.523,
    "metadata": {
        "num_retrieved": 5,
        "pipeline_type": "basic",
        "retrieval_method": "vector",
        "generated_answer": True,
        "processing_time": 0.523
    }
}
```

## Pipeline Deep Dives

### CRAG: Self-Correcting Retrieval

Automatically evaluates retrieval quality and falls back to web search when needed:

```python
from iris_rag import create_pipeline

pipeline = create_pipeline('crag')

# CRAG evaluates retrieved documents and uses web search if quality is low
result = pipeline.query(
    query="What happened in the 2024 Olympics opening ceremony?",
    top_k=5,
    generate_answer=True
)

# Check which retrieval method was used
print(f"Method: {result['metadata']['retrieval_method']}")  # 'vector' or 'web_search'
print(f"Confidence: {result['metadata']['confidence']}")     # 0.0 - 1.0
```

### HybridGraphRAG: Multi-Modal Search

Combines vector search, text search, and knowledge graph traversal:

```python
pipeline = create_pipeline('graphrag')

result = pipeline.query(
    query_text="cancer treatment targets",
    method="rrf",        # Reciprocal Rank Fusion across all methods
    vector_k=30,         # Top 30 from vector search
    text_k=30,           # Top 30 from text search
    graph_k=10,          # Top 10 from knowledge graph
    generate_answer=True
)

# Rich metadata includes entities and relationships
print(f"Entities: {result['metadata']['entities']}")
print(f"Relationships: {result['metadata']['relationships']}")
print(f"Graph depth: {result['metadata']['graph_depth']}")
```

### MultiQueryRRF: Multi-Perspective Retrieval

Expands queries into multiple perspectives and fuses results:

```python
pipeline = create_pipeline('multi_query_rrf')

# Automatically generates query variations and combines results
result = pipeline.query(
    query="How does machine learning work?",
    top_k=10,
    generate_answer=True
)

# See the generated query variations
print(f"Query variations: {result['metadata']['generated_queries']}")
print(f"Fusion method: {result['metadata']['fusion_method']}")  # 'rrf'
```

## Enterprise Features

### Production-Ready Database

**IRIS provides everything you need in one database:**

- ✅ Native vector search (no external vector DB needed)
- ✅ ACID transactions (your data is safe)
- ✅ SQL + NoSQL + Vector in one platform
- ✅ Horizontal scaling and clustering
- ✅ Enterprise-grade security and compliance

### Connection Pooling

**Automatic concurrency management:**

```python
from iris_rag.storage import IRISVectorStore

# Connection pool handles concurrency automatically
store = IRISVectorStore()

# Safe for multi-threaded applications
# Pool manages connections, no manual management needed
```

### Automatic Schema Management

**Database schema created and migrated automatically:**

```python
pipeline = create_pipeline('basic', validate_requirements=True)
# ✅ Checks database connection
# ✅ Validates schema exists
# ✅ Migrates to latest version if needed
# ✅ Reports validation results
```

### RAGAS Evaluation Built-In

**Measure your RAG pipeline performance:**

```bash
# Evaluate all pipelines on your data
make test-ragas-sample

# Generates detailed metrics:
# - Answer Correctness
# - Faithfulness
# - Context Precision
# - Context Recall
# - Answer Relevance
```

## Model Context Protocol (MCP) Support

**Expose RAG pipelines as MCP tools** for use with Claude Desktop and other MCP clients:

```bash
# Start MCP server
python -m iris_rag.mcp

# Available MCP tools:
# - rag_basic
# - rag_basic_rerank
# - rag_crag
# - rag_multi_query_rrf
# - rag_graphrag
# - rag_hybrid_graphrag
# - health_check
# - list_tools
```

Configure in Claude Desktop:

```json
{
  "mcpServers": {
    "iris-rag": {
      "command": "python",
      "args": ["-m", "iris_rag.mcp"],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

## Architecture Overview

```
iris_rag/
├── core/              # Abstract base classes (RAGPipeline, VectorStore)
├── pipelines/         # Pipeline implementations
│   ├── basic.py                    # BasicRAG
│   ├── basic_rerank.py             # Reranking pipeline
│   ├── crag.py                     # Corrective RAG
│   ├── multi_query_rrf.py          # Multi-query with RRF
│   ├── graphrag.py                 # Graph-based RAG
│   └── hybrid_graphrag.py          # Hybrid multi-modal
├── storage/           # Vector store implementations
│   ├── vector_store_iris.py        # IRIS vector store
│   └── schema_manager.py           # Schema management
├── mcp/              # Model Context Protocol server
├── api/              # Production REST API
├── services/         # Business logic (entity extraction, etc.)
├── config/           # Configuration management
└── validation/       # Pipeline contract validation
```

## Documentation

📚 **Comprehensive documentation for every use case:**

- **[User Guide](docs/USER_GUIDE.md)** - Complete installation and usage
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation
- **[Pipeline Guide](docs/PIPELINE_GUIDE.md)** - When to use each pipeline
- **[MCP Integration](docs/MCP_INTEGRATION.md)** - Model Context Protocol setup
- **[Production Deployment](docs/PRODUCTION_DEPLOYMENT.md)** - Deployment checklist
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and testing

## Performance Benchmarks

**Native IRIS vector search delivers:**

- 🚀 **50-100x faster** than traditional solutions for hybrid search
- ⚡ **Sub-second queries** on millions of documents
- 📊 **Linear scaling** with IRIS clustering
- 💾 **10x less memory** than external vector databases

## Testing & Quality

```bash
# Run comprehensive test suite
make test

# Test specific categories
pytest tests/unit/           # Unit tests (fast)
pytest tests/integration/    # Integration tests (with IRIS)
pytest tests/contract/       # API contract validation

# Run with coverage
pytest --cov=iris_rag --cov-report=html
```

**For detailed testing documentation**, see [DEVELOPMENT.md](docs/DEVELOPMENT.md)

## Research & References

This implementation is based on peer-reviewed research:

- **Basic RAG**: Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020
- **CRAG**: Yan et al., [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884), arXiv 2024
- **GraphRAG**: Edge et al., [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130), arXiv 2024
- **ColBERT**: Khattab & Zaharia, [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832), SIGIR 2020

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Testing guidelines
- Code style and standards
- Pull request process

## Community & Support

- 💬 **Discussions**: [GitHub Discussions](https://github.com/intersystems-community/iris-rag-templates/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/intersystems-community/iris-rag-templates/issues)
- 📖 **Documentation**: [Full Documentation](docs/)
- 🏢 **Enterprise Support**: [InterSystems Support](https://www.intersystems.com/support/)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ by the InterSystems Community**

*Powering intelligent applications with enterprise-grade RAG*
