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
from iris_vector_rag import create_pipeline

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
from iris_vector_rag import create_pipeline

# Start with basic
pipeline = create_pipeline('basic')
result = pipeline.query("What are the latest cancer treatment approaches?", top_k=5)

# Upgrade to basic_rerank for better accuracy
pipeline = create_pipeline('basic_rerank')
result = pipeline.query("What are the latest cancer treatment approaches?", top_k=5)

# Try graphrag for entity reasoning
pipeline = create_pipeline('graphrag')
result = pipeline.query("What are the latest cancer treatment approaches?", top_k=5)

# All pipelines return the same response format
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Retrieved: {len(result['retrieved_documents'])} documents")
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

## Pipeline Selection

**Each pipeline uses the same API** - just change the pipeline type:

- **`basic`** - Fast vector similarity search, great for getting started
- **`basic_rerank`** - Vector + cross-encoder reranking for higher accuracy
- **`crag`** - Self-correcting with web search fallback for current events
- **`graphrag`** - Multi-modal: vector + text + knowledge graph fusion
- **`multi_query_rrf`** - Query expansion with reciprocal rank fusion
- **`pylate_colbert`** - ColBERT late interaction for fine-grained matching

📖 **[Complete Pipeline Guide →](docs/PIPELINE_GUIDE.md)** - Decision tree, performance comparison, configuration examples

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

### IRIS EMBEDDING: Auto-Vectorization

**Automatic embedding generation with model caching** - eliminates repeated model loading overhead for faster document vectorization.

**Key Features**:
- ⚡ Intelligent model caching - models stay in memory across operations
- 🎯 Multi-field vectorization - combine title, abstract, and content fields
- 💾 Automatic device selection - GPU, Apple Silicon (MPS), or CPU fallback

```python
from iris_vector_rag import create_pipeline

# Enable IRIS EMBEDDING support
pipeline = create_pipeline(
    'basic',
    embedding_config='medical_embeddings_v1'
)

# Documents auto-vectorize on INSERT
pipeline.load_documents(documents=docs)
```

📖 **[Complete IRIS EMBEDDING Guide →](docs/IRIS_EMBEDDING_GUIDE.md)** - Configuration, performance tuning, multi-field vectorization, troubleshooting

## Model Context Protocol (MCP) Support

**Expose RAG pipelines as MCP tools** for Claude Desktop and other MCP clients - enables conversational RAG workflows where Claude queries your documents during conversations.

```bash
# Start MCP server
python -m iris_vector_rag.mcp
```

All pipelines available as MCP tools: `rag_basic`, `rag_basic_rerank`, `rag_crag`, `rag_graphrag`, `rag_multi_query_rrf`, `rag_pylate_colbert`.

📖 **[Complete MCP Integration Guide →](docs/MCP_INTEGRATION.md)** - Claude Desktop setup, configuration, testing, production deployment

## Architecture Overview

**Framework-first design** with abstract base classes (`RAGPipeline`, `VectorStore`) and concrete implementations for 6 production-ready pipelines.

**Key Components**: Core abstractions, pipeline implementations, IRIS vector store, MCP server, REST API, validation framework.

📖 **[Comprehensive Architecture Guide →](docs/architecture/COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md)** - System design, component interactions, extension points

## Documentation

📚 **Comprehensive documentation for every use case:**

- **[User Guide](docs/USER_GUIDE.md)** - Complete installation and usage
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation
- **[Pipeline Guide](docs/PIPELINE_GUIDE.md)** - When to use each pipeline
- **[MCP Integration](docs/MCP_INTEGRATION.md)** - Model Context Protocol setup
- **[Production Readiness](docs/PRODUCTION_READINESS_ASSESSMENT.md)** - Deployment checklist

## Testing & Quality

```bash
make test  # Run comprehensive test suite
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
```

## Research & References

This implementation is based on peer-reviewed research:

- **Basic RAG**: Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020
- **CRAG**: Yan et al., [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884), arXiv 2024
- **GraphRAG**: Edge et al., [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130), arXiv 2024
- **ColBERT**: Khattab & Zaharia, [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832), SIGIR 2020

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development setup, testing guidelines, and pull request process.

## Community & Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/intersystems-community/iris-vector-rag/issues)
- 📖 **Documentation**: [Full Documentation](docs/)
- 🏢 **Enterprise Support**: [InterSystems Support](https://www.intersystems.com/support/)

## License

MIT License - see [LICENSE](LICENSE) for details.
