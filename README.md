# IRIS Vector RAG

**Production-ready Retrieval-Augmented Generation (RAG) pipelines powered by InterSystems IRIS Vector Search**

Build intelligent applications that combine the power of large language models with your enterprise data using battle-tested RAG patterns and native vector search capabilities.

## Why IRIS Vector RAG?

- **🚀 Production-Ready Pipelines** - Six proven RAG architectures ready to deploy
- **⚡ Native Vector Search** - Leverage InterSystems IRIS's built-in vector database capabilities
- **🔧 Unified API** - Consistent interface across all pipeline types
- **📊 Enterprise-Grade** - Connection pooling, ACID transactions, and horizontal scaling built-in
- **🧪 100% Test Coverage** - Comprehensive test suite with 136 passing tests
- **🔗 Framework Compatible** - Works seamlessly with LangChain and RAGAS

## Available Pipelines

| Pipeline | Use Case | Retrieval Method | Best For |
|----------|----------|------------------|----------|
| **BasicRAG** | Standard retrieval | Vector similarity | General-purpose Q&A, simple use cases |
| **BasicRAGReranking** | Improved relevance | Vector + cross-encoder reranking | Higher precision requirements |
| **CRAG** | Self-correcting retrieval | Vector + evaluation + web search | Fact-checking, dynamic knowledge |
| **HybridGraphRAG** | Multi-modal retrieval | Vector + text + graph + RRF fusion | Complex entity relationships |
| **PyLateColBERT** | Late interaction | ColBERT contextualized embeddings | Fine-grained semantic matching |
| **IRIS-Global-GraphRAG** | Community detection | Graph communities + 3D visualization | Academic research, large corpora |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/intersystems-community/iris-vector-rag.git
cd iris-vector-rag

# Setup environment (requires uv)
make setup-env
make install
source .venv/bin/activate
```

### Database Setup

```bash
# Start IRIS database (Docker)
docker-compose up -d

# Initialize database
make setup-db
make load-data
```

### Create .env file

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS
EOF
```

### Your First Query

```python
from iris_rag import create_pipeline

# Create pipeline (validates database setup automatically)
pipeline = create_pipeline('basic', validate_requirements=True)

# Load your documents
pipeline.load_documents(documents_path="data/docs.json")

# Query with LLM-generated answer
result = pipeline.query(
    query="What is machine learning?",
    top_k=5,
    generate_answer=True
)

print(result['answer'])
print(result['sources'])
```

## Unified API

All pipelines share the same interface for easy experimentation:

```python
from iris_rag import create_pipeline

# Try different pipelines with the same code
for pipeline_type in ['basic', 'basic_rerank', 'crag', 'graphrag']:
    pipeline = create_pipeline(pipeline_type)
    result = pipeline.query("What are cancer treatment targets?", top_k=5)

    print(f"\n{pipeline_type.upper()}:")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Retrieved: {len(result['retrieved_documents'])} documents")
    print(f"Sources: {result['sources']}")
```

### Standardized Response Format

All pipelines return responses compatible with LangChain and RAGAS:

```python
{
    "query": "What is diabetes?",
    "answer": "Diabetes is a chronic condition...",           # LLM-generated answer
    "retrieved_documents": [Document(...)],                   # LangChain Document objects
    "contexts": ["context 1", "context 2"],                   # RAGAS-compatible contexts
    "sources": [{"source": "medical.pdf", "page": 12}],      # Source references
    "execution_time": 0.523,
    "metadata": {
        "num_retrieved": 5,
        "pipeline_type": "basic",
        "retrieval_method": "vector"
    }
}
```

## Enterprise Features

### Connection Pooling

IRIS Vector RAG includes built-in connection pooling for high-performance production deployments:

```python
from iris_rag.storage import IRISVectorStore

# Automatic connection pool management
store = IRISVectorStore()
# Pool handles concurrency automatically
```

### ACID Transactions

All write operations are ACID-compliant:

```python
# Load documents with transactional safety
result = pipeline.load_documents(documents)

if not result['success']:
    # Automatic rollback on failure
    print(f"Failed: {result['error']}")
```

### Horizontal Scaling

IRIS supports distributed deployment with:
- Multi-node clustering
- Automatic load balancing
- Distributed vector search
- Enterprise resilience features

## Pipeline-Specific Examples

### CRAG (Corrective RAG)

Self-correcting retrieval with web search fallback:

```python
pipeline = create_pipeline('crag')

result = pipeline.query(
    query="Latest developments in quantum computing",
    top_k=5,
    generate_answer=True
)

# CRAG automatically evaluates relevance and falls back to web search if needed
print(f"Retrieval method used: {result['metadata']['retrieval_method']}")
```

### HybridGraphRAG

Multi-modal search combining vector, text, and knowledge graph:

```python
pipeline = create_pipeline('graphrag')

result = pipeline.query(
    query_text="cancer treatment targets",
    method="rrf",        # Reciprocal Rank Fusion
    vector_k=30,
    text_k=30,
    graph_k=10
)

# Returns entities, relationships, and context
print(f"Retrieved entities: {len(result['metadata']['entities'])}")
print(f"Retrieved relationships: {len(result['metadata']['relationships'])}")
```

### PyLate ColBERT

Fine-grained late interaction retrieval:

```python
pipeline = create_pipeline('pylate_colbert')

# ColBERT computes token-level interactions
result = pipeline.query(
    query="symptoms of diabetes",
    top_k=5
)

# Higher precision through contextualized matching
print(result['answer'])
```

## Test Fixture System

IRIS Vector RAG includes a high-performance test fixture system for reproducible testing:

```bash
# List available fixtures
make fixture-list

# Load .DAT fixture (100-200x faster than JSON)
make fixture-load FIXTURE=medical-graphrag-20

# Validate fixture integrity
make fixture-validate FIXTURE=medical-graphrag-20
```

### Using Fixtures in Tests

```python
import pytest

@pytest.mark.dat_fixture("medical-graphrag-20")
def test_with_fixture():
    # Fixture automatically loaded with 21 entities, 15 relationships
    pipeline = create_pipeline("graphrag")
    result = pipeline.query("What are cancer treatment targets?")
    assert len(result["retrieved_documents"]) > 0
```

**Learn more**: See `tests/fixtures/README.md` for complete fixture documentation.

## Architecture

```
iris_rag/
├── core/           # Abstract base classes (RAGPipeline, VectorStore)
├── pipelines/      # Pipeline implementations
│   ├── basic.py
│   ├── basic_reranking.py
│   ├── crag.py
│   ├── graphrag.py
│   ├── hybrid_graphrag.py
│   └── colbert_pylate/
├── storage/        # Vector store implementations
│   └── vector_store_iris.py
├── services/       # Business logic (entity extraction, storage)
├── config/         # Configuration management
└── validation/     # Pipeline validation framework
```

## Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/contract/      # Contract tests

# Run with coverage
pytest --cov=iris_rag --cov-report=term-missing
```

## RAGAS Evaluation

Evaluate pipeline performance with RAGAS metrics:

```bash
# Quick evaluation (sample data)
make test-ragas-sample

# Full evaluation (1000 documents)
make test-ragas-1000
```

## Documentation

- **[User Guide](USER_GUIDE.md)** - Complete installation and usage guide
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation
- **[Fixture Guide](tests/fixtures/README.md)** - Test fixture system
- **[Architecture](docs/VALIDATED_ARCHITECTURE_SUMMARY.md)** - System design details
- **[Production Readiness](docs/PRODUCTION_READINESS_ASSESSMENT.md)** - Deployment checklist

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing guidelines, and contribution workflow.

## Research & References

This implementation is based on the following research:

- **Basic RAG**: Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020
- **CRAG**: Yan et al., [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884), arXiv 2024
- **GraphRAG**: Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), arXiv 2024
- **ColBERT**: Khattab & Zaharia, [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832), SIGIR 2020

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/intersystems-community/iris-vector-rag/issues)
- **Documentation**: [Full Documentation](docs/)
- **IRIS Vector Search**: [Official Documentation](https://docs.intersystems.com/iris20241/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch)
