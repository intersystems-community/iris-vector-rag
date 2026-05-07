# IRIS Vector RAG

Production-ready RAG (Retrieval-Augmented Generation) pipelines powered by InterSystems IRIS vector search.

**Author:** Thomas Dyar (thomas.dyar@intersystems.com)

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/intersystems-community/iris-vector-rag.git
cd iris-vector-rag
pip install -e .

# 2. Start IRIS
docker compose up -d

# 3. Configure
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 4. Query
python -c "
from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document

pipeline = create_pipeline('basic')
pipeline.load_documents(documents=[
    Document(page_content='RAG combines retrieval with generation for accurate AI.', metadata={'source': 'intro.pdf'}),
    Document(page_content='Vector search finds similar content using embeddings.', metadata={'source': 'vectors.pdf'}),
])
result = pipeline.query('What is RAG?', top_k=5, generate_answer=True)
print(result['answer'])
"
```

## Pipelines

All pipelines share the same interface — switch with one line:

```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline('basic')           # Vector similarity search
pipeline = create_pipeline('basic_rerank')    # + cross-encoder reranking
pipeline = create_pipeline('crag')            # + self-correction + web fallback
pipeline = create_pipeline('graphrag')        # + knowledge graph + entity reasoning
pipeline = create_pipeline('multi_query_rrf') # + query expansion + rank fusion
pipeline = create_pipeline('pylate_colbert')  # + ColBERT late interaction
```

| Pipeline | Method | Best For |
|----------|--------|----------|
| `basic` | Vector similarity | General Q&A, getting started |
| `basic_rerank` | Vector + reranking | Higher accuracy, medical/legal |
| `crag` | Vector + evaluation + web | Fact-checking, current events |
| `graphrag` | Vector + text + graph + RRF | Complex relationships, research |
| `multi_query_rrf` | Query expansion + fusion | Comprehensive coverage |
| `pylate_colbert` | ColBERT embeddings | Fine-grained matching |

## Response Format

All pipelines return the same structure (LangChain/RAGAS compatible):

```python
result = pipeline.query("What is diabetes?", top_k=5)

result["answer"]                # LLM-generated answer
result["retrieved_documents"]   # List[Document]
result["contexts"]              # List[str] — for RAGAS evaluation
result["sources"]               # Source citations
result["metadata"]              # Timing, pipeline type, method used
```

## Configuration

Environment variables (loaded automatically from `.env`):

```
OPENAI_API_KEY=sk-...          # Required for answer generation
IRIS_HOST=localhost             # IRIS SuperServer host
IRIS_PORT=1972                  # IRIS SuperServer port
IRIS_NAMESPACE=USER             # IRIS namespace
IRIS_USERNAME=_SYSTEM           # IRIS username
IRIS_PASSWORD=SYS               # IRIS password
```

## Evaluate with RAGAS

Compare pipelines side-by-side using real RAGAS metrics:

```bash
python examples/compare_pipelines.py --pipelines basic,basic_rerank
```

Or in code:

```python
from iris_vector_rag import create_pipeline
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, context_precision, context_recall

pipeline = create_pipeline('basic')
pipeline.load_documents(documents=docs)
result = pipeline.query("What is diabetes?", top_k=3, generate_answer=True)

sample = SingleTurnSample(
    user_input="What is diabetes?",
    response=result["answer"],
    retrieved_contexts=result["contexts"],
    reference="Diabetes is a chronic condition...",
)
scores = evaluate(EvaluationDataset(samples=[sample]),
                  metrics=[faithfulness, context_precision, context_recall])
```

## Optional Extras

```bash
pip install iris-vector-rag[colbert]     # ColBERT/PyLate support
pip install iris-vector-rag[dspy]        # DSPy prompt optimization
pip install iris-vector-rag[evaluation]  # RAGAS evaluation framework
pip install iris-vector-rag[api]         # REST API server (FastAPI + Redis)
```

## MCP Server

Use as an AI tool server via Model Context Protocol:

```bash
docker build -f Dockerfile.mcp -t iris-vector-rag-mcp .
docker run -p 3000:3000 --env-file .env iris-vector-rag-mcp
```

## Development

```bash
pip install -e ".[dspy,evaluation]"
pytest tests/unit/                    # Fast, no IRIS needed
pytest tests/unit/ tests/contract/    # Full suite, needs IRIS running
```

## Architecture

```
iris_vector_rag/
├── pipelines/      # 6 RAG implementations (basic, crag, graphrag, etc.)
├── core/           # Base classes, models, connection management
├── storage/        # IRIS vector store, schema management
├── embeddings/     # Embedding generation and caching
├── services/       # Entity extraction, storage adapters
├── config/         # Configuration management
├── mcp/            # MCP server implementation
└── api/            # Optional REST API (FastAPI)
```

## License

MIT
