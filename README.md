# IRIS Vector RAG

RAG (Retrieval-Augmented Generation) pipelines powered by InterSystems IRIS vector search.

**Author:** Thomas Dyar (<thomas.dyar@intersystems.com>)

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

### Already have a table?

Search your own IRIS table, with your column names, embedding model and mandatory filters,
without copying it: see [Bring your own table](docs/BRING_YOUR_OWN_TABLE.md).

### Models configured in IRIS?

On IRIS 2026.3 with AI Hub, take the answer model from a ConfigStore entry, with its key
kept in the IRIS Wallet: `create_pipeline("basic", llm_config="opsreview")`. See
[The answer model from AI Hub ConfigStore](docs/LLM_FROM_CONFIGSTORE.md).

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

| Pipeline          | Method                      | Best For                        |
| ----------------- | --------------------------- | ------------------------------- |
| `basic`           | Vector similarity           | General Q&A, getting started    |
| `basic_rerank`    | Vector + reranking          | Higher accuracy, medical/legal  |
| `crag`            | Vector + evaluation + web   | Fact-checking, current events   |
| `graphrag`        | Vector + text + graph + RRF | Complex relationships, research |
| `multi_query_rrf` | Query expansion + fusion    | Comprehensive coverage          |
| `pylate_colbert`  | ColBERT embeddings          | Fine-grained matching           |

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

```bash
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
pip install iris-vector-rag[evaluation]  # RAGAS evaluation framework
```

## Removed: the REST API

The FastAPI REST service and its `api` extra were removed in 0.16
([ADR 0001](docs/adr/0001-retire-the-rest-api.md)). Use the MCP server below, or call
the pipelines from Python. The last version of the code is at the git tag
`archive/rest-api-v1`; the design is kept in
[docs/archived/rest-api/](docs/archived/rest-api/README.md).

> **Experimental:** `pip install iris-vector-rag[dspy]` provides DSPy prompt optimization
> modules, currently under development. See
> [spec 073](specs/073-dspy-optimization-layer/spec.md) for details. Not recommended for
> production.

## MCP Server

A stdio MCP server built on the official Python SDK exposes six tools: `rag_basic`,
`rag_basic_rerank`, `rag_crag`, `rag_graphrag`, `rag_pylate_colbert` and
`rag_health_check`. It connects to IRIS through the same settings as the library
(`IRIS_HOST`, `IRIS_PORT`, `IRIS_NAMESPACE`, `IRIS_USERNAME`, `IRIS_PASSWORD`).

```bash
pip install "iris-vector-rag[mcp]"
iris-vector-rag-mcp            # or: python -m iris_vector_rag.mcp
```

An MCP client starts the command itself; see
[docs/MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md) for the Claude Desktop configuration.

> For MCP tool orchestration across IRIS packages, use
> [iris-agentic-dev](https://github.com/intersystems-community/iris-agentic-dev).

## Development

```bash
pip install -e ".[evaluation]"
pytest tests/unit/                    # Fast, no IRIS needed
pytest tests/unit/ tests/contract/    # Full suite, needs IRIS running
```

To use experimental DSPy modules:

```bash
pip install -e ".[dspy,evaluation]"  # Installs experimental dspy_modules
```

## Architecture

```text
iris_vector_rag/
├── pipelines/      # 6 RAG implementations (basic, crag, graphrag, etc.)
├── core/           # Base classes, models, connection management
├── storage/        # IRIS vector store, schema management
├── embeddings/     # Embedding generation and caching
├── services/       # Entity extraction, storage adapters
├── config/         # Configuration management
└── mcp/            # MCP server implementation
```

## License

MIT
