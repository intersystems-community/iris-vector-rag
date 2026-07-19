# User Guide

## Requirements

- Python 3.11+
- Docker
- An OpenAI API key (for answer generation)

## Installation

```bash
git clone https://github.com/intersystems-community/iris-vector-rag.git
cd iris-vector-rag
pip install -e .
```

## Start IRIS

```bash
docker compose up -d
```

This starts InterSystems IRIS Community Edition on `localhost:1972`. No license
required.

## Configure

```bash
cp .env.example .env
```

Edit `.env` and set your `OPENAI_API_KEY`. The IRIS connection defaults
(`localhost`, port `1972`, namespace `USER`, `_SYSTEM`/`SYS`) match the
`docker-compose.yml` and work without changes.

## Run your first query

```python
from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document

pipeline = create_pipeline('basic')

pipeline.load_documents(documents=[
    Document(
        page_content='RAG combines retrieval with generation for accurate AI.',
        metadata={'source': 'intro.pdf'},
    ),
    Document(
        page_content='Vector search finds similar content using embeddings.',
        metadata={'source': 'vectors.pdf'},
    ),
])

result = pipeline.query('What is RAG?', top_k=5, generate_answer=True)
print(result['answer'])
```

## Switch pipelines

All six pipelines share the same interface:

```python
pipeline = create_pipeline('basic')           # Vector similarity
pipeline = create_pipeline('basic_rerank')    # + cross-encoder reranking
pipeline = create_pipeline('crag')            # + self-correction + web fallback
pipeline = create_pipeline('graphrag')        # + knowledge graph
pipeline = create_pipeline('multi_query_rrf') # + query expansion + rank fusion
pipeline = create_pipeline('pylate_colbert')  # + ColBERT late interaction
```

## Response shape

Every pipeline returns the same dict:

```python
result['answer']               # LLM-generated answer string
result['retrieved_documents']  # List[Document]
result['contexts']             # List[str] -- for RAGAS evaluation
result['sources']              # Source citations
result['metadata']             # Timing, pipeline type, method used
```

## Evaluate with RAGAS

Compare pipelines side-by-side:

```bash
python examples/compare_pipelines.py --pipelines basic,basic_rerank
```

## Optional extras

```bash
pip install iris-vector-rag[colbert]     # ColBERT/PyLate support
pip install iris-vector-rag[evaluation]  # Full RAGAS evaluation framework
pip install iris-vector-rag[api]         # REST API server (FastAPI + Redis)
pip install iris-vector-rag[mcp]         # MCP server for Claude/AI tools
```

## MCP server

Expose all pipelines as MCP tools:

```bash
docker build -f Dockerfile.mcp -t iris-vector-rag-mcp .
docker run -p 3000:3000 --env-file .env iris-vector-rag-mcp
```

## Environment variables reference

All variables are optional except `OPENAI_API_KEY`:

| Variable         | Default     | Description                    |
| ---------------- | ----------- | ------------------------------ |
| `OPENAI_API_KEY` | --          | Required for answer generation |
| `IRIS_HOST`      | `localhost` | IRIS SuperServer host          |
| `IRIS_PORT`      | `1972`      | IRIS SuperServer port          |
| `IRIS_NAMESPACE` | `USER`      | IRIS namespace                 |
| `IRIS_USERNAME`  | `_SYSTEM`   | IRIS username                  |
| `IRIS_PASSWORD`  | `SYS`       | IRIS password                  |
