# RAG-Templates API Reference

**Version**: 0.6.0
**Last Updated**: 2026-07-24

## Overview

RAG-Templates provides a unified, standardized API across all 6 production RAG pipelines. All pipelines implement the same core methods with consistent signatures and response formats, ensuring 100% compatibility with LangChain and RAGAS evaluation frameworks.

## Quick Start

```python
from iris_vector_rag import create_pipeline

# Create any pipeline with validation
pipeline = create_pipeline("basic", validate_requirements=True)

# Load documents
result = pipeline.load_documents(documents=[...])

# Query
result = pipeline.query("What is machine learning?", top_k=5)
```

## Core API Functions

### `create_pipeline()`

Factory function to create RAG pipeline instances with automatic validation.

**Signature:**

```python
def create_pipeline(
    pipeline_type: str,
    config_path: Optional[str] = None,
    llm_func: Optional[Callable[[str], str]] = None,
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    external_connection = None,
    validate_requirements: bool = True,
    auto_setup: bool = False,
    **kwargs
) -> RAGPipeline
```

**Parameters:**

- `pipeline_type` (str): Pipeline type to create
  - `"basic"` - BasicRAG with vector similarity
  - `"basic_rerank"` - BasicRAG + cross-encoder reranking
  - `"crag"` - Corrective RAG with self-evaluation
  - `"graphrag"` - HybridGraphRAG (vector + text + graph)
  - `"pylate_colbert"` - ColBERT late interaction
- `config_path` (str, optional): Path to configuration file
- `llm_func` (callable, optional): LLM function for answer generation
- `embedding_func` (callable, optional): Embedding function for vectors
- `external_connection` (optional): Existing database connection
- `validate_requirements` (bool): Validate pipeline requirements (default: True)
- `auto_setup` (bool): Auto-fix validation issues (default: False)
- `**kwargs`: Additional pipeline-specific parameters

**Returns:** RAGPipeline instance

**Raises:**

- `ValueError`: If pipeline_type is unknown
- `PipelineValidationError`: If validation fails and auto_setup is False

**Example:**

```python
from iris_vector_rag import create_pipeline

# With validation
pipeline = create_pipeline("basic", validate_requirements=True)

# Without validation (faster, for testing)
pipeline = create_pipeline("basic", validate_requirements=False)

# With auto-setup (fixes missing tables, etc.)
pipeline = create_pipeline("crag", validate_requirements=True, auto_setup=True)
```

### `validate_pipeline()`

Validate pipeline requirements without creating an instance.

**Signature:**

```python
def validate_pipeline(
    pipeline_type: str,
    config_path: Optional[str] = None,
    external_connection = None
) -> Dict[str, Any]
```

**Returns:** Validation results dictionary with detailed status

**Example:**

```python
from iris_vector_rag import validate_pipeline

status = validate_pipeline("graphrag")
print(f"Valid: {status['is_valid']}")
print(f"Issues: {status['issues']}")
```

### `setup_pipeline()`

Set up all requirements for a pipeline type.

**Signature:**

```python
def setup_pipeline(
    pipeline_type: str,
    config_path: Optional[str] = None,
    external_connection = None
) -> Dict[str, Any]
```

**Returns:** Setup results dictionary

**Example:**

```python
from iris_vector_rag import setup_pipeline

result = setup_pipeline("basic")
print(f"Setup complete: {result['success']}")
```

## Pipeline Methods

All pipeline classes implement these core methods with identical signatures.

### `load_documents()`

Load documents into the pipeline for retrieval.

**Signature:**

```python
def load_documents(
    self,
    documents: Optional[List[Document]] = None,
    documents_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

- `documents` (List[Document], optional): List of Document objects to load
- `documents_path` (str, optional): Path to documents file (JSON)
- `**kwargs`: Pipeline-specific parameters

**Returns:**

```python
{
    "documents_loaded": int,       # Number successfully loaded
    "embeddings_generated": int,   # Number of embeddings created
    "documents_failed": int        # Number that failed to load
}
```

**Validation:**

- Requires either `documents` or `documents_path` (not both None)
- Rejects empty document lists with actionable error message
- Validates Document objects have required fields

**Example:**

```python
from iris_rag.core.models import Document

# Option 1: From Document objects
docs = [
    Document(
        page_content="Python is a programming language...",
        metadata={"source": "intro.txt", "author": "John"}
    ),
    Document(
        page_content="Machine learning uses algorithms...",
        metadata={"source": "ml.txt", "topic": "AI"}
    )
]
result = pipeline.load_documents(documents=docs)
print(f"Loaded {result['documents_loaded']} documents")

# Option 2: From file
result = pipeline.load_documents(documents_path="data/docs.json")
```

### `query()`

Execute a RAG query with document retrieval and optional answer generation.

**Signature:**

```python
def query(
    self,
    query: str,
    top_k: int = 5,
    generate_answer: bool = True,
    include_sources: bool = True,
    retrieval: str | None = None,
    weights: dict | None = None,
    rerank: bool | str | callable | None = None,
    metadata_filter: dict | None = None,
    similarity_threshold: float | None = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

- `query` (str): The query text; `query_text=` accepted as alias (backward-compat)
- `top_k` (int): Documents to retrieve, range [1-100] (default: 5)
- `generate_answer` (bool): Call LLM to generate answer (default: True)
- `include_sources` (bool): Include source metadata in response (default: True)
- `retrieval` (str | None): Retrieval mode — `"vector"` (default), `"text"`, `"hybrid"`, `"rrf"`. `"text"` and fusion modes require `iris-vector-graph`.
- `weights` (dict | None): Score-fusion weights, e.g. `{"vector": 0.7, "text": 0.3}`. Used with `"hybrid"` or `"rrf"`.
- `rerank` (bool | str | callable | None): Apply reranking after retrieval. `True` uses the default cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). A string names a strategy (`"cross-encoder"`). A callable receives `(query, docs)` and returns reordered docs. Model loads once per process (cached).
- `metadata_filter` (dict | None): Restrict results by metadata field, e.g. `{"source": "pubmed"}`. Allowed keys vary by pipeline; unknown keys raise `VectorStoreConfigurationError`.
- `similarity_threshold` (float | None): Exclude documents below this cosine similarity score (0.0–1.0).

**Returns:**

```python
{
    "query": str,                            # Original query
    "answer": str | None,                    # LLM-generated answer
    "retrieved_documents": List[Document],   # LangChain Document objects
    "contexts": List[str],                   # RAGAS-compatible contexts
    "sources": List[Dict],                   # Source references
    "execution_time": float,                 # Query execution time
    "metadata": {
        "num_retrieved": int,                # Documents retrieved
        "processing_time": float,
        "pipeline_type": str,                # Pipeline identifier
        "retrieval_mode": str,               # Retrieval strategy used ("vector", "text", "hybrid", "rrf")
        "rerank_strategy": str | None,       # "cross-encoder", custom name, or None
        "rerank_degraded": bool,             # True if reranking fell back due to error
        "weights": dict | None,              # Fusion weights applied
        "context_count": int,                # Number of contexts
        "sources": List[Dict],               # Also in metadata
        # Pipeline-specific fields...
    }
}
```

**Validation:**

- Query cannot be empty or whitespace-only
- top_k must be between 1 and 100 (inclusive)
- Raises `ValueError` with 5-part error message on validation failure

**Error Message Format:**

```
Error: <what went wrong>
Context: <where it happened>
Expected: <what was expected>
Actual: <what was received>
Fix: <how to fix it>
```

**Example:**

```python
# Basic query
result = pipeline.query("What is diabetes?", top_k=5)
print(result["answer"])

# Without answer generation (retrieval only)
result = pipeline.query(
    "diabetes symptoms",
    top_k=10,
    generate_answer=False
)

# Access retrieved documents (LangChain compatible)
for doc in result["retrieved_documents"]:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:100]}...")

# Access contexts (RAGAS compatible)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

evaluation = evaluate(
    dataset={"contexts": result["contexts"], "answer": result["answer"]},
    metrics=[faithfulness, answer_relevancy]
)
```

### `get_pipeline_info()`

Get information about the pipeline's configuration.

**Signature:**

```python
def get_pipeline_info(self) -> Dict[str, Any]
```

**Returns:** Dictionary with pipeline configuration details

**Example:**

```python
info = pipeline.get_pipeline_info()
print(f"Type: {info['pipeline_type']}")
print(f"Config: {info}")
```

## Vector Store API

### `IRISVectorStore.search_by_text()`

Explicit text-search entry point. Returns documents ranked by BM25/text relevance.

```python
docs: List[Document] = vector_store.search_by_text(query="insulin resistance", top_k=10)
```

- Requires `iris-vector-graph` with a BM25 index populated on the table.
- Returns `List[Document]` (predictable type, unlike the legacy polymorphic `similarity_search`).

### `IRISVectorStore.search_by_vector()`

Explicit vector-search entry point. Returns documents with their similarity scores.

```python
results: List[Tuple[Document, float]] = vector_store.search_by_vector(embedding=[...], top_k=10)
```

- `embedding` must match the vector dimension of the table (e.g. 384 for `all-MiniLM-L6-v2`).
- Returns `List[Tuple[Document, float]]` where the float is cosine similarity.

The legacy `similarity_search(query, k, **kwargs)` method remains unchanged.

---

## Reranker API

### `iris_vector_rag.retrieval.rerank.resolve_reranker()`

Public resolver that returns a cached reranker callable. The underlying model loads **once per `(strategy, model_name)` tuple per process** and is shared across all callers.

```python
from iris_vector_rag.retrieval.rerank import resolve_reranker

# Default cross-encoder
fn = resolve_reranker(True)

# Named strategy with custom model
fn = resolve_reranker("cross-encoder", model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

# Custom callable — passed through directly, never cached
fn = resolve_reranker(lambda query, docs: sorted(docs, key=my_score_fn))

# Use the reranker
reranked_docs = fn(query="What is insulin?", docs=retrieved_docs)
```

**Callable signature:** `(query: str, docs: List[Document]) -> List[Document]`

Thread-safe: concurrent first-load calls for the same key load the model exactly once.

---

## Pipeline-Specific Features

### BasicRAGReranking

Additional metadata fields:

```python
result["metadata"]["reranked"]            # bool: Whether reranking was applied
result["metadata"]["initial_candidates"]  # int: Initial retrieval count
result["metadata"]["rerank_factor"]       # int: Reranking multiplier
```

Configuration:

```python
pipeline = create_pipeline("basic_rerank")
# Retrieves rerank_factor * top_k documents, reranks, returns top_k
```

### CRAG (Corrective RAG)

Additional metadata fields:

```python
result["metadata"]["evaluation_score"]    # float: Relevance evaluation
result["metadata"]["corrected"]           # bool: Whether correction applied
```

### HybridGraphRAG

Pipeline-specific query parameters:

```python
result = pipeline.query(
    query_text="cancer targets",
    method="rrf",           # "rrf", "hybrid", "vector", "text", "graph"
    vector_k=30,            # Documents from vector search
    text_k=30,              # Documents from text search
    graph_k=20,             # Documents from graph traversal
    top_k=15                # Final result count after fusion
)
```

Additional metadata fields:

```python
result["metadata"]["fusion_method"]       # str: Fusion strategy used
result["metadata"]["vector_score"]        # float: Vector search contribution
result["metadata"]["text_score"]          # float: Text search contribution
result["metadata"]["graph_score"]         # float: Graph traversal contribution
```

### PyLateColBERT

Additional metadata fields:

```python
result["metadata"]["native_reranking"]    # bool: PyLate reranking used
result["metadata"]["model_name"]          # str: ColBERT model identifier
```

## Data Models

### Document

Standard document object used across all pipelines.

```python
from iris_rag.core.models import Document

doc = Document(
    page_content="The text content of the document...",
    metadata={
        "source": "filename.txt",
        "author": "John Doe",
        "date": "2024-01-01",
        # Any custom fields...
    }
)
```

**Fields:**

- `page_content` (str): The main text content
- `metadata` (dict): Dictionary of metadata fields

**LangChain Compatibility:**
This is the standard LangChain Document class, ensuring 100% compatibility.

## Error Handling

### Validation Errors

All pipelines use consistent validation with actionable error messages:

```python
try:
    result = pipeline.query("", top_k=5)
except ValueError as e:
    print(str(e))
    # Output:
    # Error: Query parameter is required and cannot be empty
    # Context: BasicRAG pipeline query operation
    # Expected: Non-empty query string
    # Actual: Empty or whitespace-only string
    # Fix: Provide a valid query string, e.g., query='What is diabetes?'
```

### Common Errors

**Empty Query:**

```python
pipeline.query("")  # Raises ValueError
```

**Invalid top_k:**

```python
pipeline.query("test", top_k=0)    # Raises ValueError (< 1)
pipeline.query("test", top_k=101)  # Raises ValueError (> 100)
```

**Empty Document List:**

```python
pipeline.load_documents(documents=[])  # Raises ValueError
```

**Missing Required Parameter:**

```python
pipeline.load_documents()  # Raises ValueError (need documents or documents_path)
```

## Configuration

### Default Configuration

Located at `iris_rag/config/default_config.yaml`:

```yaml
database:
  db_host: localhost
  db_port: 1972
  db_namespace: USER

embedding_model:
  name: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384

pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200

  basic_reranking:
    rerank_factor: 2
    reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

  crag:
    relevance_threshold: 0.5
```

### Environment Variables

Override configuration with environment variables:

```bash
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_USERNAME=SuperUser
export IRIS_PASSWORD=SYS
export OPENAI_API_KEY=your-key-here
```

## Testing

### Contract Tests

Validate API contracts without database:

```bash
pytest tests/contract/ -v
```

### Integration Tests

Full end-to-end with live database:

```bash
pytest tests/e2e/ -v
```

### Example Test

```python
from iris_vector_rag import create_pipeline
from iris_rag.core.models import Document

def test_basic_workflow():
    # Create pipeline
    pipeline = create_pipeline("basic", validate_requirements=False)

    # Load documents
    docs = [Document(page_content="Test content", metadata={"source": "test"})]
    result = pipeline.load_documents(documents=docs)
    assert result["documents_loaded"] == 1

    # Query
    result = pipeline.query("test", top_k=1)
    assert "answer" in result
    assert len(result["retrieved_documents"]) <= 1
```

## Migration from Old API

### Old API (Pre-Standardization)

```python
# OLD - Inconsistent signatures
pipeline.query(query_text="test", k=5)  # CRAG used query_text
pipeline.load_documents("path/to/docs")  # Only file paths
result["num_retrieved"]  # Inconsistent metadata
```

### New API (Standardized)

```python
# NEW - Consistent across all pipelines
pipeline.query(query="test", top_k=5)  # Unified signature
pipeline.load_documents(documents=[...])  # Supports both lists and paths
result["metadata"]["num_retrieved"]  # Standardized structure
```

## Best Practices

1. **Always use validation in production:**

   ```python
   pipeline = create_pipeline("basic", validate_requirements=True)
   ```

2. **Handle errors with specific exceptions:**

   ```python
   try:
       result = pipeline.query(user_input, top_k=5)
   except ValueError as e:
       logger.error(f"Query validation failed: {e}")
   ```

3. **Use Document objects for metadata preservation:**

   ```python
   docs = [Document(page_content=text, metadata={"source": file})]
   pipeline.load_documents(documents=docs)
   ```

4. **Access results with framework-specific interfaces:**

   ```python
   # For LangChain
   documents = result["retrieved_documents"]

   # For RAGAS
   contexts = result["contexts"]
   answer = result["answer"]
   ```

## Support

- **Documentation**: [docs/](../docs/)
- **Examples**: [scripts/](../scripts/)
- **Issues**: GitHub Issues
- **Testing**: [TEST_VALIDATION_SUMMARY.md](../TEST_VALIDATION_SUMMARY.md)
