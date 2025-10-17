# Contract: Pipeline API Behavior (API-001)

**Feature**: 036-retrofit-graphrag-s
**Requirements**: FR-001, FR-002, FR-003, FR-004
**Applies To**: BasicRAG, CRAG, BasicRerankRAG, PyLateColBERT

---

## Purpose

Validate that all 4 target pipelines implement standardized RAGPipeline API with correct input handling, response structure, and error conditions.

---

## Contract Specification

### 1. Query Method Contract

**Signature**:
```python
def query(
    self,
    query: str,
    method: str = "vector",  # May vary by pipeline
    top_k: int = 5,
    **kwargs
) -> dict
```

**Input Validation** (FR-002):
- `query` MUST be non-empty string
- `query` MUST NOT exceed 10,000 characters
- `method` MUST be valid for pipeline (e.g., "vector", "hybrid", "rrf")
- `top_k` MUST be positive integer between 1 and 100
- Invalid inputs MUST raise `ValueError` with clear message

**Response Structure** (FR-003):
```python
{
    "answer": str,                    # Required
    "contexts": list[str],            # Required (may be empty)
    "metadata": {                     # Required
        "retrieval_method": str,      # Required
        "context_count": int,         # Required
        "sources": list[str],         # Required
        "execution_time_ms": float,   # Required
        "pipeline_type": str,         # Required
        "fallback_used": bool         # Required
    }
}
```

**Error Conditions** (FR-004):
- Missing LLM API key → `ConfigurationError` with env var name
- Missing embedding model → `ConfigurationError` with model path
- Database connection failure → `ConnectionError` with retry suggestion
- Dimension mismatch → `DimensionMismatchError` with dimensions
- All errors MUST include actionable guidance (FR-010)

---

### 2. Load Documents Method Contract

**Signature**:
```python
def load_documents(
    self,
    documents: list[Document],
    batch_size: int = 100,
    **kwargs
) -> dict
```

**Input Validation** (FR-002):
- `documents` MUST be non-empty list
- Each document MUST have `content` field
- Each document MUST have `metadata` dict
- `batch_size` MUST be positive integer
- Invalid inputs MUST raise `ValueError`

**Response Structure** (FR-003):
```python
{
    "documents_loaded": int,          # Required
    "documents_failed": int,          # Required
    "embeddings_generated": int,      # Required
    "storage_complete": bool,         # Required
    "errors": list[str],              # Required (may be empty)
    "execution_time_ms": float        # Required
}
```

**Error Conditions** (FR-004):
- Empty document content → Skip document, log warning
- Embedding service failure → Use fallback (FR-016)
- Database storage failure → Retry with backoff (FR-012)
- Batch processing error → Continue with next batch, log errors

---

### 3. Embed Method Contract (Optional)

**Signature**:
```python
def embed(
    self,
    texts: list[str],
    **kwargs
) -> list[list[float]]
```

**Input Validation** (FR-002):
- `texts` MUST be non-empty list
- Each text MUST be string
- Invalid inputs MUST raise `ValueError`

**Response Structure** (FR-003):
```python
[
    [0.1, 0.2, ...],  # 384D vector for all-MiniLM-L6-v2
    [0.3, 0.4, ...],  # One vector per input text
]
```

**Dimension Validation** (FR-021):
- Output vectors MUST match expected dimensions
- BasicRAG/CRAG/BasicRerankRAG: 384 dimensions
- PyLateColBERT: Variable (ColBERT token embeddings)

**Error Conditions** (FR-004):
- Embedding model not loaded → `ConfigurationError`
- Input text too long → Truncate with warning
- Service unavailable → Use cached embeddings (FR-016)

---

## Test Implementation

### Test Files
- `tests/contract/test_basic_rag_contract.py`
- `tests/contract/test_crag_contract.py`
- `tests/contract/test_basic_rerank_contract.py`
- `tests/contract/test_pylate_colbert_contract.py`

### Test Cases (Per Pipeline)

#### Test: Query Method Validates Inputs
```python
def test_query_validates_required_parameter(pipeline):
    """FR-002: Query MUST validate required query parameter."""
    with pytest.raises(ValueError, match="query.*required"):
        pipeline.query(query=None)

    with pytest.raises(ValueError, match="query.*empty"):
        pipeline.query(query="")
```

#### Test: Query Method Returns Valid Structure
```python
def test_query_returns_valid_structure(pipeline):
    """FR-003: Query MUST return valid response structure."""
    result = pipeline.query("What are diabetes symptoms?")

    assert "answer" in result
    assert "contexts" in result
    assert "metadata" in result
    assert "retrieval_method" in result["metadata"]
    assert "context_count" in result["metadata"]
```

#### Test: Query Method Handles Errors
```python
def test_query_handles_missing_api_key(pipeline, mocker):
    """FR-004: Query MUST handle missing API key gracefully."""
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        pipeline.query("test query")
```

#### Test: Load Documents Validates Inputs
```python
def test_load_documents_validates_input(pipeline):
    """FR-002: Load documents MUST validate input."""
    with pytest.raises(ValueError, match="documents.*required"):
        pipeline.load_documents(documents=[])

    with pytest.raises(ValueError, match="content.*required"):
        pipeline.load_documents(documents=[Document(metadata={})])
```

#### Test: Load Documents Returns Valid Structure
```python
def test_load_documents_returns_valid_structure(pipeline, sample_documents):
    """FR-003: Load documents MUST return valid structure."""
    result = pipeline.load_documents(sample_documents)

    assert "documents_loaded" in result
    assert "documents_failed" in result
    assert "embeddings_generated" in result
    assert result["documents_loaded"] == len(sample_documents)
```

---

## Acceptance Criteria

- ✅ All 4 pipelines implement query() with consistent signature
- ✅ All 4 pipelines implement load_documents() with consistent signature
- ✅ All 4 pipelines validate inputs and raise clear errors
- ✅ All 4 pipelines return standardized response structures
- ✅ All 4 pipelines handle error conditions gracefully
- ✅ All contract tests complete in <30 seconds (FR-005)
- ✅ All error messages include actionable guidance (FR-010)

---

## Notes

- Pipeline-specific parameters (e.g., `method="rrf"` for CRAG) are allowed via `**kwargs`
- Response metadata may include pipeline-specific fields
- Error handling may include pipeline-specific errors (e.g., `RerankerError` for BasicRerankRAG)
- All pipelines MUST extend `RAGPipeline` base class (Constitutional requirement I)
