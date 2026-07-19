# Plan: Pipeline Interface Conformance

## Overview

Align all six RAG pipelines to a unified interface with consistent return types,
parameter names, and dependency injection. This fixes four concrete divergences:
`load_documents()` return type, `query()` parameter name, `sources` key location,
and factory kwargs pass-through.

## Key Files

### Core & Factories

- `iris_vector_rag/core/base.py` — Update `load_documents()` return type to
  `Dict[str, Any]`
- `iris_vector_rag/__init__.py` — Pass `connection_manager`, `config_manager`,
  `llm_func`, `embedding_func` to all constructors

### Pipelines

- `iris_vector_rag/pipelines/basic.py` — Rename `query()` param from `query` to
  `query_text`
- `iris_vector_rag/pipelines/basic_rerank.py` — Align return types and param
  names
- `iris_vector_rag/pipelines/crag.py` — Return stats dict from `load_documents`;
  promote `sources` to top-level in `query()`
- `iris_vector_rag/pipelines/graphrag.py` — Return stats dict from
  `load_documents`; `query()` already uses `query_text`
- `iris_vector_rag/pipelines/hybrid_graphrag.py` — Same as graphrag.py
- `iris_vector_rag/pipelines/multi_query_rrf.py` — Accept `llm_func`,
  `embedding_func` in `__init__`; return stats dict; rename param to `query_text`
- `iris_vector_rag/pipelines/colbert_pylate/pylate_pipeline.py` — Accept
  `llm_func`, `embedding_func` in `__init__`; update return types

### Tests

- `tests/unit/test_pipeline_conformance.py` — **NEW** parameterized conformance
  test suite (test first)

## Implementation Approach

### Phase 1 — Tests First

Write parameterized conformance test in
`tests/unit/test_pipeline_conformance.py`:

- **Test structure**: `pytest.mark.parametrize` over all six pipeline types
- **`load_documents()` shape**:
  - Construct each pipeline with mocked vector store
  - Call `load_documents(documents=[mock_doc])`
  - Assert result dict has keys: `documents_loaded`, `documents_failed`,
    `embeddings_generated`
- **`query()` shape**:
  - Call `query(query_text="test", generate_answer=False)` (no LLM)
  - Assert all required keys present: `answer`, `retrieved_documents`,
    `contexts`, `sources`, `metadata`, `error`
  - Assert `sources` is top-level (not inside metadata)
- **Parameter names**:
  - Both `query(query_text="test")` and `query("test")` succeed
- **Factory kwargs pass-through**:
  - Create `multi_query_rrf` pipeline with `llm_func=mock_fn`
  - Verify pipeline uses the provided function

Test location:
`/Users/tdyar/ws/iris-vector-rag-private/tests/unit/test_pipeline_conformance.py`

### Phase 2 — ABC Fix

Update `iris_vector_rag/core/base.py` lines 54–66:

Change return type from `-> None` to `-> Dict[str, Any]` and document required
keys in docstring.

```python
@abc.abstractmethod
def load_documents(self, documents_path: str, **kwargs) -> Dict[str, Any]:
    """
    Loads and processes documents into the RAG pipeline's knowledge base.

    Args:
        documents_path: Path to documents or directory.
        **kwargs: Additional keyword arguments for loading.

    Returns:
        Dict with keys:
            - documents_loaded: int (>=0)
            - documents_failed: int (>=0)
            - embeddings_generated: int (>=0)
    """
```

### Phase 3 — Pipeline Fixes

#### basic.py (line 427)

Rename `query()` parameter from `query` to `query_text`. Update all internal
references.

#### basic_rerank.py

Same as basic.py: align parameter names and return types.

#### crag.py (lines 82–232)

1. **`load_documents()` return** (line 82):
   - Return type already `Dict[str, Any]` — ensure all return paths include
     required keys

2. **`query()` sources** (line 215–232):
   - Move `sources` to top-level response (not inside metadata)

#### graphrag.py & hybrid_graphrag.py (lines 82–172, 173–232)

1. **`load_documents()` return** (line 82):
   - Change return type from `-> None` to `-> Dict[str, Any]`
   - Add return statement at end with required keys

2. **`query()` parameter**:
   - Already correct (uses `query_text`)

#### multi_query_rrf.py

1. **Constructor** (lines 58–113):
   - Add `llm_func=None, embedding_func=None` parameters
   - Store and use them if provided

2. **`query()` parameter** (line 248):
   - Rename from `query` to `query_text`

3. **`load_documents()` return** (lines 365–381):
   - Change return type from `-> None` to `-> Dict[str, Any]`
   - Return dict with required keys

#### pylate_pipeline.py

Same fixes as multi_query_rrf.py.

### Phase 4 — Factory Fix

Update `iris_vector_rag/__init__.py` lines 123–192.

Pass `connection_manager`, `config_manager`, `llm_func`, `embedding_func` to
all pipeline constructors in `_create_pipeline_legacy()`.

Example for `multi_query_rrf`:

```python
elif pipeline_type == "multi_query_rrf":
    from .pipelines.multi_query_rrf import MultiQueryRRFPipeline

    return MultiQueryRRFPipeline(
        connection_manager=connection_manager,
        config_manager=config_manager,
        llm_func=llm_func,
        embedding_func=kwargs.get("embedding_func"),
        num_queries=kwargs.get("num_queries", 4),
        retrieved_k=kwargs.get("retrieved_k", 20),
        rrf_k=kwargs.get("rrf_k", 60),
    )
```

Apply same pattern to all other pipelines.

## Risks & Constraints

- **Breaking change**: `query()` parameter rename may affect keyword-arg callers.
  Mitigation: scan tests/examples first.
- **Return type change**: Non-breaking for code that ignores return value.
- **Factory kwargs**: Ensure multi_query_rrf and pylate pipelines receive all
  expected dependencies.

## Dependencies

- Tests must run without IRIS (mock vector store, connection manager)
- Existing 219+ unit tests must continue to pass
- No changes to core query logic or retrieval behavior

## Testing Order

1. Write conformance tests (Phase 1)
2. Fix base.py (Phase 2)
3. Fix all pipelines (Phase 3)
4. Fix factory (Phase 4)
5. Run full suite: `pytest tests/unit/ tests/contract/`
