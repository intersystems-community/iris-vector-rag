# Contract: GraphRAG Setup Requirements

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03

## Purpose

Resolve 11 GraphRAG test errors by fixing setup or properly skipping with explanations.

## GraphRAG Test Errors (11 total)

### Error Pattern
```
ERROR tests/e2e/test_graphrag_pipeline_e2e.py::TestGraphRAGPipelineQuerying::test_simple_graph_query
ERROR at setup of test_simple_graph_query
```

### Root Causes (to investigate)
1. Missing dependencies (graph-ai, entity extraction libs)
2. Fixture setup failures (pipeline initialization)
3. Import errors (OntologyAwareEntityExtractor not found)
4. LLM API configuration (entity extraction requires LLM)

## Requirements

### REQ-1: Dependency Resolution
- Verify graph-ai integration available
- Verify entity extraction dependencies installed
- If missing: add to requirements OR skip with explanation

### REQ-2: Fixture Setup
- GraphRAG pipeline fixture must initialize successfully
- If setup fails: log detailed error, skip test with reason

### REQ-3: LLM Configuration
- Entity extraction may require LLM API keys
- If not configured: skip tests with "LLM not configured" reason

## Resolution Options

### Option 1: Fix Setup (Preferred)
```python
@pytest.fixture(scope="module")
def graphrag_pipeline(pipeline_dependencies):
    """GraphRAG pipeline fixture with proper setup."""
    try:
        from iris_rag.pipelines.graphrag import GraphRAGPipeline
        pipeline = GraphRAGPipeline(...)
        return pipeline
    except ImportError as e:
        pytest.skip(f"GraphRAG dependencies not available: {e}")
```

### Option 2: Skip with Explanation
```python
@pytest.mark.skip(reason="GraphRAG requires graph-ai integration (optional dependency)")
def test_graphrag_feature():
    pass
```

## Success Criteria
- 11 GraphRAG errors resolved (either pass or skip with reason)
- No ERROR status tests (only PASS, FAIL, or SKIP)
- Clear documentation of GraphRAG requirements
