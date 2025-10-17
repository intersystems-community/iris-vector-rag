# Test Contract: Hybrid Fusion Query Path

**Contract ID**: HYBRID-001
**Requirements**: FR-001, FR-002, FR-003
**Test File**: `tests/contract/test_hybrid_fusion_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG's hybrid fusion query path (`method="hybrid"`) executes multi-modal search correctly and falls back to vector search when iris_graph_core fails or returns 0 results.

## Test Cases

### TC-001: Hybrid Fusion Success Path (FR-001)
**Given**: HybridGraphRAG pipeline initialized with iris_graph_core available
**When**: Query executed with `method="hybrid"`
**Then**:
- Hybrid fusion search executes via iris_graph_core
- Returns list of relevant documents (len > 0)
- Metadata indicates `retrieval_method="hybrid_fusion"` or similar
- Documents contain content and similarity scores

**Test Method**: `test_hybrid_fusion_executes_successfully`

### TC-002: Hybrid Fusion Fallback on Zero Results (FR-002)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core hybrid fusion returns 0 results (mocked)
**When**: Query executed with `method="hybrid"`
**Then**:
- System detects 0 results from iris_graph_core
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- Warning log message indicates fallback occurred

**Test Method**: `test_hybrid_fusion_fallback_on_zero_results`

### TC-003: Hybrid Fusion Fallback on Exception (FR-003)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core hybrid fusion raises exception (mocked)
**When**: Query executed with `method="hybrid"`
**Then**:
- System catches exception from iris_graph_core
- Logs error message with exception details
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- No exception propagates to caller

**Test Method**: `test_hybrid_fusion_fallback_on_exception`

## Assertions

All test cases MUST assert:
1. Result is RAGResponse object with non-None contexts
2. Metadata contains `retrieval_method` key
3. Documents retrieved (len >= 0)
4. For fallback scenarios: retrieval_method == "vector_fallback"
5. For success scenarios: retrieval_method indicates hybrid fusion
6. Logging output contains expected messages (captured via caplog fixture)

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance connected to live IRIS
- `mocker`: pytest-mock for patching iris_graph_core methods
- `caplog`: pytest fixture for capturing log output

## Mocking Patterns

```python
# Pattern for 0 results
mocker.patch.object(
    pipeline.retrieval_methods,
    'retrieve_via_hybrid_fusion',
    return_value=([], 'hybrid_fusion')
)

# Pattern for exception
mocker.patch.object(
    pipeline.retrieval_methods,
    'retrieve_via_hybrid_fusion',
    side_effect=Exception("Connection failed")
)
```

## Success Criteria

- All 3 test cases pass
- Tests execute against live IRIS database (@pytest.mark.requires_database)
- Total execution time <30 seconds
- Tests can run in parallel with other contract test files
