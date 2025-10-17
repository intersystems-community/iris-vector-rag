# Test Contract: RRF (Reciprocal Rank Fusion) Query Path

**Contract ID**: RRF-001
**Requirements**: FR-004, FR-005, FR-006
**Test File**: `tests/contract/test_rrf_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG's RRF query path (`method="rrf"`) executes Reciprocal Rank Fusion correctly, combining vector and text search results, and falls back appropriately when failures occur.

## Test Cases

### TC-004: RRF Success Path (FR-004)
**Given**: HybridGraphRAG pipeline initialized with iris_graph_core available
**When**: Query executed with `method="rrf"`
**Then**:
- RRF search executes via iris_graph_core
- Combines vector search and text search results
- Returns list of relevant documents (len > 0)
- Metadata indicates `retrieval_method="rrf"` or similar
- Documents ranked by fused scores

**Test Method**: `test_rrf_executes_successfully`

### TC-005: RRF Fallback on Zero Results (FR-005)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core RRF returns 0 results (mocked)
**When**: Query executed with `method="rrf"`
**Then**:
- System detects 0 results from RRF
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- Warning log message indicates RRF fallback

**Test Method**: `test_rrf_fallback_on_zero_results`

### TC-006: RRF Fallback on Exception (FR-006)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core RRF raises exception (mocked)
**When**: Query executed with `method="rrf"`
**Then**:
- System catches exception from RRF
- Logs error message with exception details
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- No exception propagates to caller

**Test Method**: `test_rrf_fallback_on_exception`

## Assertions

All test cases MUST assert:
1. Result is RAGResponse object with contexts
2. Metadata contains `retrieval_method` key
3. Documents retrieved (len >= 0)
4. For fallback scenarios: retrieval_method == "vector_fallback"
5. For success scenarios: retrieval_method indicates RRF
6. Logging output validates fallback trigger conditions

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance
- `mocker`: pytest-mock for patching iris_graph_core RRF methods
- `caplog`: Log output capture

## Success Criteria

- All 3 test cases pass
- Tests use @pytest.mark.requires_database
- Execution time <30 seconds
- Can run in parallel with other contract tests
