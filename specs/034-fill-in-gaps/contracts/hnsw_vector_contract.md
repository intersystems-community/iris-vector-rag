# Test Contract: HNSW Vector Search Query Path

**Contract ID**: HNSW-001
**Requirements**: FR-010, FR-011, FR-012
**Test File**: `tests/contract/test_hnsw_vector_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG's HNSW-optimized vector search path (`method="vector"`) executes correctly via iris_graph_core and falls back to IRISVectorStore when needed.

## Test Cases

### TC-010: HNSW Vector Search Success Path (FR-010)
**Given**: HybridGraphRAG pipeline initialized with iris_graph_core available
**When**: Query executed with `method="vector"`
**Then**:
- HNSW-optimized vector search executes via iris_graph_core
- Returns list of relevant documents (len > 0)
- Metadata indicates `retrieval_method="vector"` or "hnsw_vector"
- Documents ranked by vector similarity

**Test Method**: `test_hnsw_vector_executes_successfully`

### TC-011: HNSW Vector Fallback on Zero Results (FR-011)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core HNSW search returns 0 results (mocked)
**When**: Query executed with `method="vector"`
**Then**:
- System detects 0 results from HNSW search
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- Warning log message indicates HNSW fallback

**Test Method**: `test_hnsw_vector_fallback_on_zero_results`

### TC-012: HNSW Vector Fallback on Exception (FR-012)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core HNSW search raises exception (mocked)
**When**: Query executed with `method="vector"`
**Then**:
- System catches exception from HNSW search
- Logs error message with exception details
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- No exception propagates to caller

**Test Method**: `test_hnsw_vector_fallback_on_exception`

## Assertions

All test cases MUST assert:
1. Result is RAGResponse object
2. Metadata contains `retrieval_method` key
3. Documents retrieved with vector similarity scores
4. Fallback scenarios correctly identify vector_fallback
5. Success scenarios use HNSW/vector method
6. Logging indicates fallback conditions

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance
- `mocker`: pytest-mock for patching HNSW methods
- `caplog`: Log output capture

## Success Criteria

- All 3 test cases pass
- Tests use @pytest.mark.requires_database
- Execution time <30 seconds
- Parallel execution supported
