# Test Contract: Enhanced Text Search Query Path

**Contract ID**: TEXT-001
**Requirements**: FR-007, FR-008, FR-009
**Test File**: `tests/contract/test_text_search_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG's enhanced text search path (`method="text"`) executes iFind text search via iris_graph_core correctly and falls back when necessary.

## Test Cases

### TC-007: Text Search Success Path (FR-007)
**Given**: HybridGraphRAG pipeline initialized with iris_graph_core available
**When**: Query executed with `method="text"`
**Then**:
- iFind text search executes via iris_graph_core
- Returns list of relevant documents (len > 0)
- Metadata indicates `retrieval_method="text"` or similar
- Documents matched based on textual relevance

**Test Method**: `test_text_search_executes_successfully`

### TC-008: Text Search Fallback on Zero Results (FR-008)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core text search returns 0 results (mocked)
**When**: Query executed with `method="text"`
**Then**:
- System detects 0 results from text search
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- Warning log message indicates text search fallback

**Test Method**: `test_text_search_fallback_on_zero_results`

### TC-009: Text Search Fallback on Exception (FR-009)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core text search raises exception (mocked)
**When**: Query executed with `method="text"`
**Then**:
- System catches exception from text search
- Logs error message with exception details
- Falls back to IRISVectorStore.similarity_search()
- Returns documents from fallback (len > 0)
- Metadata indicates `retrieval_method="vector_fallback"`
- No exception propagates to caller

**Test Method**: `test_text_search_fallback_on_exception`

## Assertions

All test cases MUST assert:
1. Result is RAGResponse object
2. Metadata contains `retrieval_method` key
3. Documents retrieved appropriately
4. Fallback scenarios use vector_fallback method
5. Success scenarios use text search method
6. Logging validates fallback triggers

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance
- `mocker`: pytest-mock for patching text search methods
- `caplog`: Log output capture

## Success Criteria

- All 3 test cases pass
- Tests use @pytest.mark.requires_database
- Execution time <30 seconds
- Parallel execution supported
