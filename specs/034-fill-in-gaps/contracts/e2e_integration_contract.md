# Test Contract: End-to-End Integration Testing

**Contract ID**: E2E-001
**Requirements**: FR-026, FR-027, FR-028
**Test File**: `tests/integration/test_hybridgraphrag_e2e.py`

## Contract Overview

This contract validates end-to-end query workflows for HybridGraphRAG, ensuring all 5 query methods work correctly in realistic scenarios with proper metadata and sequential execution.

## Test Cases

### TC-026: All Query Methods End-to-End (FR-026)
**Given**: HybridGraphRAG pipeline initialized with full setup
**And**: Database contains test data (2,376 documents)
**When**: Queries executed for each of 5 methods sequentially:
  - `method="hybrid"` (hybrid fusion)
  - `method="rrf"` (Reciprocal Rank Fusion)
  - `method="text"` (iFind text search)
  - `method="vector"` (HNSW vector search)
  - `method="kg"` (knowledge graph traversal)
**Then**:
- Each query completes successfully
- Each returns documents (or uses fallback)
- Metadata indicates correct retrieval method
- No exceptions during any query
- All query methods validated in single test

**Test Method**: `test_all_query_methods_end_to_end`

### TC-027: Multiple Sequential Queries (FR-027)
**Given**: HybridGraphRAG pipeline initialized
**And**: Same pipeline instance reused
**When**: 10+ sequential queries executed
**And**: Queries use different methods randomly
**Then**:
- All queries complete successfully
- Pipeline state remains consistent across queries
- No memory leaks or connection issues
- Performance remains stable (no degradation)
- Cache/state management works correctly

**Test Method**: `test_multiple_sequential_queries_consistent`

### TC-028: Retrieval Metadata Completeness (FR-028)
**Given**: HybridGraphRAG pipeline initialized
**When**: Query executed with any method
**Then**:
- Result metadata contains `retrieval_method` key
- Result metadata contains `execution_time` (in seconds)
- Result metadata contains `num_retrieved` (document count)
- Metadata values are correct and match actual results
- Metadata format consistent across all query methods

**Test Method**: `test_retrieval_metadata_completeness`

## Assertions

All test cases MUST assert:
1. Query execution succeeds for all methods
2. Results contain expected data structures
3. Metadata fields present and correct
4. Pipeline state consistent across queries
5. Performance within acceptable bounds
6. No resource leaks or connection issues

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance
- Test data: Assumes existing 2,376 documents loaded
- Configuration: Default pipeline configuration

## Success Criteria

- All 3 test cases pass
- Tests use @pytest.mark.requires_database
- Tests execute against real IRIS database
- Execution time <60 seconds for full E2E suite
- Tests validate complete integration workflows
