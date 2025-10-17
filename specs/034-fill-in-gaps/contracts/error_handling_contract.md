# Test Contract: Error Handling and Edge Cases

**Contract ID**: ERROR-001
**Requirements**: FR-023, FR-024, FR-025
**Test File**: `tests/contract/test_error_handling_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG handles error conditions correctly when required tables are missing, connections fail, or other exceptional scenarios occur.

## Test Cases

### TC-023: Missing Required Tables (FR-023)
**Given**: iris_graph_core expects tables (RDF_EDGES, kg_NodeEmbeddings_optimized)
**And**: Tables do not exist in database
**When**: Query executed via iris_graph_core methods
**Then**:
- System detects missing tables (via exception or 0 results)
- Error logged with table name and details
- Falls back to IRISVectorStore successfully
- No database schema errors propagate to caller
- System continues functioning normally

**Test Method**: `test_missing_required_tables_handled`

### TC-024: iris_graph_core Connection Failure (FR-024)
**Given**: HybridGraphRAG pipeline initialized
**And**: iris_graph_core connection fails (mocked exception)
**When**: Query executed with iris_graph_core methods
**Then**:
- Connection exception caught and logged
- Error message includes connection details
- Falls back to IRISVectorStore
- Query completes successfully via fallback
- No connection error propagates to caller

**Test Method**: `test_iris_graph_core_connection_failure_handled`

### TC-025: System Continues After Fallback Invocation (FR-025)
**Given**: HybridGraphRAG pipeline initialized
**And**: Multiple queries executed
**When**: First query triggers fallback (iris_graph_core fails)
**And**: Second query executed on same pipeline instance
**Then**:
- Second query executes successfully
- Pipeline state remains consistent after fallback
- No degradation in subsequent query performance
- Fallback mechanism reusable across multiple queries

**Test Method**: `test_system_continues_after_fallback`

## Assertions

All test cases MUST assert:
1. Errors caught and handled gracefully
2. Appropriate error logging with details
3. Fallback mechanism activates correctly
4. No exceptions propagate to user code
5. System state remains consistent
6. Subsequent operations succeed

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance
- `mocker`: pytest-mock for simulating error conditions
- `caplog`: Log output capture for error validation

## Success Criteria

- All 3 test cases pass
- Tests use @pytest.mark.requires_database
- Execution time <35 seconds
- Tests validate robust error handling
