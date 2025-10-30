# Test Contract: Fallback Mechanism Validation

**Contract ID**: FALLBACK-001
**Requirements**: FR-016, FR-017, FR-018, FR-019
**Test File**: `tests/contract/test_fallback_mechanism_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG's fallback mechanism to IRISVectorStore works correctly across all query methods, with appropriate logging and metadata.

## Test Cases

### TC-016: IRISVectorStore Fallback Retrieval (FR-016)
**Given**: HybridGraphRAG pipeline initialized
**And**: Primary retrieval method unavailable/failing (mocked)
**When**: Query executed with any method
**Then**:
- System falls back to IRISVectorStore.similarity_search()
- Fallback retrieves documents successfully (len > 0)
- Documents contain expected fields (content, metadata)
- Vector similarity search executes correctly

**Test Method**: `test_fallback_retrieves_documents_successfully`

### TC-017: Fallback Diagnostic Logging (FR-017)
**Given**: HybridGraphRAG pipeline initialized
**And**: Primary retrieval method fails (mocked exception)
**When**: Query executed and fallback triggered
**Then**:
- ERROR log message indicates primary method failure
- WARNING log message indicates fallback activation
- Log messages include method name and reason for fallback
- Log output parseable for monitoring/debugging

**Test Method**: `test_fallback_logs_diagnostic_messages`

### TC-018: Fallback Metadata Indicator (FR-018)
**Given**: HybridGraphRAG pipeline initialized
**And**: Fallback triggered for any query method
**When**: Query completes via fallback
**Then**:
- Result metadata contains `retrieval_method="vector_fallback"`
- Metadata distinguishes fallback from primary vector method
- Response clearly indicates fallback occurred
- Metadata includes execution time and document count

**Test Method**: `test_fallback_metadata_indicates_vector_fallback`

### TC-019: Graceful Degradation Without iris_graph_core (FR-019)
**Given**: iris_graph_core not installed/available (mocked unavailable)
**When**: HybridGraphRAG pipeline initializes
**Then**:
- Initialization succeeds without exception
- Pipeline gracefully degrades to standard GraphRAG functionality
- INFO log message indicates iris_graph_core unavailable
- All query methods still functional (using fallback paths)

**Test Method**: `test_graceful_degradation_without_iris_graph_core`

## Assertions

All test cases MUST assert:
1. Fallback executes successfully
2. Documents retrieved via IRISVectorStore
3. Logging output contains expected diagnostic messages
4. Metadata correctly indicates fallback
5. No exceptions propagate to caller
6. System remains functional after fallback

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline instance
- `mocker`: pytest-mock for simulating unavailable iris_graph_core
- `caplog`: Log output capture for diagnostic validation

## Success Criteria

- All 4 test cases pass
- Tests use @pytest.mark.requires_database
- Execution time <40 seconds
- Tests validate cross-cutting fallback behavior
