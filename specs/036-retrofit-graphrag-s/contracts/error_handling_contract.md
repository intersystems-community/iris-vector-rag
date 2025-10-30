# Contract: Error Handling and Diagnostic Messages (ERROR-001)

**Feature**: 036-retrofit-graphrag-s
**Requirements**: FR-009, FR-010, FR-011, FR-012, FR-013, FR-014
**Applies To**: BasicRAG, CRAG, BasicRerankRAG, PyLateColBERT

---

## Purpose

Validate that all 4 pipelines provide clear diagnostic error messages with actionable guidance and handle transient failures gracefully.

---

## Contract Specification

### 1. Configuration Errors (FR-009, FR-010, FR-011)

**Error Types**:
- `MissingAPIKeyError`: Required API key not found in environment
- `MissingEmbeddingModelError`: Embedding model not configured
- `InvalidConnectionStringError`: Database connection parameters invalid
- `DimensionMismatchError`: Embedding dimensions don't match database schema

**Error Message Template**:
```
Error: {specific_problem}
Context: {pipeline_type}, {operation}, {current_state}
Expected: {what_should_be}
Actual: {what_was_encountered}
Fix: {actionable_steps}
```

**Examples**:

**Missing API Key**:
```
Error: OpenAI API key not configured
Context: BasicRAG pipeline, query operation, initialization
Expected: OPENAI_API_KEY environment variable set
Actual: Environment variable not found
Fix: Set OPENAI_API_KEY environment variable
     export OPENAI_API_KEY="sk-..."
     Or add to .env file: OPENAI_API_KEY=sk-...
```

**Dimension Mismatch**:
```
Error: Embedding dimension mismatch
Context: BasicRAG pipeline, vector_search operation, query embedding
Expected: 384 dimensions (all-MiniLM-L6-v2)
Actual: 768 dimensions
Fix: Verify embedding model configuration in config.yaml
     Current model may be BERT-base (768D) instead of all-MiniLM-L6-v2 (384D)
     Re-index documents with correct model or update config.yaml
```

---

### 2. Transient Failure Handling (FR-012)

**Transient Errors** (MUST retry with exponential backoff):
- Database connection timeouts
- LLM API rate limits (429 status)
- Network timeouts
- Temporary service unavailability

**Retry Strategy**:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def transient_operation():
    # Operation that may fail transiently
    pass
```

**Logging Requirements**:
- Log each retry attempt at INFO level
- Log final failure at ERROR level after max retries
- Include retry count and next retry delay in logs

**Example Log Sequence**:
```
INFO: Database connection attempt 1 failed: Timeout after 5s
INFO: Retrying in 1s (attempt 1/3)
INFO: Database connection attempt 2 failed: Timeout after 5s
INFO: Retrying in 2s (attempt 2/3)
ERROR: Database connection failed after 3 attempts: Giving up
```

---

### 3. Contextual Error Messages (FR-013)

**Required Context Fields**:
```python
{
    "pipeline_type": str,        # "basic", "crag", etc.
    "operation": str,            # "query", "load_documents", "embed"
    "current_state": dict,       # Relevant state variables
    "timestamp": str,            # ISO 8601 timestamp
    "request_id": str           # Unique request identifier (optional)
}
```

**State Information to Include**:
- Pipeline configuration (model names, top-K, etc.)
- Current query or document being processed
- Retrieval method being used
- Number of documents loaded
- Connection status

---

### 4. Error Chains (FR-014)

**Error Chain Logging** (when multiple failures occur):

**Example Scenario**: Primary method fails, fallback attempted, fallback also fails

**Log Sequence**:
```
ERROR: Primary retrieval method failed: iris_graph_core connection timeout
INFO: Attempting fallback to IRISVectorStore
ERROR: Fallback failed: IRISVectorStore connection timeout
ERROR: All retrieval methods exhausted:
       1. iris_graph_core: Connection timeout after 5s
       2. IRISVectorStore: Connection timeout after 5s
       No more fallback options available
Fix: Check database connectivity, verify IRIS container is running
     docker ps | grep iris
```

**Chain Structure**:
```python
{
    "primary_error": {
        "method": "iris_graph_core",
        "error": "ConnectionTimeout",
        "message": "Connection timeout after 5s"
    },
    "fallback_attempts": [
        {
            "method": "IRISVectorStore",
            "error": "ConnectionTimeout",
            "message": "Connection timeout after 5s"
        }
    ],
    "final_state": "all_methods_exhausted",
    "fix": "Check database connectivity..."
}
```

---

## Test Implementation

### Test Files
- `tests/contract/test_basic_error_handling.py`
- `tests/contract/test_crag_error_handling.py`
- `tests/contract/test_basic_rerank_error_handling.py`
- `tests/contract/test_pylate_colbert_error_handling.py`

### Test Cases (Per Pipeline)

#### Test: Missing API Key Provides Clear Error
```python
def test_missing_api_key_error_is_actionable(pipeline, mocker):
    """
    FR-009, FR-010: Missing API key error MUST include actionable guidance.

    Given: OpenAI API key not set in environment
    When: Pipeline query executed
    Then: ConfigurationError raised with env var name and how to set it
    """
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(ConfigurationError) as exc_info:
        pipeline.query("test query")

    error_msg = str(exc_info.value).lower()

    # MUST mention specific env var
    assert "openai_api_key" in error_msg

    # MUST suggest how to fix
    assert "export" in error_msg or "set" in error_msg
```

#### Test: Database Connection Failure Retries
```python
def test_database_connection_retries_with_backoff(pipeline, mocker, caplog):
    """
    FR-012: Database connection failure MUST retry with exponential backoff.

    Given: Database connection transiently unavailable
    When: Query executed
    Then: System retries 3 times with exponential backoff
    And: Each retry attempt is logged
    """
    caplog.set_level(logging.INFO)

    # Mock connection to fail twice, succeed third time
    connection_mock = mocker.Mock(side_effect=[
        ConnectionError("Timeout"),
        ConnectionError("Timeout"),
        mocker.Mock()  # Success
    ])

    mocker.patch.object(pipeline, '_get_connection', connection_mock)

    result = pipeline.query("test query")

    # Verify retries occurred
    assert connection_mock.call_count == 3

    # Verify retry logging
    log_output = caplog.text.lower()
    assert "retry" in log_output or "attempt" in log_output
```

#### Test: Error Message Includes Context
```python
def test_error_includes_pipeline_context(pipeline, mocker):
    """
    FR-013: Error messages MUST include contextual information.

    Given: Error occurs during query
    When: Exception is raised
    Then: Error message includes pipeline type, operation, and state
    """
    # Simulate error with missing context
    mocker.patch.object(pipeline, 'vector_store', None)

    with pytest.raises(ConfigurationError) as exc_info:
        pipeline.query("test query")

    error_msg = str(exc_info.value).lower()

    # MUST include pipeline type
    assert pipeline.__class__.__name__.lower() in error_msg

    # MUST include operation
    assert "query" in error_msg or "search" in error_msg
```

#### Test: Error Chain Logs All Failures
```python
def test_error_chain_logs_all_failures(pipeline, mocker, caplog):
    """
    FR-014: Error chain MUST log all failure attempts.

    Given: Primary method fails, fallback method fails
    When: All methods exhausted
    Then: Error chain logged with all attempts and final state
    """
    caplog.set_level(logging.ERROR)

    # Mock primary method to fail
    mocker.patch.object(
        pipeline,
        '_primary_retrieval',
        side_effect=ConnectionError("Primary failed")
    )

    # Mock fallback to also fail
    mocker.patch.object(
        pipeline,
        '_fallback_retrieval',
        side_effect=ConnectionError("Fallback failed")
    )

    with pytest.raises(Exception) as exc_info:
        pipeline.query("test query")

    log_output = caplog.text

    # MUST log both failures
    assert "primary failed" in log_output.lower()
    assert "fallback failed" in log_output.lower()

    # MUST indicate exhaustion
    assert "exhausted" in log_output.lower() or "no more" in log_output.lower()
```

---

## Pipeline-Specific Error Scenarios

### BasicRAG
- Missing OpenAI/Anthropic API key
- Embedding model not found
- Database connection failure
- Dimension mismatch (384D expected)

### CRAG
- All BasicRAG errors, plus:
- Relevance evaluator API failure
- Evaluator timeout
- Web search API unavailable (if configured)

### BasicRerankRAG
- All BasicRAG errors, plus:
- Cross-encoder model not loaded
- Reranker timeout
- Reranker dimension mismatch

### PyLateColBERT
- All BasicRAG errors, plus:
- ColBERT model not loaded
- ColBERT score computation error
- Token embedding dimension mismatch

---

## Acceptance Criteria

- ✅ All configuration errors include actionable guidance (FR-010)
- ✅ All errors include pipeline context (type, operation, state) (FR-013)
- ✅ Transient failures retry with exponential backoff (FR-012)
- ✅ Retry attempts are logged at INFO level
- ✅ Error chains log all failure attempts (FR-014)
- ✅ All errors fail fast on initialization when critical config missing (FR-011)
- ✅ Error messages follow standardized template
- ✅ Tests complete in <30 seconds (FR-005)

---

## Notes

- Error templates MUST be consistent across all 4 pipelines
- Pipeline-specific errors (e.g., `RerankerError`) allowed but MUST follow template
- Retry logic MUST use exponential backoff (not linear)
- Error context MUST NOT include sensitive data (API keys, credentials)
