# Contract: Fallback Mechanism Validation (FALLBACK-001)

**Feature**: 036-retrofit-graphrag-s
**Requirements**: FR-015, FR-016, FR-017, FR-018, FR-019, FR-020
**Applies To**: CRAG, BasicRerankRAG, PyLateColBERT (BasicRAG has embedding fallback only)

---

## Purpose

Validate that pipelines with multiple retrieval methods implement automatic fallback when primary methods fail, with appropriate logging and metadata.

---

## Contract Specification

### 1. Fallback Strategies (Pipeline-Specific)

#### BasicRAG (FR-016)
**Primary**: Live embedding service
**Fallback**: Cached embeddings (if available)
**Trigger**: `EmbeddingServiceUnavailableError`, `EmbeddingTimeout`

```python
# Fallback logic
try:
    embedding = embedding_manager.generate_embedding(query)
except EmbeddingServiceUnavailableError:
    logger.info("Embedding service unavailable, using cached embeddings")
    embedding = cache.get_embedding(query)
    if embedding is None:
        raise ConfigurationError("No cached embeddings available")
```

#### CRAG (FR-015)
**Primary**: Relevance-evaluated vector search
**Fallback**: Vector + text fusion (skip evaluation)
**Trigger**: `RelevanceEvaluatorError`, `EvaluatorTimeout`, `EvaluatorAPIError`

```python
# Fallback logic
try:
    relevance_score = evaluate_relevance(query, contexts)
    if relevance_score < threshold:
        contexts = web_search_augmentation(query)
except RelevanceEvaluatorError as e:
    logger.info(f"Relevance evaluator failed: {e}, skipping evaluation")
    # Proceed with vector + text fusion
    contexts = hybrid_retrieval(query)
```

#### BasicRerankRAG
**Primary**: Cross-encoder reranking
**Fallback**: Initial vector similarity ranking
**Trigger**: `RerankerModelError`, `RerankerTimeout`, `RerankerLoadError`

```python
# Fallback logic
try:
    reranked_results = reranker.rerank(query, contexts)
except RerankerModelError as e:
    logger.info(f"Reranker failed: {e}, using initial ranking")
    reranked_results = contexts  # Keep original ranking
```

#### PyLateColBERT
**Primary**: ColBERT late interaction search
**Fallback**: Dense vector search (all-MiniLM-L6-v2)
**Trigger**: `ColBERTModelError`, `ColBERTScoreError`, `ColBERTLoadError`

```python
# Fallback logic
try:
    results = colbert_search(query)
except ColBERTModelError as e:
    logger.info(f"ColBERT search failed: {e}, falling back to dense vector")
    results = dense_vector_search(query)
```

---

### 2. Fallback Logging Requirements (FR-017)

**Required Log Fields**:
- **Level**: INFO
- **Message**: "{Pipeline} fallback activated: {reason}"
- **Context**:
  - `primary_method`: Name of failed method
  - `fallback_method`: Name of fallback method
  - `trigger_condition`: Exception type or reason
  - `query`: Query text (first 100 chars)

**Log Format**:
```
INFO: BasicRerankRAG fallback activated: RerankerTimeout
      Primary method: cross_encoder_reranking
      Fallback method: vector_similarity_ranking
      Trigger: Reranker timeout after 30s
      Query: What are the symptoms of diabetes?
```

---

### 3. Fallback Metadata (FR-017)

**Response Metadata** MUST include:
```python
{
    "metadata": {
        "retrieval_method": str,      # e.g., "vector_fallback", "rerank_fallback"
        "fallback_used": bool,        # True when fallback activated
        "fallback_reason": str,       # Exception message or reason
        "primary_method": str,        # Name of failed primary method
        "fallback_method": str        # Name of successful fallback method
    }
}
```

---

### 4. Query Semantics Preservation (FR-018)

**Requirement**: Fallback MUST preserve query semantics (return equivalent results when possible)

**Validation Criteria**:
- Fallback results MUST be relevant to query
- Result count SHOULD be within 50% of primary method count
- Source attribution MUST be maintained
- Context quality SHOULD be comparable (validated via RAGAS or manual inspection)

**Example Assertion**:
```python
primary_result_count = 10
fallback_result_count = 8

# Within 50% acceptable
assert fallback_result_count >= primary_result_count * 0.5
```

---

### 5. Fallback Configuration (FR-019)

**Configuration Option**: Allow disabling specific fallbacks

**Config Format** (config.yaml):
```yaml
fallback_config:
  enable_fallbacks: true              # Global fallback toggle
  crag:
    enable_evaluator_fallback: true
  basic_rerank:
    enable_reranker_fallback: true
  pylate_colbert:
    enable_colbert_fallback: true
```

**Behavior When Disabled**:
- If `enable_fallbacks: false` → raise exception (no fallback)
- If pipeline-specific disabled → raise exception for that pipeline only

---

### 6. Fallback Chain Termination (FR-020)

**Requirement**: Fallback chains MUST terminate gracefully when all options exhausted

**Chain Termination Logic**:
```python
try:
    result = primary_method(query)
except PrimaryMethodError:
    try:
        result = fallback_method_1(query)
    except FallbackMethod1Error:
        try:
            result = fallback_method_2(query)
        except FallbackMethod2Error:
            # All fallbacks exhausted
            raise AllFallbacksExhaustedError(
                "All retrieval methods failed. Cannot complete query.",
                chain=[
                    ("primary_method", "PrimaryMethodError"),
                    ("fallback_method_1", "FallbackMethod1Error"),
                    ("fallback_method_2", "FallbackMethod2Error")
                ],
                fix="Check database connectivity and service availability"
            )
```

**Error Message** (when all exhausted):
```
Error: All retrieval methods exhausted
Context: CRAG pipeline, query operation
Attempts:
  1. relevance_evaluated_search: RelevanceEvaluatorTimeout
  2. vector_text_fusion: IRISVectorStore connection failed
  3. web_search_augmentation: Web API unavailable
Fix: Check database connectivity (IRIS container running)
     Verify LLM API key configured
     Check network connectivity for web search
```

---

## Test Implementation

### Test Files (Pipeline-Specific)
- `tests/contract/test_basic_fallback.py` (embedding fallback only)
- `tests/contract/test_crag_fallback.py`
- `tests/contract/test_basic_rerank_fallback.py`
- `tests/contract/test_pylate_colbert_fallback.py`

### Test Cases

#### Test: Fallback Retrieves Documents Successfully
```python
def test_fallback_retrieves_documents_successfully(pipeline, mocker):
    """
    FR-016/FR-015: Fallback MUST retrieve documents successfully.

    Given: Pipeline with fallback mechanism
    And: Primary method fails (mocked)
    When: Query executed
    Then: Fallback retrieves documents successfully
    """
    query = "What are diabetes symptoms?"

    # Mock primary method to fail
    mocker.patch.object(
        pipeline,
        '_primary_retrieval_method',
        side_effect=PrimaryMethodError("Primary failed")
    )

    # Execute query (should use fallback)
    result = pipeline.query(query)

    # Verify fallback succeeded
    assert len(result['contexts']) > 0, "Fallback should retrieve documents"
    assert result['metadata']['fallback_used'] is True
```

#### Test: Fallback Activation Logged
```python
def test_fallback_activation_logged(pipeline, mocker, caplog):
    """
    FR-017: Fallback activation MUST be logged.

    Given: Pipeline with fallback mechanism
    And: Primary method fails
    When: Query executed
    Then: INFO log contains fallback activation details
    """
    caplog.set_level(logging.INFO)
    query = "test query"

    # Mock primary method to fail
    mocker.patch.object(
        pipeline,
        '_primary_retrieval_method',
        side_effect=PrimaryMethodError("Primary failed")
    )

    result = pipeline.query(query)

    log_output = caplog.text.lower()

    # Verify fallback logged
    assert "fallback" in log_output
    assert "activated" in log_output or "using" in log_output
```

#### Test: Fallback Preserves Query Semantics
```python
def test_fallback_preserves_query_semantics(pipeline, mocker):
    """
    FR-018: Fallback MUST preserve query semantics.

    Given: Pipeline with fallback mechanism
    When: Primary method returns N results
    And: Fallback returns M results
    Then: M should be within 50% of N (semantic equivalence)
    """
    query = "diabetes prevention"

    # Get baseline (primary method results)
    baseline_result = pipeline.query(query)
    baseline_count = len(baseline_result['contexts'])

    # Mock primary to fail, trigger fallback
    mocker.patch.object(
        pipeline,
        '_primary_retrieval_method',
        side_effect=PrimaryMethodError("Simulated failure")
    )

    fallback_result = pipeline.query(query)
    fallback_count = len(fallback_result['contexts'])

    # Verify semantic preservation (within 50%)
    assert fallback_count >= baseline_count * 0.5, \
        f"Fallback results ({fallback_count}) too few compared to primary ({baseline_count})"
```

#### Test: Fallback Can Be Disabled
```python
def test_fallback_can_be_disabled(pipeline, mocker):
    """
    FR-019: Fallback MUST be configurable (can be disabled).

    Given: Fallback configuration disabled
    And: Primary method fails
    When: Query executed
    Then: Exception raised (no fallback attempted)
    """
    # Disable fallback
    pipeline.config.fallback_config.enable_fallbacks = False

    # Mock primary method to fail
    mocker.patch.object(
        pipeline,
        '_primary_retrieval_method',
        side_effect=PrimaryMethodError("Primary failed")
    )

    # Should raise exception (no fallback)
    with pytest.raises(PrimaryMethodError):
        pipeline.query("test query")
```

#### Test: All Fallbacks Exhausted Handled
```python
def test_all_fallbacks_exhausted_handled(pipeline, mocker):
    """
    FR-020: System MUST handle all fallbacks exhausted gracefully.

    Given: Pipeline with fallback chain
    And: All methods fail
    When: Query executed
    Then: AllFallbacksExhaustedError raised with chain details
    """
    # Mock all methods to fail
    mocker.patch.object(
        pipeline,
        '_primary_method',
        side_effect=PrimaryMethodError("Primary failed")
    )
    mocker.patch.object(
        pipeline,
        '_fallback_method_1',
        side_effect=FallbackMethod1Error("Fallback 1 failed")
    )
    mocker.patch.object(
        pipeline,
        '_fallback_method_2',
        side_effect=FallbackMethod2Error("Fallback 2 failed")
    )

    with pytest.raises(AllFallbacksExhaustedError) as exc_info:
        pipeline.query("test query")

    # Verify error message includes chain
    error_msg = str(exc_info.value).lower()
    assert "exhausted" in error_msg
    assert "primary" in error_msg
    assert "fallback" in error_msg
```

---

## Acceptance Criteria

- ✅ Pipelines with multiple methods implement fallback (FR-015, FR-016)
- ✅ Fallback activation logged at INFO level (FR-017)
- ✅ Fallback metadata included in response (FR-017)
- ✅ Fallback preserves query semantics (FR-018)
- ✅ Fallback can be disabled via configuration (FR-019)
- ✅ Fallback chain terminates gracefully when exhausted (FR-020)
- ✅ All tests complete in <30 seconds (FR-005)

---

## Notes

- BasicRAG has minimal fallback (embedding cache only)
- CRAG, BasicRerankRAG, PyLateColBERT have rich fallback chains
- Fallback configuration MUST be documented in pipeline docs
- Fallback behavior MUST be tested in integration tests (not just contract tests)
