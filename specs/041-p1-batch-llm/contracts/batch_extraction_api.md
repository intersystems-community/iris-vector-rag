# API Contract: Batch Entity Extraction

**Feature**: 041-p1-batch-llm
**Date**: 2025-10-15
**Status**: Contract Defined (Implementation Pending)

## Batch Processing API

### EntityExtractionService.extract_batch()

**Purpose**: Process multiple documents in a single LLM call to reduce API overhead.

**Signature**:
```python
def extract_batch(
    self,
    documents: List[Document],
    token_budget: int = 8192
) -> BatchExtractionResult:
    """
    Extract entities from multiple documents in single batch.

    Args:
        documents: List of Document objects to process (1-10+ documents)
        token_budget: Maximum tokens per batch (default 8192)

    Returns:
        BatchExtractionResult with per-document entities and relationships

    Raises:
        ValueError: If documents list is empty
        TokenBudgetExceededError: If batch exceeds token_budget even after retry/split
        LLMError: If batch fails after all retries and individual processing

    Requirements:
        - FR-001: Process 5-10 documents per call (reduced to 1+ for flexibility)
        - FR-005: Retry with exponential backoff (2s, 4s, 8s), then split
        - FR-006: Respect token_budget (dynamic batch sizing)
    """
```

**Contract Test** (`tests/contract/test_batch_extraction_contract.py`):
```python
def test_extract_batch_signature():
    """Validate extract_batch() has correct signature."""
    service = EntityExtractionService(config_manager, connection_manager)

    # Verify method exists
    assert hasattr(service, 'extract_batch')

    # Verify signature
    import inspect
    sig = inspect.signature(service.extract_batch)
    assert 'documents' in sig.parameters
    assert 'token_budget' in sig.parameters
    assert sig.parameters['token_budget'].default == 8192

def test_extract_batch_returns_batch_result():
    """Validate return type is BatchExtractionResult."""
    service = EntityExtractionService(config_manager, connection_manager)
    documents = [Document(id="test1", page_content="Test document")]

    result = service.extract_batch(documents)

    assert isinstance(result, BatchExtractionResult)
    assert result.batch_id is not None
    assert result.per_document_entities is not None

def test_extract_batch_empty_input_raises():
    """Validate error on empty documents list."""
    service = EntityExtractionService(config_manager, connection_manager)

    with pytest.raises(ValueError, match="documents list cannot be empty"):
        service.extract_batch([])
```

---

## Token Counting API

### TokenCounter.estimate_tokens()

**Purpose**: Estimate token count for batch sizing decisions.

**Signature**:
```python
def estimate_tokens(
    text: str,
    model: str = "gpt-3.5-turbo"
) -> int:
    """
    Estimate token count for given text using tiktoken.

    Args:
        text: Document text to estimate
        model: Model name for tokenizer (default gpt-3.5-turbo)

    Returns:
        Estimated token count (integer)

    Raises:
        ValueError: If text is None or model unsupported
    """
```

**Contract Test** (`tests/contract/test_token_counter_contract.py`):
```python
def test_estimate_tokens_accuracy():
    """Validate token estimation within ±10% of actual."""
    test_text = "This is a test document with multiple words."

    estimated = estimate_tokens(test_text)

    # Known token count for this text (from tiktoken)
    expected = 9  # Actual count for this text
    tolerance = expected * 0.1  # ±10%

    assert abs(estimated - expected) <= tolerance

def test_estimate_tokens_empty_string():
    """Validate empty string returns 0 tokens."""
    assert estimate_tokens("") == 0

def test_estimate_tokens_large_document():
    """Validate token estimation for large documents."""
    large_text = "word " * 5000  # ~5000 tokens

    estimated = estimate_tokens(large_text)

    assert 4500 <= estimated <= 5500  # ±10% tolerance
```

---

## Batch Queue API

### BatchQueue.add_document() / get_next_batch()

**Purpose**: Manage document queue with token-aware batching.

**Signature**:
```python
class BatchQueue:
    def add_document(self, document: Document, token_count: int) -> None:
        """
        Add document to batch queue.

        Args:
            document: Document to queue
            token_count: Pre-calculated token count for this document
        """

    def get_next_batch(self, token_budget: int = 8192) -> Optional[List[Document]]:
        """
        Get next batch of documents up to token budget.

        Args:
            token_budget: Maximum tokens for batch

        Returns:
            List of documents for next batch, or None if queue empty

        Behavior:
            - Fills batch up to token_budget
            - May reorder documents for optimal packing (FR-011)
            - Returns partial batch if remaining queue < token_budget
        """
```

**Contract Test** (`tests/contract/test_batch_queue_contract.py`):
```python
def test_batch_queue_respects_token_budget():
    """Validate get_next_batch() stays within token budget."""
    queue = BatchQueue()

    # Add documents with known token counts
    queue.add_document(Document(id="1", page_content="..."), 3000)
    queue.add_document(Document(id="2", page_content="..."), 3000)
    queue.add_document(Document(id="3", page_content="..."), 3000)

    batch = queue.get_next_batch(token_budget=8000)

    # Should return 2 documents (6000 tokens), not 3 (9000 tokens)
    assert len(batch) == 2

def test_batch_queue_empty_returns_none():
    """Validate empty queue returns None."""
    queue = BatchQueue()
    assert queue.get_next_batch() is None
```

---

## Metrics API

### BatchMetricsTracker.get_statistics()

**Purpose**: Expose batch processing statistics for monitoring (FR-007).

**Signature**:
```python
def get_statistics(self) -> ProcessingMetrics:
    """
    Get current batch processing statistics.

    Returns:
        ProcessingMetrics with current counters and rates

    Requirements:
        - FR-007: Expose total docs, batches, avg time, entities/batch, zero-entity count
    """
```

**Contract Test** (`tests/contract/test_batch_metrics_contract.py`):
```python
def test_get_statistics_returns_metrics():
    """Validate get_statistics() returns ProcessingMetrics."""
    tracker = BatchMetricsTracker()

    metrics = tracker.get_statistics()

    assert isinstance(metrics, ProcessingMetrics)
    assert hasattr(metrics, 'total_documents_processed')
    assert hasattr(metrics, 'total_batches_processed')
    assert hasattr(metrics, 'zero_entity_documents_count')

def test_get_statistics_required_fields():
    """Validate FR-007 required statistics fields."""
    tracker = BatchMetricsTracker()
    metrics = tracker.get_statistics()

    # FR-007 requirements
    required_fields = [
        'total_documents_processed',
        'total_batches_processed',  # batches created
        'average_batch_processing_time',
        'entity_extraction_rate_per_batch',
        'zero_entity_documents_count'
    ]

    for field in required_fields:
        assert hasattr(metrics, field), f"Missing required field: {field}"
```

---

## Contract Summary

**Total Contracts**: 4
- Batch extraction API (extract_batch)
- Token counting API (estimate_tokens)
- Batch queue API (add_document, get_next_batch)
- Metrics API (get_statistics)

**Total Contract Tests**: 10
- 3 for batch extraction
- 3 for token counting
- 2 for batch queue
- 2 for metrics

**Test Status**: All tests MUST fail initially (no implementation yet, per TDD principle)

**Next Steps**: Generate failing contract tests, then implement to make tests pass.

---
*API contracts defined - Ready for test generation*
