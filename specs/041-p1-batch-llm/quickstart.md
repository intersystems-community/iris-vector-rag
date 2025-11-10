# Quickstart: Batch Entity Extraction

**Feature**: 041-p1-batch-llm
**Date**: 2025-10-15
**Estimated Reading Time**: 5 minutes

## Overview

This quickstart demonstrates how to use batch entity extraction to achieve 3x faster processing of large document collections (8,000+ documents) while maintaining extraction quality.

## Prerequisites

- Python 3.11+
- `rag-templates` framework installed
- IRIS database running (localhost:21972 or configured port)
- tiktoken library installed: `pip install tiktoken>=0.5.0`
- Ollama running with qwen2.5:7b model (or OpenAI API key configured)

## Quick Example

### 1. Basic Batch Processing

```python
from iris_rag.services.entity_extraction import EntityExtractionService
from iris_rag.core.models import Document
from iris_rag.config.manager import ConfigurationManager

# Initialize service with batch processing enabled
config_manager = ConfigurationManager()
config_manager.update({
    "entity_extraction": {
        "batch_processing": {
            "enabled": True,
            "token_budget": 8192,  # Default, can be adjusted
            "max_retries": 3,      # Exponential backoff retries
        }
    }
})

service = EntityExtractionService(config_manager, connection_manager)

# Prepare documents (example: 1,000 support tickets)
documents = [
    Document(id=f"ticket-{i}", page_content=load_ticket_text(i))
    for i in range(1000)
]

# Process in batches automatically
results = service.extract_batch(documents)

# Verify 3x speedup
print(f"Processed {len(documents)} documents in {results.processing_time:.1f}s")
print(f"Average entities per document: {len(results.get_all_entities()) / len(documents):.2f}")
print(f"Speedup factor: {metrics.speedup_factor:.2f}x")
```

**Expected Output**:
```
Processed 1000 documents in 180.5s
Average entities per document: 4.86
Speedup factor: 3.12x
```

---

### 2. Monitor Batch Processing Statistics

```python
# Get batch processing metrics (FR-007)
metrics = service.get_batch_metrics()

print(f"Total batches: {metrics.total_batches_processed}")
print(f"Total documents: {metrics.total_documents_processed}")
print(f"Average batch time: {metrics.average_batch_processing_time:.2f}s")
print(f"Zero-entity documents: {metrics.zero_entity_documents_count}")
print(f"Failed batches: {metrics.failed_batches_count}")
print(f"Retry attempts: {metrics.retry_attempts_total}")
```

**Expected Output**:
```
Total batches: 125
Total documents: 1000
Average batch time: 1.44s
Zero-entity documents: 7
Failed batches: 2
Retry attempts: 5
```

---

### 3. Handle Variable Document Sizes

```python
# Documents with varying sizes (100 words to 10,000 words)
small_doc = Document(id="small", page_content="Short text.")
large_doc = Document(id="large", page_content="..." * 5000)

# Batch queue automatically adjusts batch size based on token count
from common.batch_utils import BatchQueue

queue = BatchQueue(token_budget=8192)
queue.add_document(small_doc, token_count=10)
queue.add_document(large_doc, token_count=7500)

# Next batch will contain both (under budget)
batch = queue.get_next_batch()
assert len(batch) == 2  # Both documents fit in 8K token budget

# Add another large document
queue.add_document(Document(id="large2", page_content="..." * 5000), token_count=7500)

# Next batch will contain only one large document
batch = queue.get_next_batch()
assert len(batch) == 1  # Only one large doc fits
```

---

### 4. Verify Entity Traceability (FR-004)

```python
# Extract entities from batch
results = service.extract_batch(documents)

# Verify every entity links to source document
for doc_id, entities in results.per_document_entities.items():
    for entity in entities:
        assert entity.source_document_id == doc_id
        print(f"Entity '{entity.text}' from document {doc_id}")
```

**Expected Output**:
```
Entity 'TrakCare' from document ticket-0
Entity 'Database Error' from document ticket-0
Entity 'User Login' from document ticket-1
...
```

---

### 5. Test Retry Logic (FR-005)

```python
import logging
logging.basicConfig(level=logging.INFO)

# Simulate LLM failure (for testing)
with mock.patch('dspy.ChainOfThought.forward', side_effect=LLMError("Rate limit")):
    results = service.extract_batch(documents)

# Check logs for retry attempts
"""
Expected log output:
INFO: Batch attempt 1 failed: Rate limit, retrying in 2s
INFO: Batch attempt 2 failed: Rate limit, retrying in 4s
INFO: Batch attempt 3 failed: Rate limit, retrying in 8s
INFO: Batch failed after 3 retries, splitting batch
INFO: Processing 10 documents individually
"""

# Verify batch was split and processed individually
assert results.success_status == True
assert results.retry_count == 3
```

---

## Configuration Options

### memory_config.yaml

```yaml
entity_extraction:
  batch_processing:
    enabled: true              # Enable batch processing
    token_budget: 8192         # Max tokens per batch (configurable)
    max_retries: 3             # Max retry attempts before splitting
    retry_delays: [2, 4, 8]    # Exponential backoff delays (seconds)

  llm:
    model: "qwen2.5:7b"        # LLM model for extraction
    use_dspy: true             # Use DSPy for structured extraction
```

---

## Performance Validation

### Test 1: 1,000 Documents (Integration Test)

```bash
# Run integration test
pytest tests/integration/test_batch_performance.py::test_1k_documents_speedup -v

# Expected output:
# ✓ test_1k_documents_speedup PASSED
# Processing time: 180.5s (vs 562.3s single-doc baseline)
# Speedup: 3.12x ✓ (target: 3.0x)
# Quality: 4.87 entities/doc ✓ (target: 4.86)
```

### Test 2: 10,000 Documents (Performance Test)

```bash
# Run performance test (requires IRIS database)
pytest tests/integration/test_batch_performance.py::test_10k_documents_speedup -v

# Expected output:
# ✓ test_10k_documents_speedup PASSED
# Processing time: 1,810s (30.2 min vs 7.7 hours single-doc)
# Speedup: 3.05x ✓
# Quality maintained: 4.85 entities/doc ✓
```

---

## Troubleshooting

### Issue: Batch processing slower than expected

**Symptoms**: Speedup < 3.0x

**Solutions**:
1. Check token budget: `metrics.average_batch_processing_time`
   - If batches are too small → increase token_budget
   - If batches too large → LLM may be rate-limited

2. Check retry count: `metrics.retry_attempts_total`
   - High retries → LLM instability, consider different model

3. Verify connection pooling enabled (see previous P0 fix)

### Issue: Entity extraction quality degraded

**Symptoms**: `metrics.entity_extraction_rate_per_batch < 4.86`

**Solutions**:
1. Check batch size: Batches > 10 docs may confuse LLM
2. Verify JSON parsing retry logic active (see P1 fix)
3. Compare batch vs. single-doc results:
   ```python
   pytest tests/contract/test_batch_extraction_contract.py::test_batch_equals_single_doc
   ```

### Issue: High zero-entity document count

**Symptoms**: `metrics.zero_entity_documents_count > 10%`

**Solutions**:
1. Review zero-entity documents: `results.get_entity_count_by_document()`
2. Check if documents are valid (not corrupted, not empty)
3. Verify LLM prompt includes all entity types

---

## Next Steps

1. **Run contract tests**: `pytest tests/contract/test_batch_*.py`
2. **Run integration tests**: `pytest tests/integration/test_batch_*.py`
3. **Performance baseline**: Establish single-doc processing time for your data
4. **Production deployment**: Enable batch processing in production config

---

## Success Criteria

- ✓ 3x speedup vs. single-document baseline (FR-002)
- ✓ 4.86 entities/document average maintained (FR-003)
- ✓ Entity traceability preserved (FR-004)
- ✓ Batch failure retry with exponential backoff (FR-005)
- ✓ Dynamic batch sizing respects token budget (FR-006)
- ✓ Processing statistics available (FR-007)

---

*Quickstart complete - Ready for testing and implementation*
