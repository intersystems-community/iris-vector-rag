# Research: Batch LLM Entity Extraction

**Date**: 2025-10-15
**Feature**: 041-p1-batch-llm
**Status**: Complete

## Overview

This document consolidates research findings for implementing batch processing in the entity extraction pipeline to achieve 3x speedup (7.7 hours → 2.5 hours for 8,000+ documents).

## Research Areas

### 1. Token Counting for Batch Sizing

**Question**: How to accurately estimate LLM token usage for dynamic batch sizing?

**Options Evaluated**:
1. **tiktoken** (OpenAI's official tokenizer library)
2. **Hugging Face transformers tokenizers**
3. **Custom approximation** (character count / 4)

**Decision**: Use `tiktoken` library

**Rationale**:
- **Accuracy**: tiktoken provides exact token counts for OpenAI models (used via DSPy)
- **Performance**: Written in Rust, extremely fast (~1M tokens/sec)
- **Simplicity**: Single dependency, well-maintained by OpenAI
- **Model support**: Works with all models we use (GPT-3.5, GPT-4, qwen via approximation)
- **Token budget precision**: Critical for staying within 8K context window limit

**Alternatives Rejected**:
- HuggingFace tokenizers: Heavier dependency, slower for our use case
- Custom approximation: Too inaccurate (can be off by 30%), risks context overflow

**Implementation**:
```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

**References**:
- https://github.com/openai/tiktoken
- Production data: Average TrakCare ticket ~800 tokens, max ~3K tokens

---

### 2. Batch Queue Management

**Question**: What queue implementation pattern for dynamic batching with token budgets?

**Options Evaluated**:
1. **collections.deque** (built-in Python)
2. **queue.Queue** (thread-safe built-in)
3. **asyncio.Queue** (async-friendly)
4. **Custom priority queue** (sorted by document size)

**Decision**: Use `collections.deque` with custom token-aware batching logic

**Rationale**:
- **Simplicity**: Built-in, no external dependencies
- **Performance**: O(1) append/popleft operations
- **Thread safety not critical**: Entity extraction pipeline is single-threaded (workers process in parallel but don't share queue)
- **Token budget control**: Custom wrapper class can track cumulative tokens
- **Reordering acceptable**: Per clarification, documents can be reordered for optimal batching

**Alternatives Rejected**:
- queue.Queue: Thread-safety overhead unnecessary (pipeline is process-based parallelism, not thread-based)
- asyncio.Queue: Framework is not async-based, would require major refactoring
- Priority queue: Added complexity, minimal benefit (token budget matters more than size sorting)

**Implementation Sketch**:
```python
from collections import deque
from typing import List, Optional

class BatchQueue:
    def __init__(self, token_budget: int = 8192):
        self._queue = deque()
        self._token_budget = token_budget

    def add_document(self, document: Document, token_count: int) -> None:
        self._queue.append((document, token_count))

    def get_next_batch(self) -> Optional[List[Document]]:
        batch = []
        total_tokens = 0

        while self._queue:
            doc, tokens = self._queue[0]  # Peek
            if total_tokens + tokens > self._token_budget and batch:
                break  # Batch full, return what we have
            batch.append(doc)
            total_tokens += tokens
            self._queue.popleft()  # Remove from queue

        return batch if batch else None
```

**References**:
- Python collections.deque documentation
- Existing EntityExtractionService architecture

---

### 3. Exponential Backoff Implementation

**Question**: Library vs. custom retry logic for batch failures?

**Options Evaluated**:
1. **tenacity** library (full-featured retry decorator)
2. **backoff** library (simpler retry decorator)
3. **Custom implementation** (manual retry loop)

**Decision**: Custom implementation with simple exponential backoff

**Rationale**:
- **Simplicity**: Our retry logic is straightforward (3 attempts, 2s/4s/8s delays)
- **Batch splitting**: Custom requirement (split batch after max retries) not supported by libraries
- **No additional dependencies**: Constitution principle (minimize dependencies)
- **Clarity**: Explicit retry loop more readable than decorator magic for our specific use case
- **Control**: Need precise control over batch retry vs. individual document retry

**Alternatives Rejected**:
- tenacity: Too feature-rich for our needs (async retries, jitter, callbacks unnecessary)
- backoff: Simpler than tenacity but still external dependency for ~10 lines of code

**Implementation Sketch**:
```python
import time
from typing import List

RETRY_DELAYS = [2, 4, 8]  # Exponential backoff in seconds

def extract_batch_with_retry(documents: List[Document]) -> BatchExtractionResult:
    for attempt, delay in enumerate(RETRY_DELAYS + [None]):
        try:
            result = _extract_batch_impl(documents)
            if attempt > 0:
                logger.info(f"Batch succeeded on retry attempt {attempt + 1}")
            return result
        except LLMError as e:
            if delay is None:  # Last attempt failed
                logger.error(f"Batch failed after {len(RETRY_DELAYS)} retries, splitting batch")
                return _extract_batch_split(documents)  # Process individually
            logger.warning(f"Batch attempt {attempt + 1} failed: {e}, retrying in {delay}s")
            time.sleep(delay)
```

**References**:
- Production feedback: 60s/100-ticket batch overhead from connection churn (now fixed with connection pooling)
- Clarification: Retry entire batch 3 times, then split if still failing

---

### 4. DSPy Batch Module Integration

**Question**: What modifications needed for BatchEntityExtractionModule production use?

**Existing Implementation Review** (`iris_rag/dspy_modules/batch_entity_extraction.py`):

**Findings**:
1. **Already implemented**: BatchEntityExtractionModule exists with `forward()` method
2. **Input format**: Expects `List[Dict[str, str]]` with `id` and `text` keys
3. **Output format**: Returns `List[Dict[str, Any]]` with per-ticket entities/relationships
4. **JSON parsing**: Uses `json.loads()` - **CRITICAL**: Must integrate JSON retry logic from recent fix (0.7% failure rate)

**Decision**: Integrate existing module with minimal modifications

**Required Modifications**:
1. **JSON parsing robustness**: Replace `json.loads()` with `_parse_json_with_retry()` from entity_extraction.py (line 918-1000)
2. **Error handling**: Add explicit exception handling for batch failures (currently generic try/except)
3. **Logging**: Add batch-level statistics (token count, processing time, retry count)
4. **Configuration**: Make token budget configurable (currently hardcoded for TrakCare)

**No Modifications Needed**:
- Batch signature design (already optimal)
- DSPy ChainOfThought pattern (proven in production)
- Entity type extraction (generic, works for any domain)

**Implementation Notes**:
```python
# Integrate JSON retry logic from entity_extraction.py
from iris_rag.services.entity_extraction import _parse_json_with_retry

def forward(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # Existing DSPy call
    prediction = self.extract(tickets_batch=..., entity_types=...)

    # CHANGE: Use robust JSON parsing
    results = _parse_json_with_retry(
        prediction.batch_results,
        max_attempts=3,
        context="Batch entity extraction"
    )

    return results
```

**References**:
- Existing file: `iris_rag/dspy_modules/batch_entity_extraction.py`
- JSON retry fix: `iris_rag/services/entity_extraction.py:918-1000`
- Production metrics: 8,051 tickets, 4.86 entities/ticket

---

## Summary of Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Token Counting | tiktoken library | Accuracy + performance, official OpenAI tool |
| Batch Queue | collections.deque + custom logic | Simple, fast, no thread-safety overhead needed |
| Retry Logic | Custom exponential backoff | Simplicity, batch splitting requirement |
| DSPy Integration | Minimal modifications | Existing module solid, add JSON retry + config |

## Production Validation Plan

1. **Token Estimation Accuracy**: Test on 100 sample TrakCare tickets, validate ±10% accuracy
2. **Batch Queue Performance**: Benchmark queue operations (target: <1ms per add/get_next_batch)
3. **Retry Logic Correctness**: Unit tests for 3-attempt retry, batch splitting on failure
4. **End-to-End Speedup**: Integration test with 1,000 documents, validate 3x speedup

## Dependencies Added

- `tiktoken>=0.5.0` (new, for token counting)

## Dependencies NOT Added

- ~~tenacity~~ (rejected, custom retry logic instead)
- ~~backoff~~ (rejected, custom retry logic instead)

## Next Steps

Phase 1 design and contract generation:
1. Define data models (DocumentBatch, BatchExtractionResult, ProcessingMetrics)
2. Create API contracts (extract_batch, estimate_tokens, get_next_batch)
3. Generate contract tests (must fail initially, TDD principle)
4. Write quickstart guide for batch processing
5. Update CLAUDE.md with batch processing context

---
*Research complete - Ready for Phase 1 design*
