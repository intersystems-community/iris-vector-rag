# Data Model: Batch LLM Entity Extraction

**Date**: 2025-10-15
**Feature**: 041-p1-batch-llm
**Status**: Design Complete

## Overview

This document defines the data models required for batch processing integration into the entity extraction pipeline. All models extend or integrate with existing `iris_rag/core/models.py` patterns.

## Core Models

### 1. DocumentBatch

**Purpose**: Represents a collection of documents grouped for simultaneous LLM processing.

**Location**: `iris_rag/core/models.py` (add to existing models)

**Schema**:
```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List
from uuid import UUID, uuid4

class BatchStatus(Enum):
    PENDING = "pending"           # Batch queued, not yet processed
    PROCESSING = "processing"     # Currently being processed by LLM
    COMPLETED = "completed"       # Successfully processed
    FAILED = "failed"             # Failed after all retries
    SPLIT = "split"               # Split into individual documents after batch failure

@dataclass
class DocumentBatch:
    """
    A dynamically-sized batch of documents for simultaneous LLM processing.

    Attributes:
        batch_id: Unique identifier for this batch
        document_ids: List of document IDs in this batch
        batch_size: Number of documents in batch (variable 1-10+)
        total_token_count: Sum of estimated tokens for all documents
        creation_timestamp: When batch was created
        processing_status: Current status of batch processing
        retry_count: Number of retry attempts (max 3)
    """
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    document_ids: List[str] = field(default_factory=list)
    batch_size: int = 0
    total_token_count: int = 0
    creation_timestamp: datetime = field(default_factory=datetime.now)
    processing_status: BatchStatus = BatchStatus.PENDING
    retry_count: int = 0

    def add_document(self, document_id: str, token_count: int) -> None:
        """Add a document to this batch."""
        self.document_ids.append(document_id)
        self.batch_size += 1
        self.total_token_count += token_count

    def is_within_budget(self, token_budget: int = 8192) -> bool:
        """Check if batch is within token budget."""
        return self.total_token_count <= token_budget
```

**Validation Rules** (from FR-006):
- `total_token_count <= 8192` (default token budget, configurable)
- `batch_size >= 1` (at least one document)
- `retry_count <= 3` (max retries before batch splitting)

**Relationships**:
- Contains: `List[Document]` (via document_ids, not embedded)
- Produces: `BatchExtractionResult` (1:1 relationship)

---

### 2. BatchExtractionResult

**Purpose**: Output from processing a document batch, mapping entities/relationships back to source documents.

**Location**: `iris_rag/core/models.py`

**Schema**:
```python
from typing import Dict, List

@dataclass
class BatchExtractionResult:
    """
    Result of batch entity extraction with per-document breakout.

    Attributes:
        batch_id: Reference to source DocumentBatch
        per_document_entities: Map of document_id -> List[Entity]
        per_document_relationships: Map of document_id -> List[Relationship]
        processing_time: Total time to process batch (seconds)
        success_status: Whether batch processing succeeded
        retry_count: Number of retries before success/failure
        error_message: Error details if success_status=False
    """
    batch_id: str
    per_document_entities: Dict[str, List[Entity]] = field(default_factory=dict)
    per_document_relationships: Dict[str, List[Relationship]] = field(default_factory=dict)
    processing_time: float = 0.0
    success_status: bool = True
    retry_count: int = 0
    error_message: str = ""

    def get_all_entities(self) -> List[Entity]:
        """Flatten all entities across all documents."""
        return [entity for entities in self.per_document_entities.values() for entity in entities]

    def get_all_relationships(self) -> List[Relationship]:
        """Flatten all relationships across all documents."""
        return [rel for rels in self.per_document_relationships.values() for rel in rels]

    def get_entity_count_by_document(self) -> Dict[str, int]:
        """Get entity counts per document (for FR-007 statistics)."""
        return {doc_id: len(entities) for doc_id, entities in self.per_document_entities.items()}
```

**Validation Rules** (from FR-003, FR-004):
- `len(per_document_entities) == batch_size` (one entry per document, even if empty)
- Each `Entity` in results must have `source_document_id` set (FR-004 traceability)
- Average entities per document should be ~4.86 (FR-003 quality maintenance)

**Relationships**:
- References: `DocumentBatch` (via batch_id)
- Contains: Embedded `List[Entity]` and `List[Relationship]` per document

---

### 3. ProcessingMetrics

**Purpose**: Aggregate statistics for monitoring batch processing performance and quality.

**Location**: `iris_rag/core/models.py`

**Schema**:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingMetrics:
    """
    Batch processing performance and quality metrics.

    Tracks:
    - Throughput: batches/documents processed, processing time
    - Quality: entity extraction rate, zero-entity documents
    - Performance: speedup vs. single-document baseline

    Updated incrementally as batches are processed.
    """
    total_batches_processed: int = 0
    total_documents_processed: int = 0
    average_batch_processing_time: float = 0.0
    speedup_factor: Optional[float] = None  # Calculated vs. baseline
    entity_extraction_rate_per_batch: float = 0.0
    zero_entity_documents_count: int = 0
    failed_batches_count: int = 0
    retry_attempts_total: int = 0

    def update_with_batch(self, result: BatchExtractionResult, batch_size: int) -> None:
        """Incrementally update metrics with new batch result."""
        self.total_batches_processed += 1
        self.total_documents_processed += batch_size

        # Update average processing time (rolling average)
        n = self.total_batches_processed
        self.average_batch_processing_time = (
            (self.average_batch_processing_time * (n - 1) + result.processing_time) / n
        )

        # Update entity extraction rate
        total_entities = len(result.get_all_entities())
        self.entity_extraction_rate_per_batch = (
            (self.entity_extraction_rate_per_batch * (n - 1) + total_entities) / n
        )

        # Track zero-entity documents
        for count in result.get_entity_count_by_document().values():
            if count == 0:
                self.zero_entity_documents_count += 1

        # Track failures and retries
        if not result.success_status:
            self.failed_batches_count += 1
        self.retry_attempts_total += result.retry_count

    def calculate_speedup(self, single_doc_baseline_time: float) -> float:
        """
        Calculate speedup factor vs. single-document processing.

        Args:
            single_doc_baseline_time: Average time to process one document individually

        Returns:
            Speedup factor (e.g., 3.0 = 3x faster)
        """
        if self.total_documents_processed == 0:
            return 1.0

        avg_time_per_doc = (
            self.average_batch_processing_time * self.total_batches_processed
        ) / self.total_documents_processed

        self.speedup_factor = single_doc_baseline_time / avg_time_per_doc
        return self.speedup_factor
```

**Validation Rules** (from FR-002, FR-007):
- `speedup_factor >= 3.0` (target performance, FR-002)
- `entity_extraction_rate_per_batch >= 4.86` (quality maintenance, FR-003)
- Metrics must be exposed via API for monitoring (FR-007)

**Usage Pattern**:
```python
# Global metrics tracker (singleton pattern)
metrics = ProcessingMetrics()

# After each batch
result = extract_batch(documents)
metrics.update_with_batch(result, len(documents))

# Periodically calculate speedup
speedup = metrics.calculate_speedup(single_doc_baseline_time=7.2)  # seconds
assert speedup >= 3.0, "Performance target not met"
```

---

## Model Integration

### Existing Models (No Changes)

These existing models from `iris_rag/core/models.py` are reused as-is:

1. **Document**: Source documents for extraction (unchanged)
2. **Entity**: Extracted entities with traceability via `source_document_id` (unchanged)
3. **Relationship**: Entity relationships (unchanged)

### Modified Models

**EntityExtractionService** (`iris_rag/services/entity_extraction.py`):

Add methods:
```python
def extract_batch(self, documents: List[Document]) -> BatchExtractionResult:
    """
    Process multiple documents in single batch (new method).

    Implements FR-001, FR-005, FR-006.
    """
    pass  # To be implemented

def get_batch_metrics(self) -> ProcessingMetrics:
    """
    Get current batch processing statistics (new method).

    Implements FR-007.
    """
    pass  # To be implemented
```

---

## State Transitions

### DocumentBatch Status Flow

```
PENDING → PROCESSING → COMPLETED
         ↓
       (retry on failure)
         ↓
       PROCESSING (retry_count+=1)
         ↓
       (max retries reached)
         ↓
       SPLIT (fallback to individual processing)
```

**Transition Rules**:
1. `PENDING → PROCESSING`: When batch is picked up for LLM processing
2. `PROCESSING → COMPLETED`: When batch succeeds (with or without retries)
3. `PROCESSING → PROCESSING`: On retry (retry_count < 3)
4. `PROCESSING → SPLIT`: After 3rd failed retry (per FR-005 clarification)
5. `PROCESSING → FAILED`: Only for unrecoverable errors (should be rare)

---

## Persistence

### Database Storage

**Not required initially**: Batch metadata is ephemeral (exists only during processing).

**Future enhancement** (out of scope for this feature):
- Store `ProcessingMetrics` in IRIS for historical trending
- Store failed `DocumentBatch` records for debugging

### In-Memory Storage

Batch queue and metrics maintained in memory during pipeline execution:
```python
# Singleton instances
_batch_queue: Optional[BatchQueue] = None
_batch_metrics: Optional[ProcessingMetrics] = None
```

---

## Validation Contracts

### Data Integrity

1. **Entity traceability** (FR-004):
   ```python
   for entity in result.get_all_entities():
       assert entity.source_document_id in batch.document_ids
   ```

2. **Quality maintenance** (FR-003):
   ```python
   avg_entities = metrics.entity_extraction_rate_per_batch / metrics.total_batches_processed
   assert avg_entities >= 4.86, "Quality degradation detected"
   ```

3. **Token budget enforcement** (FR-006):
   ```python
   for batch in processed_batches:
       assert batch.total_token_count <= 8192, "Token budget exceeded"
   ```

---

## Summary

**New Models**: 3 (DocumentBatch, BatchExtractionResult, ProcessingMetrics)
**Modified Models**: 1 (EntityExtractionService - add 2 methods)
**Reused Models**: 3 (Document, Entity, Relationship)
**Storage**: In-memory (ephemeral during processing)
**Validation**: 3 contract tests covering traceability, quality, token budget

---
*Data model design complete - Ready for contract generation (Phase 1 continues)*
