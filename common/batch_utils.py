"""
Batch processing utilities for entity extraction.

This module provides utilities for batching documents, retry logic with exponential
backoff, and batch processing metrics tracking.

Implementation:
- BatchQueue: FIFO queue with token-aware batching (FR-006)
- Retry logic: Exponential backoff with batch splitting (FR-005)
- BatchMetricsTracker: Global singleton for metrics tracking (FR-007)
"""

from collections import deque
from typing import List, Optional, Callable
import time
import logging

from iris_rag.core.models import (
    Document,
    BatchExtractionResult,
    ProcessingMetrics,
    BatchStatus,
)


logger = logging.getLogger(__name__)


class BatchQueue:
    """
    FIFO queue for batching documents with token budget enforcement.

    This class supports FR-006 (dynamic batch sizing based on token budget) and
    provides O(1) queue operations using collections.deque.

    Attributes:
        token_budget: Default token budget for batches (default: 8192)
        _queue: Internal deque storing (document, token_count) tuples
    """

    def __init__(self, token_budget: int = 8192):
        """
        Initialize batch queue.

        Args:
            token_budget: Default token budget for batches (default: 8192 per FR-006)
        """
        self.token_budget = token_budget
        self._queue: deque = deque()

    def add_document(self, document: Document, token_count: int) -> None:
        """
        Add a document to the queue.

        Args:
            document: Document to add
            token_count: Token count for the document

        Raises:
            ValueError: If token_count is negative
        """
        if token_count < 0:
            raise ValueError("Token count must be non-negative")

        self._queue.append((document, token_count))

    def get_next_batch(self, token_budget: Optional[int] = None) -> Optional[List[Document]]:
        """
        Get next batch of documents within token budget.

        This implements first-fit batching strategy: documents are added to the batch
        in FIFO order until adding the next document would exceed the token budget.

        Special cases (per FR-006):
        - If queue is empty, returns None
        - If first document exceeds budget, returns it anyway (can't skip)
        - If token_budget is 0 or negative, returns first document

        Args:
            token_budget: Token budget for this batch (overrides default if provided)

        Returns:
            List of documents for the batch, or None if queue is empty
        """
        if not self._queue:
            return None

        # Use provided budget or default
        budget = token_budget if token_budget is not None else self.token_budget

        batch = []
        cumulative_tokens = 0

        # Build batch until we exceed budget
        while self._queue:
            doc, tokens = self._queue[0]  # Peek at first document

            # First document always goes in batch (even if exceeds budget)
            if not batch:
                batch.append(doc)
                cumulative_tokens += tokens
                self._queue.popleft()  # Remove from queue
                continue

            # Check if adding this document would exceed budget
            if cumulative_tokens + tokens > budget:
                break  # Don't add this document

            # Add document to batch
            batch.append(doc)
            cumulative_tokens += tokens
            self._queue.popleft()  # Remove from queue

        return batch if batch else None


def extract_batch_with_retry(
    documents: List[Document],
    extract_fn: Callable[[List[Document]], BatchExtractionResult],
    max_retries: int = 3,
    retry_delays: List[float] = None,
) -> BatchExtractionResult:
    """
    Extract batch with exponential backoff retry logic (FR-005).

    This function implements:
    - Exponential backoff: 2s, 4s, 8s delays (configurable)
    - Batch splitting: If all retries fail, split batch in half and retry
    - Retry tracking: Tracks retry attempts in result

    Args:
        documents: List of documents to extract entities from
        extract_fn: Function to call for extraction (takes List[Document], returns BatchExtractionResult)
        max_retries: Maximum retry attempts (default: 3 per FR-005)
        retry_delays: Retry delays in seconds (default: [2, 4, 8] per FR-005)

    Returns:
        BatchExtractionResult with extraction results or error

    Examples:
        >>> def my_extractor(docs):
        ...     return BatchExtractionResult(batch_id="test", success_status=True)
        >>> result = extract_batch_with_retry(documents, my_extractor)
    """
    if retry_delays is None:
        retry_delays = [2.0, 4.0, 8.0]  # FR-005 exponential backoff

    retry_count = 0

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Attempt extraction
            result = extract_fn(documents)

            # If successful, return result with retry count
            if result.success_status:
                result.retry_count = retry_count
                return result

            # If extraction failed, retry
            logger.warning(
                f"Batch extraction failed (attempt {attempt + 1}/{max_retries + 1}): {result.error_message}"
            )

        except Exception as e:
            logger.error(
                f"Batch extraction exception (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )

        # If not last attempt, wait and retry
        if attempt < max_retries:
            retry_count += 1
            delay = retry_delays[min(attempt, len(retry_delays) - 1)]
            logger.info(f"Retrying batch extraction after {delay}s delay...")
            time.sleep(delay)
        else:
            # All retries exhausted
            logger.error(
                f"All {max_retries} retry attempts exhausted for batch of {len(documents)} documents"
            )

            # Split batch and retry if batch has multiple documents
            if len(documents) > 1:
                logger.info(
                    f"Splitting batch of {len(documents)} documents into 2 sub-batches"
                )
                mid = len(documents) // 2
                batch1 = documents[:mid]
                batch2 = documents[mid:]

                # Recursively retry each sub-batch (with retry_count tracking)
                result1 = extract_batch_with_retry(
                    batch1, extract_fn, max_retries, retry_delays
                )
                result2 = extract_batch_with_retry(
                    batch2, extract_fn, max_retries, retry_delays
                )

                # Merge results
                merged_result = BatchExtractionResult(
                    batch_id=result1.batch_id,  # Use first sub-batch ID
                    per_document_entities={
                        **result1.per_document_entities,
                        **result2.per_document_entities,
                    },
                    per_document_relationships={
                        **result1.per_document_relationships,
                        **result2.per_document_relationships,
                    },
                    processing_time=result1.processing_time + result2.processing_time,
                    success_status=result1.success_status and result2.success_status,
                    retry_count=retry_count + result1.retry_count + result2.retry_count,
                    error_message=None
                    if (result1.success_status and result2.success_status)
                    else f"Sub-batch errors: {result1.error_message}; {result2.error_message}",
                )

                return merged_result

            # Single document batch - return failure
            return BatchExtractionResult(
                batch_id="failed",
                success_status=False,
                retry_count=retry_count,
                error_message=f"Failed after {max_retries} retries (single document)",
            )


class BatchMetricsTracker:
    """
    Global singleton for tracking batch processing metrics (FR-007).

    This class provides a global singleton instance for tracking metrics across
    all batch processing operations. It implements FR-007 statistics requirements.

    Usage:
        >>> tracker = BatchMetricsTracker.get_instance()
        >>> tracker.update_with_batch(batch_result, batch_size=10)
        >>> metrics = tracker.get_statistics()
    """

    _instance: Optional["BatchMetricsTracker"] = None

    def __init__(self):
        """Initialize metrics tracker with default values."""
        self._metrics = ProcessingMetrics()

    @classmethod
    def get_instance(cls) -> "BatchMetricsTracker":
        """
        Get global singleton instance of BatchMetricsTracker.

        Returns:
            Global BatchMetricsTracker instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset global singleton instance (useful for testing).

        This creates a fresh instance with zeroed metrics.
        """
        cls._instance = None

    def update_with_batch(
        self,
        batch_result: BatchExtractionResult,
        batch_size: int,
        single_doc_baseline_time: float = 7.2,
    ) -> None:
        """
        Update metrics with a new batch result.

        Args:
            batch_result: Result from batch processing
            batch_size: Number of documents in the batch
            single_doc_baseline_time: Baseline time per document (default: 7.2s from spec)
        """
        self._metrics.update_with_batch(
            batch_result, batch_size, single_doc_baseline_time
        )

    def get_statistics(self) -> ProcessingMetrics:
        """
        Get current processing statistics (FR-007).

        Returns:
            ProcessingMetrics with current statistics
        """
        return self._metrics
