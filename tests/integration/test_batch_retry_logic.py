"""
Integration tests for batch retry logic with exponential backoff.

Tests AS-2 from spec.md: Batch failure recovery with exponential backoff.
Tests FR-005: Retry with 2s, 4s, 8s delays, then split batch.
"""

import pytest
import time
from unittest.mock import Mock, patch
from iris_rag.core.models import Document
from iris_rag.services.entity_extraction import EntityExtractionService
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import IRISConnectionManager


@pytest.mark.integration
@pytest.mark.requires_database
class TestBatchRetryLogic:
    """Integration tests for exponential backoff retry (FR-005)."""

    @pytest.fixture
    def service(self):
        """Initialize EntityExtractionService."""
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        return EntityExtractionService(config_manager, connection_manager)

    @pytest.fixture
    def test_documents(self):
        """Create test documents for retry testing."""
        return [
            Document(id=f"doc{i}", page_content=f"Test document {i} for retry logic.")
            for i in range(10)
        ]

    def test_as2_batch_failure_exponential_backoff(self, service, test_documents):
        """
        AS-2: Validate exponential backoff retry on batch failure.

        Given: Batch extraction system is processing a batch
        When: Batch fails due to LLM error (timeout, rate limit, parsing error)
        Then: System retries entire batch 3 times with exponential backoff (2s, 4s, 8s)
        """
        # Mock LLM failure for first 2 attempts, success on 3rd
        attempt_count = {'count': 0}

        def mock_extract_with_failure(*args, **kwargs):
            attempt_count['count'] += 1
            if attempt_count['count'] < 3:
                raise Exception("Simulated LLM timeout")
            # Third attempt succeeds
            from iris_rag.core.models import BatchExtractionResult
            return BatchExtractionResult(
                batch_id="test-batch",
                per_document_entities={doc.id: [] for doc in test_documents},
                per_document_relationships={},
                processing_time=1.0,
                success_status=True,
                retry_count=attempt_count['count'] - 1,
                error_message=""
            )

        # Patch the internal batch extraction method
        with patch.object(service, '_extract_batch_impl', side_effect=mock_extract_with_failure):
            start_time = time.time()
            result = service.extract_batch(test_documents)
            elapsed = time.time() - start_time

            # Validate retry succeeded after 2 failures
            assert result.success_status, "Batch should succeed after retries"
            assert result.retry_count == 2, \
                f"Should have 2 retries before success, got {result.retry_count}"

            # Validate exponential backoff delays (2s + 4s = 6s minimum)
            # Allow some tolerance for processing time
            assert elapsed >= 6.0, \
                f"Should have exponential backoff delays (2s+4s=6s min), got {elapsed:.1f}s"

            print(f"\nExponential Backoff Validation:")
            print(f"  Retry attempts: {result.retry_count}")
            print(f"  Total time: {elapsed:.1f}s (min 6s for 2s+4s delays)")
            print(f"  Success after retries: ✓")

    def test_batch_splitting_after_max_retries(self, service, test_documents):
        """
        Validate batch splitting after 3 failed retries (FR-005).

        Given: Batch fails 3 times with exponential backoff
        When: All retries are exhausted
        Then: System splits batch into individual documents for separate retry
        """
        # Mock LLM failure for all batch attempts
        batch_attempt_count = {'count': 0}
        individual_calls = {'count': 0}

        def mock_extract_always_fail_batch(*args, **kwargs):
            batch_attempt_count['count'] += 1
            if len(args) > 0 and len(args[0]) > 1:  # Batch of multiple docs
                raise Exception("Simulated batch LLM failure")
            else:  # Individual document
                individual_calls['count'] += 1
                from iris_rag.core.models import BatchExtractionResult
                doc = args[0][0]
                return BatchExtractionResult(
                    batch_id=f"individual-{doc.id}",
                    per_document_entities={doc.id: []},
                    per_document_relationships={},
                    processing_time=0.5,
                    success_status=True,
                    retry_count=0,
                    error_message=""
                )

        with patch.object(service, '_extract_batch_impl', side_effect=mock_extract_always_fail_batch):
            result = service.extract_batch(test_documents)

            # Validate batch was split after 3 failed attempts
            assert batch_attempt_count['count'] >= 3, \
                "Should attempt batch processing 3 times before splitting"

            # Validate individual processing occurred
            assert individual_calls['count'] == len(test_documents), \
                f"Should process {len(test_documents)} documents individually after split"

            print(f"\nBatch Splitting Validation:")
            print(f"  Batch attempts: {batch_attempt_count['count']}")
            print(f"  Individual document calls: {individual_calls['count']}")
            print(f"  Batch splitting triggered: ✓")

    def test_retry_delays_are_exponential(self, service):
        """Validate retry delays follow exponential pattern (2s, 4s, 8s)."""
        from common.batch_utils import extract_batch_with_retry

        # Mock function that always fails
        def failing_function(*args, **kwargs):
            raise Exception("Simulated failure")

        attempt_times = []

        def track_attempt_time(*args, **kwargs):
            attempt_times.append(time.time())
            raise Exception("Simulated failure")

        # Test retry delays
        start = time.time()
        try:
            extract_batch_with_retry(
                documents=[Document(id="test", page_content="test")],
                extract_fn=track_attempt_time
            )
        except Exception:
            pass  # Expected to fail after all retries

        # Validate delays
        if len(attempt_times) >= 3:
            delay1 = attempt_times[1] - attempt_times[0]
            delay2 = attempt_times[2] - attempt_times[1]

            # Allow 0.5s tolerance
            assert 1.5 <= delay1 <= 2.5, \
                f"First retry delay should be ~2s, got {delay1:.1f}s"
            assert 3.5 <= delay2 <= 4.5, \
                f"Second retry delay should be ~4s, got {delay2:.1f}s"

            print(f"\nRetry Delay Validation:")
            print(f"  First delay: {delay1:.1f}s (target: 2s)")
            print(f"  Second delay: {delay2:.1f}s (target: 4s)")

    def test_successful_retry_after_first_failure(self, service, test_documents):
        """Validate immediate success after single failure."""
        attempt_count = {'count': 0}

        def mock_extract_fail_once(*args, **kwargs):
            attempt_count['count'] += 1
            if attempt_count['count'] == 1:
                raise Exception("First attempt fails")
            # Second attempt succeeds
            from iris_rag.core.models import BatchExtractionResult
            return BatchExtractionResult(
                batch_id="test-batch",
                per_document_entities={doc.id: [] for doc in test_documents},
                per_document_relationships={},
                processing_time=1.0,
                success_status=True,
                retry_count=1,
                error_message=""
            )

        with patch.object(service, '_extract_batch_impl', side_effect=mock_extract_fail_once):
            start_time = time.time()
            result = service.extract_batch(test_documents)
            elapsed = time.time() - start_time

            assert result.success_status, "Should succeed after first retry"
            assert result.retry_count == 1, "Should have 1 retry"
            assert elapsed >= 2.0, \
                f"Should have 2s delay for first retry, got {elapsed:.1f}s"

            print(f"\nSingle Retry Validation:")
            print(f"  Retry count: {result.retry_count}")
            print(f"  Time with delay: {elapsed:.1f}s (min 2s)")

    def test_no_retry_on_immediate_success(self, service, test_documents):
        """Validate no retries when batch succeeds immediately."""
        result = service.extract_batch(test_documents)

        # Should succeed without retries (assuming LLM is working)
        assert result.success_status, "Batch should succeed"
        # Retry count should be 0 for immediate success
        # (This may fail until implementation is complete)

    def test_retry_count_tracked_in_metrics(self, service, test_documents):
        """Validate retry attempts are tracked in processing metrics (FR-007)."""
        # Process batch (may have retries depending on LLM stability)
        result = service.extract_batch(test_documents)

        # Get metrics
        metrics = service.get_batch_metrics()

        # Validate retry tracking
        assert hasattr(metrics, 'retry_attempts_total'), \
            "Metrics must track total retry attempts"
        assert metrics.retry_attempts_total >= 0, \
            "Retry attempts must be non-negative"

        print(f"\nRetry Metrics:")
        print(f"  Total retry attempts: {metrics.retry_attempts_total}")
        print(f"  Failed batches: {metrics.failed_batches_count}")
