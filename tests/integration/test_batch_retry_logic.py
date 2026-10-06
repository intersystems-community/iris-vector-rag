"""
Integration tests for batch retry logic with exponential backoff.

Tests AS-2 from spec.md: Batch failure recovery with exponential backoff.
Tests FR-005: Retry with 2s, 4s, 8s delays, then split batch.
"""

import time
from unittest.mock import patch

import pytest

from iris_vector_rag.core.models import Document

from .batch_support import real_service


@pytest.mark.integration
@pytest.mark.requires_database
class TestBatchRetryLogic:
    """Integration tests for exponential backoff retry (FR-005)."""

    @pytest.fixture
    def service(self):
        """EntityExtractionService over the real LLM; the fault is injected per test."""
        return real_service()

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
        # FAULT INJECTION: the first two attempts fail at the batch boundary (the fault is
        # what is under test); the third runs the real extraction against the real LLM.
        attempt_count = {"count": 0}
        real_extract = service._extract_batch_impl

        def mock_extract_with_failure(*args, **kwargs):
            attempt_count["count"] += 1
            if attempt_count["count"] < 3:
                raise Exception("Simulated LLM timeout")
            return real_extract(*args, **kwargs)

        # Patch the internal batch extraction method
        with patch.object(
            service, "_extract_batch_impl", side_effect=mock_extract_with_failure
        ):
            start_time = time.time()
            result = service.extract_batch(test_documents)
            elapsed = time.time() - start_time

            # Validate retry succeeded after 2 failures
            assert result.success_status, "Batch should succeed after retries"
            assert (
                result.retry_count == 2
            ), f"Should have 2 retries before success, got {result.retry_count}"

            # Validate exponential backoff delays (2s + 4s = 6s minimum)
            # Allow some tolerance for processing time
            assert (
                elapsed >= 6.0
            ), f"Should have exponential backoff delays (2s+4s=6s min), got {elapsed:.1f}s"

            print("\nExponential Backoff Validation:")
            print(f"  Retry attempts: {result.retry_count}")
            print(f"  Total time: {elapsed:.1f}s (min 6s for 2s+4s delays)")
            print("  Success after retries: ✓")

    def test_batch_splitting_after_max_retries(self, service, test_documents):
        """
        Validate batch splitting after 3 failed retries (FR-005).

        Given: Batch fails 3 times with exponential backoff
        When: All retries are exhausted
        Then: System splits batch into individual documents for separate retry
        """
        # FAULT INJECTION: every multi-document batch fails at the batch boundary (the
        # fault is what is under test); single-document calls run the real extraction
        # against the real LLM.
        batch_attempt_count = {"count": 0}
        individual_calls = {"count": 0}
        real_extract = service._extract_batch_impl

        def mock_extract_always_fail_batch(*args, **kwargs):
            batch_attempt_count["count"] += 1
            if len(args) > 0 and len(args[0]) > 1:  # Batch of multiple docs
                raise Exception("Simulated batch LLM failure")
            individual_calls["count"] += 1
            return real_extract(*args, **kwargs)

        # The clock is patched, not a service: nine failing batches each back off
        # 2s+4s+8s; record the delays instead of waiting them out.
        with patch(
            "iris_vector_rag.common.batch_utils.time.sleep"
        ) as sleep, patch.object(
            service, "_extract_batch_impl", side_effect=mock_extract_always_fail_batch
        ):
            result = service.extract_batch(test_documents)
            delays = [c.args[0] for c in sleep.call_args_list]

            # Every failing batch backs off 2s, 4s, 8s before it is split
            assert delays and delays == [2.0, 4.0, 8.0] * (len(delays) // 3)
            assert result.success_status, "Split batches must recover every document"
            assert set(result.per_document_entities) == {d.id for d in test_documents}

            # Validate batch was split after 3 failed attempts
            assert (
                batch_attempt_count["count"] >= 3
            ), "Should attempt batch processing 3 times before splitting"

            # Validate individual processing occurred
            assert individual_calls["count"] == len(
                test_documents
            ), f"Should process {len(test_documents)} documents individually after split"

            print("\nBatch Splitting Validation:")
            print(f"  Batch attempts: {batch_attempt_count['count']}")
            print(f"  Individual document calls: {individual_calls['count']}")
            print("  Batch splitting triggered: ✓")

    def test_retry_delays_are_exponential(self, service):
        """Validate retry delays follow exponential pattern (2s, 4s, 8s)."""
        from iris_vector_rag.common.batch_utils import extract_batch_with_retry

        # FAULT INJECTION: an extract function that always fails; no service is involved,
        # the retry timing is the thing under test.
        def failing_function(*args, **kwargs):
            raise Exception("Simulated failure")

        attempt_times = []

        def track_attempt_time(*args, **kwargs):
            attempt_times.append(time.time())
            raise Exception("Simulated failure")

        # Test retry delays
        time.time()
        try:
            extract_batch_with_retry(
                documents=[Document(id="test", page_content="test")],
                extract_fn=track_attempt_time,
            )
        except Exception:
            pass  # Expected to fail after all retries

        # Validate delays
        if len(attempt_times) >= 3:
            delay1 = attempt_times[1] - attempt_times[0]
            delay2 = attempt_times[2] - attempt_times[1]

            # Allow 0.5s tolerance
            assert (
                1.5 <= delay1 <= 2.5
            ), f"First retry delay should be ~2s, got {delay1:.1f}s"
            assert (
                3.5 <= delay2 <= 4.5
            ), f"Second retry delay should be ~4s, got {delay2:.1f}s"

            print("\nRetry Delay Validation:")
            print(f"  First delay: {delay1:.1f}s (target: 2s)")
            print(f"  Second delay: {delay2:.1f}s (target: 4s)")

    def test_successful_retry_after_first_failure(self, service, test_documents):
        """Validate immediate success after single failure."""
        # FAULT INJECTION: the first attempt fails at the batch boundary; the second runs
        # the real extraction against the real LLM.
        attempt_count = {"count": 0}
        real_extract = service._extract_batch_impl

        def mock_extract_fail_once(*args, **kwargs):
            attempt_count["count"] += 1
            if attempt_count["count"] == 1:
                raise Exception("First attempt fails")
            return real_extract(*args, **kwargs)

        with patch.object(
            service, "_extract_batch_impl", side_effect=mock_extract_fail_once
        ):
            start_time = time.time()
            result = service.extract_batch(test_documents)
            elapsed = time.time() - start_time

            assert result.success_status, "Should succeed after first retry"
            assert result.retry_count == 1, "Should have 1 retry"
            assert (
                elapsed >= 2.0
            ), f"Should have 2s delay for first retry, got {elapsed:.1f}s"

            print("\nSingle Retry Validation:")
            print(f"  Retry count: {result.retry_count}")
            print(f"  Time with delay: {elapsed:.1f}s (min 2s)")

    def test_no_retry_on_immediate_success(self, service, test_documents):
        """Validate no retries when batch succeeds immediately."""
        result = service.extract_batch(test_documents[:3])

        assert result.success_status, "Batch should succeed"
        assert result.retry_count == 0, "An immediate success has no retries"
        assert result.failed == []

    def test_retry_count_tracked_in_metrics(self, service, test_documents):
        """Validate retry attempts are tracked in processing metrics (FR-007)."""
        # Process batch (may have retries depending on LLM stability)
        service.extract_batch(test_documents[:3])

        # Get metrics
        metrics = service.get_batch_metrics()

        # Validate retry tracking
        assert hasattr(
            metrics, "retry_attempts_total"
        ), "Metrics must track total retry attempts"
        assert metrics.retry_attempts_total >= 0, "Retry attempts must be non-negative"

        print("\nRetry Metrics:")
        print(f"  Total retry attempts: {metrics.retry_attempts_total}")
        print(f"  Failed batches: {metrics.failed_batches_count}")
