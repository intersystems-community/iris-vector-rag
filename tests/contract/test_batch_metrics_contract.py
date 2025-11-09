"""
Contract tests for batch metrics tracking.

These tests validate the BatchMetricsTracker and ProcessingMetrics API contracts.
Tests MUST fail initially, then pass after implementation (TDD).
"""

import pytest


class TestBatchMetricsContract:
    """Contract tests for batch processing metrics (FR-007)."""

    def test_processing_metrics_class_exists(self):
        """Validate ProcessingMetrics model exists."""
        from iris_vector_rag.core.models import ProcessingMetrics

        assert ProcessingMetrics is not None, "ProcessingMetrics class must exist"

    def test_processing_metrics_has_required_fields(self):
        """Validate ProcessingMetrics has all FR-007 required fields."""
        from iris_vector_rag.core.models import ProcessingMetrics
        import inspect

        sig = inspect.signature(ProcessingMetrics.__init__)
        params = sig.parameters

        # FR-007 required fields
        required_fields = [
            'total_batches_processed',
            'total_documents_processed',
            'average_batch_processing_time',
            'entity_extraction_rate_per_batch',
            'zero_entity_documents_count'
        ]

        for field in required_fields:
            assert field in params, \
                f"ProcessingMetrics must have '{field}' per FR-007"

    def test_processing_metrics_additional_fields(self):
        """Validate ProcessingMetrics has additional tracking fields."""
        from iris_vector_rag.core.models import ProcessingMetrics
        import inspect

        sig = inspect.signature(ProcessingMetrics.__init__)
        params = sig.parameters

        # Additional fields from data-model.md
        additional_fields = [
            'speedup_factor',
            'failed_batches_count',
            'retry_attempts_total'
        ]

        for field in additional_fields:
            assert field in params, \
                f"ProcessingMetrics should have '{field}' for comprehensive tracking"

    def test_processing_metrics_helper_methods(self):
        """Validate ProcessingMetrics has required helper methods."""
        from iris_vector_rag.core.models import ProcessingMetrics

        required_methods = [
            'update_with_batch',
            'calculate_speedup'
        ]

        for method in required_methods:
            assert hasattr(ProcessingMetrics, method), \
                f"ProcessingMetrics must have '{method}' method"

    def test_batch_metrics_tracker_class_exists(self):
        """Validate BatchMetricsTracker class exists."""
        from iris_vector_rag.common.batch_utils import BatchMetricsTracker

        assert BatchMetricsTracker is not None, "BatchMetricsTracker class must exist"

    def test_batch_metrics_tracker_get_statistics_method(self):
        """Validate get_statistics() method exists."""
        from iris_vector_rag.common.batch_utils import BatchMetricsTracker

        tracker = BatchMetricsTracker()
        assert hasattr(tracker, 'get_statistics'), \
            "BatchMetricsTracker must have get_statistics() method"

    def test_get_statistics_returns_processing_metrics(self):
        """Validate get_statistics() returns ProcessingMetrics instance."""
        from iris_vector_rag.common.batch_utils import BatchMetricsTracker
        from iris_vector_rag.core.models import ProcessingMetrics

        tracker = BatchMetricsTracker()
        metrics = tracker.get_statistics()

        assert isinstance(metrics, ProcessingMetrics), \
            "get_statistics() must return ProcessingMetrics instance"

    def test_metrics_update_with_batch(self):
        """Validate metrics update incrementally with batch results."""
        from iris_vector_rag.core.models import ProcessingMetrics, BatchExtractionResult

        metrics = ProcessingMetrics(
            total_batches_processed=0,
            total_documents_processed=0,
            average_batch_processing_time=0.0,
            speedup_factor=None,
            entity_extraction_rate_per_batch=0.0,
            zero_entity_documents_count=0,
            failed_batches_count=0,
            retry_attempts_total=0
        )

        # Create mock batch result
        batch_result = BatchExtractionResult(
            batch_id="test-batch",
            per_document_entities={"doc1": [], "doc2": []},
            per_document_relationships={},
            processing_time=1.5,
            success_status=True,
            retry_count=0,
            error_message=""
        )

        # Update metrics
        metrics.update_with_batch(batch_result, batch_size=2)

        # Validate updates
        assert metrics.total_batches_processed == 1, \
            "total_batches_processed should increment"
        assert metrics.total_documents_processed == 2, \
            "total_documents_processed should increment by batch_size"

    def test_metrics_calculate_speedup(self):
        """Validate calculate_speedup() computes correct speedup factor."""
        from iris_vector_rag.core.models import ProcessingMetrics

        metrics = ProcessingMetrics(
            total_batches_processed=10,
            total_documents_processed=100,
            average_batch_processing_time=1.0,  # 1 second per batch
            speedup_factor=None,
            entity_extraction_rate_per_batch=5.0,
            zero_entity_documents_count=0,
            failed_batches_count=0,
            retry_attempts_total=0
        )

        # Baseline: 3 seconds per document (single-doc processing)
        speedup = metrics.calculate_speedup(single_doc_baseline_time=3.0)

        # Expected speedup: 3.0 / (1.0 * 10 / 100) = 3.0 / 0.1 = 30x
        # (This is very high because batch processes 10 docs/batch)
        # More realistic: 3.0 / (10 * 1.0 / 100) = 3.0 / 0.1 = 30x
        assert speedup > 1.0, "Speedup should be positive"
        assert metrics.speedup_factor is not None, \
            "speedup_factor should be set after calculation"

    def test_entity_extraction_service_get_batch_metrics_method(self):
        """Validate EntityExtractionService.get_batch_metrics() exists."""
        from iris_vector_rag.services.entity_extraction import EntityExtractionService

        assert hasattr(EntityExtractionService, 'get_batch_metrics'), \
            "EntityExtractionService must have get_batch_metrics() method (FR-007)"

    def test_get_batch_metrics_returns_metrics(self):
        """Validate get_batch_metrics() returns ProcessingMetrics."""
        from iris_vector_rag.services.entity_extraction import EntityExtractionService
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.common.iris_connection_manager import IRISConnectionManager
        from iris_vector_rag.core.models import ProcessingMetrics

        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        service = EntityExtractionService(config_manager, connection_manager)

        metrics = service.get_batch_metrics()

        assert isinstance(metrics, ProcessingMetrics), \
            "get_batch_metrics() must return ProcessingMetrics instance"

    def test_metrics_track_zero_entity_documents(self):
        """Validate metrics correctly track zero-entity documents."""
        from iris_vector_rag.core.models import ProcessingMetrics, BatchExtractionResult

        metrics = ProcessingMetrics(
            total_batches_processed=0,
            total_documents_processed=0,
            average_batch_processing_time=0.0,
            speedup_factor=None,
            entity_extraction_rate_per_batch=0.0,
            zero_entity_documents_count=0,
            failed_batches_count=0,
            retry_attempts_total=0
        )

        # Batch with one zero-entity document
        batch_result = BatchExtractionResult(
            batch_id="test",
            per_document_entities={"doc1": [], "doc2": ["Entity1"]},  # doc1 has 0
            per_document_relationships={},
            processing_time=1.0,
            success_status=True,
            retry_count=0,
            error_message=""
        )

        metrics.update_with_batch(batch_result, batch_size=2)

        assert metrics.zero_entity_documents_count == 1, \
            "Should track documents with zero entities"

    def test_metrics_track_failed_batches(self):
        """Validate metrics track failed batches and retry attempts."""
        from iris_vector_rag.core.models import ProcessingMetrics, BatchExtractionResult

        metrics = ProcessingMetrics(
            total_batches_processed=0,
            total_documents_processed=0,
            average_batch_processing_time=0.0,
            speedup_factor=None,
            entity_extraction_rate_per_batch=0.0,
            zero_entity_documents_count=0,
            failed_batches_count=0,
            retry_attempts_total=0
        )

        # Failed batch with 2 retry attempts
        batch_result = BatchExtractionResult(
            batch_id="failed-batch",
            per_document_entities={},
            per_document_relationships={},
            processing_time=1.0,
            success_status=False,
            retry_count=2,
            error_message="LLM timeout"
        )

        metrics.update_with_batch(batch_result, batch_size=0)

        assert metrics.failed_batches_count == 1, "Should track failed batches"
        assert metrics.retry_attempts_total == 2, "Should track total retry attempts"
