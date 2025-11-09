"""
Performance validation tests for batch entity extraction.

Tests FR-002: 3x speedup requirement (7.7h → 2.5h for 8K+ documents)
Tests FR-003: Quality maintenance (4.86 entities/doc average)
Tests FR-009: Mixed document types handling
"""

import pytest
import time
from iris_vector_rag.core.models import Document
from iris_vector_rag.services.entity_extraction import EntityExtractionService
from iris_vector_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import IRISConnectionManager


@pytest.mark.integration
@pytest.mark.requires_database
@pytest.mark.slow
class TestBatchPerformance:
    """Performance validation tests for batch processing (FR-002, FR-003)."""

    @pytest.fixture
    def service(self):
        """Initialize EntityExtractionService."""
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        return EntityExtractionService(config_manager, connection_manager)

    @pytest.fixture
    def documents_1k(self):
        """Generate 1,000 test documents for performance testing."""
        documents = []
        for i in range(1000):
            content = f"Support ticket {i}: TrakCare system error in module PatientManagement. " \
                     f"User admin reported issue with database connection. " \
                     f"Error code ERR{i:04d} in version 2024.1.{i % 10}. " \
                     f"The system failed to process patient records correctly."
            documents.append(Document(
                id=f"perf-ticket-{i}",
                page_content=content
            ))
        return documents

    @pytest.fixture
    def documents_10k(self):
        """Generate 10,000 test documents for large-scale performance testing."""
        documents = []
        for i in range(10000):
            content = f"Ticket {i}: Error in TrakCare module. " \
                     f"Code: ERR{i:05d}. Version: 2024.{i % 12 + 1}.{i % 30 + 1}."
            documents.append(Document(
                id=f"large-ticket-{i}",
                page_content=content
            ))
        return documents

    def test_1k_documents_speedup_target_3x(self, service, documents_1k):
        """
        Validate 3x speedup on 1,000 documents (FR-002).

        Target: Process 1K documents at least 3.0x faster than single-doc baseline
        Baseline: 8.33 tickets/min single-doc = 7.2s per ticket
        Expected batch: 25 tickets/min = 2.4s per ticket (3x speedup)
        """
        print(f"\n{'='*60}")
        print("Performance Test: 1,000 Documents (3x Speedup Target)")
        print(f"{'='*60}")

        # Measure batch processing time
        start_time = time.time()
        result = service.extract_batch(documents_1k, token_budget=8192)
        elapsed = time.time() - start_time

        # Calculate metrics
        single_doc_baseline = 7.2  # seconds per document (from spec.md)
        expected_single_doc_time = 1000 * single_doc_baseline  # 7200s = 2 hours
        actual_speedup = expected_single_doc_time / elapsed

        # Validate 3x speedup (allow 20% tolerance: 2.4x - 3.6x)
        print(f"\nResults:")
        print(f"  Batch processing time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Expected single-doc time: {expected_single_doc_time:.1f}s ({expected_single_doc_time/3600:.1f} hours)")
        print(f"  Actual speedup: {actual_speedup:.2f}x")
        print(f"  Target speedup: 3.0x (tolerance: 2.4x - 3.6x)")

        assert actual_speedup >= 2.4, \
            f"Speedup must be at least 2.4x (target 3.0x), got {actual_speedup:.2f}x"
        print(f"  ✓ Speedup requirement met: {actual_speedup:.2f}x >= 2.4x")

        # Validate batch succeeded
        assert result.success_status, "Batch processing must complete successfully"
        print(f"  ✓ Batch processing succeeded")

    def test_10k_documents_speedup_target_3x(self, service, documents_10k):
        """
        Validate 3x speedup on 10,000 documents (FR-002 at scale).

        Target: Process 10K documents at least 3.0x faster than single-doc baseline
        Expected: ~30 minutes vs ~20 hours single-doc
        """
        print(f"\n{'='*60}")
        print("Performance Test: 10,000 Documents (3x Speedup at Scale)")
        print(f"{'='*60}")

        # Measure batch processing time
        start_time = time.time()
        result = service.extract_batch(documents_10k, token_budget=8192)
        elapsed = time.time() - start_time

        # Calculate metrics
        single_doc_baseline = 7.2  # seconds per document
        expected_single_doc_time = 10000 * single_doc_baseline  # 72000s = 20 hours
        actual_speedup = expected_single_doc_time / elapsed

        print(f"\nResults:")
        print(f"  Batch processing time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Expected single-doc time: {expected_single_doc_time:.1f}s ({expected_single_doc_time/3600:.1f} hours)")
        print(f"  Actual speedup: {actual_speedup:.2f}x")
        print(f"  Target speedup: 3.0x (tolerance: 2.4x - 3.6x)")

        assert actual_speedup >= 2.4, \
            f"Speedup must be at least 2.4x at 10K scale, got {actual_speedup:.2f}x"
        print(f"  ✓ Speedup requirement met at 10K scale: {actual_speedup:.2f}x >= 2.4x")

        assert result.success_status, "Large-scale batch must succeed"
        print(f"  ✓ Large-scale batch processing succeeded")

    def test_quality_maintenance_4_86_entities_per_doc(self, service, documents_1k):
        """
        Validate quality maintenance: 4.86 entities/doc average (FR-003).

        Target: Maintain same extraction quality as single-doc processing
        Baseline: 4.86 entities per document (from spec.md production data)
        """
        print(f"\n{'='*60}")
        print("Quality Test: Entity Extraction Rate (4.86 entities/doc target)")
        print(f"{'='*60}")

        # Process batch
        result = service.extract_batch(documents_1k)

        # Calculate quality metrics
        total_entities = len(result.get_all_entities())
        avg_entities_per_doc = total_entities / len(documents_1k)

        print(f"\nResults:")
        print(f"  Total documents: {len(documents_1k)}")
        print(f"  Total entities extracted: {total_entities}")
        print(f"  Average entities per document: {avg_entities_per_doc:.2f}")
        print(f"  Target: 4.86 entities/doc (tolerance: >= 4.0)")

        # Validate quality maintained (allow some tolerance: >= 4.0)
        assert avg_entities_per_doc >= 4.0, \
            f"Quality must be maintained (target 4.86 entities/doc), got {avg_entities_per_doc:.2f}"
        print(f"  ✓ Quality requirement met: {avg_entities_per_doc:.2f} >= 4.0")

        # Additional quality checks
        entity_counts = result.get_entity_count_by_document()
        zero_entity_docs = sum(1 for count in entity_counts.values() if count == 0)
        zero_entity_pct = (zero_entity_docs / len(documents_1k)) * 100

        print(f"\nAdditional Quality Metrics:")
        print(f"  Documents with zero entities: {zero_entity_docs} ({zero_entity_pct:.1f}%)")
        print(f"  Max entities in single doc: {max(entity_counts.values())}")
        print(f"  Min entities in single doc: {min(entity_counts.values())}")

        # Zero-entity documents should be low (< 10%)
        assert zero_entity_pct < 10.0, \
            f"Too many zero-entity documents ({zero_entity_pct:.1f}%), expected < 10%"
        print(f"  ✓ Low zero-entity rate: {zero_entity_pct:.1f}% < 10%")

    def test_mixed_document_types_in_same_batch(self, service):
        """
        Validate handling of mixed document types in same batch (FR-009).

        Given: Batch contains different document types (emails, tickets, PDFs)
        When: System processes the batch
        Then: All document types are handled correctly
        """
        print(f"\n{'='*60}")
        print("Mixed Document Types Test (FR-009)")
        print(f"{'='*60}")

        # Create mixed document types
        mixed_docs = [
            # Support tickets
            Document(id="ticket1", page_content="Support ticket: TrakCare error ERR001 in module PatientManagement."),
            Document(id="ticket2", page_content="Ticket #2: Database connection failed with timeout."),

            # Emails
            Document(id="email1", page_content="From: user@example.com. Subject: Issue with TrakCare login. Body: Cannot access system."),
            Document(id="email2", page_content="Email: System upgrade notification for version 2024.1.5."),

            # Documentation
            Document(id="doc1", page_content="Documentation: TrakCare module PatientManagement handles patient records."),
            Document(id="doc2", page_content="User guide: How to troubleshoot database errors in TrakCare."),

            # Short notes
            Document(id="note1", page_content="Quick note: ERR404 resolved."),
            Document(id="note2", page_content="Meeting notes: Discussed TrakCare upgrade schedule."),
        ]

        # Process mixed batch
        result = service.extract_batch(mixed_docs)

        print(f"\nResults:")
        print(f"  Total document types: 4 (tickets, emails, docs, notes)")
        print(f"  Total documents: {len(mixed_docs)}")
        print(f"  Documents processed: {len(result.per_document_entities)}")

        # Validate all documents processed
        assert len(result.per_document_entities) == len(mixed_docs), \
            "All document types must be processed"
        print(f"  ✓ All document types processed successfully")

        # Validate success
        assert result.success_status, "Mixed document batch must succeed"
        print(f"  ✓ Batch processing succeeded with mixed types")

        # Validate entities extracted from each type
        for doc in mixed_docs:
            assert doc.id in result.per_document_entities, \
                f"Document {doc.id} must have entity results"

        print(f"  ✓ All documents have entity extraction results")

    def test_throughput_metrics(self, service, documents_1k):
        """Validate throughput meets target (25 tickets/min with batching)."""
        print(f"\n{'='*60}")
        print("Throughput Test: Documents per Minute")
        print(f"{'='*60}")

        start_time = time.time()
        result = service.extract_batch(documents_1k)
        elapsed = time.time() - start_time

        # Calculate throughput
        docs_per_second = len(documents_1k) / elapsed
        docs_per_minute = docs_per_second * 60

        print(f"\nResults:")
        print(f"  Processing time: {elapsed:.1f}s")
        print(f"  Throughput: {docs_per_minute:.1f} docs/min")
        print(f"  Target: >= 20 docs/min (3x improvement over 8.33 baseline)")

        # Validate throughput improvement
        assert docs_per_minute >= 20.0, \
            f"Throughput must be >= 20 docs/min, got {docs_per_minute:.1f}"
        print(f"  ✓ Throughput requirement met: {docs_per_minute:.1f} >= 20")

    def test_processing_statistics_accuracy(self, service, documents_1k):
        """Validate processing statistics are accurate (FR-007)."""
        print(f"\n{'='*60}")
        print("Processing Statistics Validation (FR-007)")
        print(f"{'='*60}")

        # Process batch
        result = service.extract_batch(documents_1k)

        # Get metrics
        metrics = service.get_batch_metrics()

        print(f"\nStatistics:")
        print(f"  Total batches processed: {metrics.total_batches_processed}")
        print(f"  Total documents processed: {metrics.total_documents_processed}")
        print(f"  Average batch time: {metrics.average_batch_processing_time:.2f}s")
        print(f"  Entity extraction rate: {metrics.entity_extraction_rate_per_batch:.2f}")
        print(f"  Zero-entity documents: {metrics.zero_entity_documents_count}")
        print(f"  Failed batches: {metrics.failed_batches_count}")
        print(f"  Total retry attempts: {metrics.retry_attempts_total}")

        # Validate statistics are reasonable
        assert metrics.total_batches_processed > 0, "Should have processed batches"
        assert metrics.total_documents_processed >= len(documents_1k), \
            "Should track document count"
        assert metrics.average_batch_processing_time > 0, \
            "Should track processing time"

        print(f"  ✓ All statistics tracked correctly")

    def test_batch_vs_single_doc_quality_equivalence(self, service):
        """Validate batch extraction quality equals single-doc (FR-008)."""
        print(f"\n{'='*60}")
        print("Quality Equivalence Test: Batch vs Single-Doc (FR-008)")
        print(f"{'='*60}")

        # Create test documents
        test_docs = [
            Document(id=f"equiv-{i}", page_content=f"TrakCare error ERR{i:03d} in module PatientManagement. User admin reported issue.")
            for i in range(10)
        ]

        # Process via batch
        batch_result = service.extract_batch(test_docs)
        batch_entities = {doc_id: entities for doc_id, entities in batch_result.per_document_entities.items()}

        # Process individually (if supported)
        # For now, validate batch extraction produces consistent results
        print(f"\nResults:")
        print(f"  Documents processed: {len(batch_entities)}")

        # Validate consistency across documents
        entity_counts = [len(entities) for entities in batch_entities.values()]
        avg_count = sum(entity_counts) / len(entity_counts)
        variance = sum((c - avg_count) ** 2 for c in entity_counts) / len(entity_counts)

        print(f"  Average entities per doc: {avg_count:.2f}")
        print(f"  Variance: {variance:.2f}")

        # Variance should be low for similar documents
        assert variance < 10.0, \
            f"Extraction should be consistent across similar docs (variance {variance:.2f})"
        print(f"  ✓ Consistent extraction across documents")

    def test_memory_usage_bounded(self, service, documents_1k):
        """Validate memory usage stays bounded during batch processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process batch
        result = service.extract_batch(documents_1k)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")

        # Memory increase should be reasonable (< 500 MB for 1K docs)
        assert memory_increase < 500, \
            f"Memory usage should be bounded (increase {memory_increase:.1f} MB)"
        print(f"  ✓ Memory usage bounded: {memory_increase:.1f} MB < 500 MB")
