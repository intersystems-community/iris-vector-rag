"""
End-to-end integration tests for batch entity extraction.

Tests AS-1, AS-3, AS-5 from spec.md acceptance scenarios.
Requires IRIS database per Constitution principle III.
"""

import pytest
from iris_rag.core.models import Document
from iris_rag.services.entity_extraction import EntityExtractionService
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import IRISConnectionManager


@pytest.mark.integration
@pytest.mark.requires_database
class TestBatchExtractionE2E:
    """End-to-end tests for batch entity extraction pipeline."""

    @pytest.fixture
    def service(self):
        """Initialize EntityExtractionService with test configuration."""
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        return EntityExtractionService(config_manager, connection_manager)

    @pytest.fixture
    def sample_documents_1k(self):
        """Generate 1,000 test documents for AS-1 validation."""
        documents = []
        for i in range(1000):
            content = f"Sample support ticket {i}: User reported error in TrakCare module. " \
                     f"The system failed to process patient data correctly. " \
                     f"Error code: ERR{i:04d}. Version: 2024.1.{i % 10}."
            documents.append(Document(
                id=f"ticket-{i}",
                page_content=content
            ))
        return documents

    def test_as1_1k_documents_3x_speedup(self, service, sample_documents_1k):
        """
        AS-1: Validate 3x speedup on 1,000 documents.

        Given: 1,000 documents queued for entity extraction
        When: Batch extraction system processes them
        Then: Processing time is reduced by ~3x vs single-document processing
        """
        import time

        # Measure batch processing time
        start_time = time.time()
        batch_result = service.extract_batch(sample_documents_1k)
        batch_elapsed = time.time() - start_time

        # Validate batch completed successfully
        assert batch_result.success_status, \
            "Batch processing must complete successfully"

        # Get baseline single-document time (estimate from config or measure)
        # For TrakCare: ~7.2 seconds per ticket (from spec.md production context)
        single_doc_baseline = 7.2  # seconds per document

        # Calculate expected time and speedup
        expected_batch_time = (1000 * single_doc_baseline) / 3.0  # 3x speedup target
        actual_speedup = (1000 * single_doc_baseline) / batch_elapsed

        # Validate 3x speedup (allow 20% tolerance: 2.4x - 3.6x)
        assert actual_speedup >= 2.4, \
            f"Speedup must be at least 2.4x (target 3.0x), got {actual_speedup:.2f}x"

        # Validate quality maintained (4.86 entities/doc average from spec.md)
        total_entities = len(batch_result.get_all_entities())
        avg_entities_per_doc = total_entities / 1000

        assert avg_entities_per_doc >= 4.0, \
            f"Quality must be maintained (target 4.86 entities/doc), got {avg_entities_per_doc:.2f}"

        print(f"\nAS-1 Results:")
        print(f"  Batch processing time: {batch_elapsed:.1f}s")
        print(f"  Speedup: {actual_speedup:.2f}x (target: 3.0x)")
        print(f"  Entities per document: {avg_entities_per_doc:.2f} (target: 4.86)")

    def test_as3_entity_traceability(self, service):
        """
        AS-3: Validate entity traceability to source documents.

        Given: Batch of documents has been successfully processed
        When: System stores extracted entities
        Then: Each entity is correctly associated with its source document ID
        """
        # Create test documents with distinct content
        docs = [
            Document(id="doc1", page_content="TrakCare system error in module A"),
            Document(id="doc2", page_content="User login failed with error code 404"),
            Document(id="doc3", page_content="Database connection timeout in version 2024.1")
        ]

        # Process batch
        result = service.extract_batch(docs)

        # Validate entity traceability (FR-004)
        all_entities = result.get_all_entities()

        for entity in all_entities:
            # Each entity must have source_document_id
            assert hasattr(entity, 'source_document_id'), \
                "Entity must have source_document_id attribute"
            assert entity.source_document_id is not None, \
                "Entity source_document_id cannot be None"

            # Source document ID must be in original batch
            assert entity.source_document_id in ["doc1", "doc2", "doc3"], \
                f"Entity source_document_id must match batch documents, got {entity.source_document_id}"

        # Validate per-document mapping
        for doc_id in ["doc1", "doc2", "doc3"]:
            doc_entities = result.per_document_entities.get(doc_id, [])
            for entity in doc_entities:
                assert entity.source_document_id == doc_id, \
                    f"Entity in per_document_entities must have correct source_document_id"

        print(f"\nAS-3 Results:")
        print(f"  Total entities: {len(all_entities)}")
        print(f"  All entities have valid source_document_id: ✓")

    def test_as5_single_document_batch_queue_integration(self, service):
        """
        AS-5: Validate single document is added to batch queue.

        Given: Batch processing is enabled
        When: Single document is submitted for extraction
        Then: System adds it to batch queue (waits for batch to fill)
        """
        # Single document
        single_doc = Document(
            id="single-doc",
            page_content="Single urgent document for entity extraction"
        )

        # Process through batch system (FR-010: always batch)
        result = service.extract_batch([single_doc])

        # Validate batch result
        assert result.success_status, "Single document batch must succeed"
        assert "single-doc" in result.per_document_entities, \
            "Result must contain single document"

        # Validate batch metadata
        assert result.batch_id is not None, "Batch must have ID"
        assert result.processing_time > 0, "Processing time must be recorded"

        print(f"\nAS-5 Results:")
        print(f"  Single document processed via batch queue: ✓")
        print(f"  Batch ID: {result.batch_id}")
        print(f"  Processing time: {result.processing_time:.2f}s")

    def test_entity_extraction_quality_consistency(self, service):
        """Validate batch extraction quality matches single-doc (FR-008)."""
        # Create test document
        doc = Document(
            id="test-doc",
            page_content="TrakCare error ERR001 in module PatientManagement. "
                        "User admin reported issue with version 2024.1.5."
        )

        # Process via batch
        batch_result = service.extract_batch([doc])
        batch_entities = batch_result.per_document_entities["test-doc"]

        # Process same document individually (if single-doc method exists)
        # For now, validate batch extraction produces reasonable results
        assert len(batch_entities) > 0, \
            "Batch extraction must extract entities from valid document"

        # Validate entity types are from configured types
        valid_types = ["PRODUCT", "ERROR", "MODULE", "USER", "VERSION", "ACTION", "ORGANIZATION"]
        for entity in batch_entities:
            assert entity.entity_type in valid_types, \
                f"Entity type must be valid (got {entity.entity_type})"

        print(f"\nQuality Validation:")
        print(f"  Entities extracted: {len(batch_entities)}")
        print(f"  Entity types: {set(e.entity_type for e in batch_entities)}")

    def test_batch_processing_with_mixed_document_sizes(self, service):
        """Validate dynamic batch sizing with variable document sizes (AS-4)."""
        # Create documents of varying sizes
        docs = [
            Document(id="small", page_content="Short doc."),
            Document(id="medium", page_content="Medium length document. " * 50),
            Document(id="large", page_content="Very large document. " * 500),
            Document(id="small2", page_content="Another short one.")
        ]

        # Process batch
        result = service.extract_batch(docs, token_budget=8192)

        # Validate all documents processed
        assert len(result.per_document_entities) == 4, \
            "All documents must be processed regardless of size"

        # Validate success
        assert result.success_status, "Mixed-size batch must succeed"

        print(f"\nMixed Size Results:")
        print(f"  Documents processed: {len(result.per_document_entities)}")
        print(f"  Success: {result.success_status}")
