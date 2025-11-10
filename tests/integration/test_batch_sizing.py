"""
Integration tests for dynamic batch sizing based on token budget.

Tests AS-4 from spec.md: Dynamic batch sizing with variable document sizes.
Tests FR-006: Token budget enforcement (8,192 default).
"""

import pytest
from iris_vector_rag.core.models import Document
from iris_vector_rag.services.entity_extraction import EntityExtractionService
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.common.iris_connection_manager import IRISConnectionManager
from iris_vector_rag.common.batch_utils import BatchQueue
from iris_vector_rag.utils.token_counter import estimate_tokens


@pytest.mark.integration
@pytest.mark.requires_database
class TestBatchSizing:
    """Integration tests for dynamic batch sizing (FR-006)."""

    @pytest.fixture
    def service(self):
        """Initialize EntityExtractionService."""
        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager()
        return EntityExtractionService(config_manager, connection_manager)

    def test_as4_variable_document_sizes_dynamic_batching(self, service):
        """
        AS-4: Validate dynamic batch sizing with variable document sizes.

        Given: User configures batch processing for entity extraction
        When: System encounters documents of varying sizes (100 words to 10,000 words)
        Then: System dynamically adjusts batch size based on token count
        """
        # Create documents with varying sizes
        small_docs = [
            Document(id=f"small-{i}", page_content="Short document. " * 10)
            for i in range(5)
        ]  # ~100 words each

        medium_docs = [
            Document(id=f"medium-{i}", page_content="Medium length document. " * 100)
            for i in range(3)
        ]  # ~1000 words each

        large_docs = [
            Document(id=f"large-{i}", page_content="Very large document. " * 1000)
            for i in range(2)
        ]  # ~10,000 words each

        all_docs = small_docs + medium_docs + large_docs

        # Process with default token budget (8192)
        result = service.extract_batch(all_docs, token_budget=8192)

        # Validate all documents processed
        assert len(result.per_document_entities) == len(all_docs), \
            "All documents must be processed regardless of size variation"

        # Validate success
        assert result.success_status, "Variable-size batch must succeed"

        print(f"\nVariable Size Batch Results:")
        print(f"  Small docs (5): {len([d for d in small_docs])}")
        print(f"  Medium docs (3): {len([d for d in medium_docs])}")
        print(f"  Large docs (2): {len([d for d in large_docs])}")
        print(f"  Total processed: {len(result.per_document_entities)}")

    def test_token_budget_enforcement_8k_default(self):
        """Validate BatchQueue enforces 8,192 token budget (FR-006 default)."""
        queue = BatchQueue(token_budget=8192)

        # Add documents with known token counts
        doc1 = Document(id="1", page_content="Document 1")
        doc2 = Document(id="2", page_content="Document 2")
        doc3 = Document(id="3", page_content="Document 3")

        # Each document: 4000 tokens
        queue.add_document(doc1, token_count=4000)
        queue.add_document(doc2, token_count=4000)
        queue.add_document(doc3, token_count=4000)

        # Get first batch (8K budget = 2 docs max)
        batch = queue.get_next_batch(token_budget=8192)

        assert len(batch) == 2, \
            f"Batch should contain 2 docs (8000 tokens), got {len(batch)}"

        # Get second batch (remaining doc)
        batch2 = queue.get_next_batch(token_budget=8192)

        assert len(batch2) == 1, "Second batch should have remaining document"

    def test_custom_token_budget(self):
        """Validate custom token budget configuration."""
        queue = BatchQueue(token_budget=4096)

        # Add documents
        docs = [
            Document(id=f"doc{i}", page_content="Test document")
            for i in range(5)
        ]

        for doc in docs:
            queue.add_document(doc, token_count=1500)

        # Get batch with 4K budget (2 docs max at 1500 tokens each)
        batch = queue.get_next_batch(token_budget=4096)

        assert len(batch) == 2, \
            f"4K budget should fit 2 docs at 1500 tokens each, got {len(batch)}"

    def test_batch_queue_optimal_packing(self):
        """Validate batch queue packs documents optimally within budget."""
        queue = BatchQueue()

        # Add documents with varying token counts
        docs_and_tokens = [
            (Document(id="1", page_content="Doc1"), 2000),
            (Document(id="2", page_content="Doc2"), 3000),
            (Document(id="3", page_content="Doc3"), 1000),
            (Document(id="4", page_content="Doc4"), 2500),
            (Document(id="5", page_content="Doc5"), 1500),
        ]

        for doc, tokens in docs_and_tokens:
            queue.add_document(doc, tokens)

        # Get batch with 8K budget
        # Optimal: Doc1 (2000) + Doc2 (3000) + Doc3 (1000) + Doc5 (1500) = 7500 tokens
        # or: Doc1 (2000) + Doc2 (3000) + Doc3 (1000) = 6000 tokens (FIFO)
        batch = queue.get_next_batch(token_budget=8192)

        # Validate batch respects budget
        total_tokens = sum(tokens for doc, tokens in docs_and_tokens[:len(batch)])
        assert total_tokens <= 8192, \
            f"Batch must respect token budget (got {total_tokens} tokens)"

    def test_single_large_document_exceeds_budget(self):
        """Validate handling of single document exceeding token budget."""
        queue = BatchQueue()

        # Add very large document (exceeds 8K budget)
        large_doc = Document(id="large", page_content="Large document")
        queue.add_document(large_doc, token_count=10000)

        # Should still return the document (can't split)
        batch = queue.get_next_batch(token_budget=8192)

        assert batch is not None, "Must return batch even if single doc exceeds budget"
        assert len(batch) == 1, "Should contain single large document"
        assert batch[0].id == "large", "Should return the large document"

    def test_token_estimation_accuracy(self):
        """Validate token estimation is accurate for batch sizing decisions."""
        test_cases = [
            ("Short text.", 3),
            ("This is a medium length sentence with multiple words.", 10),
            ("A very long document. " * 100, 400),  # ~400 tokens
        ]

        for text, expected_approx in test_cases:
            estimated = estimate_tokens(text)

            # Allow ±20% tolerance
            tolerance = expected_approx * 0.2
            assert abs(estimated - expected_approx) <= tolerance, \
                f"Token estimation accuracy (expected ~{expected_approx}, got {estimated})"

    def test_batch_respects_configured_token_budget(self, service):
        """Validate service respects token_budget parameter in extract_batch()."""
        # Create documents
        docs = [
            Document(id=f"doc{i}", page_content="Test document content. " * 50)
            for i in range(10)
        ]

        # Process with custom token budget
        result = service.extract_batch(docs, token_budget=4096)

        # Validate result
        assert result.success_status, "Batch with custom budget must succeed"
        assert len(result.per_document_entities) == len(docs), \
            "All documents must be processed"

    def test_token_budget_from_config(self):
        """Validate token budget can be configured in memory_config.yaml."""
        config_manager = ConfigurationManager()

        # Load batch processing config
        batch_config = config_manager.get_config('rag_memory_config.knowledge_extraction.entity_extraction.batch_processing')

        assert batch_config is not None, "Batch processing config must exist"
        assert 'token_budget' in batch_config, "Config must have token_budget"
        assert batch_config['token_budget'] == 8192, \
            f"Default token budget must be 8192 per FR-006, got {batch_config['token_budget']}"

    def test_empty_document_token_count(self):
        """Validate empty documents handled correctly in batch queue."""
        queue = BatchQueue()

        empty_doc = Document(id="empty", page_content="")
        queue.add_document(empty_doc, token_count=0)

        batch = queue.get_next_batch()

        assert batch is not None, "Should return batch with empty document"
        assert len(batch) == 1, "Should contain empty document"

    def test_batch_size_varies_with_document_sizes(self):
        """Validate batch size automatically adjusts based on document token counts."""
        queue = BatchQueue()

        # Scenario 1: Small documents (many fit in batch)
        small_docs = [
            Document(id=f"small{i}", page_content="Small")
            for i in range(20)
        ]

        for doc in small_docs:
            queue.add_document(doc, token_count=200)  # 200 tokens each

        batch1 = queue.get_next_batch(token_budget=8192)
        # Should fit many small documents (8192 / 200 = ~40 docs)
        assert len(batch1) >= 20, \
            f"Should fit many small documents in batch, got {len(batch1)}"

        # Scenario 2: Large documents (few fit in batch)
        queue2 = BatchQueue()
        large_docs = [
            Document(id=f"large{i}", page_content="Large")
            for i in range(5)
        ]

        for doc in large_docs:
            queue2.add_document(doc, token_count=3000)  # 3000 tokens each

        batch2 = queue2.get_next_batch(token_budget=8192)
        # Should fit only 2 large documents (8192 / 3000 = ~2 docs)
        assert len(batch2) == 2, \
            f"Should fit only 2 large documents, got {len(batch2)}"
