"""
Contract tests for batch queue utility.

These tests validate the BatchQueue API contract before implementation.
Tests MUST fail initially, then pass after implementation (TDD).
"""

import pytest
from iris_vector_rag.core.models import Document


class TestBatchQueueContract:
    """Contract tests for BatchQueue class."""

    def test_batch_queue_class_exists(self):
        """Validate BatchQueue class exists."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        assert BatchQueue is not None, "BatchQueue class must exist"

    def test_batch_queue_init_signature(self):
        """Validate BatchQueue.__init__() has correct signature."""
        from iris_vector_rag.common.batch_utils import BatchQueue
        import inspect

        sig = inspect.signature(BatchQueue.__init__)
        params = sig.parameters

        # Should accept optional token_budget parameter
        assert 'token_budget' in params or len(params) == 1, \
            "BatchQueue should accept optional token_budget parameter"

    def test_batch_queue_add_document_method_exists(self):
        """Validate add_document() method exists."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue()
        assert hasattr(queue, 'add_document'), \
            "BatchQueue must have add_document() method"

    def test_batch_queue_get_next_batch_method_exists(self):
        """Validate get_next_batch() method exists."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue()
        assert hasattr(queue, 'get_next_batch'), \
            "BatchQueue must have get_next_batch() method"

    def test_batch_queue_respects_token_budget(self):
        """Validate get_next_batch() respects token budget (FR-006)."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue(token_budget=8000)

        # Add documents with known token counts
        doc1 = Document(id="1", page_content="Document 1")
        doc2 = Document(id="2", page_content="Document 2")
        doc3 = Document(id="3", page_content="Document 3")

        queue.add_document(doc1, token_count=3000)
        queue.add_document(doc2, token_count=3000)
        queue.add_document(doc3, token_count=3000)

        # Get next batch (should contain only 2 documents, not 3)
        batch = queue.get_next_batch(token_budget=8000)

        # Should return 2 documents (6000 tokens), not 3 (9000 tokens)
        assert len(batch) == 2, \
            f"Batch should contain 2 docs (6000 tokens < 8000 budget), got {len(batch)}"
        assert batch[0].id == "1", "First document should be doc1"
        assert batch[1].id == "2", "Second document should be doc2"

    def test_batch_queue_empty_returns_none(self):
        """Validate empty queue returns None or empty list."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue()

        batch = queue.get_next_batch()

        # Should return None or empty list for empty queue
        assert batch is None or batch == [], \
            "Empty queue must return None or empty list"

    def test_batch_queue_add_document_signature(self):
        """Validate add_document() has correct signature."""
        from iris_vector_rag.common.batch_utils import BatchQueue
        import inspect

        sig = inspect.signature(BatchQueue.add_document)
        params = sig.parameters

        # Required parameters
        assert 'self' in params, "add_document must be instance method"
        assert 'document' in params, "add_document must accept 'document' parameter"
        assert 'token_count' in params, "add_document must accept 'token_count' parameter"

    def test_batch_queue_get_next_batch_signature(self):
        """Validate get_next_batch() has correct signature."""
        from iris_vector_rag.common.batch_utils import BatchQueue
        import inspect

        sig = inspect.signature(BatchQueue.get_next_batch)
        params = sig.parameters

        # Should accept optional token_budget parameter
        assert 'self' in params, "get_next_batch must be instance method"
        # token_budget can be optional with default

    def test_batch_queue_fifo_ordering(self):
        """Validate queue maintains FIFO ordering."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue(token_budget=10000)

        # Add documents in specific order
        docs = [
            Document(id=f"doc{i}", page_content=f"Document {i}")
            for i in range(5)
        ]

        for doc in docs:
            queue.add_document(doc, token_count=500)

        # Get batch (all should fit in 10K budget)
        batch = queue.get_next_batch(token_budget=10000)

        # Should maintain order
        assert len(batch) == 5, "All documents should fit in batch"
        for i, doc in enumerate(batch):
            assert doc.id == f"doc{i}", \
                f"Document order not maintained (expected doc{i}, got {doc.id})"

    def test_batch_queue_handles_single_large_document(self):
        """Validate queue handles single document exceeding budget."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue(token_budget=5000)

        # Add large document (exceeds budget)
        large_doc = Document(id="large", page_content="Large document")
        queue.add_document(large_doc, token_count=7000)

        # Should still return the document (can't split)
        batch = queue.get_next_batch(token_budget=5000)

        assert batch is not None, "Must return batch even if single doc exceeds budget"
        assert len(batch) == 1, "Should contain single large document"
        assert batch[0].id == "large", "Should return the large document"

    def test_batch_queue_multiple_batches(self):
        """Validate queue can produce multiple batches."""
        from iris_vector_rag.common.batch_utils import BatchQueue

        queue = BatchQueue()

        # Add 6 documents (3000 tokens each)
        for i in range(6):
            doc = Document(id=f"doc{i}", page_content=f"Document {i}")
            queue.add_document(doc, token_count=3000)

        # Get first batch (8K budget = 2 docs)
        batch1 = queue.get_next_batch(token_budget=8000)
        assert len(batch1) == 2, "First batch should have 2 documents"

        # Get second batch
        batch2 = queue.get_next_batch(token_budget=8000)
        assert len(batch2) == 2, "Second batch should have 2 documents"

        # Get third batch
        batch3 = queue.get_next_batch(token_budget=8000)
        assert len(batch3) == 2, "Third batch should have 2 documents"

        # Queue should now be empty
        batch4 = queue.get_next_batch(token_budget=8000)
        assert batch4 is None or batch4 == [], "Queue should be empty"
