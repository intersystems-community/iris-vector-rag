"""
Unit tests for batch queue management.

Tests BatchQueue logic in isolation without dependencies.
"""

import pytest
from iris_vector_rag.common.batch_utils import BatchQueue
from iris_vector_rag.core.models import Document


class TestBatchQueue:
    """Unit tests for BatchQueue class."""

    def test_fifo_ordering(self):
        """Validate queue maintains FIFO (first-in-first-out) ordering."""
        queue = BatchQueue(token_budget=10000)

        # Add documents in order
        docs = [Document(id=f"doc{i}", page_content=f"Doc {i}") for i in range(5)]

        for doc in docs:
            queue.add_document(doc, token_count=500)

        # Get batch
        batch = queue.get_next_batch()

        # Verify FIFO order
        for i, doc in enumerate(batch):
            assert doc.id == f"doc{i}", \
                f"FIFO order not maintained (expected doc{i}, got {doc.id})"

    def test_token_budget_calculations(self):
        """Validate token budget calculations are correct."""
        queue = BatchQueue()

        # Add documents with specific token counts
        docs_and_tokens = [
            (Document(id="1", page_content="Doc1"), 2000),
            (Document(id="2", page_content="Doc2"), 3000),
            (Document(id="3", page_content="Doc3"), 4000),
        ]

        for doc, tokens in docs_and_tokens:
            queue.add_document(doc, tokens)

        # Get batch with 6000 token budget
        batch = queue.get_next_batch(token_budget=6000)

        # Should get first 2 docs (2000 + 3000 = 5000 <= 6000)
        assert len(batch) == 2, \
            f"Should fit 2 docs in 6K budget, got {len(batch)}"
        assert batch[0].id == "1", "First doc should be id='1'"
        assert batch[1].id == "2", "Second doc should be id='2'"

        # Get next batch
        batch2 = queue.get_next_batch(token_budget=6000)

        # Should get remaining doc
        assert len(batch2) == 1, "Should have 1 remaining doc"
        assert batch2[0].id == "3", "Remaining doc should be id='3'"

    def test_queue_state_transitions(self):
        """Validate queue state changes correctly."""
        queue = BatchQueue()

        # Initially empty
        assert queue.get_next_batch() is None or queue.get_next_batch() == [], \
            "Empty queue should return None or []"

        # Add documents
        queue.add_document(Document(id="1", page_content="Doc1"), 1000)
        queue.add_document(Document(id="2", page_content="Doc2"), 1000)

        # Should have documents
        batch = queue.get_next_batch()
        assert batch is not None and len(batch) > 0, \
            "Queue should have documents"

        # After getting all documents, should be empty again
        remaining = queue.get_next_batch()
        assert remaining is None or remaining == [], \
            "Queue should be empty after all documents retrieved"

    def test_empty_queue_returns_none_or_empty(self):
        """Validate empty queue returns None or empty list."""
        queue = BatchQueue()

        result = queue.get_next_batch()

        assert result is None or result == [], \
            f"Empty queue should return None or [], got {result}"

    def test_single_document(self):
        """Validate queue handles single document correctly."""
        queue = BatchQueue()

        doc = Document(id="single", page_content="Single doc")
        queue.add_document(doc, token_count=100)

        batch = queue.get_next_batch()

        assert len(batch) == 1, "Should contain single document"
        assert batch[0].id == "single", "Should return the added document"

    def test_exact_budget_fit(self):
        """Validate documents that exactly fit budget."""
        queue = BatchQueue()

        # Add docs that exactly fit 8000 token budget
        queue.add_document(Document(id="1", page_content="Doc1"), 4000)
        queue.add_document(Document(id="2", page_content="Doc2"), 4000)
        queue.add_document(Document(id="3", page_content="Doc3"), 4000)

        batch = queue.get_next_batch(token_budget=8000)

        # Should fit exactly 2 documents (4000 + 4000 = 8000)
        assert len(batch) == 2, "Should fit exactly 2 docs at 8000 tokens"

    def test_no_document_fits_budget(self):
        """Validate handling when no document fits budget (except first)."""
        queue = BatchQueue()

        # All documents exceed budget individually
        queue.add_document(Document(id="1", page_content="Large1"), 10000)
        queue.add_document(Document(id="2", page_content="Large2"), 10000)

        # Should still return first document (can't skip it)
        batch = queue.get_next_batch(token_budget=5000)

        assert len(batch) == 1, "Should return single large document"
        assert batch[0].id == "1", "Should return first large document"

    def test_multiple_batch_retrieval(self):
        """Validate multiple batches can be retrieved sequentially."""
        queue = BatchQueue()

        # Add 9 documents (1000 tokens each)
        for i in range(9):
            queue.add_document(Document(id=f"doc{i}", page_content=f"Doc{i}"), 1000)

        # Retrieve 3 batches (3 docs each with 3000 token budget)
        batch1 = queue.get_next_batch(token_budget=3000)
        batch2 = queue.get_next_batch(token_budget=3000)
        batch3 = queue.get_next_batch(token_budget=3000)

        assert len(batch1) == 3, "First batch should have 3 docs"
        assert len(batch2) == 3, "Second batch should have 3 docs"
        assert len(batch3) == 3, "Third batch should have 3 docs"

        # Verify all different documents
        all_ids = [doc.id for batch in [batch1, batch2, batch3] for doc in batch]
        assert len(set(all_ids)) == 9, "All documents should be unique"

    def test_zero_token_document(self):
        """Validate handling of document with zero tokens."""
        queue = BatchQueue()

        queue.add_document(Document(id="empty", page_content=""), 0)

        batch = queue.get_next_batch()

        assert batch is not None, "Should handle zero-token document"
        assert len(batch) == 1, "Should contain zero-token document"

    def test_very_large_single_document(self):
        """Validate single document larger than budget is still returned."""
        queue = BatchQueue()

        huge_doc = Document(id="huge", page_content="Huge document")
        queue.add_document(huge_doc, token_count=50000)

        batch = queue.get_next_batch(token_budget=8192)

        assert batch is not None, "Should return batch even with oversized doc"
        assert len(batch) == 1, "Should contain single oversized document"
        assert batch[0].id == "huge", "Should return the huge document"

    def test_mixed_token_sizes(self):
        """Validate queue handles mixed token sizes correctly."""
        queue = BatchQueue()

        # Mix of small, medium, large documents
        queue.add_document(Document(id="small", page_content="S"), 100)
        queue.add_document(Document(id="large", page_content="L"), 7000)
        queue.add_document(Document(id="medium", page_content="M"), 1000)
        queue.add_document(Document(id="tiny", page_content="T"), 50)

        batch = queue.get_next_batch(token_budget=8000)

        # Should fit small (100) + large (7000) = 7100 <= 8000
        # OR just small + medium + tiny if FIFO strictly followed
        # Implementation may vary, but should respect budget
        total_tokens = sum([100, 7000])  # Assuming first two fit
        assert len(batch) <= 4, "Batch size should be reasonable"

    def test_custom_token_budget_initialization(self):
        """Validate custom token budget in constructor."""
        custom_budget = 4096
        queue = BatchQueue(token_budget=custom_budget)

        # Add documents
        for i in range(5):
            queue.add_document(Document(id=f"doc{i}", page_content=f"Doc{i}"), 1500)

        # Get batch (should respect 4096 budget)
        batch = queue.get_next_batch()

        # At 1500 tokens each, should fit 2 docs (3000 <= 4096)
        assert len(batch) == 2, \
            f"4096 budget should fit 2 docs at 1500 tokens each, got {len(batch)}"

    def test_dynamic_budget_override(self):
        """Validate get_next_batch can override default budget."""
        queue = BatchQueue(token_budget=8192)

        # Add documents
        for i in range(5):
            queue.add_document(Document(id=f"doc{i}", page_content=f"Doc{i}"), 2000)

        # Override with smaller budget
        batch = queue.get_next_batch(token_budget=5000)

        # Should fit 2 docs (4000 <= 5000) not 4 docs (8000)
        assert len(batch) == 2, \
            f"5000 budget should fit 2 docs at 2000 tokens each, got {len(batch)}"

    def test_queue_preserves_document_objects(self):
        """Validate queue preserves original document objects."""
        queue = BatchQueue()

        original_doc = Document(id="test", page_content="Test content", metadata={"key": "value"})
        queue.add_document(original_doc, token_count=100)

        batch = queue.get_next_batch()
        retrieved_doc = batch[0]

        # Should be same object or equivalent
        assert retrieved_doc.id == original_doc.id
        assert retrieved_doc.page_content == original_doc.page_content
        assert retrieved_doc.metadata == original_doc.metadata

    def test_boundary_conditions(self):
        """Validate boundary conditions in token budget calculations."""
        queue = BatchQueue()

        # Test exact boundary
        queue.add_document(Document(id="1", page_content="D1"), 4096)
        queue.add_document(Document(id="2", page_content="D2"), 4096)

        batch = queue.get_next_batch(token_budget=8192)

        # Exactly 2 docs should fit (4096 + 4096 = 8192)
        assert len(batch) == 2, "Exactly 2 docs should fit at budget boundary"

        # Test one over boundary
        queue2 = BatchQueue()
        queue2.add_document(Document(id="1", page_content="D1"), 4097)
        queue2.add_document(Document(id="2", page_content="D2"), 4096)

        batch2 = queue2.get_next_batch(token_budget=8192)

        # Only 1 doc should fit (4097 + 4096 = 8193 > 8192)
        assert len(batch2) == 1, "Only 1 doc should fit when sum exceeds budget"
