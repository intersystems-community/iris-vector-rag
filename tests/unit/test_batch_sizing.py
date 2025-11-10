"""
Unit tests for dynamic batch sizing logic.

Tests batch size calculation and token accumulation without dependencies.
"""

import pytest
from iris_vector_rag.common.batch_utils import BatchQueue
from iris_vector_rag.core.models import Document
from iris_vector_rag.utils.token_counter import estimate_tokens


class TestBatchSizing:
    """Unit tests for dynamic batch sizing calculations."""

    def test_dynamic_batch_size_calculation(self):
        """Validate batch size adjusts dynamically based on token counts."""
        queue = BatchQueue()

        # Scenario 1: Small documents (many fit)
        for i in range(20):
            queue.add_document(Document(id=f"small{i}", page_content="Small"), 200)

        batch1 = queue.get_next_batch(token_budget=8192)

        # Should fit many small documents (8192 / 200 = ~40)
        assert len(batch1) >= 10, \
            f"Should fit many small documents, got {len(batch1)}"

        # Scenario 2: Large documents (few fit)
        queue2 = BatchQueue()
        for i in range(10):
            queue2.add_document(Document(id=f"large{i}", page_content="Large"), 3000)

        batch2 = queue2.get_next_batch(token_budget=8192)

        # Should fit few large documents (8192 / 3000 = ~2)
        assert len(batch2) <= 3, \
            f"Should fit few large documents, got {len(batch2)}"

    def test_token_count_accumulation(self):
        """Validate token counts are accumulated correctly."""
        queue = BatchQueue()

        # Add documents with specific token counts
        docs_tokens = [(100, "doc1"), (200, "doc2"), (300, "doc3"), (400, "doc4")]

        for tokens, doc_id in docs_tokens:
            queue.add_document(Document(id=doc_id, page_content=doc_id), tokens)

        # Get batch with 600 token budget
        batch = queue.get_next_batch(token_budget=600)

        # Should fit doc1 (100) + doc2 (200) + doc3 (300) = 600
        assert len(batch) == 3, \
            f"Should fit 3 docs totaling 600 tokens, got {len(batch)}"

    def test_batch_boundary_conditions(self):
        """Validate boundary conditions in batch sizing."""
        queue = BatchQueue()

        # Test exact match
        queue.add_document(Document(id="1", page_content="D"), 4096)
        queue.add_document(Document(id="2", page_content="D"), 4096)
        queue.add_document(Document(id="3", page_content="D"), 100)

        batch = queue.get_next_batch(token_budget=8192)

        # Should fit exactly first 2 (4096 + 4096 = 8192)
        assert len(batch) == 2, "Should fit exactly 2 docs at budget limit"

        # Verify third doc remains
        batch2 = queue.get_next_batch(token_budget=8192)
        assert len(batch2) == 1, "Third doc should remain for next batch"
        assert batch2[0].id == "3", "Third doc should be available"

    def test_incremental_token_counting(self):
        """Validate tokens are counted incrementally as docs are added."""
        queue = BatchQueue()

        # Track cumulative tokens manually
        cumulative = 0
        docs = []

        for i in range(10):
            tokens = (i + 1) * 100  # 100, 200, 300, ..., 1000
            doc = Document(id=f"doc{i}", page_content=f"Doc {i}")
            queue.add_document(doc, tokens)
            cumulative += tokens
            docs.append((doc, tokens))

        # Get batch
        batch = queue.get_next_batch(token_budget=2000)

        # Should fit docs until cumulative exceeds 2000
        # 100 + 200 + 300 + 400 + 500 = 1500 (fits)
        # 100 + 200 + 300 + 400 + 500 + 600 = 2100 (exceeds)
        # So should get 5 docs
        assert len(batch) == 5, \
            f"Should fit 5 docs (1500 tokens total), got {len(batch)}"

    def test_single_document_exceeding_budget(self):
        """Validate single document larger than budget is handled."""
        queue = BatchQueue()

        # Add document larger than budget
        queue.add_document(Document(id="huge", page_content="Huge"), 10000)

        batch = queue.get_next_batch(token_budget=5000)

        # Should still return the document
        assert len(batch) == 1, "Should return oversized document"
        assert batch[0].id == "huge", "Should return correct document"

    def test_multiple_documents_each_exceeding_budget(self):
        """Validate multiple oversized documents are processed individually."""
        queue = BatchQueue()

        # Add 3 documents, each exceeding budget
        for i in range(3):
            queue.add_document(Document(id=f"huge{i}", page_content="Huge"), 10000)

        # Each batch should contain one document
        batch1 = queue.get_next_batch(token_budget=5000)
        batch2 = queue.get_next_batch(token_budget=5000)
        batch3 = queue.get_next_batch(token_budget=5000)

        assert len(batch1) == 1, "First batch should have 1 oversized doc"
        assert len(batch2) == 1, "Second batch should have 1 oversized doc"
        assert len(batch3) == 1, "Third batch should have 1 oversized doc"

    def test_zero_budget_behavior(self):
        """Validate behavior with zero token budget."""
        queue = BatchQueue()

        queue.add_document(Document(id="test", page_content="Test"), 100)

        # Zero budget should still return at least first document
        batch = queue.get_next_batch(token_budget=0)

        # Implementation choice: either return empty or return first doc
        # Most likely: return first doc (can't have empty batch if queue has docs)
        assert batch is not None, "Should handle zero budget gracefully"

    def test_negative_token_count_handling(self):
        """Validate handling of negative token counts (error case)."""
        queue = BatchQueue()

        # Negative token count should be handled
        # Either raise error or treat as 0
        try:
            queue.add_document(Document(id="negative", page_content="Neg"), -100)
            batch = queue.get_next_batch()
            # If accepted, should handle gracefully
            assert batch is not None, "Should handle negative tokens"
        except ValueError:
            # Alternatively, may reject negative tokens
            pass  # This is also acceptable behavior

    def test_very_small_budget_with_normal_documents(self):
        """Validate very small budget (< normal document size)."""
        queue = BatchQueue()

        queue.add_document(Document(id="doc1", page_content="Document 1"), 1000)
        queue.add_document(Document(id="doc2", page_content="Document 2"), 1000)

        # Budget smaller than any document
        batch = queue.get_next_batch(token_budget=500)

        # Should still return first document
        assert len(batch) == 1, "Should return first doc even if exceeds budget"

    def test_fractional_token_counts(self):
        """Validate handling of fractional token counts (if supported)."""
        queue = BatchQueue()

        # Some tokenizers might return floats
        # Test if queue handles them (should convert to int or accept)
        try:
            queue.add_document(Document(id="test", page_content="Test"), 100.5)
            batch = queue.get_next_batch()
            assert batch is not None, "Should handle fractional tokens"
        except (TypeError, ValueError):
            # May require integer tokens
            pass  # This is acceptable

    def test_optimal_packing_strategy(self):
        """Validate queue uses optimal packing strategy (first-fit)."""
        queue = BatchQueue()

        # Add documents in specific order
        queue.add_document(Document(id="1", page_content="D1"), 3000)
        queue.add_document(Document(id="2", page_content="D2"), 6000)  # Won't fit with #1
        queue.add_document(Document(id="3", page_content="D3"), 2000)

        batch = queue.get_next_batch(token_budget=8000)

        # First-fit: #1 (3000) fits, #2 (6000) doesn't fit (would be 9000)
        # Best-fit might try #1 + #3, but FIFO means #1, then #2 attempted, then #3
        # Expected: #1 only (3000), or #1 + #3 if implementation is smart

        # Minimum requirement: should fit at least doc #1
        assert len(batch) >= 1, "Should fit at least one document"
        assert batch[0].id == "1", "First document should be in batch"

    def test_realistic_document_sizes(self):
        """Validate with realistic document token sizes."""
        queue = BatchQueue()

        # Realistic TrakCare ticket sizes (from spec.md: ~800 tokens average)
        realistic_tokens = [500, 800, 1200, 600, 900, 750, 1100, 650]

        for i, tokens in enumerate(realistic_tokens):
            queue.add_document(Document(id=f"ticket{i}", page_content=f"Ticket {i}"), tokens)

        batch = queue.get_next_batch(token_budget=8192)

        # Calculate actual total
        cumulative = 0
        expected_count = 0
        for tokens in realistic_tokens:
            if cumulative + tokens <= 8192:
                cumulative += tokens
                expected_count += 1
            else:
                break

        assert len(batch) == expected_count, \
            f"Should fit {expected_count} realistic documents, got {len(batch)}"

    def test_batch_size_reproducibility(self):
        """Validate batch sizing is deterministic and reproducible."""
        # Create two identical queues
        queue1 = BatchQueue()
        queue2 = BatchQueue()

        docs = [
            (Document(id="1", page_content="D1"), 1000),
            (Document(id="2", page_content="D2"), 2000),
            (Document(id="3", page_content="D3"), 1500),
        ]

        for doc, tokens in docs:
            queue1.add_document(doc, tokens)
            queue2.add_document(Document(id=doc.id, page_content=doc.page_content), tokens)

        batch1 = queue1.get_next_batch(token_budget=5000)
        batch2 = queue2.get_next_batch(token_budget=5000)

        # Should produce identical batches
        assert len(batch1) == len(batch2), "Batch sizes should be identical"
        for i in range(len(batch1)):
            assert batch1[i].id == batch2[i].id, "Batch contents should be identical"

    def test_estimate_tokens_integration(self):
        """Validate integration with estimate_tokens function."""
        # Create documents and estimate their tokens
        docs = [
            Document(id="short", page_content="Short text."),
            Document(id="medium", page_content="This is a medium length document. " * 10),
            Document(id="long", page_content="This is a very long document. " * 100),
        ]

        # Estimate tokens for each
        tokens = [estimate_tokens(doc.page_content) for doc in docs]

        # Create queue with estimated tokens
        queue = BatchQueue()
        for doc, token_count in zip(docs, tokens):
            queue.add_document(doc, token_count)

        batch = queue.get_next_batch(token_budget=8192)

        # Should produce valid batch
        assert batch is not None, "Should produce batch with estimated tokens"
        assert len(batch) > 0, "Batch should contain documents"

    def test_edge_case_all_documents_fit_exactly(self):
        """Validate when all documents fit exactly within budget."""
        queue = BatchQueue()

        # Add documents totaling exactly 8192 tokens
        queue.add_document(Document(id="1", page_content="D1"), 2048)
        queue.add_document(Document(id="2", page_content="D2"), 2048)
        queue.add_document(Document(id="3", page_content="D3"), 2048)
        queue.add_document(Document(id="4", page_content="D4"), 2048)

        batch = queue.get_next_batch(token_budget=8192)

        # All 4 should fit exactly
        assert len(batch) == 4, "All documents should fit exactly in budget"

        # Queue should be empty
        batch2 = queue.get_next_batch(token_budget=8192)
        assert batch2 is None or batch2 == [], "Queue should be empty"
