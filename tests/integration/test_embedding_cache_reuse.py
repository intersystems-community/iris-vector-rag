"""
Integration tests for actual SentenceTransformer model caching.

These tests use real sentence-transformers models (no mocks) to verify:
- Actual model instances are cached and reused
- Performance improvement (10x+ faster on cache hits)
- Same embeddings from same model instance

IMPORTANT: These tests are slower (~400ms first load) but verify real caching behavior.

Test Coverage:
- T005: Actual model caching with real SentenceTransformer models
"""

import pytest
import time
from typing import List

from iris_vector_rag.embeddings.manager import EmbeddingManager
from iris_vector_rag.config.manager import ConfigurationManager


class TestActualModelCaching:
    """Integration tests with real SentenceTransformer models."""

    def test_actual_model_caching(self):
        """
        T005: Verify real sentence-transformers models are cached (no mocks).

        Given: Real sentence-transformers models
        When: Create multiple managers with same model configuration
        Then: Embeddings identical, second initialization much faster

        Expected to FAIL before implementation with:
        - Both initializations taking ~400ms (no caching)
        - Assertion error on time2 < time1 / 10
        """
        # Create first EmbeddingManager (should load model from disk)
        config1 = ConfigurationManager()
        start1 = time.time()
        manager1 = EmbeddingManager(config1)
        emb1 = manager1.embed_text("hello world")
        time1 = time.time() - start1

        # Create second EmbeddingManager (should use cached model)
        config2 = ConfigurationManager()
        start2 = time.time()
        manager2 = EmbeddingManager(config2)
        emb2 = manager2.embed_text("hello world")
        time2 = time.time() - start2

        # Verify same embeddings (same model instance)
        assert emb1 == emb2, "Embeddings should be identical when using same model instance"

        # Verify second initialization is at least 10x faster
        print(f"\nFirst initialization: {time1:.3f}s")
        print(f"Second initialization: {time2:.3f}s")
        print(f"Speedup: {time1/time2 if time2 > 0 else 'infinite'}x")

        # This assertion will FAIL before caching is implemented
        # because both will take ~400ms
        assert time2 < time1 / 10, (
            f"Expected at least 10x speedup, but got {time1/time2:.1f}x "
            f"(first={time1:.3f}s, second={time2:.3f}s)"
        )

    def test_multiple_managers_sequential(self):
        """
        Verify sequential creation of multiple managers uses cache.

        Given: Clean process or warm cache
        When: Create 5 managers sequentially
        Then: All subsequent managers use cache (consistent fast times)
        """
        times: List[float] = []

        for i in range(5):
            config = ConfigurationManager()
            start = time.time()
            manager = EmbeddingManager(config)
            _ = manager.embed_text(f"test {i}")
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Manager {i+1}: {elapsed:.4f}s")

        # Verify cache is working: subsequent managers should be consistently fast
        # If cache wasn't working, all would take ~400ms
        avg_time = sum(times) / len(times)

        print(f"\nAverage time: {avg_time:.4f}s")
        print(f"Max time: {max(times):.4f}s")

        # With caching, average should be well under 100ms
        # Without caching, all would be ~400ms (average ~400ms)
        assert avg_time < 0.1, (
            f"Average time {avg_time:.3f}s suggests no caching. "
            f"Expected <0.1s with caching, would be ~0.4s without"
        )

        # All times should be consistently fast (< 50ms)
        for i, t in enumerate(times, start=1):
            assert t < 0.05, (
                f"Manager {i} took {t:.3f}s, expected <0.05s with caching"
            )

    def test_cache_persistence_across_embeddings(self):
        """
        Verify cache persists across multiple embedding operations.

        Given: Cached model from first manager
        When: Create second manager and generate multiple embeddings
        Then: All operations are fast, model not reloaded
        """
        # First manager loads model
        config1 = ConfigurationManager()
        manager1 = EmbeddingManager(config1)
        _ = manager1.embed_text("initial load")

        # Second manager should use cache
        config2 = ConfigurationManager()
        start = time.time()
        manager2 = EmbeddingManager(config2)

        # Generate multiple embeddings
        for i in range(10):
            _ = manager2.embed_text(f"test {i}")

        total_time = time.time() - start

        # All 10 embeddings + manager init should be fast (<100ms total)
        # This will FAIL before caching because manager init alone takes 400ms
        print(f"\nTotal time for manager2 + 10 embeddings: {total_time:.3f}s")
        assert total_time < 0.1, (
            f"Expected <0.1s for cached manager + 10 embeddings, got {total_time:.3f}s"
        )


class TestCacheBehaviorIntegration:
    """Integration tests for cache behavior with real models."""

    def test_embedding_consistency(self):
        """
        Verify embeddings are identical across multiple manager instances.

        Given: Multiple managers using cached model
        When: Generate embeddings for same text
        Then: All embeddings are identical (same model instance)
        """
        test_text = "The quick brown fox jumps over the lazy dog"

        # Create 3 managers
        embeddings = []
        for i in range(3):
            config = ConfigurationManager()
            manager = EmbeddingManager(config)
            emb = manager.embed_text(test_text)
            embeddings.append(emb)

        # All embeddings should be identical
        for i in range(1, len(embeddings)):
            assert embeddings[i] == embeddings[0], (
                f"Embedding {i} differs from embedding 0 - "
                "indicates different model instances (cache not working)"
            )

    @pytest.mark.slow
    def test_performance_with_batch_operations(self):
        """
        Verify cache works correctly during batch-like operations.

        Given: Scenario mimicking batch document processing
        When: Create 20 EmbeddingManagers sequentially
        Then: Total time is consistent with caching (not 20 × 400ms)
        """
        total_time = 0
        num_managers = 20

        print(f"\nCreating {num_managers} managers sequentially:")
        for i in range(num_managers):
            config = ConfigurationManager()
            start = time.time()
            manager = EmbeddingManager(config)
            _ = manager.embed_text(f"document {i}")
            elapsed = time.time() - start
            total_time += elapsed

            if i < 5 or i >= num_managers - 2:  # Print first 5 and last 2
                print(f"Manager {i+1:2d}: {elapsed:.4f}s")
            elif i == 5:
                print("...")

        avg_time = total_time / num_managers

        print(f"\nTotal time: {total_time:.3f}s")
        print(f"Average per manager: {avg_time:.4f}s")
        print(f"\nWithout caching: ~{num_managers * 0.4:.1f}s (20 × 0.4s)")
        print(f"With caching: ~0.6s (1 × 0.4s + 19 × 0.01s)")

        # This will FAIL before caching implementation
        # Expected: ~0.6s total, Actual before fix: ~8.0s
        assert total_time < 1.0, (
            f"Total time {total_time:.3f}s suggests no caching. "
            f"Expected <1.0s with caching, would be ~{num_managers * 0.4:.1f}s without"
        )
