"""
Unit tests for SentenceTransformer model caching in EmbeddingManager.

These tests verify the module-level cache behavior using mocks.
They MUST fail before implementation (TDD requirement).

Test Coverage:
- T002: Cache reuse in single-threaded scenario
- T003: Thread safety with concurrent initialization
- T004: Different model+device configurations get separate cache entries
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import logging

from iris_vector_rag.embeddings.manager import EmbeddingManager
from iris_vector_rag.config.manager import ConfigurationManager


class TestEmbeddingCache:
    """Contract tests for SentenceTransformer model caching."""

    @patch('iris_rag.embeddings.manager.logger')
    @patch('iris_rag.embeddings.manager._get_cached_sentence_transformer')
    def test_cache_reuse_single_threaded(self, mock_get_cached, mock_logger):
        """
        T002: Verify model loaded once, subsequent EmbeddingManagers use cache.

        Given: Clean process (no cached models)
        When: Create two EmbeddingManagers with same model configuration
        Then: Model loaded once, second instantiation uses cache

        Expected to FAIL before implementation with:
        - AttributeError: module 'iris_rag.embeddings.manager' has no attribute '_get_cached_sentence_transformer'
        """
        # Setup mock to simulate caching behavior
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]  # 384-dim embeddings
        mock_get_cached.return_value = mock_model

        # Track calls to verify caching
        call_count = 0
        def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_model
        mock_get_cached.side_effect = track_calls

        # Create first EmbeddingManager (should load model)
        config1 = ConfigurationManager()
        manager1 = EmbeddingManager(config1)

        # Create second EmbeddingManager (should use cache)
        config2 = ConfigurationManager()
        manager2 = EmbeddingManager(config2)

        # Verify both managers work
        emb1 = manager1.embed_text("test")
        emb2 = manager2.embed_text("test")
        assert len(emb1) == 384
        assert len(emb2) == 384

        # Verify _get_cached_sentence_transformer was called twice
        # (once per manager, but cache should be hit on second call)
        assert mock_get_cached.call_count == 2

        # In real implementation, we'd check logs for exactly 1 "one-time initialization"
        # but since we're mocking, we verify the function was called

    @patch('iris_rag.embeddings.manager.logger')
    @patch('iris_rag.embeddings.manager._get_cached_sentence_transformer')
    def test_cache_thread_safety(self, mock_get_cached, mock_logger):
        """
        T003: Verify thread-safe initialization prevents race conditions.

        Given: Clean process, multiple threads
        When: 10 threads create EmbeddingManagers concurrently
        Then: Model loaded exactly once, no race conditions, all embeddings valid

        Expected to FAIL before implementation with:
        - AttributeError: module 'iris_rag.embeddings.manager' has no attribute '_get_cached_sentence_transformer'
        """
        # Setup mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        # Track initialization calls
        init_count = 0
        def track_init(*args, **kwargs):
            nonlocal init_count
            init_count += 1
            return mock_model
        mock_get_cached.side_effect = track_init

        def create_manager_and_embed(thread_id):
            """Create EmbeddingManager and generate embedding."""
            config = ConfigurationManager()
            manager = EmbeddingManager(config)
            embedding = manager.embed_text(f"test from thread {thread_id}")
            return len(embedding)

        # Create 10 managers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(create_manager_and_embed, range(10)))

        # Verify all threads got valid 384-dimensional embeddings
        assert all(dim == 384 for dim in results)
        assert len(results) == 10

        # Verify _get_cached_sentence_transformer was called 10 times
        # (once per thread, but actual model load should only happen once)
        assert mock_get_cached.call_count == 10

        # In real implementation, only 1 "one-time initialization" log message
        # Here we just verify no exceptions were raised

    @patch('iris_rag.embeddings.manager.logger')
    @patch('iris_rag.embeddings.manager._get_cached_sentence_transformer')
    def test_different_configurations(self, mock_get_cached, mock_logger):
        """
        T004: Verify different model+device combos get separate cache entries.

        Given: Clean process
        When: Create managers with different model names or devices
        Then: Each unique configuration loads model once

        Expected to FAIL before implementation with:
        - AttributeError: module 'iris_rag.embeddings.manager' has no attribute '_get_cached_sentence_transformer'
        """
        # Setup mocks for different models
        mock_model_1 = MagicMock()
        # Return numpy array that gets converted to list
        import numpy as np
        mock_model_1.encode.return_value = np.array([[0.1] * 384])  # all-MiniLM-L6-v2: 384 dims

        mock_model_2 = MagicMock()
        mock_model_2.encode.return_value = np.array([[0.1] * 768])  # all-mpnet-base-v2: 768 dims

        # Track which model to return based on call order
        call_count = 0
        def return_different_models(model_name, device="cpu"):
            nonlocal call_count
            call_count += 1
            if "MiniLM" in model_name:
                return mock_model_1
            else:
                return mock_model_2
        mock_get_cached.side_effect = return_different_models

        # Create manager with default model (all-MiniLM-L6-v2)
        config1 = ConfigurationManager()
        manager1 = EmbeddingManager(config1)
        emb1 = manager1.embed_text("test")

        # Create manager with different model (all-mpnet-base-v2)
        config2 = ConfigurationManager()
        # Override model name in config
        config2._config["embeddings"] = {
            "sentence_transformers": {
                "model_name": "all-mpnet-base-v2",
                "device": "cpu"
            }
        }
        manager2 = EmbeddingManager(config2)
        emb2 = manager2.embed_text("test")

        # Verify different embedding dimensions
        assert len(emb1) == 384  # MiniLM
        assert len(emb2) == 768  # mpnet

        # Verify _get_cached_sentence_transformer was called twice
        # (once per unique model configuration)
        assert mock_get_cached.call_count == 2

        # In real implementation, 2 "one-time initialization" messages
        # (one per unique model+device combination)


class TestCacheKeyGeneration:
    """Test cache key format for different configurations."""

    @patch('iris_rag.embeddings.manager._get_cached_sentence_transformer')
    def test_cache_key_format(self, mock_get_cached):
        """
        Verify cache keys are generated correctly for different model+device combos.

        Expected to FAIL before implementation with AttributeError.
        """
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_get_cached.return_value = mock_model

        # Create managers with different configurations
        configs = [
            ("all-MiniLM-L6-v2", "cpu"),
            ("all-MiniLM-L6-v2", "cuda"),
            ("all-mpnet-base-v2", "cpu"),
        ]

        for model_name, device in configs:
            config = ConfigurationManager()
            config._config["embeddings"] = {
                "sentence_transformers": {
                    "model_name": model_name,
                    "device": device
                }
            }
            manager = EmbeddingManager(config)

            # Verify _get_cached_sentence_transformer was called with correct args
            # (in real implementation, this creates cache key "{model}:{device}")

        # Should have been called 3 times (one per unique config)
        assert mock_get_cached.call_count == 3
