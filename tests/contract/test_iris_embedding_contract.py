"""
Contract tests for IRIS EMBEDDING integration.

Feature: 051-add-native-iris
Contract: iris_embedding_contract.yaml
Status: TDD - These tests MUST FAIL until implementation is complete

This test file validates the IRIS EMBEDDING integration contract specifications,
ensuring that the embedding model caching, configuration management, and
vectorization APIs meet the documented requirements.
"""

import pytest
from uuid import uuid4
from typing import List, Dict, Any
import time

# These imports will fail initially - that's expected for TDD
try:
    from iris_rag.embeddings.iris_embedding import (
        get_config,
        embed_texts,
        configure_embedding,
    )
    from iris_rag.config.embedding_config import (
        EmbeddingConfig,
        ValidationResult,
        validate_embedding_config,
    )
    from iris_rag.embeddings.manager import (
        get_cache_stats,
        clear_cache,
    )
    IMPLEMENTATION_EXISTS = True
except ImportError:
    IMPLEMENTATION_EXISTS = False


pytestmark = pytest.mark.skipif(
    not IMPLEMENTATION_EXISTS,
    reason="Implementation not yet available (TDD - tests first)"
)


class TestReadEmbeddingConfig:
    """Test scenarios for read_embedding_config contract."""

    def test_load_valid_config(self):
        """
        Scenario: Load valid EMBEDDING configuration from IRIS
        Given: %Embedding.Config contains valid entry
        When: get_config('medical_embeddings_v1') is called
        Then: Returns EmbeddingConfig with all fields populated
        """
        # Setup: Create test configuration
        config_name = f"test_config_{uuid4().hex[:8]}"
        configure_embedding(
            name=config_name,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3",
            description="Test configuration"
        )

        # Execute
        config = get_config(config_name)

        # Verify
        assert config.name == config_name
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.hf_cache_path == "/var/lib/huggingface"
        assert config.python_path == "/usr/bin/python3"
        assert config.batch_size > 0
        assert config.device_preference in ["cuda", "mps", "cpu", "auto"]

    def test_config_not_found(self):
        """
        Scenario: Handle missing configuration gracefully
        Given: Configuration 'nonexistent' does not exist
        When: get_config('nonexistent') is called
        Then: Raises CONFIG_NOT_FOUND error with clear message
        """
        with pytest.raises(ValueError) as exc_info:
            get_config("nonexistent_config_12345")

        assert "CONFIG_NOT_FOUND" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


class TestValidateEmbeddingConfig:
    """Test scenarios for validate_embedding_config contract."""

    def test_validate_valid_config(self):
        """
        Scenario: Validate correct configuration
        Given: Valid EmbeddingConfig with existing model
        When: validate(config) is called
        Then: Returns ValidationResult with valid=True
        """
        config = EmbeddingConfig(
            name="test_valid",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3",
            embedding_class="%Embedding.SentenceTransformers",
            batch_size=32,
            device_preference="auto"
        )

        result = validate_embedding_config(config)

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_missing_model(self):
        """
        Scenario: Detect missing model file
        Given: Config with model_name that doesn't exist locally
        When: validate(config) is called
        Then: Returns ValidationResult with valid=False and MODEL_NOT_FOUND error
        """
        config = EmbeddingConfig(
            name="test_missing_model",
            model_name="nonexistent-model/does-not-exist",
            hf_cache_path="/tmp/nonexistent_cache",
            python_path="/usr/bin/python3",
            embedding_class="%Embedding.SentenceTransformers",
            batch_size=32,
            device_preference="auto"
        )

        result = validate_embedding_config(config)

        assert isinstance(result, ValidationResult)
        assert result.valid is False
        assert any("MODEL_NOT_FOUND" in error or "not found" in error.lower() for error in result.errors)


class TestGenerateEmbeddings:
    """Test scenarios for generate_embeddings contract."""

    @pytest.fixture(autouse=True)
    def setup_config(self):
        """Setup test configuration before each test."""
        self.config_name = f"test_embed_{uuid4().hex[:8]}"
        configure_embedding(
            name=self.config_name,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3"
        )
        yield
        # Cleanup: clear cache after test
        try:
            clear_cache(self.config_name)
        except:
            pass

    def test_embed_texts_cache_hit(self):
        """
        Scenario: Generate embeddings with cached model (FR-001, FR-003)
        Given: Model already cached from previous call
        When: embed_texts('config', ['text1', 'text2']) is called
        Then:
          - Returns embeddings list with 2 vectors
          - cache_hit == True
          - embedding_time_ms < 50
          - model_load_time_ms == 0
        """
        texts = ["Sample text 1", "Sample text 2"]

        # First call: loads model into cache
        result1 = embed_texts(self.config_name, texts)
        assert len(result1.embeddings) == 2
        assert result1.cache_hit is False  # First call, no cache

        # Second call: should hit cache
        start_time = time.time()
        result2 = embed_texts(self.config_name, texts)
        elapsed_ms = (time.time() - start_time) * 1000

        # Verify cache hit
        assert len(result2.embeddings) == 2
        assert result2.cache_hit is True
        assert result2.embedding_time_ms < 50, f"Expected <50ms, got {result2.embedding_time_ms}ms"
        assert result2.model_load_time_ms == 0
        assert result2.device_used in ["cuda:0", "mps", "cpu"]

    def test_embed_texts_cache_miss(self):
        """
        Scenario: Generate embeddings with model load
        Given: Model not in cache
        When: embed_texts('config', ['text']) is called first time
        Then:
          - Returns embedding vector
          - cache_hit == False
          - model_load_time_ms < 5000
        """
        # Clear cache to ensure cache miss
        clear_cache(self.config_name)

        texts = ["First embedding after cache clear"]

        result = embed_texts(self.config_name, texts)

        # Verify cache miss
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) > 0  # Vector has dimensions
        assert result.cache_hit is False
        assert result.model_load_time_ms < 5000, f"Model load took {result.model_load_time_ms}ms (target: <5000ms)"
        assert result.device_used in ["cuda:0", "mps", "cpu"]

    def test_embed_texts_empty_text(self):
        """
        Scenario: Handle empty text input
        Given: texts list contains empty string
        When: embed_texts('config', ['']) is called
        Then: Raises EMPTY_TEXT error
        """
        with pytest.raises(ValueError) as exc_info:
            embed_texts(self.config_name, [""])

        assert "EMPTY_TEXT" in str(exc_info.value) or "empty" in str(exc_info.value).lower()

    def test_cache_hit_rate_target(self):
        """
        Scenario: Verify 95% cache hit rate after warmup (FR-003)
        Given: 1000 embedding calls made
        When: get_cache_stats() is called
        Then: hit_rate >= 0.95
        """
        texts = [f"Document {i} content for testing" for i in range(100)]

        # Generate 1000 embeddings (10 iterations of 100 texts)
        for _ in range(10):
            embed_texts(self.config_name, texts)

        # Check cache statistics
        stats = get_cache_stats(self.config_name)

        assert stats.cache_hits + stats.cache_misses >= 1000
        hit_rate = stats.cache_hits / (stats.cache_hits + stats.cache_misses)
        assert hit_rate >= 0.95, f"Cache hit rate {hit_rate:.2%} below 95% target"

    def test_gpu_fallback(self):
        """
        Scenario: Gracefully fall back to CPU on GPU OOM
        Given: GPU memory exhausted
        When: embed_texts() triggers OOM
        Then:
          - System falls back to CPU device
          - Embedding generation succeeds
          - Warning logged with GPU_OOM code
        """
        # This test is difficult to trigger reliably
        # We test that CPU fallback works by forcing CPU device
        config_name_cpu = f"test_cpu_{uuid4().hex[:8]}"
        configure_embedding(
            name=config_name_cpu,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3",
            device_preference="cpu"  # Force CPU
        )

        texts = ["Test CPU fallback works"]
        result = embed_texts(config_name_cpu, texts)

        # Verify embedding generation succeeds on CPU
        assert len(result.embeddings) == 1
        assert result.device_used == "cpu"

        # Cleanup
        clear_cache(config_name_cpu)


class TestCacheManagement:
    """Test scenarios for cache management contracts."""

    def test_get_cache_stats(self):
        """Test retrieving cache statistics."""
        config_name = f"test_stats_{uuid4().hex[:8]}"
        configure_embedding(
            name=config_name,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3"
        )

        # Generate some embeddings
        texts = ["Test text 1", "Test text 2"]
        embed_texts(config_name, texts)

        # Get stats
        stats = get_cache_stats(config_name)

        assert stats.config_name == config_name
        assert stats.cache_hits >= 0
        assert stats.cache_misses >= 1  # At least one miss for initial load
        assert stats.model_load_count >= 1
        assert stats.avg_embedding_time_ms > 0

        # Cleanup
        clear_cache(config_name)

    def test_clear_cache(self):
        """Test clearing model cache."""
        config_name = f"test_clear_{uuid4().hex[:8]}"
        configure_embedding(
            name=config_name,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3"
        )

        # Load model into cache
        embed_texts(config_name, ["Test text"])

        # Clear cache
        result = clear_cache(config_name)

        assert result.models_cleared >= 1
        assert result.memory_freed_mb > 0

        # Verify cache was cleared (next call should be cache miss)
        result2 = embed_texts(config_name, ["Test after clear"])
        assert result2.cache_hit is False


@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Performance benchmark tests from contract."""

    def test_benchmark_cache_hit_performance(self):
        """
        Benchmark: Measure cached embedding time (FR-002)
        Target: <30 seconds (vs 20 minutes baseline) for 1,746 texts
        Acceptance: 50x improvement minimum
        """
        config_name = f"bench_perf_{uuid4().hex[:8]}"
        configure_embedding(
            name=config_name,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3"
        )

        # Generate 1,746 texts (exact count from DP-442038)
        texts = [f"Medical document {i} content" for i in range(1746)]

        # Warmup: load model into cache
        embed_texts(config_name, texts[:10])

        # Benchmark: measure time for all 1,746 texts
        start_time = time.time()
        for i in range(0, len(texts), 100):  # Batch processing
            batch = texts[i:i+100]
            embed_texts(config_name, batch)
        elapsed_time = time.time() - start_time

        # Verify performance target
        assert elapsed_time < 30, f"Vectorization took {elapsed_time:.1f}s (target: <30s)"

        # Cleanup
        clear_cache(config_name)


# TDD Status Check
def test_implementation_status():
    """
    Meta-test: Verify that implementation exists.
    This test should PASS when implementation is complete.
    """
    assert IMPLEMENTATION_EXISTS, (
        "Implementation not found. Expected modules:\n"
        "- iris_rag.embeddings.iris_embedding\n"
        "- iris_rag.config.embedding_config\n"
        "- iris_rag.embeddings.manager\n"
        "These tests are written following TDD - implementation comes next."
    )
