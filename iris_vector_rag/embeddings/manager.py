"""
Embedding manager with fallback support for RAG templates.

This module provides a unified interface for embedding generation with support
for multiple backends and graceful fallback mechanisms.

Extended for Feature 051: IRIS EMBEDDING support with cache statistics tracking.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level cache for SentenceTransformer models (singleton pattern)
# Prevents repeated 400MB model loads from disk
# ============================================================================
_SENTENCE_TRANSFORMER_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

# ============================================================================
# Cache statistics tracking (Feature 051)
# ============================================================================
@dataclass
class CachedModelInstance:
    """
    Represents in-memory embedding model with device allocation, reference count,
    and performance metrics.

    Extended from Feature 050's basic cache to track detailed statistics.
    """
    config_name: str
    model: Any  # SentenceTransformer instance
    device: str  # "cuda:0", "mps", "cpu"
    load_time_ms: float
    reference_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_embeddings_generated: int = 0
    total_embedding_time_ms: float = 0.0  # Cumulative embedding time for averaging


@dataclass
class CacheStatistics:
    """Aggregate performance metrics for cache monitoring."""
    config_name: str
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_embedding_time_ms: float
    model_load_count: int
    memory_usage_mb: float
    device: str
    total_embeddings: int


# Statistics storage
_CACHE_STATS: Dict[str, CachedModelInstance] = {}
_STATS_LOCK = threading.Lock()


def _get_cached_sentence_transformer(model_name: str, device: str = "cpu"):
    """Get or create cached SentenceTransformer model.

    Performance improvement: 10-20x faster for repeated model access.

    Args:
        model_name: Name of the sentence-transformers model
        device: Device to load model on ('cpu', 'cuda', etc.)

    Returns:
        Cached SentenceTransformer model instance
    """
    cache_key = f"{model_name}:{device}"

    # Fast path: Check cache without lock (99.99% of calls after first load)
    if cache_key in _SENTENCE_TRANSFORMER_CACHE:
        return _SENTENCE_TRANSFORMER_CACHE[cache_key]

    # Slow path: Load model with lock (only on cache miss)
    with _CACHE_LOCK:
        # Double-check after acquiring lock (prevents race condition)
        if cache_key in _SENTENCE_TRANSFORMER_CACHE:
            return _SENTENCE_TRANSFORMER_CACHE[cache_key]

        # Load model from disk (one-time operation per cache key)
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model (one-time initialization): {model_name} on {device}")
        model = SentenceTransformer(model_name, device=device)

        # Cache for future use
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = model
        logger.info(f"✅ SentenceTransformer model '{model_name}' loaded and cached")

        return model


class EmbeddingManager:
    """
    Manages embedding generation with multiple backends and fallback support.

    Provides a unified interface for generating embeddings from text, with
    automatic fallback to alternative backends if the primary fails.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the embedding manager.

        Args:
            config_manager: Configuration manager for embedding settings
        """
        self.config_manager = config_manager
        self.embedding_config = self.config_manager.get("embeddings", {})

        # Get primary and fallback backends
        self.primary_backend = self.embedding_config.get(
            "primary_backend", "sentence_transformers"
        )
        self.fallback_backends = self.embedding_config.get(
            "fallback_backends", ["openai"]
        )

        # Cache for loaded embedding functions
        self._embedding_functions: Dict[str, Callable] = {}

        # Initialize primary backend
        self._initialize_backend(self.primary_backend)

    def _initialize_backend(self, backend_name: str) -> bool:
        """
        Initialize a specific embedding backend.

        Args:
            backend_name: Name of the backend to initialize

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if backend_name == "sentence_transformers":
                self._embedding_functions[backend_name] = (
                    self._create_sentence_transformers_function()
                )
            elif backend_name == "openai":
                self._embedding_functions[backend_name] = self._create_openai_function()
            elif backend_name == "huggingface":
                self._embedding_functions[backend_name] = (
                    self._create_huggingface_function()
                )
            else:
                logger.warning(f"Unknown embedding backend: {backend_name}")
                return False

            logger.info(f"Successfully initialized embedding backend: {backend_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embedding backend {backend_name}: {e}")
            return False

    def _create_sentence_transformers_function(self) -> Callable:
        """Create sentence transformers embedding function."""
        try:
            model_name = self.embedding_config.get("sentence_transformers", {}).get(
                "model_name", "all-MiniLM-L6-v2"
            )
            # Get device from config (default to 'cpu' to avoid GPU contention)
            device = self.embedding_config.get("sentence_transformers", {}).get(
                "device", "cpu"
            )
            model = _get_cached_sentence_transformer(model_name, device)
            logger.info(f"✅ SentenceTransformer initialized on device: {device}")

            def embed_texts(texts: List[str]) -> List[List[float]]:
                embeddings = model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()

            return embed_texts

        except ImportError:
            logger.error(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to create sentence transformers function: {e}")
            raise

    def _create_openai_function(self) -> Callable:
        """Create OpenAI embedding function."""
        try:
            import openai

            openai_config = self.embedding_config.get("openai", {})
            api_key = openai_config.get("api_key") or self.config_manager.get(
                "openai:api_key"
            )
            model_name = openai_config.get("model_name", "text-embedding-ada-002")

            if not api_key:
                raise ValueError("OpenAI API key not found in configuration")

            client = openai.OpenAI(api_key=api_key)

            def embed_texts(texts: List[str]) -> List[List[float]]:
                response = client.embeddings.create(input=texts, model=model_name)
                return [embedding.embedding for embedding in response.data]

            return embed_texts

        except ImportError:
            logger.error("openai not available. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to create OpenAI function: {e}")
            raise

    def _create_huggingface_function(self) -> Callable:
        """Create Hugging Face embedding function."""
        try:
            import torch

            from iris_vector_rag.common.huggingface_utils import download_huggingface_model

            hf_config = self.embedding_config.get("huggingface", {})
            model_name = hf_config.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )

            tokenizer, model = download_huggingface_model(model_name)

            def embed_texts(texts: List[str]) -> List[List[float]]:
                # Tokenize and encode
                encoded_input = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                )

                # Generate embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                    # Use mean pooling
                    embeddings = model_output.last_hidden_state.mean(dim=1)

                return embeddings.tolist()

            return embed_texts

        except ImportError:
            logger.error(
                "transformers not available. Install with: pip install transformers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to create Hugging Face function: {e}")
            raise

    def _create_fallback_function(self) -> Callable:
        """Create a simple fallback embedding function."""

        def embed_texts(texts: List[str]) -> List[List[float]]:
            """
            Simple fallback that creates basic embeddings based on text length and hash.
            This is not suitable for production but allows the system to continue functioning.
            """
            import hashlib

            embeddings = []
            for text in texts:
                # Handle None or empty text
                if text is None:
                    text = ""
                elif not isinstance(text, str):
                    text = str(text)

                # Create a simple embedding based on text characteristics
                text_hash = hashlib.md5(text.encode()).hexdigest()

                # Convert hash to numbers and normalize
                hash_numbers = [
                    int(text_hash[i : i + 2], 16) for i in range(0, len(text_hash), 2)
                ]

                # Pad or truncate to desired dimension (get from config or use 384 fallback)
                target_dim = self.embedding_config.get("dimension", 384)
                while len(hash_numbers) < target_dim:
                    hash_numbers.extend(hash_numbers[: target_dim - len(hash_numbers)])
                hash_numbers = hash_numbers[:target_dim]

                # Normalize to [-1, 1] range
                normalized = [(x - 127.5) / 127.5 for x in hash_numbers]
                embeddings.append(normalized)

            logger.warning(f"Using fallback embeddings for {len(texts)} texts")
            return embeddings

        return embed_texts

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (alias for embed_text for compatibility).

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_text(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (alias for embed_texts for HybridGraphRAG compatibility).

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with fallback support.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Try primary backend first
        if self.primary_backend in self._embedding_functions:
            try:
                return self._embedding_functions[self.primary_backend](texts)
            except Exception as e:
                logger.warning(f"Primary backend {self.primary_backend} failed: {e}")

        # Try fallback backends
        for backend_name in self.fallback_backends:
            if backend_name not in self._embedding_functions:
                if not self._initialize_backend(backend_name):
                    continue

            try:
                logger.info(f"Using fallback backend: {backend_name}")
                return self._embedding_functions[backend_name](texts)
            except Exception as e:
                logger.warning(f"Fallback backend {backend_name} failed: {e}")
                continue

        # If all backends fail, use simple fallback
        logger.warning("All embedding backends failed, using simple fallback")
        fallback_func = self._create_fallback_function()
        return fallback_func(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current backend.

        Returns:
            Embedding dimension
        """
        # Try to get dimension from config first
        dimension = self.embedding_config.get("dimension")
        if dimension:
            return dimension

        # Try to get from embedding config's model mapping
        model_name = self.embedding_config.get("sentence_transformers", {}).get(
            "model_name", "all-MiniLM-L6-v2"
        )

        # Use direct model-to-dimension mapping instead of dimension utils
        known_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "bert-base-uncased": 768,
            "bert-large-uncased": 1024,
        }

        if model_name in known_dimensions:
            return known_dimensions[model_name]

        # Otherwise, generate a test embedding to determine dimension
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            # HARD FAIL - no fallback dimensions to hide configuration issues
            error_msg = f"CRITICAL: Cannot determine embedding dimension: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def is_available(self, backend_name: Optional[str] = None) -> bool:
        """
        Check if a specific backend or any backend is available.

        Args:
            backend_name: Optional specific backend to check

        Returns:
            True if backend(s) available, False otherwise
        """
        if backend_name:
            return backend_name in self._embedding_functions

        # Check if any backend is available
        return len(self._embedding_functions) > 0

    def get_available_backends(self) -> List[str]:
        """
        Get list of currently available backends.

        Returns:
            List of available backend names
        """
        return list(self._embedding_functions.keys())

    def switch_backend(self, backend_name: str) -> bool:
        """
        Switch to a different primary backend.

        Args:
            backend_name: Name of backend to switch to

        Returns:
            True if switch successful, False otherwise
        """
        if backend_name not in self._embedding_functions:
            if not self._initialize_backend(backend_name):
                return False

        self.primary_backend = backend_name
        logger.info(f"Switched to primary backend: {backend_name}")
        return True


# ============================================================================
# Cache Statistics API (Feature 051)
# ============================================================================

def get_cache_stats(config_name: Optional[str] = None) -> CacheStatistics:
    """
    Retrieve model cache statistics (FR-022).

    Args:
        config_name: Optional specific configuration to get stats for.
                    If None, returns aggregate stats for all cached models.

    Returns:
        CacheStatistics with performance metrics

    Example:
        >>> stats = get_cache_stats("medical_embeddings_v1")
        >>> print(f"Cache hit rate: {stats.hit_rate:.2%}")
        Cache hit rate: 99.50%
    """
    with _STATS_LOCK:
        if config_name:
            if config_name not in _CACHE_STATS:
                # Return empty stats if not found
                return CacheStatistics(
                    config_name=config_name,
                    cache_hits=0,
                    cache_misses=0,
                    hit_rate=0.0,
                    avg_embedding_time_ms=0.0,
                    model_load_count=0,
                    memory_usage_mb=0.0,
                    device="unknown",
                    total_embeddings=0
                )

            instance = _CACHE_STATS[config_name]
            total_calls = instance.cache_hits + instance.cache_misses
            hit_rate = instance.cache_hits / total_calls if total_calls > 0 else 0.0
            avg_time = (
                instance.total_embedding_time_ms / total_calls
                if total_calls > 0
                else 0.0
            )

            return CacheStatistics(
                config_name=instance.config_name,
                cache_hits=instance.cache_hits,
                cache_misses=instance.cache_misses,
                hit_rate=hit_rate,
                avg_embedding_time_ms=avg_time,
                model_load_count=instance.cache_misses,  # Approximation
                memory_usage_mb=instance.memory_usage_mb,
                device=instance.device,
                total_embeddings=instance.total_embeddings_generated
            )
        else:
            # Aggregate stats for all configs
            total_hits = sum(inst.cache_hits for inst in _CACHE_STATS.values())
            total_misses = sum(inst.cache_misses for inst in _CACHE_STATS.values())
            total_embeddings = sum(inst.total_embeddings_generated for inst in _CACHE_STATS.values())
            total_memory = sum(inst.memory_usage_mb for inst in _CACHE_STATS.values())
            total_calls = total_hits + total_misses
            hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

            return CacheStatistics(
                config_name="<all>",
                cache_hits=total_hits,
                cache_misses=total_misses,
                hit_rate=hit_rate,
                avg_embedding_time_ms=0.0,
                model_load_count=total_misses,
                memory_usage_mb=total_memory,
                device="mixed",
                total_embeddings=total_embeddings
            )


def clear_cache(config_name: Optional[str] = None):
    """
    Clear model cache (for testing or memory management).

    Args:
        config_name: Optional specific configuration to clear.
                    If None, clears all cached models.

    Returns:
        ClearCacheResult with models cleared and memory freed

    Example:
        >>> result = clear_cache("medical_embeddings_v1")
        >>> print(f"Freed {result.memory_freed_mb}MB")
        Freed 384.2MB
    """
    from ..config.embedding_config import ClearCacheResult

    with _CACHE_LOCK:
        if config_name:
            # Clear specific model by config_name
            # Need to get config to find model_name, then build cache keys
            from ..embeddings.iris_embedding import get_config

            try:
                config = get_config(config_name)
                model_name = config.model_name
            except ValueError:
                # Config not found, return empty result
                logger.warning(f"Config '{config_name}' not found, nothing to clear")
                return ClearCacheResult(models_cleared=0, memory_freed_mb=0.0)

            # Cache keys are "model_name:device", check all possible devices
            possible_devices = ["cuda:0", "mps", "cpu"]
            cache_keys_to_remove = []

            for device in possible_devices:
                cache_key = f"{model_name}:{device}"
                if cache_key in _SENTENCE_TRANSFORMER_CACHE:
                    cache_keys_to_remove.append(cache_key)

            models_cleared = len(cache_keys_to_remove)
            memory_freed = 0.0

            for cache_key in cache_keys_to_remove:
                del _SENTENCE_TRANSFORMER_CACHE[cache_key]
                logger.info(f"Cleared cache for: {cache_key}")

            with _STATS_LOCK:
                if config_name in _CACHE_STATS:
                    memory_freed = _CACHE_STATS[config_name].memory_usage_mb
                    del _CACHE_STATS[config_name]

            return ClearCacheResult(
                models_cleared=models_cleared,
                memory_freed_mb=memory_freed
            )
        else:
            # Clear all models
            models_cleared = len(_SENTENCE_TRANSFORMER_CACHE)
            _SENTENCE_TRANSFORMER_CACHE.clear()
            logger.info(f"Cleared all cached models ({models_cleared} total)")

            with _STATS_LOCK:
                memory_freed = sum(inst.memory_usage_mb for inst in _CACHE_STATS.values())
                _CACHE_STATS.clear()

            return ClearCacheResult(
                models_cleared=models_cleared,
                memory_freed_mb=memory_freed
            )


def _record_cache_hit(config_name: str):
    """Record a cache hit for statistics tracking."""
    with _STATS_LOCK:
        if config_name in _CACHE_STATS:
            _CACHE_STATS[config_name].cache_hits += 1
            _CACHE_STATS[config_name].last_access_time = time.time()


def _record_cache_miss(config_name: str, device: str, load_time_ms: float):
    """Record a cache miss (model load) for statistics tracking."""
    with _STATS_LOCK:
        if config_name not in _CACHE_STATS:
            # Create new stats entry
            _CACHE_STATS[config_name] = CachedModelInstance(
                config_name=config_name,
                model=None,  # We don't store the model here, just stats
                device=device,
                load_time_ms=load_time_ms,
                cache_misses=1,
                memory_usage_mb=400.0  # Approximate model size
            )
        else:
            _CACHE_STATS[config_name].cache_misses += 1
            _CACHE_STATS[config_name].last_access_time = time.time()


def _record_embeddings_generated(config_name: str, count: int):
    """Record number of embeddings generated."""
    with _STATS_LOCK:
        if config_name in _CACHE_STATS:
            _CACHE_STATS[config_name].total_embeddings_generated += count


def _record_embedding_time(config_name: str, time_ms: float):
    """Record embedding generation time for averaging."""
    with _STATS_LOCK:
        if config_name in _CACHE_STATS:
            _CACHE_STATS[config_name].total_embedding_time_ms += time_ms
