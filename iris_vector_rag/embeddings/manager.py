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
from typing import Any, Callable, Dict, List, Optional, cast

from ..config.manager import ConfigurationManager
from ..exceptions import EmbeddingError

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
    embedding_calls: int = 0  # Timed embedding calls: the divisor for the average


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

    Raises whatever the load raises (`ImportError`, `OSError`, ...). Nothing stands in for
    a model that would not load, and nothing failed is cached.

    Args:
        model_name: Name of the sentence-transformers model
        device: Device to load model on ('cpu', 'cuda', etc.)

    Returns:
        Cached SentenceTransformer model instance
    """
    cache_key = f"{model_name}:{device}"

    cached = _SENTENCE_TRANSFORMER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Slow path: Load model with lock (only on cache miss)
    with _CACHE_LOCK:
        # Double-check after acquiring lock (prevents race condition)
        if cache_key in _SENTENCE_TRANSFORMER_CACHE:
            return _SENTENCE_TRANSFORMER_CACHE[cache_key]

        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading SentenceTransformer model (one-time initialization): %s on %s",
            model_name,
            device,
        )
        model = SentenceTransformer(model_name, device=device)
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = model
        logger.info("SentenceTransformer model '%s' loaded and cached", model_name)

        return model


# Known output widths, so a configured model needs no probe.
KNOWN_MODEL_DIMENSIONS: Dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
}

DEFAULT_OPENAI_TIMEOUT_SECONDS = 60
DEFAULT_OPENAI_BATCH_SIZE = 256

_OFFLINE_HINT = (
    "If this host cannot reach the model hub, set HF_HUB_OFFLINE=1 and point "
    "embeddings.sentence_transformers.model_name at a model already on disk, or "
    "configure a reachable real backend in embeddings.fallback_backends."
)

STUB_BACKEND = "stub"


def _mean_pool(hidden_states, attention_mask):
    """Mean of the token vectors that are real text, then unit length.

    Padding positions are masked out, so a text's vector does not depend on how long the
    longest text in its batch happens to be.
    """
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(summed / counts, p=2, dim=1)


class EmbeddingManager:
    """
    Manages embedding generation with multiple backends and fallback support.

    Provides a unified interface for generating embeddings from text. The primary backend
    is tried first, then each configured `fallback_backends` entry in order. When none
    produces vectors it raises `EmbeddingError`: nothing is ever substituted for a real
    model. The test embedder is reachable only by naming it (`provider: stub`).
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the embedding manager.

        Args:
            config_manager: Configuration manager for embedding settings
        """
        self.config_manager = config_manager
        self.embedding_config = self.config_manager.get("embeddings", {}) or {}

        # Get primary and fallback backends
        self.primary_backend = self.embedding_config.get(
            "primary_backend", "sentence_transformers"
        )
        if self.embedding_config.get("provider") == STUB_BACKEND:
            self.primary_backend = STUB_BACKEND
        self.fallback_backends = list(
            self.embedding_config.get("fallback_backends", ["openai"])
        )
        if STUB_BACKEND in self.fallback_backends:
            raise ValueError(
                "The test embedder ('stub') cannot be a fallback backend: its vectors "
                "would be mixed with a real model's. Name it as the provider instead "
                "(embeddings.provider: stub)."
            )

        # Cache for loaded embedding functions, and why the others are not there
        self._embedding_functions: Dict[str, Callable] = {}
        self._backend_errors: Dict[str, str] = {}
        self._backend_widths: Dict[str, Callable[[], Optional[int]]] = {}
        self._backend_models: Dict[str, str] = {}
        self._dimension: Optional[int] = None

        # Initialize primary backend
        self._initialize_backend(self.primary_backend)

    def _initialize_backend(self, backend_name: str) -> bool:
        """
        Initialize a specific embedding backend.

        Args:
            backend_name: Name of the backend to initialize

        Returns:
            True if initialization successful, False otherwise. The reason a backend
            could not start is kept in `_backend_errors` for `embed_texts` to report.
        """
        creators = {
            "sentence_transformers": self._create_sentence_transformers_function,
            "openai": self._create_openai_function,
            "huggingface": self._create_huggingface_function,
            STUB_BACKEND: self._create_stub_function,
        }
        creator = creators.get(backend_name)
        if creator is None:
            self._backend_errors[backend_name] = (
                f"unknown embedding backend {backend_name!r} "
                f"(known: {', '.join(sorted(creators))})"
            )
            logger.error(self._backend_errors[backend_name])
            return False

        try:
            self._embedding_functions[backend_name] = creator()
        except Exception as e:
            self._backend_errors[backend_name] = f"{type(e).__name__}: {e}"
            logger.error(f"Failed to initialize embedding backend {backend_name}: {e}")
            return False

        self._backend_errors.pop(backend_name, None)
        logger.info(f"Successfully initialized embedding backend: {backend_name}")
        return True

    def _create_stub_function(self) -> Callable:
        """The test embedder, at the configured dimension. Reachable by name only."""
        from ..testing.embedder import MODEL_KEY, TestEmbedder

        dimension = self.embedding_config.get("dimension")
        if not dimension:
            raise EmbeddingError(
                "The test embedder needs embeddings.dimension: it has no model to "
                "take a width from."
            )
        embedder = TestEmbedder(dimension=int(dimension))
        self._backend_widths[STUB_BACKEND] = lambda: embedder.dimension
        self._backend_models[STUB_BACKEND] = MODEL_KEY
        return embedder.embed_texts

    def _create_sentence_transformers_function(self) -> Callable:
        """Create sentence transformers embedding function."""
        st_config = self.embedding_config.get("sentence_transformers", {})
        model_name = st_config.get("model_name", "all-MiniLM-L6-v2")
        device = st_config.get("device", "cpu")
        _get_cached_sentence_transformer(model_name, device)
        logger.info(f"SentenceTransformer initialized on device: {device}")

        def embed_texts(texts: List[str]) -> List[List[float]]:
            m = _get_cached_sentence_transformer(model_name, device)
            embeddings: Any = m.encode(
                texts, convert_to_tensor=False, show_progress_bar=False
            )
            to_list = getattr(embeddings, "tolist", None)
            if callable(to_list):
                return cast(List[List[float]], to_list())
            return cast(List[List[float]], list(embeddings))

        def width() -> Optional[int]:
            m = _get_cached_sentence_transformer(model_name, device)
            reported = m.get_sentence_embedding_dimension()
            return int(reported) if reported else None

        self._backend_widths["sentence_transformers"] = width
        self._backend_models["sentence_transformers"] = model_name
        return embed_texts

    def _create_openai_function(self) -> Callable:
        """Create OpenAI embedding function."""
        import openai

        openai_config = self.embedding_config.get("openai", {})
        api_key = openai_config.get("api_key") or self.config_manager.get(
            "openai:api_key"
        )
        model_name = openai_config.get("model_name", "text-embedding-ada-002")
        timeout = openai_config.get("timeout", DEFAULT_OPENAI_TIMEOUT_SECONDS)
        batch_size = int(openai_config.get("batch_size", DEFAULT_OPENAI_BATCH_SIZE))

        if not api_key:
            raise ValueError("OpenAI API key not found in configuration")

        client = openai.OpenAI(api_key=api_key, timeout=timeout)

        def embed_texts(texts: List[str]) -> List[List[float]]:
            vectors: List[List[float]] = []
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                response = client.embeddings.create(input=chunk, model=model_name)
                if len(response.data) != len(chunk):
                    raise EmbeddingError(
                        f"OpenAI returned {len(response.data)} embeddings for "
                        f"{len(chunk)} texts",
                        model_name=model_name,
                    )
                vectors.extend(item.embedding for item in response.data)
            return vectors

        def width() -> Optional[int]:
            if model_name in KNOWN_MODEL_DIMENSIONS:
                return KNOWN_MODEL_DIMENSIONS[model_name]
            return len(embed_texts(["dimension probe"])[0])

        self._backend_widths["openai"] = width
        self._backend_models["openai"] = model_name
        return embed_texts

    def _create_huggingface_function(self) -> Callable:
        """Create Hugging Face embedding function."""
        import torch

        from iris_vector_rag.common.huggingface_utils import download_huggingface_model

        hf_config = self.embedding_config.get("huggingface", {})
        model_name = hf_config.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )

        tokenizer, model = download_huggingface_model(model_name)

        def embed_texts(texts: List[str]) -> List[List[float]]:
            encoded_input = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = model(**encoded_input)
                embeddings = _mean_pool(
                    model_output.last_hidden_state, encoded_input["attention_mask"]
                )
            return cast(List[List[float]], embeddings.tolist())

        def width() -> Optional[int]:
            hidden = getattr(getattr(model, "config", None), "hidden_size", None)
            return int(hidden) if hidden else None

        self._backend_widths["huggingface"] = width
        self._backend_models["huggingface"] = model_name
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

    def _backend_order(self) -> List[str]:
        order = [self.primary_backend]
        order.extend(b for b in self.fallback_backends if b not in order)
        return order

    def _run_backend(self, backend_name: str, texts: List[str]) -> List[List[float]]:
        """One backend over all `texts`, in `batch_size` slices when configured."""
        function = self._embedding_functions[backend_name]
        batch_size = self.embedding_config.get("batch_size")
        if not batch_size or len(texts) <= int(batch_size):
            return function(texts)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), int(batch_size)):
            vectors.extend(function(texts[start : start + int(batch_size)]))
        return vectors

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with fallback support.

        The primary backend first, then each configured fallback in order. Raises
        `EmbeddingError` naming every backend tried and why it failed when none produces
        vectors of the dimension `get_embedding_dimension()` reports.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        expected = self.get_embedding_dimension()
        failures: List[str] = []
        for backend_name in self._backend_order():
            if backend_name not in self._embedding_functions:
                if not self._initialize_backend(backend_name):
                    failures.append(
                        f"{backend_name}: {self._backend_errors[backend_name]}"
                    )
                    continue
            try:
                vectors = self._run_backend(backend_name, texts)
                self._check_width(backend_name, vectors, expected)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.warning(f"Embedding backend {backend_name} failed: {reason}")
                failures.append(f"{backend_name}: {reason}")
                continue
            if backend_name != self.primary_backend:
                logger.warning(
                    f"Primary backend {self.primary_backend} unavailable; "
                    f"embedded with fallback backend {backend_name}"
                )
            return vectors

        raise EmbeddingError(
            "No embedding backend produced vectors. "
            + "; ".join(failures)
            + f". {_OFFLINE_HINT}",
            backends=",".join(self._backend_order()),
            text_count=len(texts),
        )

    @staticmethod
    def _check_width(
        backend_name: str, vectors: List[List[float]], expected: int
    ) -> None:
        """A backend whose vectors are not the manager's one dimension has failed."""
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    f"backend {backend_name} returned a {len(vector)}-dimensional vector "
                    f"but this manager's embedding dimension is {expected}",
                    backend=backend_name,
                )

    def _probe_dimension(self) -> int:
        """The first backend that can say what width it produces decides."""
        failures: List[str] = []
        for backend_name in self._backend_order():
            if backend_name not in self._embedding_functions:
                if not self._initialize_backend(backend_name):
                    failures.append(
                        f"{backend_name}: {self._backend_errors[backend_name]}"
                    )
                    continue
            try:
                width = self._backend_widths[backend_name]()
            except Exception as e:
                failures.append(f"{backend_name}: {type(e).__name__}: {e}")
                continue
            if width:
                configured = self.embedding_config.get("dimension")
                if configured and int(configured) != width:
                    raise EmbeddingError(
                        f"embeddings.dimension is {configured} but backend "
                        f"{backend_name} ({self._backend_models.get(backend_name)}) "
                        f"produces {width}-dimensional vectors. Remove the setting or "
                        f"set it to {width}.",
                        configured=configured,
                        actual=width,
                    )
                return int(width)
            failures.append(f"{backend_name}: reported no dimension")
        raise EmbeddingError(
            "Cannot determine the embedding dimension: "
            + "; ".join(failures)
            + f". {_OFFLINE_HINT}"
        )

    def get_embedding_dimension(self) -> int:
        """
        The dimension of the vectors this manager produces: the active backend's real
        output width, computed once. `embeddings.dimension` is an override checked
        against it, never a substitute for it.

        Raises:
            EmbeddingError: no backend can report its width, or the override disagrees.
        """
        if self._dimension is None:
            self._dimension = self._probe_dimension()
        return self._dimension

    @property
    def model_key(self) -> str:
        """The embedding identity of the active backend's model (`stub` for the test embedder)."""
        for backend_name in self._backend_order():
            if backend_name in self._backend_models:
                return self._backend_models[backend_name]
        if self._initialize_backend(self.primary_backend):
            return self._backend_models[self.primary_backend]
        raise EmbeddingError(
            f"No embedding model is available: {self._backend_errors.get(self.primary_backend)}"
        )

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
        self._dimension = None
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
                    total_embeddings=0,
                )

            instance = _CACHE_STATS[config_name]
            total_calls = instance.cache_hits + instance.cache_misses
            hit_rate = instance.cache_hits / total_calls if total_calls > 0 else 0.0
            avg_time = (
                instance.total_embedding_time_ms / instance.embedding_calls
                if instance.embedding_calls > 0
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
                total_embeddings=instance.total_embeddings_generated,
            )
        else:
            # Aggregate stats for all configs
            total_hits = sum(inst.cache_hits for inst in _CACHE_STATS.values())
            total_misses = sum(inst.cache_misses for inst in _CACHE_STATS.values())
            total_embeddings = sum(
                inst.total_embeddings_generated for inst in _CACHE_STATS.values()
            )
            total_memory = sum(inst.memory_usage_mb for inst in _CACHE_STATS.values())
            total_calls = total_hits + total_misses
            hit_rate = total_hits / total_calls if total_calls > 0 else 0.0
            timed_calls = sum(inst.embedding_calls for inst in _CACHE_STATS.values())
            total_time = sum(
                inst.total_embedding_time_ms for inst in _CACHE_STATS.values()
            )

            return CacheStatistics(
                config_name="<all>",
                cache_hits=total_hits,
                cache_misses=total_misses,
                hit_rate=hit_rate,
                avg_embedding_time_ms=total_time / timed_calls if timed_calls else 0.0,
                model_load_count=total_misses,
                memory_usage_mb=total_memory,
                device="mixed",
                total_embeddings=total_embeddings,
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

            # Cache keys are "model_name:device", and the device is whatever string the
            # loader was given ("cuda", "cuda:0", "mps", "cpu", ...): match on the model.
            cache_keys_to_remove = [
                key
                for key in _SENTENCE_TRANSFORMER_CACHE
                if key.startswith(f"{model_name}:")
            ]

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
                models_cleared=models_cleared, memory_freed_mb=memory_freed
            )
        else:
            # Clear all models
            models_cleared = len(_SENTENCE_TRANSFORMER_CACHE)
            _SENTENCE_TRANSFORMER_CACHE.clear()
            logger.info(f"Cleared all cached models ({models_cleared} total)")

            with _STATS_LOCK:
                memory_freed = sum(
                    inst.memory_usage_mb for inst in _CACHE_STATS.values()
                )
                _CACHE_STATS.clear()

            return ClearCacheResult(
                models_cleared=models_cleared, memory_freed_mb=memory_freed
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
                memory_usage_mb=400.0,  # Approximate model size
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
            _CACHE_STATS[config_name].embedding_calls += 1
