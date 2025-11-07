"""
IRIS EMBEDDING Integration Layer.

Feature: 051-add-native-iris
Purpose: Integrate with IRIS %Embedding.Config and provide Python embedding
         functions callable by IRIS %Embedding.SentenceTransformers class.

This module bridges IRIS EMBEDDING columns with the Python embedding cache,
solving the 720x slowdown issue (DP-442038) by caching models in memory.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import logging

from ..config.embedding_config import EmbeddingConfig, create_embedding_config
from .manager import (
    _get_cached_sentence_transformer,
    _record_cache_hit,
    _record_cache_miss,
    _record_embeddings_generated,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Result of embedding generation with performance metrics.
    
    Returned by embed_texts() to provide embeddings along with
    cache hit/miss information and timing data.
    """
    embeddings: List[List[float]]
    cache_hit: bool
    embedding_time_ms: float
    model_load_time_ms: float
    device_used: str
    
    def __post_init__(self):
        """Validate embedding result."""
        if not self.embeddings:
            raise ValueError("embeddings must not be empty")
        if self.embedding_time_ms < 0:
            raise ValueError("embedding_time_ms must be non-negative")
        if self.model_load_time_ms < 0:
            raise ValueError("model_load_time_ms must be non-negative")


# ============================================================================
# Configuration Storage (simulated - in production, queries %Embedding.Config)
# ============================================================================
_CONFIG_STORE: Dict[str, EmbeddingConfig] = {}


def configure_embedding(
    name: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_cache_path: str = "/var/lib/huggingface",
    python_path: str = "/usr/bin/python3",
    description: str = "",
    enable_entity_extraction: bool = False,
    entity_types: Optional[List[str]] = None,
    batch_size: int = 32,
    device_preference: str = "auto",
) -> EmbeddingConfig:
    """
    Create embedding configuration (simulates INSERT into %Embedding.Config).
    
    In production, this would insert into IRIS %Embedding.Config table.
    For testing/development, stores in-memory.
    
    Args:
        name: Unique configuration identifier
        model_name: HuggingFace model ID
        hf_cache_path: Path to HuggingFace cache directory
        python_path: Path to Python executable
        description: Human-readable description
        enable_entity_extraction: Whether to extract entities
        entity_types: List of entity types to extract
        batch_size: Documents per batch
        device_preference: GPU preference (cuda, mps, cpu, auto)
    
    Returns:
        EmbeddingConfig instance
    
    Example:
        >>> config = configure_embedding(
        ...     name="medical_embeddings",
        ...     enable_entity_extraction=True,
        ...     entity_types=["Disease", "Medication"]
        ... )
    """
    config = create_embedding_config(
        name=name,
        model_name=model_name,
        hf_cache_path=hf_cache_path,
        python_path=python_path,
        description=description,
        enable_entity_extraction=enable_entity_extraction,
        entity_types=entity_types or [],
        batch_size=batch_size,
        device_preference=device_preference,
    )
    
    # Store configuration (in production, INSERT into %Embedding.Config)
    _CONFIG_STORE[name] = config
    logger.info(f"Created embedding configuration: {name}")
    
    return config


def get_config(config_name: str) -> EmbeddingConfig:
    """
    Read EMBEDDING configuration from IRIS %Embedding.Config table.
    
    In production, this queries IRIS:
        SELECT Configuration FROM %Embedding.Config WHERE Name = :config_name
    
    For testing/development, reads from in-memory store.
    
    Args:
        config_name: Name of configuration to retrieve
    
    Returns:
        EmbeddingConfig instance
    
    Raises:
        ValueError: If configuration not found (CONFIG_NOT_FOUND)
    
    Example:
        >>> config = get_config("medical_embeddings_v1")
        >>> print(config.model_name)
        sentence-transformers/all-MiniLM-L6-v2
    """
    if config_name not in _CONFIG_STORE:
        raise ValueError(
            f"CONFIG_NOT_FOUND: Embedding configuration '{config_name}' "
            f"not found in %Embedding.Config"
        )
    
    return _CONFIG_STORE[config_name]


def _detect_device(config: EmbeddingConfig) -> str:
    """
    Detect actual device to use based on device preference and availability.
    
    Args:
        config: EmbeddingConfig with device_preference setting
    
    Returns:
        Device string: "cuda:0", "mps", or "cpu"
    """
    try:
        import torch
        
        if config.device_preference == "cuda":
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        
        elif config.device_preference == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
        
        elif config.device_preference == "cpu":
            return "cpu"
        
        elif config.device_preference == "auto":
            # Auto-detect: CUDA > MPS > CPU
            if torch.cuda.is_available():
                return "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        else:
            logger.warning(f"Unknown device preference: {config.device_preference}, using CPU")
            return "cpu"
    
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        return "cpu"


def embed_texts(config_name: str, texts: List[str]) -> EmbeddingResult:
    """
    Generate embeddings for text using cached model (FR-001, FR-002).

    This is the core function called by IRIS EMBEDDING columns to vectorize text.
    Implements model caching to solve the 720x slowdown (DP-442038).

    Performance targets:
        - Cache hit: <50ms for batch of 32
        - Cache miss: <5000ms (includes model load)
        - Cache hit rate: >=95% after warmup

    Args:
        config_name: Name of embedding configuration
        texts: List of texts to embed (1-1000 texts)

    Returns:
        EmbeddingResult with embeddings and performance metrics

    Raises:
        ValueError: If texts empty or config not found

    Example:
        >>> result = embed_texts("medical_embeddings", ["Patient has diabetes"])
        >>> print(f"Embedding dimension: {len(result.embeddings[0])}")
        Embedding dimension: 384
        >>> print(f"Cache hit: {result.cache_hit}")
        Cache hit: True
    """
    # Structured logging: Function entry
    logger.debug(
        "EMBEDDING_REQUEST",
        extra={
            "config_name": config_name,
            "num_texts": len(texts),
            "operation": "embed_texts",
            "feature": "051_iris_embedding",
        }
    )

    # Validate inputs
    if not texts:
        logger.error(
            "EMBEDDING_VALIDATION_ERROR: Empty texts list",
            extra={"config_name": config_name, "error_type": "EMPTY_TEXT"}
        )
        raise ValueError("EMPTY_TEXT: texts list must not be empty")

    for i, text in enumerate(texts):
        if not text or not text.strip():
            logger.error(
                "EMBEDDING_VALIDATION_ERROR: Empty text at index",
                extra={
                    "config_name": config_name,
                    "text_index": i,
                    "error_type": "EMPTY_TEXT"
                }
            )
            raise ValueError(f"EMPTY_TEXT: Text at index {i} is empty")

    # Load configuration
    try:
        config = get_config(config_name)
        logger.debug(
            "EMBEDDING_CONFIG_LOADED",
            extra={
                "config_name": config_name,
                "model_name": config.model_name,
                "device_preference": config.device_preference,
            }
        )
    except ValueError as e:
        logger.error(
            "EMBEDDING_CONFIG_ERROR",
            extra={"config_name": config_name, "error": str(e)}
        )
        raise

    # Detect device
    device = _detect_device(config)
    logger.info(
        "EMBEDDING_DEVICE_SELECTED",
        extra={
            "config_name": config_name,
            "device": device,
            "device_preference": config.device_preference,
        }
    )

    # Track timing
    total_start = time.time()
    model_load_time_ms = 0.0
    cache_hit = False
    
    # Build cache key
    cache_key = f"{config.model_name}:{device}"
    
    # Check if model is cached
    from .manager import _SENTENCE_TRANSFORMER_CACHE
    
    if cache_key in _SENTENCE_TRANSFORMER_CACHE:
        cache_hit = True
        _record_cache_hit(config_name)
    else:
        # Model not cached - will load from disk
        cache_hit = False
        load_start = time.time()
    
    # Get or load model (uses double-checked locking in manager)
    try:
        model = _get_cached_sentence_transformer(config.model_name, device)

        if not cache_hit:
            # Record model load time
            model_load_time_ms = (time.time() - load_start) * 1000
            _record_cache_miss(config_name, device, model_load_time_ms)
            logger.info(
                "EMBEDDING_MODEL_LOADED",
                extra={
                    "config_name": config_name,
                    "model_name": config.model_name,
                    "device": device,
                    "load_time_ms": model_load_time_ms,
                    "cache_hit": False,
                }
            )

    except Exception as e:
        logger.error(
            "EMBEDDING_MODEL_LOAD_FAILED",
            extra={
                "config_name": config_name,
                "model_name": config.model_name,
                "device": device,
                "error": str(e),
            }
        )
        raise ValueError(f"MODEL_LOAD_FAILED: Failed to load model: {e}") from e

    # Generate embeddings
    try:
        embeddings = model.encode(texts, convert_to_tensor=False)
        embeddings_list = embeddings.tolist()

        # Record embeddings generated
        _record_embeddings_generated(config_name, len(texts))

        logger.debug(
            "EMBEDDING_GENERATION_SUCCESS",
            extra={
                "config_name": config_name,
                "num_texts": len(texts),
                "device": device,
                "cache_hit": cache_hit,
            }
        )

    except Exception as e:
        # Check if GPU OOM error
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            logger.warning(
                "EMBEDDING_GPU_OOM",
                extra={
                    "config_name": config_name,
                    "device": device,
                    "num_texts": len(texts),
                    "error": str(e),
                    "fallback": "cpu",
                }
            )

            # Try CPU fallback
            try:
                model_cpu = _get_cached_sentence_transformer(config.model_name, "cpu")
                embeddings = model_cpu.encode(texts, convert_to_tensor=False)
                embeddings_list = embeddings.tolist()
                device = "cpu"

                logger.info(
                    "EMBEDDING_CPU_FALLBACK_SUCCESS",
                    extra={
                        "config_name": config_name,
                        "num_texts": len(texts),
                        "original_device": "gpu",
                        "fallback_device": "cpu",
                    }
                )

            except Exception as cpu_error:
                logger.error(
                    "EMBEDDING_CPU_FALLBACK_FAILED",
                    extra={
                        "config_name": config_name,
                        "num_texts": len(texts),
                        "gpu_error": str(e),
                        "cpu_error": str(cpu_error),
                    }
                )
                raise ValueError(
                    f"EMBEDDING_FAILED: Failed on both GPU and CPU: {cpu_error}"
                ) from cpu_error
        else:
            logger.error(
                "EMBEDDING_GENERATION_FAILED",
                extra={
                    "config_name": config_name,
                    "device": device,
                    "num_texts": len(texts),
                    "error": str(e),
                }
            )
            raise ValueError(f"EMBEDDING_FAILED: {e}") from e

    # Calculate total time
    total_time_ms = (time.time() - total_start) * 1000
    embedding_time_ms = total_time_ms - model_load_time_ms

    # Structured logging: Performance metrics
    logger.info(
        "EMBEDDING_PERFORMANCE_METRICS",
        extra={
            "config_name": config_name,
            "num_texts": len(texts),
            "cache_hit": cache_hit,
            "device": device,
            "total_time_ms": total_time_ms,
            "embedding_time_ms": embedding_time_ms,
            "model_load_time_ms": model_load_time_ms,
            "feature": "051_iris_embedding",
        }
    )

    # Legacy log messages for backward compatibility
    if cache_hit:
        logger.debug(
            f"Generated {len(texts)} embeddings in {embedding_time_ms:.1f}ms "
            f"(cache hit, device: {device})"
        )
    else:
        logger.info(
            f"Generated {len(texts)} embeddings in {total_time_ms:.1f}ms "
            f"(model load: {model_load_time_ms:.1f}ms, "
            f"embedding: {embedding_time_ms:.1f}ms, device: {device})"
        )
    
    return EmbeddingResult(
        embeddings=embeddings_list,
        cache_hit=cache_hit,
        embedding_time_ms=embedding_time_ms,
        model_load_time_ms=model_load_time_ms,
        device_used=device,
    )


def embed_text(config_name: str, text: str) -> List[float]:
    """
    Generate embedding for single text (convenience method).
    
    Args:
        config_name: Name of embedding configuration
        text: Text to embed
    
    Returns:
        Embedding vector as list of floats
    
    Example:
        >>> embedding = embed_text("medical_embeddings", "Patient has diabetes")
        >>> len(embedding)
        384
    """
    result = embed_texts(config_name, [text])
    return result.embeddings[0]


# ============================================================================
# IRIS Integration Points
# ============================================================================

def iris_embedding_callback(config_name: str, text: str) -> List[float]:
    """
    Callback function for IRIS %Embedding.SentenceTransformers.
    
    This is the function IRIS calls when EMBEDDING columns auto-vectorize text.
    
    IRIS Usage:
        CREATE TABLE documents (
            content VARCHAR(5000),
            content_vector EMBEDDING
                REFERENCES %Embedding.Config('medical_embeddings')
                USING content
        );
    
    When rows are inserted, IRIS calls:
        iris_embedding_callback('medical_embeddings', row.content)
    
    Args:
        config_name: Configuration name from EMBEDDING column definition
        text: Text content to vectorize
    
    Returns:
        Embedding vector as list of floats
    """
    return embed_text(config_name, text)


def iris_batch_embedding_callback(
    config_name: str,
    texts: List[str]
) -> List[List[float]]:
    """
    Batch callback function for optimized bulk vectorization.
    
    Used during bulk INSERT operations for better performance.
    
    Args:
        config_name: Configuration name
        texts: List of texts to vectorize
    
    Returns:
        List of embedding vectors
    """
    result = embed_texts(config_name, texts)
    return result.embeddings
