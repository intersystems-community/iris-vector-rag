"""
HuggingFace model download utilities with rate limiting and retry logic.
"""

import logging
import random
import time
from functools import wraps
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if this is a rate limiting error
                    error_str = str(e).lower()
                    is_rate_limit = any(
                        indicator in error_str
                        for indicator in [
                            "rate limit",
                            "too many requests",
                            "429",
                            "quota exceeded",
                            "service unavailable",
                            "503",
                            "timeout",
                        ]
                    )

                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise last_exception

                    if is_rate_limit:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (exponential_base**attempt), max_delay)

                        # Add jitter to prevent thundering herd
                        if jitter:
                            delay *= 0.5 + random.random() * 0.5

                        logger.warning(
                            f"Rate limit detected (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.2f} seconds: {e}"
                        )
                        time.sleep(delay)
                    else:
                        # For non-rate-limit errors, fail immediately
                        logger.error(f"Non-rate-limit error encountered: {e}")
                        raise e

            raise last_exception

        return wrapper

    return decorator


@retry_with_exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
def download_huggingface_model(
    model_name: str, trust_remote_code: bool = False, **kwargs
) -> Tuple[Any, Any]:
    """
    Download HuggingFace model and tokenizer with retry logic.

    Args:
        model_name: Name of the model to download
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for model loading

    Returns:
        Tuple of (tokenizer, model)
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Downloading HuggingFace model: {model_name}")

        # Download tokenizer first
        logger.debug(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **kwargs
        )

        # Download model
        logger.debug(f"Loading model for {model_name}")
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **kwargs
        )

        logger.info(f"Successfully downloaded HuggingFace model: {model_name}")
        return tokenizer, model

    except ImportError as e:
        logger.error(f"transformers library not available: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise


def get_cached_model(
    model_name: str, cache_dict: dict, trust_remote_code: bool = False, **kwargs
) -> Tuple[Any, Any]:
    """
    Get model from cache or download if not cached.

    Args:
        model_name: Name of the model
        cache_dict: Dictionary to use for caching
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for model loading

    Returns:
        Tuple of (tokenizer, model)
    """
    if model_name not in cache_dict:
        logger.info(f"Model {model_name} not in cache, downloading...")
        tokenizer, model = download_huggingface_model(
            model_name, trust_remote_code=trust_remote_code, **kwargs
        )
        cache_dict[model_name] = (tokenizer, model)
        logger.info(f"Cached model {model_name}")
    else:
        logger.info(f"Using cached model {model_name}")
        tokenizer, model = cache_dict[model_name]

    return tokenizer, model


def clear_model_cache(cache_dict: dict, model_name: Optional[str] = None):
    """
    Clear model cache.

    Args:
        cache_dict: Dictionary containing cached models
        model_name: Specific model to clear, or None to clear all
    """
    if model_name:
        if model_name in cache_dict:
            del cache_dict[model_name]
            logger.info(f"Cleared cache for model {model_name}")
    else:
        cache_dict.clear()
        logger.info("Cleared all model cache")


# Global cache for models
_global_model_cache = {}


def get_global_cached_model(
    model_name: str, trust_remote_code: bool = False, **kwargs
) -> Tuple[Any, Any]:
    """
    Get model from global cache or download if not cached.

    Args:
        model_name: Name of the model
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for model loading

    Returns:
        Tuple of (tokenizer, model)
    """
    return get_cached_model(
        model_name, _global_model_cache, trust_remote_code, **kwargs
    )


def clear_global_cache(model_name: Optional[str] = None):
    """
    Clear global model cache.

    Args:
        model_name: Specific model to clear, or None to clear all
    """
    clear_model_cache(_global_model_cache, model_name)
