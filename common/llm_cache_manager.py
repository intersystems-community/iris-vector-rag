"""
LLM Cache Manager

This module provides the main cache management functionality, integrating
with Langchain's caching system and coordinating between different cache backends.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
from langchain_core.outputs import Generation

from common.llm_cache_config import CacheConfig, load_cache_config
from common.llm_cache_iris import IRISCacheBackend, create_iris_cache_backend

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    avg_response_time_cached: float = 0.0
    avg_response_time_uncached: float = 0.0

    def record_hit(self, response_time: float = 0.0) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.total_requests += 1
        self._update_hit_rate()
        if response_time > 0:
            self._update_cached_response_time(response_time)

    def record_miss(self, response_time: float = 0.0) -> None:
        """Record a cache miss."""
        self.misses += 1
        self.total_requests += 1
        self._update_hit_rate()
        if response_time > 0:
            self._update_uncached_response_time(response_time)

    def _update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        if self.total_requests > 0:
            self.hit_rate = self.hits / self.total_requests

    def _update_cached_response_time(self, response_time: float) -> None:
        """Update average cached response time."""
        if self.hits > 1:
            self.avg_response_time_cached = (
                self.avg_response_time_cached * (self.hits - 1) + response_time
            ) / self.hits
        else:
            self.avg_response_time_cached = response_time

    def _update_uncached_response_time(self, response_time: float) -> None:
        """Update average uncached response time."""
        if self.misses > 1:
            self.avg_response_time_uncached = (
                self.avg_response_time_uncached * (self.misses - 1) + response_time
            ) / self.misses
        else:
            self.avg_response_time_uncached = response_time


class LangchainCacheManager:
    """Manages Langchain cache configuration and lifecycle."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.metrics = CacheMetrics()
        self.cache_backend = None
        self._langchain_cache_configured = False

    def setup_cache(self) -> Optional[Any]:
        """Configure Langchain's global cache."""
        if not self.config.enabled:
            logger.info("LLM caching is disabled")
            return None

        try:
            import langchain
            from langchain_community.cache import InMemoryCache

            if self.config.backend == "memory":
                cache = InMemoryCache()
                langchain.llm_cache = cache
                logger.info("Langchain memory cache configured")

            elif self.config.backend == "iris":
                # Try to reuse existing IRIS connection first, fallback to URL-based connection
                try:
                    iris_connector = self._get_iris_connection_for_cache()

                    # Validate connection before creating cache backend
                    if iris_connector is None:
                        raise ConnectionError(
                            "Failed to setup IRIS cache table: _handle is NULL"
                        )

                    self.cache_backend = create_iris_cache_backend(
                        self.config, iris_connector
                    )

                    # Create Langchain-compatible cache wrapper
                    cache = LangchainIRISCacheWrapper(self.cache_backend)
                    langchain.llm_cache = cache
                    logger.info("Langchain IRIS cache configured")

                except Exception as e:
                    logger.error(f"Failed to setup IRIS cache table: {e}")
                    if self.config.graceful_fallback:
                        logger.info(
                            "Falling back to memory cache due to IRIS connection failure"
                        )
                        cache = InMemoryCache()
                        langchain.llm_cache = cache
                        logger.info("Langchain memory cache configured as fallback")
                    else:
                        raise

            else:
                logger.error(
                    f"Unsupported cache backend: {self.config.backend}. Supported backends: memory, iris"
                )
                return None

            self._langchain_cache_configured = True
            return cache

        except Exception as e:
            logger.error(f"Failed to setup Langchain cache: {e}")
            if self.config.graceful_fallback:
                logger.info("Continuing without cache due to graceful fallback")
                return None
            raise

    def _get_iris_connection_for_cache(self):
        """
        Get IRIS connection for cache using DBAPI interface.

        Returns:
            IRIS connection object
        """
        # Always use DBAPI for cache to avoid JDBC compatibility issues
        try:
            from common.iris_dbapi_connector import get_iris_dbapi_connection

            iris_connector = get_iris_dbapi_connection()

            # Validate the connection handle
            if iris_connector is None:
                raise ConnectionError("DBAPI connection returned NULL handle")

            logger.info("Using DBAPI IRIS connection for cache")
            return iris_connector
        except Exception as e:
            logger.debug(f"Could not get DBAPI connection: {e}")

        # Fallback to URL-based connection (original behavior)
        try:
            from common.utils import get_iris_connector

            iris_connector = get_iris_connector()

            # Validate the connection handle
            if iris_connector is None:
                raise ConnectionError("URL-based connection returned NULL handle")

            logger.info("Using URL-based IRIS connection for cache")
            return iris_connector
        except Exception as e:
            logger.error(f"Failed to establish IRIS connection for cache: {e}")
            raise

    def _setup_iris_fallback(self) -> Optional[Any]:
        """Setup IRIS cache as fallback."""
        try:
            iris_connector = self._get_iris_connection_for_cache()

            self.cache_backend = create_iris_cache_backend(self.config, iris_connector)
            cache = LangchainIRISCacheWrapper(self.cache_backend)

            import langchain

            langchain.llm_cache = cache

            logger.info("IRIS cache configured as fallback")
            return cache

        except Exception as e:
            logger.error(f"IRIS fallback cache setup failed: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = {
            "enabled": self.config.enabled,
            "backend": self.config.backend,
            "configured": self._langchain_cache_configured,
            "metrics": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "total_requests": self.metrics.total_requests,
                "hit_rate": self.metrics.hit_rate,
                "avg_response_time_cached": self.metrics.avg_response_time_cached,
                "avg_response_time_uncached": self.metrics.avg_response_time_uncached,
            },
        }

        if self.cache_backend:
            stats["backend_stats"] = self.cache_backend.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        if self.cache_backend:
            self.cache_backend.clear()

        # Also clear Langchain cache if configured
        try:
            import langchain

            if hasattr(langchain, "llm_cache") and langchain.llm_cache:
                if hasattr(langchain.llm_cache, "clear"):
                    langchain.llm_cache.clear()
        except Exception as e:
            logger.warning(f"Failed to clear Langchain cache: {e}")


class LangchainIRISCacheWrapper:
    """Wrapper to make IRIS cache backend compatible with Langchain."""

    def __init__(self, iris_cache_backend: IRISCacheBackend):
        self.backend = iris_cache_backend

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """Langchain cache lookup interface."""
        cache_key = self._generate_langchain_key(prompt, llm_string)
        result = self.backend.get(cache_key)

        if result:
            # Extract the response from cached data
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            elif isinstance(result, str):
                return result

        return None

    async def alookup(
        self, prompt: str, llm_string: str
    ) -> Optional[List["Generation"]]:
        """
        Asynchronous Langchain cache lookup interface.

        Args:
            prompt: The input prompt
            llm_string: The LLM string identifier

        Returns:
            Optional[List[Generation]]: List of Generation objects if found, None otherwise
        """
        try:
            # Run the synchronous lookup logic in a separate thread
            result = await asyncio.to_thread(self._lookup_sync, prompt, llm_string)
            return result
        except Exception as e:
            logger.error(f"Error during async cache lookup: {e}")
            return None

    def _lookup_sync(
        self, prompt: str, llm_string: str
    ) -> Optional[List["Generation"]]:
        """
        Synchronous helper method for database lookup logic.

        Args:
            prompt: The input prompt
            llm_string: The LLM string identifier

        Returns:
            Optional[List[Generation]]: List of Generation objects if found, None otherwise
        """
        try:
            # Import Generation and load here to avoid circular imports
            from langchain_core.outputs import Generation
            from langchain_core.load import load

            cache_key = self._generate_langchain_key(prompt, llm_string)
            result = self.backend.get(cache_key)

            if result:
                # Extract the response from cached data
                if isinstance(result, dict) and "response" in result:
                    response_data = result["response"]
                elif isinstance(result, str):
                    response_data = result
                else:
                    return None

                # Parse the JSON string containing the list of generations
                try:
                    generations_data = json.loads(response_data)

                    # Ensure it's a list
                    if not isinstance(generations_data, list):
                        raise TypeError(
                            f"Expected list of generations, got {type(generations_data)}"
                        )

                    # Convert each generation dict back to Generation object
                    generations = []
                    for gen_data in generations_data:
                        if isinstance(gen_data, dict):
                            try:
                                # Check if it's a langchain serialized format (has 'lc' and 'kwargs')
                                if "lc" in gen_data and "kwargs" in gen_data:
                                    # Use langchain's load function to reconstruct
                                    generation = load(gen_data)
                                    generations.append(generation)
                                elif "text" in gen_data and "type" in gen_data:
                                    # Handle fallback string representation
                                    logger.warning(
                                        f"Loading generation from string fallback: {gen_data['type']}"
                                    )
                                    generation = Generation(text=gen_data["text"])
                                    generations.append(generation)
                                else:
                                    # Direct construction from kwargs
                                    generation = Generation(**gen_data)
                                    generations.append(generation)
                            except Exception as load_error:
                                logger.warning(
                                    f"Failed to load generation from cache data, skipping: {load_error}"
                                )
                                # Try to create a basic Generation object with available text
                                if "text" in gen_data:
                                    generation = Generation(text=gen_data["text"])
                                    generations.append(generation)
                                elif isinstance(gen_data, str):
                                    generation = Generation(text=gen_data)
                                    generations.append(generation)
                        else:
                            logger.warning(
                                f"Expected dict for generation data, got {type(gen_data)}, attempting string conversion"
                            )
                            generation = Generation(text=str(gen_data))
                            generations.append(generation)

                    return generations

                except json.JSONDecodeError as e:
                    logger.error("Error during cache lookup JSON decode: %s", e)
                    return None
                except (TypeError, AttributeError, ValueError) as e:
                    logger.error("Error during cache lookup data processing: %s", e)
                    return None
                except Exception as e:
                    # Catch ValidationError and other pydantic errors
                    logger.error("Error during cache lookup: %s", e)
                    return None

            return None

        except Exception as e:
            logger.error("Error during async cache lookup: %s", e)
            return None

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: List["Generation"]
    ) -> None:
        """
        Asynchronous Langchain cache update interface.

        Args:
            prompt: The input prompt
            llm_string: The LLM string identifier
            return_val: List of Generation objects to cache
        """
        try:
            # Run the synchronous update logic in a separate thread
            await asyncio.to_thread(self._update_sync, prompt, llm_string, return_val)
        except Exception as e:
            logger.error(f"Error during async cache update: {e}")

    def _update_sync(
        self, prompt: str, llm_string: str, return_val: List["Generation"]
    ) -> None:
        """
        Synchronous helper method for database update logic.

        Args:
            prompt: The input prompt
            llm_string: The LLM string identifier
            return_val: List of Generation objects to cache
        """
        try:
            import json
            from langchain_core.load import dumpd
            from langchain_core.messages import BaseMessage

            # Convert Generation objects to serializable format with robust error handling
            generations_data = []

            for i, generation in enumerate(return_val):
                try:
                    # Log generation details for debugging
                    logger.debug(
                        f"Serializing generation {i}: {type(generation).__name__}"
                    )

                    # Use Langchain's dumpd function for proper serialization
                    gen_data = dumpd(generation)

                    # Additional validation: ensure the result is JSON serializable
                    # This catches cases where dumpd succeeds but produces non-serializable objects
                    json.dumps(gen_data)

                    # Special handling for ChatGeneration with BaseMessage that might not be fully serialized
                    if hasattr(generation, "message") and isinstance(
                        generation.message, BaseMessage
                    ):
                        # Double-check that the message in the serialized data is also serializable
                        if "message" in gen_data and isinstance(
                            gen_data["message"], BaseMessage
                        ):
                            logger.debug(
                                f"Re-serializing BaseMessage in generation {i}"
                            )
                            gen_data["message"] = dumpd(generation.message)
                            # Validate again
                            json.dumps(gen_data)

                    generations_data.append(gen_data)
                    logger.debug(f"Successfully serialized generation {i}")

                except Exception as e:
                    logger.warning(
                        f"Failed to serialize generation {i} with dumpd: {e}"
                    )
                    logger.debug(
                        f"Generation {i} type: {type(generation)}, attributes: {dir(generation)}"
                    )

                    # Fallback 1: Try dict method if available
                    try:
                        if hasattr(generation, "dict"):
                            gen_data = generation.dict()

                            # Handle BaseMessage in dict output
                            if "message" in gen_data and hasattr(
                                gen_data["message"], "dict"
                            ):
                                gen_data["message"] = gen_data["message"].dict()

                            # Validate JSON serialization
                            json.dumps(gen_data)
                            generations_data.append(gen_data)
                            logger.debug(
                                f"Successfully serialized generation {i} using dict fallback"
                            )
                            continue

                    except Exception as dict_error:
                        logger.warning(
                            f"Dict fallback failed for generation {i}: {dict_error}"
                        )

                    # Fallback 2: Manual extraction of key attributes
                    try:
                        gen_data = {
                            "text": getattr(generation, "text", str(generation)),
                            "type": type(generation).__name__,
                            "generation_info": getattr(
                                generation, "generation_info", None
                            ),
                        }

                        # Handle message attribute for ChatGeneration
                        if hasattr(generation, "message"):
                            message = generation.message
                            if hasattr(message, "content"):
                                gen_data["message"] = {
                                    "content": getattr(message, "content", ""),
                                    "type": type(message).__name__,
                                }
                            else:
                                gen_data["message"] = str(message)

                        # Validate JSON serialization
                        json.dumps(gen_data)
                        generations_data.append(gen_data)
                        logger.debug(
                            f"Successfully serialized generation {i} using manual extraction"
                        )

                    except Exception as manual_error:
                        logger.error(
                            f"Manual extraction failed for generation {i}: {manual_error}"
                        )

                        # Fallback 3: Minimal safe representation
                        gen_data = {
                            "text": str(generation) if generation is not None else "",
                            "type": type(generation).__name__,
                            "error": "serialization_failed",
                            "error_details": str(e),
                        }
                        generations_data.append(gen_data)
                        logger.warning(
                            f"Using minimal safe representation for generation {i}"
                        )

            # Final validation: ensure the entire list is JSON serializable
            try:
                serialized_generations = json.dumps(generations_data)
                logger.debug(
                    f"Successfully serialized {len(generations_data)} generations to JSON ({len(serialized_generations)} chars)"
                )
            except Exception as final_error:
                logger.error(f"Final JSON serialization failed: {final_error}")
                # Create a safe fallback representation
                safe_generations = []
                for i, gen_data in enumerate(generations_data):
                    try:
                        json.dumps(gen_data)
                        safe_generations.append(gen_data)
                    except:
                        safe_generations.append(
                            {
                                "text": f"Generation {i} serialization failed",
                                "type": "unknown",
                                "error": "json_serialization_failed",
                            }
                        )
                serialized_generations = json.dumps(safe_generations)
                logger.warning(
                    f"Used safe fallback serialization for {len(safe_generations)} generations"
                )

            # Use the existing update method with serialized data
            self.update(prompt, llm_string, serialized_generations)
            logger.debug("Cache update completed successfully")

        except Exception as e:
            logger.error(f"Error during sync cache update: {e}")
            # Don't re-raise the exception to avoid breaking the LLM call

    def update(
        self, prompt: str, llm_string: str, return_val: Union[str, List["Generation"]]
    ) -> None:
        """Langchain cache update interface."""
        cache_key = self._generate_langchain_key(prompt, llm_string)

        # Handle both string and List[Generation] inputs
        if isinstance(return_val, list):
            # This is a List[Generation] - serialize it properly
            try:
                serialized_value = self._serialize_generations_for_sync_update(
                    return_val
                )
            except Exception as e:
                logger.error(f"Error serializing generations for sync update: {e}")
                # Fallback to string representation
                serialized_value = str(return_val)
        else:
            # Already a string
            serialized_value = return_val

        # Store with metadata
        cache_data = {
            "response": serialized_value,
            "llm_string": llm_string,
            "timestamp": time.time(),
        }

        # Extract model name from llm_string if possible
        model_name = self._extract_model_name(llm_string)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        self.backend.set(
            cache_key=cache_key,
            value=cache_data,
            model_name=model_name,
            prompt_hash=prompt_hash,
        )

    def _serialize_generations_for_sync_update(
        self, return_val: List["Generation"]
    ) -> str:
        """
        Serialize List[Generation] for synchronous cache update.

        Args:
            return_val: List of Generation objects to serialize

        Returns:
            JSON string representation of the generations
        """
        try:
            import json
            from langchain_core.load import dumpd
            from langchain_core.messages import BaseMessage

            # Convert Generation objects to serializable format with robust error handling
            generations_data = []

            for i, generation in enumerate(return_val):
                try:
                    # Log generation details for debugging
                    logger.debug(
                        f"Serializing generation {i}: {type(generation).__name__}"
                    )

                    # Use Langchain's dumpd function for proper serialization
                    gen_data = dumpd(generation)

                    # Additional validation: ensure the result is JSON serializable
                    # This catches cases where dumpd succeeds but produces non-serializable objects
                    json.dumps(gen_data)

                    # Special handling for ChatGeneration with BaseMessage that might not be fully serialized
                    if hasattr(generation, "message") and isinstance(
                        generation.message, BaseMessage
                    ):
                        # Double-check that the message in the serialized data is also serializable
                        if "message" in gen_data and isinstance(
                            gen_data["message"], BaseMessage
                        ):
                            logger.debug(
                                f"Re-serializing BaseMessage in generation {i}"
                            )
                            gen_data["message"] = dumpd(generation.message)
                            # Validate again
                            json.dumps(gen_data)

                    generations_data.append(gen_data)
                    logger.debug(f"Successfully serialized generation {i}")

                except Exception as e:
                    logger.warning(
                        f"Failed to serialize generation {i} with dumpd: {e}"
                    )
                    logger.debug(
                        f"Generation {i} type: {type(generation)}, attributes: {dir(generation)}"
                    )

                    # Fallback 1: Try dict method if available
                    try:
                        if hasattr(generation, "dict"):
                            gen_data = generation.dict()

                            # Handle BaseMessage in dict output
                            if "message" in gen_data and hasattr(
                                gen_data["message"], "dict"
                            ):
                                gen_data["message"] = gen_data["message"].dict()

                            # Validate JSON serialization
                            json.dumps(gen_data)
                            generations_data.append(gen_data)
                            logger.debug(
                                f"Successfully serialized generation {i} using dict fallback"
                            )
                            continue

                    except Exception as dict_error:
                        logger.warning(
                            f"Dict fallback failed for generation {i}: {dict_error}"
                        )

                    # Fallback 2: Manual extraction of key attributes
                    try:
                        gen_data = {
                            "text": getattr(generation, "text", str(generation)),
                            "type": type(generation).__name__,
                            "generation_info": getattr(
                                generation, "generation_info", None
                            ),
                        }

                        # Handle message attribute for ChatGeneration
                        if hasattr(generation, "message"):
                            message = generation.message
                            if hasattr(message, "content"):
                                gen_data["message"] = {
                                    "content": getattr(message, "content", ""),
                                    "type": type(message).__name__,
                                }
                            else:
                                gen_data["message"] = str(message)

                        # Validate JSON serialization
                        json.dumps(gen_data)
                        generations_data.append(gen_data)
                        logger.debug(
                            f"Successfully serialized generation {i} using manual extraction"
                        )

                    except Exception as manual_error:
                        logger.error(
                            f"Manual extraction failed for generation {i}: {manual_error}"
                        )

                        # Fallback 3: Minimal safe representation
                        gen_data = {
                            "text": str(generation) if generation is not None else "",
                            "type": type(generation).__name__,
                            "error": "serialization_failed",
                            "error_details": str(e),
                        }
                        generations_data.append(gen_data)
                        logger.warning(
                            f"Using minimal safe representation for generation {i}"
                        )

            # Final validation: ensure the entire list is JSON serializable
            try:
                serialized_generations = json.dumps(generations_data)
                logger.debug(
                    f"Successfully serialized {len(generations_data)} generations to JSON ({len(serialized_generations)} chars)"
                )
                return serialized_generations
            except Exception as final_error:
                logger.error(f"Final JSON serialization failed: {final_error}")
                # Create a safe fallback representation
                safe_generations = []
                for i, gen_data in enumerate(generations_data):
                    try:
                        json.dumps(gen_data)
                        safe_generations.append(gen_data)
                    except:
                        safe_generations.append(
                            {
                                "text": f"Generation {i} serialization failed",
                                "type": "unknown",
                                "error": "json_serialization_failed",
                            }
                        )
                serialized_generations = json.dumps(safe_generations)
                logger.warning(
                    f"Used safe fallback serialization for {len(safe_generations)} generations"
                )
                return serialized_generations

        except Exception as e:
            logger.error(f"Error during sync cache update serialization: {e}")
            # Ultimate fallback - return a safe JSON string
            return json.dumps(
                [
                    {
                        "text": "Serialization completely failed",
                        "type": "unknown",
                        "error": str(e),
                    }
                ]
            )

    def clear(self) -> None:
        """Clear all cache entries."""
        self.backend.clear()

    def _generate_langchain_key(self, prompt: str, llm_string: str) -> str:
        """Generate cache key compatible with Langchain format."""
        combined = f"{prompt}|{llm_string}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _extract_model_name(self, llm_string: str) -> str:
        """Extract model name from Langchain LLM string."""
        # Try to parse model name from llm_string
        # Format is usually like "OpenAI\nmodel_name=gpt-3.5-turbo\n..."
        try:
            for line in llm_string.split("\n"):
                if "model_name=" in line:
                    return line.split("model_name=")[1].split("\n")[0].strip()
                elif "model=" in line:
                    return line.split("model=")[1].split("\n")[0].strip()
        except Exception:
            pass

        return "unknown"


def generate_cache_key(prompt: str, model_name: str = "default", **kwargs) -> str:
    """
    Generate a cache key from prompt and parameters.

    Args:
        prompt: The input prompt
        model_name: LLM model name
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        SHA256 hash as cache key
    """
    # Load config to determine what to include in key
    config = load_cache_config()

    cache_data = {"prompt": prompt.strip()}

    if config.include_model_name:
        cache_data["model"] = model_name

    if config.include_temperature and "temperature" in kwargs:
        cache_data["temperature"] = kwargs["temperature"]

    if config.include_max_tokens and "max_tokens" in kwargs:
        cache_data["max_tokens"] = kwargs["max_tokens"]

    # Add other specified parameters
    for key, value in kwargs.items():
        if key not in ["temperature", "max_tokens"]:  # Already handled above
            cache_data[key] = value

    # Apply normalization if configured
    if config.normalize_whitespace:
        cache_data["prompt"] = " ".join(cache_data["prompt"].split())

    if config.normalize_case:
        cache_data["prompt"] = cache_data["prompt"].lower()

    # Sort keys for deterministic hashing
    cache_str = json.dumps(cache_data, sort_keys=True)

    if config.hash_algorithm == "sha256":
        return hashlib.sha256(cache_str.encode()).hexdigest()
    elif config.hash_algorithm == "md5":
        return hashlib.md5(cache_str.encode()).hexdigest()
    elif config.hash_algorithm == "sha1":
        return hashlib.sha1(cache_str.encode()).hexdigest()
    else:
        # Default to sha256
        return hashlib.sha256(cache_str.encode()).hexdigest()


def setup_langchain_cache(config: Optional[CacheConfig] = None) -> Optional[Any]:
    """
    Setup Langchain cache based on configuration.

    Args:
        config: Cache configuration. If None, loads from default location.

    Returns:
        Cache instance or None if disabled/failed
    """
    if config is None:
        config = load_cache_config()

    if not config.enabled:
        return None

    manager = LangchainCacheManager(config)
    return manager.setup_cache()


def is_langchain_cache_configured() -> bool:
    """Check if Langchain cache is already configured."""
    try:
        import langchain

        return hasattr(langchain, "llm_cache") and langchain.llm_cache is not None
    except ImportError:
        return False


# Global cache manager instance
_global_cache_manager: Optional[LangchainCacheManager] = None


def get_global_cache_manager() -> Optional[LangchainCacheManager]:
    """Get or create global cache manager instance."""
    global _global_cache_manager

    if _global_cache_manager is None:
        config = load_cache_config()
        if config.enabled:
            _global_cache_manager = LangchainCacheManager(config)
            _global_cache_manager.setup_cache()

    return _global_cache_manager


def clear_global_cache() -> None:
    """Clear the global cache."""
    manager = get_global_cache_manager()
    if manager:
        manager.clear_cache()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    manager = get_global_cache_manager()
    if manager:
        return manager.get_cache_stats()
    else:
        return {
            "enabled": False,
            "backend": "none",
            "configured": False,
            "metrics": {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0},
        }
