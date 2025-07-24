"""
LLM Response Caching Module

DEPRECATED: This module is deprecated in favor of the new Langchain-based caching system.
Please use the new caching system in common/llm_cache_manager.py instead.

Provides caching functionality for LLM responses to reduce API costs and improve performance.
Supports multiple cache backends: memory, file, and Redis.
"""

import hashlib
import json
import logging
import os
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "iris_rag.llm.cache is deprecated. Please use the new Langchain-based caching system "
    "in common/llm_cache_manager.py instead. This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            if entry.get('expires_at') and time.time() > entry['expires_at']:
                del self.cache[key]
                return None
            return entry['value']
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Implement simple LRU eviction
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        entry = {'value': value}
        if ttl:
            entry['expires_at'] = time.time() + ttl
        
        self.cache[key] = entry
    
    def delete(self, key: str) -> None:
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        self.cache.clear()


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: str = ".llm_cache", max_files: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_files = max_files
    
    def _get_file_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check TTL
                if entry.get('expires_at') and time.time() > entry['expires_at']:
                    file_path.unlink()
                    return None
                
                return entry['value']
            except Exception as e:
                logger.warning(f"Failed to read cache file {file_path}: {e}")
                # Remove corrupted file
                file_path.unlink(missing_ok=True)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Clean up old files if we're at the limit
        self._cleanup_if_needed()
        
        entry = {'value': value}
        if ttl:
            entry['expires_at'] = time.time() + ttl
        
        file_path = self._get_file_path(key)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to write cache file {file_path}: {e}")
    
    def delete(self, key: str) -> None:
        file_path = self._get_file_path(key)
        file_path.unlink(missing_ok=True)
    
    def clear(self) -> None:
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink()
    
    def _cleanup_if_needed(self) -> None:
        """Remove oldest files if we're at the limit."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        if len(cache_files) >= self.max_files:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda p: p.stat().st_mtime)
            for file_path in cache_files[:len(cache_files) - self.max_files + 1]:
                file_path.unlink(missing_ok=True)

class LLMCache:
    """Main LLM cache class."""
    
    def __init__(self, backend: CacheBackend, default_ttl: int = 3600):
        self.backend = backend
        self.default_ttl = default_ttl
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
    
    def _make_cache_key(self, prompt: str, model: str = "default", 
                       temperature: float = 0.0, **kwargs) -> str:
        """Create a cache key from prompt and parameters."""
        # Create a deterministic key from prompt and parameters
        cache_data = {
            'prompt': prompt,
            'model': model,
            'temperature': temperature,
            **kwargs
        }
        
        # Sort keys for deterministic hashing
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, model: str = "default", **kwargs) -> Optional[str]:
        """Get cached response for prompt."""
        key = self._make_cache_key(prompt, model, **kwargs)
        result = self.backend.get(key)
        
        if result is not None:
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return result
        else:
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for prompt: {prompt[:50]}...")
            return None
    
    def set(self, prompt: str, response: str, model: str = "default", 
            ttl: Optional[int] = None, **kwargs) -> None:
        """Cache response for prompt."""
        key = self._make_cache_key(prompt, model, **kwargs)
        ttl = ttl or self.default_ttl
        self.backend.set(key, response, ttl)
        self.stats['sets'] += 1
        logger.debug(f"Cached response for prompt: {prompt[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.backend.clear()
        logger.info("LLM cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate
        }


def cached_llm_function(cache: LLMCache, model: str = "default", **cache_kwargs):
    """Decorator to add caching to LLM functions."""
    
    def decorator(llm_func):
        @wraps(llm_func)
        def wrapper(prompt: str, **kwargs):
            # Check cache first
            cached_response = cache.get(prompt, model=model, **cache_kwargs, **kwargs)
            if cached_response is not None:
                return cached_response
            
            # Call original function
            response = llm_func(prompt, **kwargs)
            
            # Cache the response
            if response:
                cache.set(prompt, response, model=model, **cache_kwargs, **kwargs)
            
            return response
        
        return wrapper
    return decorator


# Global cache instance (can be configured)
_global_cache: Optional[LLMCache] = None


def get_global_cache() -> LLMCache:
    """Get or create global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        # Default to file cache
        cache_type = os.getenv("LLM_CACHE_TYPE", "file")
        
        if cache_type == "memory":
            backend = MemoryCache(max_size=int(os.getenv("LLM_CACHE_SIZE", "1000")))
        else:  # file
            cache_dir = os.getenv("LLM_CACHE_DIR", ".llm_cache")
            backend = FileCache(cache_dir=cache_dir)
        
        ttl = int(os.getenv("LLM_CACHE_TTL", "3600"))  # 1 hour default
        _global_cache = LLMCache(backend, default_ttl=ttl)
        
        logger.info(f"Initialized global LLM cache with {cache_type} backend")
    
    return _global_cache


def configure_global_cache(backend: CacheBackend, default_ttl: int = 3600) -> None:
    """Configure the global cache instance."""
    global _global_cache
    _global_cache = LLMCache(backend, default_ttl)


def cached_llm_call(prompt: str, llm_func, model: str = "default", **kwargs) -> str:
    """Make a cached LLM call using the global cache."""
    cache = get_global_cache()
    
    # Check cache first
    cached_response = cache.get(prompt, model=model, **kwargs)
    if cached_response is not None:
        return cached_response
    
    # Call LLM function
    response = llm_func(prompt)
    
    # Cache the response
    if response:
        cache.set(prompt, response, model=model, **kwargs)
    
    return response