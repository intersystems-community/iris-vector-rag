"""
Disk-based LLM Cache Backend

This module provides a local filesystem backend for the LLM caching layer,
storing responses as JSON files. This is ideal for offline development
and reducing API costs without a running database.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from iris_vector_rag.common.llm_cache_config import CacheConfig

logger = logging.getLogger(__name__)


class DiskCacheBackend:
    """Filesystem backend for LLM response caching."""

    def __init__(
        self,
        cache_dir: str = ".cache/iris_rag/llm",
        ttl_seconds: int = 3600
    ):
        """
        Initialize Disk cache backend.

        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Default time-to-live for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "errors": 0}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Disk cache initialized at {self.cache_dir}")

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve value from disk cache.

        Args:
            cache_key: Unique hash for the prompt/model
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            self.stats["misses"] += 1
            return None
            
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                
            # Check expiration
            expires_at_str = data.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    logger.debug(f"Cache entry expired for {cache_key}")
                    self.delete(cache_key)
                    self.stats["misses"] += 1
                    return None
            
            self.stats["hits"] += 1
            return data.get("value")
            
        except Exception as e:
            logger.error(f"Error reading disk cache file {cache_file}: {e}")
            self.stats["errors"] += 1
            return None

    def set(
        self,
        cache_key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Store value in disk cache.

        Args:
            cache_key: Unique hash
            value: Response to cache
            ttl: Override default TTL
        """
        try:
            ttl_to_use = ttl or self.ttl_seconds
            expires_at = datetime.now() + timedelta(seconds=ttl_to_use)
            
            cache_data = {
                "cache_key": cache_key,
                "value": value,
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat(),
                "metadata": kwargs
            }
            
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
                
            self.stats["sets"] += 1
            logger.debug(f"Stored disk cache entry: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
            self.stats["errors"] += 1

    def delete(self, cache_key: str) -> None:
        """Delete specific cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cache_file.unlink()
                self.stats["deletes"] += 1
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")

    def clear(self) -> None:
        """Clear all entries in the cache directory."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared disk cache at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache usage statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0.0
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_dir": str(self.cache_dir)
        }


def create_disk_cache_backend(config: CacheConfig) -> DiskCacheBackend:
    """Factory for DiskCacheBackend."""
    return DiskCacheBackend(
        cache_dir=config.cache_directory,
        ttl_seconds=config.ttl_seconds
    )
