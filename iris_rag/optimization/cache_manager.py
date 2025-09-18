"""
GraphRAG Cache Manager for Performance Optimization.

This module provides multi-layer caching for GraphRAG operations to achieve
sub-200ms response times. Includes LRU cache for query results, entity extraction
cache, graph traversal path cache, and TTL-based invalidation.
"""

import time
import hashlib
import logging
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field
from ..core.models import Document

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with TTL and metadata."""
    value: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds <= 0:  # No expiration
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_removals': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returning None if not found or expired."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._stats['expired_removals'] += 1
                self._stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache with optional TTL override."""
        with self._lock:
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                
                # Evict if necessary
                while len(self._cache) > self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._stats['evictions'] += 1
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {k: 0 for k in self._stats}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_size_bytes': total_size
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self._stats['expired_removals'] += 1
            
            return len(expired_keys)
    
    def _estimate_size(self, value: Any) -> int:
        """Rough size estimation for cache entry."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                return sum(len(str(k)) + self._estimate_size(v) for k, v in list(value.items())[:10])
            elif hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                return 1024  # Default estimate
        except:
            return 1024


class GraphRAGCacheManager:
    """
    Comprehensive cache manager for GraphRAG operations.
    
    Provides multi-layer caching:
    - Query result cache (full response caching)
    - Entity extraction cache (document -> entities mapping)
    - Graph traversal path cache (seed entities -> relevant entities)
    - Document retrieval cache (entity IDs -> documents)
    """
    
    def __init__(self, config_manager=None):
        """Initialize cache manager with configuration."""
        self.config = self._load_config(config_manager)
        
        # Initialize cache layers
        self.query_cache = LRUCache(
            max_size=self.config['query_cache']['max_size'],
            default_ttl=self.config['query_cache']['ttl']
        )
        
        self.entity_cache = LRUCache(
            max_size=self.config['entity_cache']['max_size'],
            default_ttl=self.config['entity_cache']['ttl']
        )
        
        self.graph_path_cache = LRUCache(
            max_size=self.config['graph_path_cache']['max_size'],
            default_ttl=self.config['graph_path_cache']['ttl']
        )
        
        self.document_cache = LRUCache(
            max_size=self.config['document_cache']['max_size'],
            default_ttl=self.config['document_cache']['ttl']
        )
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        if self.config['background_cleanup']['enabled']:
            self._start_cleanup_thread()
        
        logger.info(f"GraphRAG cache manager initialized with {len(self._get_all_caches())} cache layers")
    
    def _load_config(self, config_manager) -> Dict[str, Any]:
        """Load cache configuration with defaults."""
        default_config = {
            'query_cache': {
                'max_size': 500,
                'ttl': 3600  # 1 hour
            },
            'entity_cache': {
                'max_size': 1000,
                'ttl': 7200  # 2 hours
            },
            'graph_path_cache': {
                'max_size': 2000,
                'ttl': 1800  # 30 minutes
            },
            'document_cache': {
                'max_size': 1000,
                'ttl': 3600  # 1 hour
            },
            'background_cleanup': {
                'enabled': True,
                'interval_seconds': 300  # 5 minutes
            }
        }
        
        if config_manager:
            cache_config = config_manager.get('graphrag_cache', {})
            # Deep merge with defaults
            for layer, settings in default_config.items():
                if layer in cache_config:
                    settings.update(cache_config[layer])
        
        return default_config
    
    def get_query_result(self, query_text: str, top_k: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        cache_key = self._generate_query_key(query_text, top_k, **kwargs)
        return self.query_cache.get(cache_key)
    
    def cache_query_result(self, query_text: str, top_k: int, result: Dict[str, Any], **kwargs) -> None:
        """Cache query result."""
        cache_key = self._generate_query_key(query_text, top_k, **kwargs)
        # Remove non-serializable metadata for caching
        cacheable_result = self._make_cacheable(result)
        self.query_cache.put(cache_key, cacheable_result)
        logger.debug(f"Cached query result for key: {cache_key[:50]}...")
    
    def get_entity_extraction(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached entity extraction results."""
        return self.entity_cache.get(f"entity_extraction:{document_id}")
    
    def cache_entity_extraction(self, document_id: str, extraction_result: Dict[str, Any]) -> None:
        """Cache entity extraction results."""
        cache_key = f"entity_extraction:{document_id}"
        self.entity_cache.put(cache_key, extraction_result)
        logger.debug(f"Cached entity extraction for document: {document_id}")
    
    def get_graph_traversal(self, seed_entities: List[Tuple[str, str, float]], max_depth: int, max_entities: int) -> Optional[Set[str]]:
        """Get cached graph traversal results."""
        cache_key = self._generate_graph_traversal_key(seed_entities, max_depth, max_entities)
        result = self.graph_path_cache.get(cache_key)
        return set(result) if result else None
    
    def cache_graph_traversal(self, seed_entities: List[Tuple[str, str, float]], max_depth: int, max_entities: int, relevant_entities: Set[str]) -> None:
        """Cache graph traversal results."""
        cache_key = self._generate_graph_traversal_key(seed_entities, max_depth, max_entities)
        # Convert set to list for JSON serialization
        self.graph_path_cache.put(cache_key, list(relevant_entities))
        logger.debug(f"Cached graph traversal with {len(relevant_entities)} entities")
    
    def get_documents_for_entities(self, entity_ids: Set[str], top_k: int) -> Optional[List[Document]]:
        """Get cached documents for entity IDs."""
        cache_key = self._generate_document_retrieval_key(entity_ids, top_k)
        return self.document_cache.get(cache_key)
    
    def cache_documents_for_entities(self, entity_ids: Set[str], top_k: int, documents: List[Document]) -> None:
        """Cache documents for entity IDs."""
        cache_key = self._generate_document_retrieval_key(entity_ids, top_k)
        # Convert documents to cacheable format
        cacheable_docs = [self._document_to_dict(doc) for doc in documents]
        self.document_cache.put(cache_key, cacheable_docs)
        logger.debug(f"Cached {len(documents)} documents for {len(entity_ids)} entities")
    
    def invalidate_query_cache(self) -> None:
        """Invalidate all query caches (e.g., after document updates)."""
        self.query_cache.clear()
        logger.info("Query cache invalidated")
    
    def invalidate_entity_cache(self, document_id: Optional[str] = None) -> None:
        """Invalidate entity cache for specific document or all."""
        if document_id:
            self.entity_cache.invalidate(f"entity_extraction:{document_id}")
            logger.info(f"Entity cache invalidated for document: {document_id}")
        else:
            self.entity_cache.clear()
            logger.info("Entity cache completely invalidated")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'query_cache': self.query_cache.get_stats(),
            'entity_cache': self.entity_cache.get_stats(),
            'graph_path_cache': self.graph_path_cache.get_stats(),
            'document_cache': self.document_cache.get_stats()
        }
        
        # Calculate totals
        total_size = sum(cache_stats['size'] for cache_stats in stats.values())
        total_hits = sum(cache_stats['hits'] for cache_stats in stats.values())
        total_misses = sum(cache_stats['misses'] for cache_stats in stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        stats['overall'] = {
            'total_cache_entries': total_size,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': overall_hit_rate
        }
        
        return stats
    
    def warm_cache(self, warm_queries: List[str]) -> Dict[str, Any]:
        """Pre-warm cache with common queries (for production startup)."""
        start_time = time.time()
        warmed_count = 0
        
        logger.info(f"Starting cache warming with {len(warm_queries)} queries")
        
        # Note: This would need to be integrated with the actual GraphRAG pipeline
        # For now, just prepare the cache keys
        for query in warm_queries:
            cache_key = self._generate_query_key(query, 10)
            # In a real implementation, we'd execute the query and cache the result
            warmed_count += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Cache warming completed: {warmed_count} queries in {elapsed:.2f}s")
        
        return {
            'warmed_queries': warmed_count,
            'elapsed_seconds': elapsed
        }
    
    def _generate_query_key(self, query_text: str, top_k: int, **kwargs) -> str:
        """Generate cache key for query results."""
        # Include relevant parameters in key
        key_parts = [
            query_text.lower().strip(),
            str(top_k),
            str(kwargs.get('include_sources', True)),
            str(kwargs.get('generate_answer', True))
        ]
        key_string = '|'.join(key_parts)
        return f"query:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _generate_graph_traversal_key(self, seed_entities: List[Tuple[str, str, float]], max_depth: int, max_entities: int) -> str:
        """Generate cache key for graph traversal."""
        # Sort seed entities for consistent key generation
        sorted_seeds = sorted(seed_entities, key=lambda x: x[0])
        key_parts = [
            str(sorted_seeds),
            str(max_depth),
            str(max_entities)
        ]
        key_string = '|'.join(key_parts)
        return f"graph_traversal:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _generate_document_retrieval_key(self, entity_ids: Set[str], top_k: int) -> str:
        """Generate cache key for document retrieval."""
        # Sort entity IDs for consistent key generation
        sorted_ids = sorted(entity_ids)
        key_parts = [
            str(sorted_ids),
            str(top_k)
        ]
        key_string = '|'.join(key_parts)
        return f"documents:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _make_cacheable(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to cacheable format (remove non-serializable objects)."""
        cacheable = result.copy()
        
        # Convert documents to dictionaries
        if 'retrieved_documents' in cacheable:
            cacheable['retrieved_documents'] = [
                self._document_to_dict(doc) for doc in cacheable['retrieved_documents']
            ]
        
        return cacheable
    
    def _document_to_dict(self, doc: Document) -> Dict[str, Any]:
        """Convert Document object to dictionary for caching."""
        return {
            'id': doc.id,
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
    
    def _dict_to_document(self, doc_dict: Dict[str, Any]) -> Document:
        """Convert dictionary back to Document object."""
        return Document(
            id=doc_dict['id'],
            page_content=doc_dict['page_content'],
            metadata=doc_dict['metadata']
        )
    
    def _get_all_caches(self) -> List[LRUCache]:
        """Get all cache instances for bulk operations."""
        return [
            self.query_cache,
            self.entity_cache,
            self.graph_path_cache,
            self.document_cache
        ]
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread for cache cleanup."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.config['background_cleanup']['interval_seconds']):
                try:
                    total_cleaned = 0
                    for cache in self._get_all_caches():
                        total_cleaned += cache.cleanup_expired()
                    
                    if total_cleaned > 0:
                        logger.debug(f"Background cleanup removed {total_cleaned} expired cache entries")
                except Exception as e:
                    logger.error(f"Error in background cache cleanup: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info("Background cache cleanup thread started")
    
    def shutdown(self) -> None:
        """Shutdown cache manager and cleanup threads."""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("Cache manager shutdown completed")