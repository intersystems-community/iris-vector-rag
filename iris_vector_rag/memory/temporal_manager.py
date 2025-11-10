#!/usr/bin/env python3
"""
Temporal Memory Manager for RAG applications.

Provides generic temporal storage patterns adaptable to any time-based memory needs.
Demonstrates configurable time windows, retention policies, and performance optimization.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.memory.models import (
    MemoryItem,
    TemporalContext,
    TemporalQuery,
    TemporalWindow,
    TemporalWindowConfig,
)

logger = logging.getLogger(__name__)


class TemporalMemoryManager:
    """
    Generic temporal memory patterns for RAG applications.

    Features:
    - Configurable temporal windows (7/30/90 day patterns)
    - Memory storage with automatic expiration
    - Performance-optimized retrieval with caching
    - Cleanup policies for memory management
    - Performance target: <100ms for time-window queries
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        window_configs: List[TemporalWindowConfig],
    ):
        """
        Initialize temporal memory manager.

        Args:
            connection_manager: Database connection manager
            window_configs: List of temporal window configurations
        """
        self.connection_manager = connection_manager
        self.window_configs = {config.name: config for config in window_configs}

        # Performance optimization
        self._memory_cache: Dict[str, List[MemoryItem]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)  # 30-minute cache TTL

        # Performance tracking
        self._retrieval_times: List[float] = []
        self._storage_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0

        # Ensure memory table exists
        self._ensure_temporal_memory_table()

        logger.info(
            f"Temporal memory manager initialized with {len(window_configs)} windows"
        )

    async def store_with_window(
        self,
        content: Any,
        window: TemporalWindow,
        context: Dict[str, Any] = None,
        source_query: str = "",
    ) -> str:
        """
        Store content with configurable temporal window.

        Args:
            content: Content to store (any serializable type)
            window: Temporal window for storage
            context: Additional context information
            source_query: Query that generated this content

        Returns:
            Memory item ID
        """
        start_time = time.perf_counter()

        try:
            # Generate deterministic ID
            item_id = self._generate_memory_id(content, source_query, window)

            # Get window configuration
            window_config = self._get_window_config(window)

            # Create memory item
            memory_item = MemoryItem(
                item_id=item_id,
                content=str(content),
                context=context or {},
                temporal_window=window,
                source_query=source_query,
                created_at=datetime.utcnow(),
            )

            # Calculate expiration time
            if window_config:
                expiration_time = window_config.get_expiration_time()
            else:
                expiration_time = datetime.utcnow() + timedelta(days=30)  # Default

            # Store in database
            await self._store_memory_item(memory_item, expiration_time)

            # Invalidate relevant cache entries
            self._invalidate_cache_for_window(window)

            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self._storage_times.append(processing_time)

            logger.debug(
                f"Stored memory item {item_id} in {window.value} window ({processing_time:.2f}ms)"
            )
            return item_id

        except Exception as e:
            logger.error(f"Error storing memory with window: {e}")
            raise

    async def retrieve_temporal_context(self, query: TemporalQuery) -> TemporalContext:
        """
        Retrieve relevant memories within temporal context.

        Performance target: <100ms for time-window queries

        Args:
            query: Temporal query specification

        Returns:
            Temporal context with relevant memories
        """
        start_time = time.perf_counter()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._cache_hits += 1
                processing_time = (time.perf_counter() - start_time) * 1000
                self._retrieval_times.append(processing_time)
                return cached_result

            self._cache_misses += 1

            # Calculate time window boundaries
            window_end = datetime.utcnow()
            window_start = self._calculate_window_start(query.window, window_end)

            # Retrieve memories from database
            memories = await self._retrieve_memories_in_window(
                query, window_start, window_end
            )

            # Calculate relevance and filter
            relevant_memories = self._filter_by_relevance(memories, query)

            # Create temporal context
            context = TemporalContext(
                window=query.window,
                items=relevant_memories[: query.max_results],
                total_items_in_window=len(memories),
                avg_relevance_score=(
                    sum(m.relevance_score for m in relevant_memories)
                    / len(relevant_memories)
                    if relevant_memories
                    else 0.0
                ),
                window_start=window_start,
                window_end=window_end,
            )

            # Cache result
            self._set_in_cache(cache_key, context)

            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self._retrieval_times.append(processing_time)

            # Log performance warning if target exceeded
            if processing_time > 100.0:  # 100ms target
                logger.warning(
                    f"Temporal retrieval took {processing_time:.2f}ms (target: <100ms)"
                )

            logger.debug(
                f"Retrieved {len(relevant_memories)} relevant memories from {query.window.value} "
                f"window in {processing_time:.2f}ms"
            )

            return context

        except Exception as e:
            logger.error(f"Error retrieving temporal context: {e}")
            return TemporalContext(window=query.window)

    async def cleanup_expired_memories(
        self, window: Optional[TemporalWindow] = None
    ) -> int:
        """
        Clean up memories based on configurable retention policies.

        Args:
            window: Specific window to clean up, or None for all windows

        Returns:
            Number of memories cleaned up
        """
        try:
            cleanup_count = 0
            windows_to_clean = [window] if window else list(TemporalWindow)

            for temp_window in windows_to_clean:
                window_config = self._get_window_config(temp_window)
                if not window_config:
                    continue

                # Calculate cleanup threshold
                cleanup_threshold = datetime.utcnow() - timedelta(
                    days=window_config.duration_days
                )

                # Perform cleanup based on retention policy
                if window_config.retention_policy == "expire":
                    count = await self._delete_expired_memories(
                        temp_window, cleanup_threshold
                    )
                elif window_config.retention_policy == "archive":
                    count = await self._archive_expired_memories(
                        temp_window, cleanup_threshold
                    )
                else:  # keep
                    continue

                cleanup_count += count

                # Invalidate cache for cleaned windows
                self._invalidate_cache_for_window(temp_window)

            logger.info(f"Cleaned up {cleanup_count} expired memories")
            return cleanup_count

        except Exception as e:
            logger.error(f"Error cleaning up expired memories: {e}")
            return 0

    async def get_window_statistics(self, window: TemporalWindow) -> Dict[str, Any]:
        """
        Get statistics for a temporal window.

        Args:
            window: Temporal window to analyze

        Returns:
            Statistics dictionary
        """
        try:
            window_start = self._calculate_window_start(window, datetime.utcnow())
            window_end = datetime.utcnow()

            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()

            try:
                # Count total memories in window
                cursor.execute(
                    """
                    SELECT COUNT(*), AVG(relevance_score)
                    FROM RAG.TemporalMemory 
                    WHERE temporal_window = ? 
                    AND created_at BETWEEN ? AND ?
                    AND (expires_at IS NULL OR expires_at > ?)
                """,
                    [window.value, window_start, window_end, datetime.utcnow()],
                )

                result = cursor.fetchone()
                total_count = result[0] if result else 0
                avg_relevance = result[1] if result and result[1] else 0.0

                # Get memory distribution by hour
                cursor.execute(
                    """
                    SELECT strftime('%H', created_at) as hour, COUNT(*)
                    FROM RAG.TemporalMemory 
                    WHERE temporal_window = ? 
                    AND created_at BETWEEN ? AND ?
                    GROUP BY hour
                    ORDER BY hour
                """,
                    [window.value, window_start, window_end],
                )

                hourly_distribution = {
                    str(hour): count for hour, count in cursor.fetchall()
                }

                return {
                    "window": window.value,
                    "total_memories": total_count,
                    "average_relevance": round(avg_relevance, 3),
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "hourly_distribution": hourly_distribution,
                }

            finally:
                cursor.close()

        except Exception as e:
            logger.error(f"Error getting window statistics: {e}")
            return {"error": str(e)}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for temporal memory operations."""
        total_cache_ops = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        )

        return {
            "retrieval_performance": {
                "avg_time_ms": (
                    sum(self._retrieval_times) / len(self._retrieval_times)
                    if self._retrieval_times
                    else 0
                ),
                "max_time_ms": (
                    max(self._retrieval_times) if self._retrieval_times else 0
                ),
                "total_retrievals": len(self._retrieval_times),
                "target_met_percentage": (
                    len([t for t in self._retrieval_times if t <= 100.0])
                    / len(self._retrieval_times)
                    * 100
                    if self._retrieval_times
                    else 0
                ),
            },
            "storage_performance": {
                "avg_time_ms": (
                    sum(self._storage_times) / len(self._storage_times)
                    if self._storage_times
                    else 0
                ),
                "total_stores": len(self._storage_times),
            },
            "cache_performance": {
                "hit_rate": cache_hit_rate,
                "total_hits": self._cache_hits,
                "total_misses": self._cache_misses,
            },
        }

    def _get_window_config(
        self, window: TemporalWindow
    ) -> Optional[TemporalWindowConfig]:
        """Get configuration for temporal window."""
        return self.window_configs.get(window.value)

    def _calculate_window_start(
        self, window: TemporalWindow, end_time: datetime
    ) -> datetime:
        """Calculate start time for temporal window."""
        window_config = self._get_window_config(window)
        if window_config:
            return end_time - timedelta(days=window_config.duration_days)

        # Default window sizes
        window_days = {
            TemporalWindow.SHORT_TERM: 7,
            TemporalWindow.MEDIUM_TERM: 30,
            TemporalWindow.LONG_TERM: 90,
            TemporalWindow.PERMANENT: 365 * 10,  # 10 years
        }

        days = window_days.get(window, 30)
        return end_time - timedelta(days=days)

    async def _store_memory_item(
        self, memory_item: MemoryItem, expiration_time: datetime
    ) -> None:
        """Store memory item in database."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO RAG.TemporalMemory 
                (item_id, content, context, temporal_window, relevance_score, 
                 created_at, source_query, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    memory_item.item_id,
                    memory_item.content,
                    str(memory_item.context),  # JSON string
                    memory_item.temporal_window.value,
                    memory_item.relevance_score,
                    memory_item.created_at,
                    memory_item.source_query,
                    expiration_time,
                ],
            )

            connection.commit()

        finally:
            cursor.close()

    async def _retrieve_memories_in_window(
        self, query: TemporalQuery, window_start: datetime, window_end: datetime
    ) -> List[MemoryItem]:
        """Retrieve memories within time window."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Build query with filters
            sql_query = """
                SELECT item_id, content, context, temporal_window, relevance_score, 
                       created_at, source_query
                FROM RAG.TemporalMemory 
                WHERE temporal_window = ? 
                AND created_at BETWEEN ? AND ?
            """
            params = [query.window.value, window_start, window_end]

            # Add expiration filter
            if not query.include_expired:
                sql_query += " AND (expires_at IS NULL OR expires_at > ?)"
                params.append(datetime.utcnow())

            # Add context filters
            for filter_key, filter_value in query.context_filters.items():
                sql_query += f" AND context LIKE ?"
                params.append(f"%{filter_key}:{filter_value}%")

            sql_query += " ORDER BY created_at DESC"

            cursor.execute(sql_query, params)
            rows = cursor.fetchall()

            memories = []
            for row in rows:
                memory_item = MemoryItem(
                    item_id=str(row[0]),
                    content=str(row[1]),
                    context=eval(str(row[2])) if row[2] else {},  # Parse JSON
                    temporal_window=TemporalWindow(str(row[3])),
                    relevance_score=float(row[4]) if row[4] else 1.0,
                    created_at=row[5],
                    source_query=str(row[6]) if row[6] else "",
                )
                memories.append(memory_item)

            return memories

        finally:
            cursor.close()

    def _filter_by_relevance(
        self, memories: List[MemoryItem], query: TemporalQuery
    ) -> List[MemoryItem]:
        """Filter memories by relevance to query."""
        if not query.query_text:
            return memories

        # Simple relevance calculation based on content similarity
        query_words = set(query.query_text.lower().split())

        for memory in memories:
            content_words = set(memory.content.lower().split())
            source_words = set(memory.source_query.lower().split())

            # Calculate word overlap
            content_overlap = (
                len(query_words & content_words) / len(query_words | content_words)
                if query_words or content_words
                else 0
            )
            source_overlap = (
                len(query_words & source_words) / len(query_words | source_words)
                if query_words or source_words
                else 0
            )

            # Combine relevance scores
            memory.relevance_score = (content_overlap + source_overlap) / 2.0

        # Filter by threshold and sort by relevance
        relevant_memories = [
            m for m in memories if m.relevance_score >= query.relevance_threshold
        ]
        relevant_memories.sort(key=lambda m: m.relevance_score, reverse=True)

        return relevant_memories

    async def _delete_expired_memories(
        self, window: TemporalWindow, threshold: datetime
    ) -> int:
        """Delete expired memories."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                DELETE FROM RAG.TemporalMemory 
                WHERE temporal_window = ? AND created_at < ?
            """,
                [window.value, threshold],
            )

            deleted_count = cursor.rowcount
            connection.commit()
            return deleted_count

        finally:
            cursor.close()

    async def _archive_expired_memories(
        self, window: TemporalWindow, threshold: datetime
    ) -> int:
        """Archive expired memories (placeholder for future implementation)."""
        # For now, just mark as archived rather than delete
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                UPDATE RAG.TemporalMemory 
                SET context = json_set(context, '$.archived', 'true')
                WHERE temporal_window = ? AND created_at < ?
            """,
                [window.value, threshold],
            )

            archived_count = cursor.rowcount
            connection.commit()
            return archived_count

        finally:
            cursor.close()

    def _generate_memory_id(
        self, content: Any, source_query: str, window: TemporalWindow
    ) -> str:
        """Generate deterministic memory ID."""
        content_str = f"{content}:{source_query}:{window.value}"
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        return f"mem_{content_hash[:16]}"

    def _generate_cache_key(self, query: TemporalQuery) -> str:
        """Generate cache key for temporal query."""
        key_parts = [
            query.query_text,
            query.window.value,
            str(query.max_results),
            str(query.relevance_threshold),
            str(sorted(query.context_filters.items())),
        ]
        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_from_cache(self, cache_key: str) -> Optional[TemporalContext]:
        """Get temporal context from cache with TTL check."""
        if cache_key in self._memory_cache:
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp and (datetime.utcnow() - timestamp) < self._cache_ttl:
                return self._memory_cache[cache_key]
            else:
                # Expired - remove from cache
                del self._memory_cache[cache_key]
                if cache_key in self._cache_timestamps:
                    del self._cache_timestamps[cache_key]
        return None

    def _set_in_cache(self, cache_key: str, context: TemporalContext) -> None:
        """Set temporal context in cache."""
        self._memory_cache[cache_key] = context
        self._cache_timestamps[cache_key] = datetime.utcnow()

    def _invalidate_cache_for_window(self, window: TemporalWindow) -> None:
        """Invalidate cache entries for specific window."""
        keys_to_remove = []
        for cache_key in self._memory_cache.keys():
            # Simple cache invalidation - could be more sophisticated
            if window.value in cache_key:
                keys_to_remove.append(cache_key)

        for key in keys_to_remove:
            del self._memory_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    def _ensure_temporal_memory_table(self) -> None:
        """Ensure temporal memory table exists."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS RAG.TemporalMemory (
                    item_id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    context TEXT,
                    temporal_window VARCHAR(50) NOT NULL,
                    relevance_score DECIMAL(3,2) DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_query TEXT,
                    expires_at TIMESTAMP,
                    INDEX idx_temporal_window (temporal_window, created_at),
                    INDEX idx_expiration (expires_at)
                )
            """
            )
            connection.commit()
            logger.debug("Temporal memory table ensured")

        except Exception as e:
            logger.warning(f"Could not ensure temporal memory table: {e}")
            # This will be handled properly by schema extensions
        finally:
            cursor.close()
