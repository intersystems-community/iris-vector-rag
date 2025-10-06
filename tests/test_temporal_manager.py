"""
Comprehensive tests for TemporalMemoryManager functionality.

This test suite covers the temporal memory management system for RAG applications,
including configurable time windows, retention policies, and performance optimization.
"""

import asyncio
import time
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from iris_rag.core.connection import ConnectionManager
from iris_rag.memory.models import MemoryItem, TemporalWindow, TemporalWindowConfig

# Standard imports
from iris_rag.memory.temporal_manager import (
    TemporalContext,
    TemporalMemoryManager,
    TemporalQuery,
)


class TestTemporalQuery(unittest.TestCase):
    """Test the TemporalQuery data class."""

    def test_temporal_query_creation(self):
        """Test creating temporal query objects."""
        window = TemporalWindow.SHORT_TERM
        query = TemporalQuery(
            query_text="test query",
            window=window,
            max_results=5,
            relevance_threshold=0.7,
            include_expired=True,
            context_filters={"category": "research"},
        )

        self.assertEqual(query.query_text, "test query")
        self.assertEqual(query.window, window)
        self.assertEqual(query.max_results, 5)
        self.assertEqual(query.relevance_threshold, 0.7)
        self.assertTrue(query.include_expired)
        self.assertEqual(query.context_filters["category"], "research")

    def test_temporal_query_defaults(self):
        """Test temporal query with default values."""
        window = TemporalWindow.SHORT_TERM
        query = TemporalQuery(query_text="test", window=window)

        self.assertEqual(query.max_results, 10)
        self.assertEqual(query.relevance_threshold, 0.5)
        self.assertFalse(query.include_expired)
        self.assertEqual(query.context_filters, {})


class TestTemporalContext(unittest.TestCase):
    """Test the TemporalContext data class."""

    def test_temporal_context_creation(self):
        """Test creating temporal context objects."""
        window = TemporalWindow.MEDIUM_TERM
        context = TemporalContext(
            window=window, total_items_in_window=25, avg_relevance_score=0.85
        )

        self.assertEqual(context.window, window)
        self.assertEqual(context.items, [])
        self.assertEqual(context.total_items_in_window, 25)
        self.assertEqual(context.avg_relevance_score, 0.85)
        self.assertIsInstance(context.window_start, datetime)
        self.assertIsInstance(context.window_end, datetime)

    def test_temporal_context_with_items(self):
        """Test temporal context with memory items."""
        window = TemporalWindow.SHORT_TERM
        memory_item = MemoryItem(
            item_id="test_item",
            content="test content",
            context={},
            temporal_window=window,
            source_query="test query",
        )

        context = TemporalContext(window=window, items=[memory_item])

        self.assertEqual(len(context.items), 1)
        self.assertEqual(context.items[0], memory_item)


class TestTemporalMemoryManager(unittest.TestCase):
    """Test the main TemporalMemoryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)

        # Create test window configurations
        self.window_configs = [
            TemporalWindowConfig(name="recent", duration_days=7),
            TemporalWindowConfig(name="medium_term", duration_days=30),
            TemporalWindowConfig(name="long_term", duration_days=90),
        ]

        # Mock database operations
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection_manager.get_connection.return_value = self.mock_connection

        # Patch the table creation method to avoid database calls
        with patch.object(TemporalMemoryManager, "_ensure_temporal_memory_table"):
            self.manager = TemporalMemoryManager(
                connection_manager=self.mock_connection_manager,
                window_configs=self.window_configs,
            )

    def test_initialization(self):
        """Test TemporalMemoryManager initialization."""
        self.assertEqual(self.manager.connection_manager, self.mock_connection_manager)
        self.assertEqual(len(self.manager.window_configs), 3)
        self.assertIn("recent", self.manager.window_configs)
        self.assertIn("medium_term", self.manager.window_configs)
        self.assertIn("long_term", self.manager.window_configs)

        # Check performance tracking initialization
        self.assertEqual(self.manager._retrieval_times, [])
        self.assertEqual(self.manager._storage_times, [])
        self.assertEqual(self.manager._cache_hits, 0)
        self.assertEqual(self.manager._cache_misses, 0)

        # Check caching initialization
        self.assertEqual(self.manager._memory_cache, {})
        self.assertEqual(self.manager._cache_timestamps, {})
        self.assertEqual(self.manager._cache_ttl, timedelta(minutes=30))

    @patch.object(TemporalMemoryManager, "_store_memory_item", new_callable=AsyncMock)
    @patch.object(TemporalMemoryManager, "_invalidate_cache_for_window")
    def test_store_with_window(self, mock_invalidate_cache, mock_store_item):
        """Test storing content with temporal window."""

        async def run_test():
            content = "Test content for storage"
            window = TemporalWindow.SHORT_TERM
            context = {"category": "test"}
            source_query = "test query"

            mock_store_item.return_value = None

            result = await self.manager.store_with_window(
                content=content,
                window=window,
                context=context,
                source_query=source_query,
            )

            # Verify result is a string (item ID)
            self.assertIsInstance(result, str)

            # Verify storage was called
            mock_store_item.assert_called_once()

            # Verify cache invalidation was called
            mock_invalidate_cache.assert_called_once_with(window)

            # Check performance tracking
            self.assertEqual(len(self.manager._storage_times), 1)
            self.assertGreater(self.manager._storage_times[0], 0)

        asyncio.run(run_test())

    @patch.object(
        TemporalMemoryManager, "_retrieve_memories_in_window", new_callable=AsyncMock
    )
    @patch.object(TemporalMemoryManager, "_filter_by_relevance")
    def test_retrieve_temporal_context(
        self, mock_filter_relevance, mock_retrieve_memories
    ):
        """Test retrieving temporal context."""

        async def run_test():
            # Setup test data
            test_memory_item = MemoryItem(
                item_id="test_id",
                content="test content",
                context={},
                temporal_window=TemporalWindow.SHORT_TERM,
                source_query="test query",
            )

            mock_retrieve_memories.return_value = [test_memory_item]
            mock_filter_relevance.return_value = [test_memory_item]

            query = TemporalQuery(
                query_text="test query",
                window=TemporalWindow.SHORT_TERM,
                max_results=10,
            )

            result = await self.manager.retrieve_temporal_context(query)

            # Verify result structure
            self.assertIsInstance(result, TemporalContext)
            self.assertEqual(result.window, TemporalWindow.SHORT_TERM)
            self.assertEqual(len(result.items), 1)
            self.assertEqual(result.items[0], test_memory_item)

            # Verify methods were called
            mock_retrieve_memories.assert_called_once()
            mock_filter_relevance.assert_called_once()

            # Check performance tracking
            self.assertEqual(len(self.manager._retrieval_times), 1)

        asyncio.run(run_test())

    @patch.object(
        TemporalMemoryManager, "_delete_expired_memories", new_callable=AsyncMock
    )
    @patch.object(
        TemporalMemoryManager, "_archive_expired_memories", new_callable=AsyncMock
    )
    def test_cleanup_expired_memories(self, mock_archive, mock_delete):
        """Test cleaning up expired memories."""

        async def run_test():
            mock_delete.return_value = 5  # 5 items deleted
            mock_archive.return_value = 3  # 3 items archived

            result = await self.manager.cleanup_expired_memories(
                archive_before_deletion=True
            )

            # Verify cleanup results
            self.assertIn("deleted_count", result)
            self.assertIn("archived_count", result)
            self.assertEqual(result["deleted_count"], 5)
            self.assertEqual(result["archived_count"], 3)

            # Verify both operations were called
            mock_archive.assert_called_once()
            mock_delete.assert_called_once()

        asyncio.run(run_test())

    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Add some mock performance data
        self.manager._retrieval_times = [50.0, 75.0, 25.0, 100.0]
        self.manager._storage_times = [20.0, 30.0, 15.0]
        self.manager._cache_hits = 10
        self.manager._cache_misses = 3

        metrics = self.manager.get_performance_metrics()

        # Verify metric structure
        self.assertIn("average_retrieval_time_ms", metrics)
        self.assertIn("average_storage_time_ms", metrics)
        self.assertIn("cache_hit_rate", metrics)
        self.assertIn("total_retrievals", metrics)
        self.assertIn("total_storages", metrics)

        # Verify calculations
        self.assertEqual(metrics["average_retrieval_time_ms"], 62.5)  # (50+75+25+100)/4
        self.assertEqual(
            metrics["average_storage_time_ms"], 21.67
        )  # (20+30+15)/3, rounded
        self.assertAlmostEqual(metrics["cache_hit_rate"], 0.769, places=2)  # 10/(10+3)

    def test_get_window_config(self):
        """Test getting window configuration."""
        # Test existing window
        config = self.manager._get_window_config(TemporalWindow.SHORT_TERM)
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "recent")
        self.assertEqual(config.retention_days, 7)

        # Test non-existent window
        config = self.manager._get_window_config(TemporalWindow.ARCHIVE)
        self.assertIsNone(config)

    def test_calculate_window_start(self):
        """Test calculating window start time."""
        config = TemporalWindowConfig(
            name="test", duration_days=7, window=TemporalWindow.SHORT_TERM
        )

        # Mock current time for consistent testing
        test_time = datetime(2023, 1, 15, 12, 0, 0)
        with patch("iris_rag.memory.temporal_manager.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = test_time

            start_time = self.manager._calculate_window_start(config)
            expected_start = test_time - timedelta(days=7)

            self.assertEqual(start_time, expected_start)

    def test_generate_memory_id(self):
        """Test generating memory IDs."""
        content = "test content"
        source_query = "test query"
        window = TemporalWindow.SHORT_TERM

        # Generate multiple IDs with same inputs
        id1 = self.manager._generate_memory_id(content, source_query, window)
        id2 = self.manager._generate_memory_id(content, source_query, window)

        # Should be deterministic (same inputs = same ID)
        self.assertEqual(id1, id2)
        self.assertIsInstance(id1, str)
        self.assertGreater(len(id1), 0)

        # Different inputs should generate different IDs
        id3 = self.manager._generate_memory_id(
            "different content", source_query, window
        )
        self.assertNotEqual(id1, id3)

    def test_generate_cache_key(self):
        """Test generating cache keys for queries."""
        query1 = TemporalQuery(
            query_text="test query", window=TemporalWindow.SHORT_TERM, max_results=10
        )
        query2 = TemporalQuery(
            query_text="test query", window=TemporalWindow.SHORT_TERM, max_results=10
        )
        query3 = TemporalQuery(
            query_text="different query",
            window=TemporalWindow.SHORT_TERM,
            max_results=10,
        )

        key1 = self.manager._generate_cache_key(query1)
        key2 = self.manager._generate_cache_key(query2)
        key3 = self.manager._generate_cache_key(query3)

        # Same queries should generate same cache key
        self.assertEqual(key1, key2)

        # Different queries should generate different cache keys
        self.assertNotEqual(key1, key3)

    def test_filter_by_relevance(self):
        """Test filtering memory items by relevance."""
        # Create test memory items
        high_relevance_item = MemoryItem(
            item_id="high",
            content="very relevant test content for the query",
            context={},
            temporal_window=TemporalWindow.SHORT_TERM,
            source_query="test content",
        )

        low_relevance_item = MemoryItem(
            item_id="low",
            content="completely unrelated information",
            context={},
            temporal_window=TemporalWindow.SHORT_TERM,
            source_query="other topic",
        )

        memory_items = [high_relevance_item, low_relevance_item]
        query_text = "test content"
        threshold = 0.3

        filtered_items = self.manager._filter_by_relevance(
            memory_items, query_text, threshold
        )

        # Should have at least the high relevance item
        self.assertGreater(len(filtered_items), 0)

        # Results should be sorted by relevance (descending)
        if len(filtered_items) > 1:
            for i in range(len(filtered_items) - 1):
                self.assertGreaterEqual(filtered_items[i][1], filtered_items[i + 1][1])

    @patch.object(
        TemporalMemoryManager, "_retrieve_memories_in_window", new_callable=AsyncMock
    )
    def test_get_window_statistics(self, mock_retrieve):
        """Test getting window statistics."""

        async def run_test():
            # Mock memory items in window
            test_items = [
                MemoryItem(
                    item_id=f"item_{i}",
                    content=f"content {i}",
                    context={},
                    temporal_window=TemporalWindow.SHORT_TERM,
                    source_query=f"query {i}",
                )
                for i in range(5)
            ]
            mock_retrieve.return_value = test_items

            stats = await self.manager.get_window_statistics(TemporalWindow.SHORT_TERM)

            # Verify statistics structure
            self.assertIn("window_name", stats)
            self.assertIn("total_items", stats)
            self.assertIn("window_start", stats)
            self.assertIn("window_end", stats)
            self.assertIn("avg_content_length", stats)

            # Verify values
            self.assertEqual(stats["total_items"], 5)
            self.assertIsInstance(stats["window_start"], datetime)
            self.assertIsInstance(stats["window_end"], datetime)

        asyncio.run(run_test())


class TestTemporalMemoryManagerIntegration(unittest.TestCase):
    """Integration tests for temporal memory manager workflows."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)

        self.window_configs = [
            TemporalWindowConfig(name="recent", duration_days=7),
            TemporalWindowConfig(name="medium_term", duration_days=30),
        ]

        with patch.object(TemporalMemoryManager, "_ensure_temporal_memory_table"):
            self.manager = TemporalMemoryManager(
                connection_manager=self.mock_connection_manager,
                window_configs=self.window_configs,
            )

    @patch.object(TemporalMemoryManager, "_store_memory_item", new_callable=AsyncMock)
    @patch.object(
        TemporalMemoryManager, "_retrieve_memories_in_window", new_callable=AsyncMock
    )
    def test_store_and_retrieve_workflow(self, mock_retrieve, mock_store):
        """Test complete store and retrieve workflow."""

        async def run_test():
            # Store content
            content = "Important research findings about temporal patterns"
            window = TemporalWindow.SHORT_TERM
            context = {"project": "research", "priority": "high"}
            source_query = "temporal patterns research"

            mock_store.return_value = None

            item_id = await self.manager.store_with_window(
                content=content,
                window=window,
                context=context,
                source_query=source_query,
            )

            # Simulate stored item for retrieval
            stored_item = MemoryItem(
                item_id=item_id,
                content=content,
                context=context,
                temporal_window=window,
                source_query=source_query,
            )
            mock_retrieve.return_value = [stored_item]

            # Retrieve content
            query = TemporalQuery(
                query_text="research findings temporal",
                window=window,
                max_results=10,
                relevance_threshold=0.5,
            )

            context_result = await self.manager.retrieve_temporal_context(query)

            # Verify workflow
            self.assertIsInstance(item_id, str)
            self.assertIsInstance(context_result, TemporalContext)
            self.assertEqual(len(context_result.items), 1)
            self.assertEqual(context_result.items[0].content, content)

            # Verify performance tracking
            self.assertGreater(len(self.manager._storage_times), 0)
            self.assertGreater(len(self.manager._retrieval_times), 0)

        asyncio.run(run_test())

    def test_performance_monitoring_workflow(self):
        """Test performance monitoring across operations."""
        # Simulate some operations
        self.manager._storage_times = [25.0, 30.0, 20.0, 35.0]
        self.manager._retrieval_times = [45.0, 55.0, 40.0, 50.0]
        self.manager._cache_hits = 8
        self.manager._cache_misses = 2

        metrics = self.manager.get_performance_metrics()

        # Check that metrics meet performance targets
        self.assertLess(metrics["average_retrieval_time_ms"], 100)  # <100ms target
        self.assertLess(metrics["average_storage_time_ms"], 50)  # Should be fast
        self.assertGreater(metrics["cache_hit_rate"], 0.7)  # Good cache performance


if __name__ == "__main__":
    unittest.main()
