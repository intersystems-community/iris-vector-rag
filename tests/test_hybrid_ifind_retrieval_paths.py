#!/usr/bin/env python3
"""
Comprehensive tests for Hybrid IFind pipeline retrieval paths.

This test file explicitly tests different retrieval paths:
1. IFind working path
2. IFind fallback to LIKE search
3. Vector-only results
4. Fusion of results from both systems
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.core.models import Document
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestHybridIFindRetrievalPaths:
    """
    Test different retrieval paths in Hybrid IFind pipeline.
    
    These tests ensure that fallback behaviors are explicitly tested,
    not buried in every test.
    """
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        connection = Mock()
        cursor = Mock()
        
        manager.get_connection.return_value = connection
        connection.cursor.return_value = cursor
        
        return manager, connection, cursor
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config = Mock(spec=ConfigurationManager)
        config.get.side_effect = lambda key, default=None: {
            "pipelines:hybrid_ifind": {
                "top_k": 5,
                "vector_weight": 0.6,
                "ifind_weight": 0.4,
                "min_ifind_score": 0.1
            }
        }.get(key, default)
        return config
    
    @pytest.fixture
    def pipeline(self, mock_connection_manager, mock_config_manager):
        """Create pipeline instance with mocks."""
        manager, _, _ = mock_connection_manager
        return HybridIFindRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
    
    def test_ifind_working_path(self, pipeline, mock_connection_manager):
        """
        Test when IFind is working properly.
        
        This test verifies that when IFind is functional:
        - IFind SQL query executes successfully
        - Results are returned with proper IFind scores
        - No fallback to LIKE search occurs
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            # Mock successful vector search
            cursor.fetchall.side_effect = [
                # Vector search results
                [
                    ("doc1", "Title 1", "Content 1", 0.9),
                    ("doc2", "Title 2", "Content 2", 0.8)
                ],
                # IFind search results (working)
                [
                    ("doc1", "Title 1", "Content 1", 0.85),
                    ("doc3", "Title 3", "Content 3", 0.75)
                ]
            ]
            
            # Execute query
            result = pipeline.query("test query", top_k=3)
            
            # Verify IFind SQL was executed
            calls = cursor.execute.call_args_list
            ifind_call = calls[1]  # Second call should be IFind
            assert "$FIND" in ifind_call[0][0]
            assert "$SCORE" in ifind_call[0][0]
            
            # Verify results include IFind scores
            docs = result["retrieved_documents"]
            assert len(docs) == 3
            
            # Check that doc1 has both vector and IFind scores
            doc1 = next(d for d in docs if d.id == "doc1")
            assert doc1.metadata["has_vector"] is True
            assert doc1.metadata["has_ifind"] is True
            assert "vector_score" in doc1.metadata
            assert "ifind_score" in doc1.metadata
            
            # Check that doc3 only has IFind score
            doc3 = next(d for d in docs if d.id == "doc3")
            assert doc3.metadata["has_vector"] is False
            assert doc3.metadata["has_ifind"] is True
            assert "ifind_score" in doc3.metadata
            
            logger.info("✅ IFind working path test passed")
    
    def test_ifind_fallback_to_like_search(self, pipeline, mock_connection_manager):
        """
        Test IFind fallback to LIKE search.
        
        This test verifies that when IFind fails:
        - An exception is caught from IFind query
        - System falls back to LIKE search
        - Results are returned with fallback indication
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            # Mock IFind failure and LIKE success
            def execute_side_effect(sql, params=None):
                if "$FIND" in sql:
                    raise Exception("IFind not configured")
                # Return results for other queries
                return None
            
            cursor.execute.side_effect = execute_side_effect
            
            cursor.fetchall.side_effect = [
                # Vector search results
                [
                    ("doc1", "Title 1", "Content 1", 0.9),
                    ("doc2", "Title 2", "Content 2", 0.8)
                ],
                # LIKE search results (fallback)
                [
                    ("doc1", "Title 1", "Content 1", 1.0),
                    ("doc4", "Title 4", "Content 4", 1.0)
                ]
            ]
            
            # Execute query
            result = pipeline.query("test query", top_k=3)
            
            # Verify LIKE SQL was executed after IFind failed
            calls = cursor.execute.call_args_list
            # Should have vector search, failed IFind, then LIKE search
            assert len(calls) >= 3
            
            # Find the LIKE query
            like_query_found = False
            for call in calls:
                if "LIKE" in call[0][0]:
                    like_query_found = True
                    assert "%test query%" in str(call[0][1])
                    break
            assert like_query_found, "LIKE query not found in execute calls"
            
            # Verify results indicate fallback
            docs = result["retrieved_documents"]
            
            # Check that results have text_fallback search type
            fallback_docs = [d for d in docs if d.metadata.get("search_type") == "text_fallback"]
            assert len(fallback_docs) > 0, "No documents marked as text_fallback"
            
            logger.info("✅ IFind fallback test passed")
    
    def test_vector_only_results(self, pipeline, mock_connection_manager):
        """
        Test when only vector search returns results.
        
        This verifies:
        - Vector search returns results
        - IFind/LIKE returns no results
        - Final results are vector-only
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            cursor.fetchall.side_effect = [
                # Vector search results
                [
                    ("doc1", "Title 1", "Content 1", 0.9),
                    ("doc2", "Title 2", "Content 2", 0.8)
                ],
                # Empty IFind results
                []
            ]
            
            # Execute query
            result = pipeline.query("test query", top_k=3)
            
            # Verify results are vector-only
            docs = result["retrieved_documents"]
            assert len(docs) == 2
            
            for doc in docs:
                assert doc.metadata["has_vector"] is True
                assert doc.metadata["has_ifind"] is False
                assert "vector_score" in doc.metadata
                assert doc.metadata.get("ifind_score") is None or doc.metadata.get("ifind_score") == 0.0
            
            logger.info("✅ Vector-only results test passed")
    
    def test_result_fusion(self, pipeline, mock_connection_manager):
        """
        Test fusion of results from both systems.
        
        This verifies:
        - Overlapping documents get combined scores
        - Non-overlapping documents are included
        - Hybrid scores are calculated correctly
        - Results are ranked by hybrid score
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            cursor.fetchall.side_effect = [
                # Vector search results
                [
                    ("doc1", "Title 1", "Content 1", 0.9),  # High vector score
                    ("doc2", "Title 2", "Content 2", 0.7),  # Medium vector score
                    ("doc3", "Title 3", "Content 3", 0.5)   # Low vector score
                ],
                # IFind results
                [
                    ("doc2", "Title 2", "Content 2", 0.95), # High IFind score
                    ("doc3", "Title 3", "Content 3", 0.8),  # Medium IFind score
                    ("doc4", "Title 4", "Content 4", 0.6)   # IFind only
                ]
            ]
            
            # Execute query
            result = pipeline.query("test query", top_k=4)
            
            docs = result["retrieved_documents"]
            assert len(docs) == 4
            
            # Find specific documents
            doc1 = next((d for d in docs if d.id == "doc1"), None)
            doc2 = next((d for d in docs if d.id == "doc2"), None)
            doc3 = next((d for d in docs if d.id == "doc3"), None)
            doc4 = next((d for d in docs if d.id == "doc4"), None)
            
            # Doc1: Vector only
            assert doc1 is not None
            assert doc1.metadata["has_vector"] is True
            assert doc1.metadata["has_ifind"] is False
            
            # Doc2: Both systems (should have highest hybrid score)
            assert doc2 is not None
            assert doc2.metadata["has_vector"] is True
            assert doc2.metadata["has_ifind"] is True
            assert doc2.metadata["hybrid_score"] > doc1.metadata["hybrid_score"]
            
            # Doc3: Both systems
            assert doc3 is not None
            assert doc3.metadata["has_vector"] is True
            assert doc3.metadata["has_ifind"] is True
            
            # Doc4: IFind only
            assert doc4 is not None
            assert doc4.metadata["has_vector"] is False
            assert doc4.metadata["has_ifind"] is True
            
            # Verify ordering by hybrid score
            scores = [d.metadata["hybrid_score"] for d in docs]
            assert scores == sorted(scores, reverse=True)
            
            logger.info("✅ Result fusion test passed")
    
    def test_empty_results_handling(self, pipeline, mock_connection_manager):
        """
        Test handling when both systems return no results.
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            cursor.fetchall.side_effect = [
                [],  # Empty vector results
                []   # Empty IFind results
            ]
            
            # Execute query
            result = pipeline.query("test query", top_k=3)
            
            # Verify empty results handled gracefully
            assert result["retrieved_documents"] == []
            assert result["vector_results_count"] == 0
            assert result["ifind_results_count"] == 0
            assert result["answer"] is None
            
            logger.info("✅ Empty results test passed")
    
    def test_score_normalization(self, pipeline, mock_connection_manager):
        """
        Test score normalization in fusion process.
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            cursor.fetchall.side_effect = [
                # Vector results with varying scores
                [
                    ("doc1", "Title 1", "Content 1", 0.9),
                    ("doc2", "Title 2", "Content 2", 0.5),
                    ("doc3", "Title 3", "Content 3", 0.1)
                ],
                # IFind results with different scale
                [
                    ("doc1", "Title 1", "Content 1", 100.0),
                    ("doc2", "Title 2", "Content 2", 50.0),
                    ("doc4", "Title 4", "Content 4", 10.0)
                ]
            ]
            
            # Execute query
            result = pipeline.query("test query", top_k=4)
            
            docs = result["retrieved_documents"]
            
            # Verify all scores are normalized (between 0 and 1)
            for doc in docs:
                if doc.metadata["has_vector"]:
                    assert 0 <= doc.metadata["vector_score"] <= 1
                if doc.metadata["has_ifind"]:
                    assert 0 <= doc.metadata["ifind_score"] <= 1
                assert 0 <= doc.metadata["hybrid_score"] <= 1
            
            logger.info("✅ Score normalization test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])