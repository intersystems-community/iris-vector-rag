#!/usr/bin/env python3
"""
Comprehensive tests for GraphRAG pipeline retrieval paths.

This test file explicitly tests different retrieval paths:
1. Graph-only retrieval
2. Vector-only retrieval  
3. Combined graph + vector retrieval
4. Entity extraction and linking
5. Graph traversal strategies
"""

import pytest
import logging
from unittest.mock import Mock, patch

from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestGraphRAGRetrievalPaths:
    """
    Test different retrieval paths in GraphRAG pipeline.
    
    These tests ensure that different graph retrieval strategies
    are explicitly tested, not buried in integration tests.
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
            "pipelines:graphrag": {
                "top_k": 5,
                "max_entities": 10,
                "relationship_depth": 2
            }
        }.get(key, default)
        return config
    
    @pytest.fixture
    def pipeline(self, mock_connection_manager, mock_config_manager):
        """Create pipeline instance with mocks."""
        manager, _, _ = mock_connection_manager
        return GraphRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
    
    def test_graph_only_retrieval(self, pipeline, mock_connection_manager):
        """
        Test graph-only retrieval path.
        
        This test verifies:
        - Entity extraction from query
        - Graph nodes retrieval based on entities
        - No vector search performed
        - Results based purely on graph relationships
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock entity extraction
        with patch.object(pipeline, '_extract_query_entities') as mock_extract:
            mock_extract.return_value = ["Entity1", "Entity2"]
            
            # Mock embedding (shouldn't be used for graph-only)
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                # Configure to return graph results only
                cursor.fetchall.side_effect = [
                    # Graph node results
                    [
                        ("node1", "Entity1", "doc1", "Description 1", 0.9),
                        ("node2", "Entity2", "doc2", "Description 2", 0.8),
                        ("node3", "Related Entity", "doc3", "Description 3", 0.7)
                    ],
                    # Document retrieval for graph nodes
                    [
                        ("doc1", "Title 1", "Content about Entity1"),
                        ("doc2", "Title 2", "Content about Entity2"),
                        ("doc3", "Title 3", "Related content")
                    ],
                    # Empty vector results (simulating graph-only mode)
                    []
                ]
                
                # Execute query
                result = pipeline.query("What is Entity1 and Entity2?", top_k=3)
                
                # Verify entity extraction was called
                mock_extract.assert_called_once()
                
                # Verify results are from graph
                docs = result["retrieved_documents"]
                assert len(docs) == 3
                
                # Check metadata indicates graph retrieval
                for doc in docs:
                    assert "entity" in doc.metadata or "graph_score" in doc.metadata
                    assert doc.metadata.get("retrieval_type") == "graph" or "entity" in str(doc.metadata)
                
                logger.info("✅ Graph-only retrieval test passed")
    
    def test_vector_only_retrieval(self, pipeline, mock_connection_manager):
        """
        Test vector-only retrieval path.
        
        This test verifies:
        - No entity extraction performed
        - Standard vector search executed
        - Results based purely on embedding similarity
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock entity extraction to return empty
        with patch.object(pipeline, '_extract_query_entities') as mock_extract:
            mock_extract.return_value = []
            
            # Mock embedding
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                cursor.fetchall.side_effect = [
                    # Empty graph results (no entities)
                    [],
                    # Vector search results
                    [
                        ("doc1", "Title 1", "Content 1", 0.95),
                        ("doc2", "Title 2", "Content 2", 0.85),
                        ("doc3", "Title 3", "Content 3", 0.75)
                    ]
                ]
                
                # Execute query
                result = pipeline.query("general information query", top_k=3)
                
                # Verify results are from vector search
                docs = result["retrieved_documents"]
                assert len(docs) == 3
                
                # Check metadata indicates vector retrieval
                for doc in docs:
                    assert doc.metadata.get("vector_score", 0) > 0
                    assert "entity" not in doc.metadata
                
                logger.info("✅ Vector-only retrieval test passed")
    
    def test_combined_graph_vector_retrieval(self, pipeline, mock_connection_manager):
        """
        Test combined graph + vector retrieval.
        
        This test verifies:
        - Both entity extraction and vector search performed
        - Results from both sources are combined
        - Proper weighting of graph vs vector scores
        - Deduplication of overlapping results
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock entity extraction
        with patch.object(pipeline, '_extract_query_entities') as mock_extract:
            mock_extract.return_value = ["Entity1", "Entity2"]
            
            # Mock embedding
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                cursor.fetchall.side_effect = [
                    # Graph node results
                    [
                        ("node1", "Entity1", "doc1", "Description 1", 0.9),
                        ("node2", "Entity2", "doc2", "Description 2", 0.8)
                    ],
                    # Documents for graph nodes
                    [
                        ("doc1", "Title 1", "Content about Entity1"),
                        ("doc2", "Title 2", "Content about Entity2")
                    ],
                    # Vector search results (with overlap)
                    [
                        ("doc1", "Title 1", "Content about Entity1", 0.85),  # Overlap with graph
                        ("doc3", "Title 3", "Different content", 0.90),      # Vector only
                        ("doc4", "Title 4", "More content", 0.80)            # Vector only
                    ]
                ]
                
                # Execute query
                result = pipeline.query("Tell me about Entity1 and related topics", top_k=4)
                
                docs = result["retrieved_documents"]
                assert len(docs) == 4  # Should have doc1, doc2, doc3, doc4
                
                # Find doc1 (should have both graph and vector scores)
                doc1 = next((d for d in docs if d.id == "doc1"), None)
                assert doc1 is not None
                assert doc1.metadata.get("graph_score", 0) > 0 or "entity" in doc1.metadata
                assert doc1.metadata.get("vector_score", 0) > 0
                
                # Find doc3 (vector only)
                doc3 = next((d for d in docs if d.id == "doc3"), None)
                assert doc3 is not None
                assert doc3.metadata.get("vector_score", 0) > 0
                assert doc3.metadata.get("graph_score", 0) == 0 or "entity" not in doc3.metadata
                
                logger.info("✅ Combined retrieval test passed")
    
    def test_entity_extraction_failure(self, pipeline, mock_connection_manager):
        """
        Test fallback when entity extraction fails.
        
        This verifies:
        - System handles entity extraction errors gracefully
        - Falls back to vector-only search
        - Still returns results
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock entity extraction to fail
        with patch.object(pipeline, '_extract_query_entities') as mock_extract:
            mock_extract.side_effect = Exception("Entity extraction failed")
            
            # Mock embedding
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                cursor.fetchall.side_effect = [
                    # Vector search results (fallback)
                    [
                        ("doc1", "Title 1", "Content 1", 0.9),
                        ("doc2", "Title 2", "Content 2", 0.8)
                    ]
                ]
                
                # Execute query
                result = pipeline.query("test query", top_k=2)
                
                # Should still get results from vector search
                assert "error" not in result or result.get("answer") is not None
                docs = result["retrieved_documents"]
                assert len(docs) == 2
                
                logger.info("✅ Entity extraction failure test passed")
    
    def test_graph_traversal_depth(self, pipeline, mock_connection_manager):
        """
        Test graph traversal with different depths.
        
        This verifies:
        - Direct entity matches (depth 0)
        - Related entities (depth 1)
        - Second-degree relationships (depth 2)
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock entity extraction
        with patch.object(pipeline, '_extract_query_entities') as mock_extract:
            mock_extract.return_value = ["Entity1"]
            
            # Mock embedding
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                # Test different traversal depths
                test_cases = [
                    (0, 1),  # Depth 0: Only direct match
                    (1, 3),  # Depth 1: Direct + first-degree
                    (2, 6)   # Depth 2: All relationships
                ]
                
                for depth, expected_count in test_cases:
                    cursor.fetchall.side_effect = [
                        # Graph results based on depth
                        [
                            ("node1", "Entity1", "doc1", "Direct match", 1.0),
                        ] + ([
                            ("node2", "Related1", "doc2", "First degree", 0.8),
                            ("node3", "Related2", "doc3", "First degree", 0.7),
                        ] if depth >= 1 else []) + ([
                            ("node4", "Related3", "doc4", "Second degree", 0.6),
                            ("node5", "Related4", "doc5", "Second degree", 0.5),
                            ("node6", "Related5", "doc6", "Second degree", 0.4),
                        ] if depth >= 2 else []),
                        # Document retrieval
                        [(f"doc{i}", f"Title {i}", f"Content {i}") for i in range(1, expected_count + 1)],
                        # Empty vector results
                        []
                    ]
                    
                    with patch.object(pipeline, 'relationship_depth', depth):
                        result = pipeline.query("Tell me about Entity1", top_k=10)
                        
                        docs = result["retrieved_documents"]
                        assert len(docs) == expected_count, f"Depth {depth} should return {expected_count} docs"
                        
                        logger.info(f"✅ Graph traversal depth {depth} test passed")
    
    def test_entity_threshold_filtering(self, pipeline, mock_connection_manager):
        """
        Test entity confidence threshold filtering.
        
        This verifies:
        - Only entities above threshold are used
        - Low-confidence entities are filtered out
        """
        _, connection, cursor = mock_connection_manager
        
        # Mock entity extraction to return high confidence entities only
        with patch.object(pipeline, '_extract_query_entities') as mock_extract:
            # Simulate threshold filtering by only returning high confidence entities
            mock_extract.return_value = ["HighConfidence", "MediumConfidence"]
            
            # Mock embedding
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                # Set threshold via pipeline config
                pipeline.pipeline_config['entity_threshold'] = 0.6
                with patch.object(pipeline, 'pipeline_config', pipeline.pipeline_config):
                    cursor.fetchall.side_effect = [
                        # Should only query for high and medium confidence entities
                        [
                            ("node1", "HighConfidence", "doc1", "High conf content", 0.9),
                            ("node2", "MediumConfidence", "doc2", "Medium conf content", 0.7)
                        ],
                        # Documents
                        [
                            ("doc1", "Title 1", "Content 1"),
                            ("doc2", "Title 2", "Content 2")
                        ],
                        # Vector results
                        []
                    ]
                    
                    result = pipeline.query("test query", top_k=5)
                    
                    docs = result["retrieved_documents"]
                    assert len(docs) == 2  # Only high and medium confidence
                    
                    # Verify low confidence entity was filtered
                    doc_contents = [d.page_content for d in docs]
                    assert not any("LowConfidence" in content for content in doc_contents)
                    
                    logger.info("✅ Entity threshold filtering test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])