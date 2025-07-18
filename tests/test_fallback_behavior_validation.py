#!/usr/bin/env python3
"""
Fallback behavior validation tests.

This test suite validates that all pipelines handle failures gracefully
and have appropriate fallback mechanisms.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFallbackBehaviors:
    """
    Test fallback behaviors across all pipelines.
    """
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        manager = Mock(spec=ConnectionManager)
        connection = Mock()
        cursor = Mock()
        
        manager.get_connection.return_value = connection
        connection.cursor.return_value = cursor
        connection.commit.return_value = None
        
        return manager, connection, cursor
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config = Mock(spec=ConfigurationManager)
        config.get.return_value = {}
        return config
    
    def test_hybrid_ifind_index_creation_failure(self, mock_connection_manager, mock_config_manager):
        """
        Test Hybrid IFind handles index creation failure gracefully.
        """
        manager, connection, cursor = mock_connection_manager
        
        # Mock index creation to fail
        def execute_side_effect(sql, params=None):
            if "CREATE INDEX" in sql and "IFIND" in sql:
                raise Exception("IFind not supported in this IRIS version")
            return None
        
        cursor.execute.side_effect = execute_side_effect
        cursor.fetchone.return_value = [0]  # No existing index
        
        # Create pipeline
        pipeline = HybridIFindRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
        
        # Should not raise exception during init
        assert pipeline is not None
        
        # Pipeline should still work with vector search only
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            cursor.execute.side_effect = None  # Reset
            cursor.fetchall.return_value = [
                [("doc1", "Title", "Content", 0.9)],  # Vector results
                []  # Empty IFind results due to no index
            ]
            
            result = pipeline.query("test query")
            assert len(result["retrieved_documents"]) > 0
            
        logger.info("✅ Hybrid IFind index failure test passed")
    
    def test_graphrag_entity_extraction_failure(self, mock_connection_manager, mock_config_manager):
        """
        Test GraphRAG handles entity extraction failure.
        """
        manager, _, cursor = mock_connection_manager
        
        pipeline = GraphRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
        
        # Mock entity extraction to fail
        with patch.object(pipeline, '_extract_entities') as mock_extract:
            mock_extract.side_effect = Exception("NER model not available")
            
            # Mock embedding to work
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                cursor.fetchall.return_value = [
                    [("doc1", "Title", "Content", 0.9)]  # Vector fallback
                ]
                
                # Should fall back to vector search
                result = pipeline.query("test query")
                
                # Should not raise exception
                assert "error" not in result or result["retrieved_documents"] is not None
                assert len(result["retrieved_documents"]) > 0
                
        logger.info("✅ GraphRAG entity extraction failure test passed")
    
    def test_crag_chunking_failure(self, mock_connection_manager, mock_config_manager):
        """
        Test CRAG handles chunking failure.
        """
        manager, _, cursor = mock_connection_manager
        
        pipeline = CRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
        
        # Mock chunking to fail
        with patch.object(pipeline, '_chunk_document') as mock_chunk:
            mock_chunk.side_effect = Exception("Chunking failed")
            
            # Test document ingestion
            from iris_rag.core.models import Document
            docs = [Document(page_content="Test content", metadata={})]
            
            # Should handle chunking failure gracefully
            result = pipeline.ingest_documents(docs)
            
            # Should report error but not crash
            assert result["status"] == "error" or result.get("chunks_created", 0) == 0
            
        logger.info("✅ CRAG chunking failure test passed")
    
    def test_hyde_hypothesis_generation_failure(self, mock_connection_manager, mock_config_manager):
        """
        Test HyDE handles hypothesis generation failure.
        """
        manager, _, cursor = mock_connection_manager
        
        # Create pipeline without LLM function
        pipeline = HyDERAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager,
            llm_func=None  # No LLM available
        )
        
        # Mock embedding
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            cursor.fetchall.return_value = [
                [("doc1", "Title", "Content", 0.9)]  # Direct search results
            ]
            
            # Should fall back to direct search
            result = pipeline.query("test query")
            
            # Should still return results
            assert len(result["retrieved_documents"]) > 0
            # Should indicate no hypothesis was used
            assert result.get("hypotheses") is None or len(result.get("hypotheses", [])) == 0
            
        logger.info("✅ HyDE hypothesis generation failure test passed")
    
    def test_colbert_token_embedding_missing(self, mock_connection_manager, mock_config_manager):
        """
        Test ColBERT handles missing token embeddings.
        """
        manager, _, cursor = mock_connection_manager
        
        pipeline = ColBERTRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
        
        # Mock to simulate no token embeddings
        cursor.fetchone.return_value = [0]  # No token embeddings
        
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            # Mock tokenizer
            with patch.object(pipeline, '_tokenize_text') as mock_tokenize:
                mock_tokenize.return_value = ["test", "query"]
                
                cursor.fetchall.side_effect = [
                    [],  # No token matches
                    [("doc1", "Title", "Content", 0.9)]  # Vector fallback
                ]
                
                # Should fall back to vector search
                result = pipeline.query("test query")
                
                # Should still return results from vector search
                assert len(result["retrieved_documents"]) > 0
                
        logger.info("✅ ColBERT missing token embeddings test passed")
    
    def test_embedding_generation_failure(self, mock_connection_manager, mock_config_manager):
        """
        Test all pipelines handle embedding generation failure.
        """
        manager, _, cursor = mock_connection_manager
        
        pipelines = [
            HybridIFindRAGPipeline(manager, mock_config_manager),
            GraphRAGPipeline(manager, mock_config_manager),
            CRAGPipeline(manager, mock_config_manager),
            HyDERAGPipeline(manager, mock_config_manager),
            ColBERTRAGPipeline(manager, mock_config_manager)
        ]
        
        for pipeline in pipelines:
            # Mock embedding to fail
            with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
                mock_embed.side_effect = Exception("Embedding model not available")
                
                # Each pipeline should handle this gracefully
                result = pipeline.query("test query")
                
                # Should return error result, not crash
                assert "error" in result or result["retrieved_documents"] == []
                assert result.get("pipeline_type") is not None
                
                logger.info(f"✅ {pipeline.__class__.__name__} embedding failure test passed")
    
    def test_database_connection_failure(self, mock_config_manager):
        """
        Test all pipelines handle database connection failure.
        """
        # Create manager that fails to connect
        manager = Mock(spec=ConnectionManager)
        manager.get_connection.side_effect = Exception("Database connection failed")
        
        pipelines = [
            HybridIFindRAGPipeline,
            GraphRAGPipeline,
            CRAGPipeline,
            HyDERAGPipeline,
            ColBERTRAGPipeline
        ]
        
        for pipeline_class in pipelines:
            # Should handle connection failure during query
            pipeline = pipeline_class(
                connection_manager=manager,
                config_manager=mock_config_manager
            )
            
            result = pipeline.query("test query")
            
            # Should return error result
            assert "error" in result or result["retrieved_documents"] == []
            assert result.get("pipeline_type") is not None
            
            logger.info(f"✅ {pipeline_class.__name__} connection failure test passed")
    
    def test_partial_results_handling(self, mock_connection_manager, mock_config_manager):
        """
        Test pipelines return partial results when some components fail.
        """
        manager, _, cursor = mock_connection_manager
        
        # Test Hybrid IFind with partial results
        pipeline = HybridIFindRAGPipeline(
            connection_manager=manager,
            config_manager=mock_config_manager
        )
        
        with patch.object(pipeline.embedding_manager, 'embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 384
            
            # Mock IFind to fail but vector to succeed
            def execute_side_effect(sql, params=None):
                if "$FIND" in sql:
                    raise Exception("IFind query failed")
                return None
            
            cursor.execute.side_effect = execute_side_effect
            cursor.fetchall.side_effect = [
                [("doc1", "Title", "Content", 0.9)],  # Vector results
                []  # Empty due to IFind failure
            ]
            
            result = pipeline.query("test query")
            
            # Should still return vector results
            assert len(result["retrieved_documents"]) > 0
            assert result["vector_results_count"] > 0
            assert result["ifind_results_count"] == 0
            
        logger.info("✅ Partial results handling test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])