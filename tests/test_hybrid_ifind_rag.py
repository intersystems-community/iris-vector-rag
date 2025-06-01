"""
Test suite for Hybrid iFind+Graph+Vector RAG Pipeline

This module contains comprehensive tests for the hybrid RAG pipeline that combines
iFind keyword search, graph-based retrieval, and vector similarity search with
reciprocal rank fusion.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import sys # Added for sys.path manipulation
import os # Added for os.path manipulation

# Add project root to sys.path to allow for absolute imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline, create_hybrid_ifind_rag_pipeline # Updated import

logger = logging.getLogger(__name__)


class TestHybridiFindRAGPipeline:
    """Test cases for the Hybrid iFind RAG Pipeline."""
    
    @pytest.fixture
    def mock_iris_connector(self):
        """Create a mock IRIS connector for testing."""
        connector = Mock()
        connector.execute_query = Mock()
        return connector
    
    @pytest.fixture
    def mock_embedding_func(self):
        """Create a mock embedding function."""
        def mock_embed(text):
            # Return a simple mock embedding based on text length
            return [0.1] * min(len(text), 384)
        return mock_embed
    
    @pytest.fixture
    def mock_llm_func(self):
        """Create a mock LLM function."""
        def mock_llm(prompt):
            return f"Mock response for: {prompt[:50]}..."
        return mock_llm
    
    @pytest.fixture
    def pipeline(self, mock_iris_connector, mock_embedding_func, mock_llm_func):
        """Create a test pipeline instance."""
        return HybridiFindRAGPipeline(
            iris_connector=mock_iris_connector,
            embedding_func=mock_embedding_func,
            llm_func=mock_llm_func
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization with default configuration."""
        assert pipeline.config['ifind_weight'] == 0.33
        assert pipeline.config['graph_weight'] == 0.33
        assert pipeline.config['vector_weight'] == 0.34
        assert pipeline.config['rrf_k'] == 60
        assert pipeline.config['max_results_per_method'] == 20
        assert pipeline.config['final_results_limit'] == 10
    
    def test_config_update(self, pipeline):
        """Test configuration parameter updates."""
        pipeline.update_config(
            ifind_weight=0.5,
            graph_weight=0.3,
            vector_weight=0.2,
            rrf_k=100
        )
        
        assert pipeline.config['ifind_weight'] == 0.5
        assert pipeline.config['graph_weight'] == 0.3
        assert pipeline.config['vector_weight'] == 0.2
        assert pipeline.config['rrf_k'] == 100
    
    def test_extract_keywords(self, pipeline):
        """Test keyword extraction from queries."""
        query = "What are the effects of machine learning on healthcare?"
        keywords = pipeline._extract_keywords(query)
        
        # Should extract meaningful keywords, excluding stop words
        assert "effects" in keywords
        assert "machine" in keywords
        assert "learning" in keywords
        assert "healthcare" in keywords
        
        # Should exclude stop words
        assert "what" not in keywords
        assert "are" not in keywords
        assert "the" not in keywords
        assert "of" not in keywords
        assert "on" not in keywords
    
    def test_extract_keywords_empty_query(self, pipeline):
        """Test keyword extraction with empty query."""
        keywords = pipeline._extract_keywords("")
        assert keywords == []
    
    def test_extract_keywords_short_words(self, pipeline):
        """Test that short words are filtered out."""
        query = "AI ML is good"
        keywords = pipeline._extract_keywords(query)
        
        # Short words should be filtered out
        assert "AI" not in keywords  # Too short
        assert "ML" not in keywords  # Too short
        assert "is" not in keywords  # Stop word and short
        assert "good" in keywords    # Should be included
    
    def test_ifind_keyword_search(self, pipeline):
        """Test iFind keyword search functionality."""
        # Mock database response
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "Test Document 1", "Content about machine learning", '{"type": "research"}', 1),
            (2, "Test Document 2", "Content about healthcare AI", '{"type": "medical"}', 2)
        ]
        pipeline.iris_connector.execute_query.return_value = mock_cursor
        
        keywords = ["machine", "learning"]
        results = pipeline._ifind_keyword_search(keywords)
        
        assert len(results) == 2
        assert results[0]['document_id'] == 1
        assert results[0]['title'] == "Test Document 1"
        assert results[0]['method'] == 'ifind'
        assert results[0]['rank_position'] == 1
        
        # Verify SQL query was called
        pipeline.iris_connector.execute_query.assert_called_once()
    
    def test_ifind_keyword_search_empty_keywords(self, pipeline):
        """Test iFind search with empty keywords."""
        results = pipeline._ifind_keyword_search([])
        assert results == []
        
        # Should not call database
        pipeline.iris_connector.execute_query.assert_not_called()
    
    def test_graph_retrieval(self, pipeline):
        """Test graph-based retrieval functionality."""
        # Mock database response
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "Graph Document 1", "Content with entities", '{"type": "research"}', 0.8, 1),
            (3, "Graph Document 2", "Related content", '{"type": "analysis"}', 0.6, 2)
        ]
        pipeline.iris_connector.execute_query.return_value = mock_cursor
        
        query = "machine learning healthcare"
        results = pipeline._graph_retrieval(query)
        
        assert len(results) == 2
        assert results[0]['document_id'] == 1
        assert results[0]['method'] == 'graph'
        assert results[0]['relationship_strength'] == 0.8
        assert results[0]['rank_position'] == 1
        
        # Verify SQL query was called
        pipeline.iris_connector.execute_query.assert_called_once()
    
    def test_vector_similarity_search(self, pipeline):
        """Test vector similarity search functionality."""
        # Mock database response
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "Vector Document 1", "Similar content", '{"type": "research"}', 0.95, 1),
            (4, "Vector Document 2", "Related content", '{"type": "analysis"}', 0.87, 2)
        ]
        pipeline.iris_connector.execute_query.return_value = mock_cursor
        
        query = "machine learning applications"
        results = pipeline._vector_similarity_search(query)
        
        assert len(results) == 2
        assert results[0]['document_id'] == 1
        assert results[0]['method'] == 'vector'
        assert results[0]['similarity_score'] == 0.95
        assert results[0]['rank_position'] == 1
        
        # Verify embedding function was called
        assert pipeline.iris_connector.execute_query.called
    
    def test_reciprocal_rank_fusion(self, pipeline):
        """Test reciprocal rank fusion algorithm."""
        # Create mock results from different methods
        ifind_results = [
            {'document_id': 1, 'title': 'Doc 1', 'content': 'Content 1', 'metadata': '{}', 'rank_position': 1},
            {'document_id': 2, 'title': 'Doc 2', 'content': 'Content 2', 'metadata': '{}', 'rank_position': 2}
        ]
        
        graph_results = [
            {'document_id': 1, 'title': 'Doc 1', 'content': 'Content 1', 'metadata': '{}', 'rank_position': 2, 'relationship_strength': 0.8},
            {'document_id': 3, 'title': 'Doc 3', 'content': 'Content 3', 'metadata': '{}', 'rank_position': 1, 'relationship_strength': 0.9}
        ]
        
        vector_results = [
            {'document_id': 2, 'title': 'Doc 2', 'content': 'Content 2', 'metadata': '{}', 'rank_position': 1, 'similarity_score': 0.95},
            {'document_id': 3, 'title': 'Doc 3', 'content': 'Content 3', 'metadata': '{}', 'rank_position': 2, 'similarity_score': 0.87}
        ]
        
        fused_results = pipeline._reciprocal_rank_fusion(ifind_results, graph_results, vector_results)
        
        # Should have 3 unique documents
        assert len(fused_results) == 3
        
        # Results should be sorted by RRF score
        assert fused_results[0]['rrf_score'] >= fused_results[1]['rrf_score']
        assert fused_results[1]['rrf_score'] >= fused_results[2]['rrf_score']
        
        # Check that method tracking works
        doc1_result = next(r for r in fused_results if r['document_id'] == 1)
        assert 'ifind' in doc1_result['methods_used']
        assert 'graph' in doc1_result['methods_used']
        assert doc1_result['method_count'] == 2
        
        doc2_result = next(r for r in fused_results if r['document_id'] == 2)
        assert 'ifind' in doc2_result['methods_used']
        assert 'vector' in doc2_result['methods_used']
        assert doc2_result['method_count'] == 2
        
        doc3_result = next(r for r in fused_results if r['document_id'] == 3)
        assert 'graph' in doc3_result['methods_used']
        assert 'vector' in doc3_result['methods_used']
        assert doc3_result['method_count'] == 2
    
    def test_reciprocal_rank_fusion_empty_results(self, pipeline):
        """Test RRF with empty result sets."""
        fused_results = pipeline._reciprocal_rank_fusion([], [], [])
        assert fused_results == []
    
    def test_retrieve_documents(self, pipeline):
        """Test complete document retrieval process."""
        # Mock all three retrieval methods
        with patch.object(pipeline, '_ifind_keyword_search') as mock_ifind, \
             patch.object(pipeline, '_graph_retrieval') as mock_graph, \
             patch.object(pipeline, '_vector_similarity_search') as mock_vector, \
             patch.object(pipeline, '_reciprocal_rank_fusion') as mock_fusion:
            
            # Set up mock returns
            mock_ifind.return_value = [{'document_id': 1, 'method': 'ifind'}]
            mock_graph.return_value = [{'document_id': 2, 'method': 'graph'}]
            mock_vector.return_value = [{'document_id': 3, 'method': 'vector'}]
            mock_fusion.return_value = [{'document_id': 1, 'rrf_score': 0.8}]
            
            query = "test query"
            results = pipeline.retrieve_documents(query)
            
            # Verify all methods were called
            mock_ifind.assert_called_once()
            mock_graph.assert_called_once_with(query)
            mock_vector.assert_called_once_with(query)
            mock_fusion.assert_called_once()
            
            # Verify results
            assert len(results) == 1
            assert results[0]['document_id'] == 1
    
    def test_generate_response(self, pipeline):
        """Test response generation with retrieved documents."""
        retrieved_docs = [
            {
                'document_id': 1,
                'title': 'Test Document',
                'content': 'This is test content about machine learning.',
                'methods_used': ['ifind', 'vector'],
                'rrf_score': 0.85
            }
        ]
        
        query = "What is machine learning?"
        response = pipeline.generate_response(query, retrieved_docs)
        
        # Should return a mock response
        assert response.startswith("Mock response for:")
        assert "What is machine learning?" in response
    
    def test_generate_response_no_documents(self, pipeline):
        """Test response generation with no retrieved documents."""
        response = pipeline.generate_response("test query", [])
        assert "couldn't find any relevant documents" in response
    
    def test_complete_query_pipeline(self, pipeline):
        """Test the complete query pipeline end-to-end."""
        # Mock the retrieve_documents method
        mock_docs = [
            {
                'document_id': 1,
                'title': 'Test Document',
                'content': 'Test content',
                'metadata': '{}',
                'methods_used': ['ifind', 'vector'],
                'rrf_score': 0.85
            }
        ]
        
        with patch.object(pipeline, 'retrieve_documents') as mock_retrieve:
            mock_retrieve.return_value = mock_docs
            
            query = "test query"
            result = pipeline.query(query)
            
            # Verify result structure
            assert result['query'] == query
            assert 'answer' in result
            assert result['retrieved_documents'] == mock_docs
            assert 'metadata' in result
            assert 'total_time' in result['metadata']
            assert result['metadata']['num_documents'] == 1
    
    def test_query_with_error(self, pipeline):
        """Test query handling when errors occur."""
        # Mock retrieve_documents to raise an exception
        with patch.object(pipeline, 'retrieve_documents') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Test error")
            
            query = "test query"
            result = pipeline.query(query)
            
            # Should handle error gracefully
            assert result['query'] == query
            assert "Error processing query" in result['answer']
            assert result['retrieved_documents'] == []
            assert 'error' in result['metadata']
    
    def test_factory_function(self, mock_iris_connector):
        """Test the factory function for creating pipeline instances."""
        pipeline = create_hybrid_ifind_rag_pipeline(mock_iris_connector)
        
        assert isinstance(pipeline, HybridiFindRAGPipeline)
        assert pipeline.iris_connector == mock_iris_connector
    
    def test_rrf_score_calculation(self, pipeline):
        """Test RRF score calculation with different rank combinations."""
        # Test case where document appears in all three methods
        ifind_results = [{'document_id': 1, 'title': 'Doc 1', 'content': 'Content', 'metadata': '{}', 'rank_position': 1}]
        graph_results = [{'document_id': 1, 'title': 'Doc 1', 'content': 'Content', 'metadata': '{}', 'rank_position': 2, 'relationship_strength': 0.8}]
        vector_results = [{'document_id': 1, 'title': 'Doc 1', 'content': 'Content', 'metadata': '{}', 'rank_position': 3, 'similarity_score': 0.9}]
        
        fused_results = pipeline._reciprocal_rank_fusion(ifind_results, graph_results, vector_results)
        
        assert len(fused_results) == 1
        result = fused_results[0]
        
        # Calculate expected RRF score
        expected_score = (0.33 / (60 + 1)) + (0.33 / (60 + 2)) + (0.34 / (60 + 3))
        assert abs(result['rrf_score'] - expected_score) < 0.001
        
        # Should have all three methods
        assert len(result['methods_used']) == 3
        assert result['method_count'] == 3
    
    def test_keyword_extraction_special_characters(self, pipeline):
        """Test keyword extraction with special characters and numbers."""
        query = "COVID-19 and machine-learning algorithms for healthcare-AI"
        keywords = pipeline._extract_keywords(query)
        
        # Should handle hyphenated terms and numbers
        assert any("covid" in k.lower() for k in keywords)
        assert any("machine" in k.lower() for k in keywords)
        assert any("learning" in k.lower() for k in keywords)
        assert any("algorithms" in k.lower() for k in keywords)
        assert any("healthcare" in k.lower() for k in keywords)
    
    def test_configuration_validation(self, pipeline):
        """Test that configuration weights are properly applied."""
        # Update to keyword-focused configuration
        pipeline.update_config(
            ifind_weight=0.6,
            graph_weight=0.2,
            vector_weight=0.2
        )
        
        # Create test results where iFind has lower rank but higher weight
        ifind_results = [{'document_id': 1, 'title': 'Doc 1', 'content': 'Content', 'metadata': '{}', 'rank_position': 3}]
        graph_results = [{'document_id': 1, 'title': 'Doc 1', 'content': 'Content', 'metadata': '{}', 'rank_position': 1, 'relationship_strength': 0.8}]
        vector_results = [{'document_id': 1, 'title': 'Doc 1', 'content': 'Content', 'metadata': '{}', 'rank_position': 2, 'similarity_score': 0.9}]
        
        fused_results = pipeline._reciprocal_rank_fusion(ifind_results, graph_results, vector_results)
        
        # Calculate expected score with new weights
        expected_score = (0.6 / (60 + 3)) + (0.2 / (60 + 1)) + (0.2 / (60 + 2))
        assert abs(fused_results[0]['rrf_score'] - expected_score) < 0.001


class TestHybridiFindRAGIntegration:
    """Integration tests for the Hybrid iFind RAG Pipeline."""
    
    @pytest.mark.integration
    def test_with_real_iris_connection(self, iris_connector):
        """Test pipeline with real IRIS connection (requires test database)."""
        # This test requires a real IRIS connection and test data
        # Skip if not in integration test environment
        pytest.skip("Integration test requires real IRIS database")
    
    @pytest.mark.performance
    def test_performance_with_large_dataset(self, pipeline):
        """Test pipeline performance with large result sets."""
        # Mock large result sets
        large_ifind_results = [
            {'document_id': i, 'title': f'Doc {i}', 'content': f'Content {i}', 'metadata': '{}', 'rank_position': i}
            for i in range(1, 101)  # 100 results
        ]
        
        large_graph_results = [
            {'document_id': i, 'title': f'Doc {i}', 'content': f'Content {i}', 'metadata': '{}', 'rank_position': i, 'relationship_strength': 0.5}
            for i in range(50, 150)  # 100 results, 50% overlap
        ]
        
        large_vector_results = [
            {'document_id': i, 'title': f'Doc {i}', 'content': f'Content {i}', 'metadata': '{}', 'rank_position': i-49, 'similarity_score': 0.8}
            for i in range(75, 175)  # 100 results, different overlap
        ]
        
        import time
        start_time = time.time()
        
        fused_results = pipeline._reciprocal_rank_fusion(
            large_ifind_results, large_graph_results, large_vector_results
        )
        
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for this size)
        assert end_time - start_time < 1.0
        
        # Should return limited results
        assert len(fused_results) <= pipeline.config['final_results_limit']
        
        # Results should be properly sorted
        for i in range(len(fused_results) - 1):
            assert fused_results[i]['rrf_score'] >= fused_results[i + 1]['rrf_score']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])