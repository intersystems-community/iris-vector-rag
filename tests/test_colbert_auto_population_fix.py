"""
Test ColBERT Auto-Population Fix.

This test validates that the ColBERT token embedding auto-population
fixes work correctly:
- Proper 768D token embeddings
- Auto-population during pipeline initialization
- Integration with TokenEmbeddingService
- No manual token embedding population required
"""

import pytest
import logging
from unittest.mock import Mock, patch
from pathlib import Path

from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from iris_rag.services.token_embedding_service import TokenEmbeddingService
from iris_rag.embeddings.colbert_interface import get_colbert_interface_from_config
from common.iris_connection_manager import get_iris_connection

logger = logging.getLogger(__name__)


class TestColBERTAutoPopulationFix:
    """Test suite for ColBERT auto-population fixes."""
    
    @pytest.fixture
    def config_manager(self):
        """Create configuration manager for testing."""
        return ConfigurationManager()
    
    @pytest.fixture
    def connection_manager(self):
        """Create connection manager for testing."""
        return type('ConnectionManager', (), {
            'get_connection': lambda: get_iris_connection()
        })()
    
    @pytest.fixture
    def mock_iris_connector(self):
        """Create mock IRIS connector."""
        mock_connector = Mock()
        mock_connection = Mock()
        mock_cursor = Mock()
        
        # Setup mock chain
        mock_connector.get_connection.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = [0]  # No existing token embeddings
        
        return mock_connector
    
    def test_colbert_interface_uses_768d_embeddings(self, config_manager, connection_manager):
        """Test that ColBERT interface uses proper 768D token embeddings."""
        # Get ColBERT interface from config
        colbert_interface = get_colbert_interface_from_config(config_manager, connection_manager)
        
        # Verify token dimension is 768D
        assert colbert_interface.get_token_dimension() == 768, \
            f"Expected 768D token embeddings, got {colbert_interface.get_token_dimension()}D"
        
        # Test query encoding produces 768D embeddings
        query_embeddings = colbert_interface.encode_query("test query")
        assert len(query_embeddings) > 0, "Query encoding should produce embeddings"
        assert len(query_embeddings[0]) == 768, \
            f"Query token embeddings should be 768D, got {len(query_embeddings[0])}D"
        
        # Test document encoding produces 768D embeddings
        doc_embeddings = colbert_interface.encode_document("test document content")
        assert len(doc_embeddings) > 0, "Document encoding should produce embeddings"
        assert len(doc_embeddings[0]) == 768, \
            f"Document token embeddings should be 768D, got {len(doc_embeddings[0])}D"
    
    def test_token_embedding_service_initialization(self, config_manager, connection_manager):
        """Test that TokenEmbeddingService initializes correctly."""
        service = TokenEmbeddingService(config_manager, connection_manager)
        
        # Verify service has proper token dimension
        assert service.token_dimension == 768, \
            f"TokenEmbeddingService should use 768D, got {service.token_dimension}D"
        
        # Verify ColBERT interface is properly initialized
        assert service.colbert_interface is not None, "ColBERT interface should be initialized"
        assert service.colbert_interface.get_token_dimension() == 768, \
            "ColBERT interface should use 768D embeddings"
    
    @patch('iris_rag.services.token_embedding_service.insert_vector')
    def test_token_embedding_service_auto_population(self, mock_insert_vector, 
                                                    config_manager, connection_manager):
        """Test that TokenEmbeddingService can auto-populate token embeddings."""
        # Setup mock to simulate successful vector insertion
        mock_insert_vector.return_value = True
        
        service = TokenEmbeddingService(config_manager, connection_manager)
        
        # Mock database queries to return test documents
        with patch.object(service.connection_manager, 'get_connection') as mock_get_conn:
            mock_connection = Mock()
            mock_cursor = Mock()
            mock_get_conn.return_value = mock_connection
            mock_connection.cursor.return_value = mock_cursor
            
            # Mock documents missing token embeddings
            mock_cursor.fetchall.return_value = [
                ('doc1', 'Test Title 1', 'Test Abstract 1', 'Test Content 1'),
                ('doc2', 'Test Title 2', 'Test Abstract 2', 'Test Content 2')
            ]
            mock_cursor.description = [
                ('doc_id',), ('title',), ('abstract',), ('content',)
            ]
            
            # Test auto-population
            stats = service.ensure_token_embeddings_exist(['doc1', 'doc2'])
            
            # Verify processing occurred
            assert stats.documents_processed >= 0, "Should process documents"
            assert stats.tokens_generated >= 0, "Should generate tokens"
            assert stats.processing_time >= 0, "Should track processing time"
    
    def test_colbert_pipeline_auto_population_integration(self, config_manager, mock_iris_connector):
        """Test that ColBERT pipeline integrates auto-population correctly."""
        # Create pipeline with mock connector
        pipeline = ColBERTRAGPipeline(
            iris_connector=mock_iris_connector,
            config_manager=config_manager
        )
        
        # Verify pipeline has proper dimensions
        assert pipeline.token_embedding_dim == 768, \
            f"Pipeline should use 768D token embeddings, got {pipeline.token_embedding_dim}D"
        
        # Verify ColBERT interface is properly configured
        assert pipeline.colbert_interface is not None, "ColBERT interface should be initialized"
        assert pipeline.colbert_interface.get_token_dimension() == 768, \
            "ColBERT interface should use 768D embeddings"
    
    @patch('iris_rag.services.token_embedding_service.TokenEmbeddingService')
    def test_colbert_pipeline_validate_setup_auto_population(self, mock_service_class, 
                                                            config_manager, mock_iris_connector):
        """Test that ColBERT pipeline validate_setup() triggers auto-population."""
        # Setup mock service
        mock_service = Mock()
        mock_stats = Mock()
        mock_stats.documents_processed = 5
        mock_stats.tokens_generated = 100
        mock_stats.processing_time = 2.5
        mock_stats.errors = 0
        mock_service.ensure_token_embeddings_exist.return_value = mock_stats
        mock_service_class.return_value = mock_service
        
        # Setup mock cursor to simulate no existing token embeddings, then some after auto-population
        mock_cursor = mock_iris_connector.get_connection.return_value.cursor.return_value
        mock_cursor.fetchone.side_effect = [
            [1],  # Table exists
            [0],  # No token embeddings initially
            [100] # Token embeddings exist after auto-population
        ]
        
        # Create pipeline
        pipeline = ColBERTRAGPipeline(
            iris_connector=mock_iris_connector,
            config_manager=config_manager
        )
        
        # Test validate_setup triggers auto-population
        result = pipeline.validate_setup()
        
        # Verify auto-population was triggered
        assert result is True, "validate_setup should succeed after auto-population"
        mock_service_class.assert_called_once()
        mock_service.ensure_token_embeddings_exist.assert_called_once()
    
    @patch('iris_rag.services.token_embedding_service.TokenEmbeddingService')
    def test_colbert_pipeline_load_documents_auto_population(self, mock_service_class,
                                                           config_manager, mock_iris_connector):
        """Test that ColBERT pipeline load_documents() triggers auto-population."""
        # Setup mock service
        mock_service = Mock()
        mock_stats = Mock()
        mock_stats.documents_processed = 3
        mock_stats.tokens_generated = 75
        mock_stats.processing_time = 1.8
        mock_stats.errors = 0
        mock_service.ensure_token_embeddings_exist.return_value = mock_stats
        mock_service_class.return_value = mock_service
        
        # Create pipeline with mock vector store
        mock_vector_store = Mock()
        pipeline = ColBERTRAGPipeline(
            iris_connector=mock_iris_connector,
            config_manager=config_manager,
            vector_store=mock_vector_store
        )
        
        # Test load_documents triggers auto-population
        pipeline.load_documents("test/path")
        
        # Verify vector store load_documents was called
        mock_vector_store.load_documents.assert_called_once_with("test/path")
        
        # Verify auto-population was triggered
        mock_service_class.assert_called_once()
        mock_service.ensure_token_embeddings_exist.assert_called_once()
    
    def test_dimension_consistency_across_components(self, config_manager, connection_manager):
        """Test that all components use consistent 768D token embeddings."""
        # Test ColBERT interface
        colbert_interface = get_colbert_interface_from_config(config_manager, connection_manager)
        assert colbert_interface.get_token_dimension() == 768
        
        # Test TokenEmbeddingService
        token_service = TokenEmbeddingService(config_manager, connection_manager)
        assert token_service.token_dimension == 768
        
        # Test schema manager token dimension
        from iris_rag.storage.schema_manager import SchemaManager
        schema_manager = SchemaManager(connection_manager, config_manager)
        assert schema_manager.get_colbert_token_dimension() == 768
        
        logger.info("All components consistently use 768D token embeddings")
    
    def test_no_manual_population_required(self, config_manager, mock_iris_connector):
        """Test that ColBERT works without requiring manual token embedding population."""
        # Setup mock to simulate auto-population success
        mock_cursor = mock_iris_connector.get_connection.return_value.cursor.return_value
        mock_cursor.fetchone.side_effect = [
            [1],   # Table exists
            [0],   # No token embeddings initially
            [100]  # Token embeddings exist after auto-population
        ]
        
        with patch('iris_rag.services.token_embedding_service.TokenEmbeddingService') as mock_service_class:
            mock_service = Mock()
            mock_stats = Mock()
            mock_stats.documents_processed = 10
            mock_stats.tokens_generated = 200
            mock_stats.processing_time = 3.0
            mock_stats.errors = 0
            mock_service.ensure_token_embeddings_exist.return_value = mock_stats
            mock_service_class.return_value = mock_service
            
            # Create and validate pipeline
            pipeline = ColBERTRAGPipeline(
                iris_connector=mock_iris_connector,
                config_manager=config_manager
            )
            
            # Pipeline should work without manual intervention
            assert pipeline.validate_setup() is True, \
                "ColBERT pipeline should work without manual token embedding population"
            
            # Verify auto-population was used
            mock_service.ensure_token_embeddings_exist.assert_called_once()
            
            logger.info("ColBERT pipeline works without manual token embedding population")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])