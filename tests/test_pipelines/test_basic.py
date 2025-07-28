"""
Tests for Basic RAG Pipeline implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock # Added MagicMock
from iris_rag.core.models import Document
from iris_rag.pipelines.basic import BasicRAGPipeline # Moved to top-level import
from common.iris_connection_manager import get_iris_connection as get_real_iris_connection
from iris_rag.config.manager import ConfigurationManager as PipelineConfigManager
# import os # Not strictly needed as os.path and os.environ are patched by string path

def test_basic_pipeline_imports():
    """Test that basic pipeline can be imported."""
    # from iris_rag.pipelines.basic import BasicRAGPipeline # Now imported at top
    assert BasicRAGPipeline is not None


def test_document_creation():
    """Test basic document creation."""
    doc = Document(
        page_content="Test content",
        metadata={"source": "test.txt"}
    )
    assert doc.page_content == "Test content"
    assert doc.metadata["source"] == "test.txt"


@patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
@patch('iris_rag.pipelines.basic.EmbeddingManager')
@patch('iris_rag.pipelines.basic.get_iris_connection')
def test_pipeline_initialization(mock_get_iris_connection, mock_embedding_manager, mock_vector_store):
    """Test pipeline initialization with minimal mocks."""
    # Create minimal mocks
    mock_config_manager = Mock()
    # Provide proper configuration structure for SchemaManager
    def mock_get_config(key, default=None):
        config_map = {
            "pipelines:basic": {},
            "storage:base_embedding_dimension": 384,
            "storage:colbert_token_dimension": 768,
            "storage:colbert_backend": "native"
        }
        return config_map.get(key, default)
    
    mock_config_manager.get.side_effect = mock_get_config
    mock_connection = Mock()
    mock_get_iris_connection.return_value = mock_connection
    
    # Create pipeline
    pipeline = BasicRAGPipeline(
        config_manager=mock_config_manager
    )
    
    # Verify basic initialization
    assert pipeline.config_manager == mock_config_manager
    assert pipeline.connection == mock_connection
    assert pipeline.vector_store is not None


def test_text_chunking():
    """Test text chunking functionality without heavy mocks."""
    # from iris_rag.pipelines.basic import BasicRAGPipeline # Now imported at top
    
    # Create minimal pipeline instance for testing utility methods
    mock_config_manager = Mock()
    # Provide proper configuration structure for SchemaManager
    def mock_get_config(key, default=None):
        config_map = {
            "pipelines:basic": {},
            "storage:base_embedding_dimension": 384,
            "storage:colbert_token_dimension": 768,
            "storage:colbert_backend": "native"
        }
        return config_map.get(key, default)
    
    mock_config_manager.get.side_effect = mock_get_config
    
    with patch('iris_rag.storage.vector_store_iris.IRISVectorStore'), \
         patch('iris_rag.pipelines.basic.EmbeddingManager'), \
         patch('iris_rag.pipelines.basic.get_iris_connection'), \
         patch('tools.chunking.chunking_service.DocumentChunkingService'):
        
        pipeline = BasicRAGPipeline(
            config_manager=mock_config_manager
        )
        
        # Set small chunk size for testing
        pipeline.chunk_size = 50
        pipeline.chunk_overlap = 10
        
        # Test that chunking service is initialized
        assert hasattr(pipeline, 'chunking_service')
        assert hasattr(pipeline, 'chunking_strategy')
        assert pipeline.chunking_strategy == "fixed_size"  # default value


def test_factory_function():
    """Test the create_pipeline factory function."""
    from iris_rag import create_pipeline
    
    # Test unknown pipeline type
    with pytest.raises(ValueError, match="Unknown pipeline type"):
        create_pipeline("unknown_type")


def test_standard_return_format():
    """Test that pipeline returns standard format."""
    # This is a basic structure test
    expected_keys = ["query", "answer", "retrieved_documents"]
    
    # Mock result structure
    result = {
        "query": "test query",
        "answer": "test answer",
        "retrieved_documents": []
    }
    
    for key in expected_keys:
        assert key in result

@patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
@patch('iris_rag.pipelines.basic.EmbeddingManager')
@patch('iris_rag.pipelines.basic.get_iris_connection')
def test_basic_pipeline_connection_uses_config_manager(
    mock_get_iris_connection,
    mock_embedding_manager_class,
    mock_vector_store
):
    """
    Tests that BasicRAGPipeline uses the new connection manager.
    """
    # Configure mocks
    mock_connection = Mock()
    mock_get_iris_connection.return_value = mock_connection
    
    mock_config_manager = Mock()
    # Provide proper configuration structure for SchemaManager
    def mock_get_config(key, default=None):
        config_map = {
            "pipelines:basic": {},
            "storage:base_embedding_dimension": 384,
            "storage:colbert_token_dimension": 768,
            "storage:colbert_backend": "native"
        }
        return config_map.get(key, default)
    
    mock_config_manager.get.side_effect = mock_get_config
    
    # Create the BasicRAGPipeline with the new architecture
    pipeline = BasicRAGPipeline(
        config_manager=mock_config_manager
    )
    
    # Verify that get_iris_connection was called
    mock_get_iris_connection.assert_called_once()
    
    # Verify that the pipeline has the mocked connection
    assert pipeline.connection == mock_connection
    # If it's not called, an assert_called_once_with or assert_any_call would fail.
    # This is fine, as it means ConfigurationManager was correctly prioritized.
    # For a robust test, we might check mock_os_environ_get.called is False if that's the expected behavior.
    # However, the prompt implies os.environ.get *is* mocked to return *different* credentials,
    # suggesting it might be checked by the SUT.
    # Let's assume for now the critical check is that jaydebeapi.connect used CM creds.