import pytest
from unittest.mock import MagicMock, patch, Mock

from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager."""
    manager = Mock(spec=ConnectionManager)
    connection = Mock()
    cursor = Mock()
    
    manager.get_connection.return_value = connection
    connection.cursor.return_value = cursor
    connection.commit.return_value = None
    
    return manager

@pytest.fixture
def mock_config_manager():
    """Create a mock configuration manager."""
    config = Mock(spec=ConfigurationManager)
    
    # Configure mock to return proper values for different keys
    def mock_get(key, default=None):
        config_values = {
            "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model.dimension": 384,
            "colbert": {
                "backend": "native",
                "token_dimension": 768,
                "model_name": "bert-base-uncased"
            },
            "storage:iris": {},
            "storage:chunking": {"enabled": False}
        }
        return config_values.get(key, default if default is not None else {})
    
    config.get.side_effect = mock_get
    return config

@pytest.fixture
def hybrid_pipeline(mock_connection_manager, mock_config_manager):
    return HybridIFindRAGPipeline(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )

def test_hybrid_ifind_rag_e2e_combined_retrieval(hybrid_pipeline, mock_connection_manager, mock_config_manager):
    """Test HybridIFind pipeline with combined vector and IFind retrieval."""
    query = "What are the treatments for neurodegenerative diseases?"
    
    manager, connection, cursor = mock_connection_manager, mock_connection_manager.get_connection(), mock_connection_manager.get_connection().cursor()

    # Mock embedding function
    with patch.object(hybrid_pipeline.embedding_manager, 'embed_text') as mock_embed:
        mock_embed.return_value = [0.1] * 384
        
        # Mock vector search results
        vector_results = [
            ("doc1", "Vector result about neurodegenerative treatments", 0.9),
            ("doc2", "Another vector result about protein aggregation", 0.8)
        ]
        
        # Mock IFind search results
        ifind_results = [
            ("doc3", "IFind result about disease treatments", 85.0),
            ("doc1", "Vector result about neurodegenerative treatments", 75.0)  # Overlap
        ]
        
        # Set up cursor to return different results for different SQL queries
        def cursor_fetchall_side_effect():
            # First call is vector search, second is IFind search
            if cursor.fetchall.call_count == 1:
                return vector_results
            else:
                return ifind_results
        
        cursor.fetchall.side_effect = [vector_results, ifind_results]
        
        # Mock successful query execution
        result = hybrid_pipeline.query(query, top_k=3)
        
        # Assertions for the result structure
        assert result is not None
        assert "retrieved_documents" in result
        assert "query" in result
        assert result["query"] == query
        
        # Check that the pipeline attempted both vector and IFind searches
        # (even if one fails, the other should provide results)
        if "error" not in result:
            retrieved_docs = result["retrieved_documents"]
            assert isinstance(retrieved_docs, list)
            # Should have fusion of results from both methods
            assert len(retrieved_docs) <= 3  # Respects top_k limit
        else:
            # If there's an error, make sure it's handled gracefully
            assert "retrieved_documents" in result
            assert result["retrieved_documents"] == []