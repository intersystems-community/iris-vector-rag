"""
Test to validate SQLRAG pipeline constructor fix.

This test ensures that SQLRAG can be initialized with the standard RAG pipeline pattern
like other pipelines (BasicRAG, HyDE, etc.).
"""

import pytest
from unittest.mock import Mock, MagicMock
from iris_rag.pipelines.sql_rag import SQLRAGPipeline
from iris_rag.config.manager import ConfigurationManager


def create_mock_config_manager():
    """Create a properly configured mock config manager."""
    mock_config_manager = Mock(spec=ConfigurationManager)
    
    # Mock configuration responses for SchemaManager
    def mock_get(key, default=None):
        config_map = {
            "storage:base_embedding_dimension": 384,
            "storage:colbert_token_dimension": 128,
            "storage:colbert_backend": "native",
            "pipelines:sql_rag": {},
            "pipeline_overrides:sql_rag:chunking": {}
        }
        return config_map.get(key, default if default is not None else {})
    
    mock_config_manager.get.side_effect = mock_get
    mock_config_manager.get_config.side_effect = mock_get
    return mock_config_manager


class TestSQLRAGConstructorFix:
    """Test SQLRAG constructor follows standard RAG pipeline pattern."""
    
    def test_sqlrag_constructor_with_config_manager_only(self):
        """Test SQLRAG can be initialized with just config_manager like other pipelines."""
        # Create mock config manager with proper configuration values
        mock_config_manager = create_mock_config_manager()
        
        # This should work without throwing "missing required positional argument" error
        try:
            pipeline = SQLRAGPipeline(config_manager=mock_config_manager)
            assert pipeline is not None
            assert hasattr(pipeline, 'config_manager')
            assert hasattr(pipeline, 'vector_store')
            assert hasattr(pipeline, 'llm_func')
            print("✓ SQLRAG initialized successfully with config_manager only")
        except TypeError as e:
            if "missing 1 required positional argument: 'connection_manager'" in str(e):
                pytest.fail(f"SQLRAG constructor still has the old signature issue: {e}")
            else:
                pytest.fail(f"Unexpected TypeError: {e}")
    
    def test_sqlrag_constructor_with_all_standard_params(self):
        """Test SQLRAG can be initialized with standard RAG pipeline parameters."""
        # Create mock config manager with proper configuration values
        mock_config_manager = create_mock_config_manager()
        
        # Create mock LLM function
        mock_llm_func = Mock()
        mock_llm_func.return_value = "Mock LLM response"
        
        # Create mock vector store
        mock_vector_store = Mock()
        
        # This should work with the standard pattern used by BasicRAG and HyDE
        try:
            pipeline = SQLRAGPipeline(
                config_manager=mock_config_manager,
                llm_func=mock_llm_func,
                vector_store=mock_vector_store
            )
            assert pipeline is not None
            assert pipeline.config_manager == mock_config_manager
            assert pipeline.llm_func == mock_llm_func
            assert pipeline.vector_store == mock_vector_store
            print("✓ SQLRAG initialized successfully with all standard parameters")
        except Exception as e:
            pytest.fail(f"SQLRAG failed to initialize with standard parameters: {e}")
    
    def test_sqlrag_constructor_matches_basicrag_pattern(self):
        """Test SQLRAG constructor signature matches BasicRAG pattern."""
        # Create mock config manager with proper configuration values
        mock_config_manager = create_mock_config_manager()
        
        # Test the exact same pattern used by BasicRAG
        try:
            # This is how BasicRAG is typically initialized in tests
            pipeline = SQLRAGPipeline(config_manager=mock_config_manager)
            
            # Verify it inherits from RAGPipeline properly
            from iris_rag.core.base import RAGPipeline
            assert isinstance(pipeline, RAGPipeline)
            
            # Verify it has the required abstract methods implemented
            assert hasattr(pipeline, 'execute')
            assert hasattr(pipeline, 'query')
            assert callable(pipeline.execute)
            assert callable(pipeline.query)
            
            print("✓ SQLRAG follows the same constructor pattern as BasicRAG")
        except Exception as e:
            pytest.fail(f"SQLRAG constructor doesn't match BasicRAG pattern: {e}")
    
    def test_sqlrag_constructor_backward_compatibility(self):
        """Test that old-style initialization fails with clear error message."""
        # Create mock managers
        mock_connection_manager = Mock()
        mock_config_manager = create_mock_config_manager()
        
        # The old broken pattern should not work
        with pytest.raises(TypeError):
            # This was the old broken signature that caused the issue
            SQLRAGPipeline(mock_connection_manager, mock_config_manager)
        
        print("✓ Old constructor pattern correctly rejected")


if __name__ == "__main__":
    # Run the tests directly for quick validation
    test_instance = TestSQLRAGConstructorFix()
    
    print("Testing SQLRAG constructor fix...")
    
    try:
        test_instance.test_sqlrag_constructor_with_config_manager_only()
        test_instance.test_sqlrag_constructor_with_all_standard_params()
        test_instance.test_sqlrag_constructor_matches_basicrag_pattern()
        test_instance.test_sqlrag_constructor_backward_compatibility()
        print("\n🎉 All SQLRAG constructor tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise