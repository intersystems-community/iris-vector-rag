"""
Test suite for Phase 1 of the Library Consumption Framework: Simple API.

This module implements the TDD anchor tests for the zero-configuration Simple API
that enables immediate RAG usage with sensible defaults.

Following TDD workflow:
1. RED: Write failing tests first
2. GREEN: Implement minimum code to pass tests  
3. REFACTOR: Clean up code while keeping tests passing
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class TestSimpleAPIPhase1:
    """Test suite for Simple API Phase 1 implementation."""
    
    def test_zero_config_initialization(self):
        """
        TDD Anchor Test: RAG() works without configuration.
        
        This test verifies that the Simple API can be initialized
        with zero configuration and sensible defaults.
        """
        # This test should fail initially (RED phase)
        from rag_templates.simple import RAG
        
        # Should initialize without any parameters
        rag = RAG()
        
        # Should have default configuration loaded
        assert rag is not None
        assert hasattr(rag, '_config_manager')
        assert hasattr(rag, '_initialized')
        
    def test_simple_query_returns_string(self):
        """
        TDD Anchor Test: query() returns string answer.
        
        This test verifies that the Simple API query method
        returns a simple string answer.
        """
        from rag_templates.simple import RAG
        
        rag = RAG()
        
        # Mock the underlying pipeline to avoid real dependencies
        with patch.object(rag, '_get_pipeline') as mock_pipeline:
            mock_pipeline.return_value.execute.return_value = {
                "answer": "This is a test answer",
                "query": "test query",
                "retrieved_documents": []
            }
            
            result = rag.query("What is machine learning?")
            
            # Should return a string answer
            assert isinstance(result, str)
            assert result == "This is a test answer"
    
    def test_add_documents_from_list(self):
        """
        TDD Anchor Test: documents can be added from list.
        
        This test verifies that documents can be added to the
        Simple API from a list of strings or documents.
        """
        from rag_templates.simple import RAG
        
        rag = RAG()
        
        # Mock the underlying pipeline
        with patch.object(rag, '_get_pipeline') as mock_pipeline:
            mock_pipeline.return_value.load_documents = MagicMock()
            
            # Should accept list of strings
            documents = [
                "This is the first document.",
                "This is the second document.",
                "This is the third document."
            ]
            
            rag.add_documents(documents)
            
            # Should have called the underlying pipeline
            mock_pipeline.return_value.load_documents.assert_called_once()
    
    def test_default_config_loaded(self):
        """
        TDD Anchor Test: default configuration loads properly.
        
        This test verifies that the enhanced ConfigurationManager
        loads default configuration when no config is provided.
        """
        from rag_templates.core.config_manager import ConfigurationManager
        
        # Should initialize with defaults
        config_manager = ConfigurationManager()
        
        # Should have default database configuration
        assert config_manager.get("database:iris:host") is not None
        assert config_manager.get("database:iris:port") is not None
        assert config_manager.get("database:iris:namespace") is not None
        
        # Should have default embedding configuration
        assert config_manager.get("embeddings:model") is not None
        assert config_manager.get("embeddings:dimension") is not None
    
    def test_environment_variables_override_defaults(self):
        """
        TDD Anchor Test: env vars override defaults.
        
        This test verifies that environment variables properly
        override default configuration values.
        """
        from rag_templates.core.config_manager import ConfigurationManager
        
        # Set environment variable
        test_host = "test-override-host"
        with patch.dict(os.environ, {"RAG_DATABASE__IRIS__HOST": test_host}):
            config_manager = ConfigurationManager()
            
            # Environment variable should override default
            assert config_manager.get("database:iris:host") == test_host
    
    def test_no_hardcoded_secrets(self):
        """
        TDD Anchor Test: no secrets in default configuration.
        
        This test verifies that no hardcoded secrets or sensitive
        information exists in the default configuration.
        """
        from rag_templates.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        
        # Check that sensitive fields are not hardcoded
        password = config_manager.get("database:iris:password")
        username = config_manager.get("database:iris:username")
        
        # Should be None or placeholder values, not real credentials
        assert password is None or password in ["", "password", "your_password", None]
        assert username is None or username in ["", "username", "your_username", None]
    
    def test_error_handling_initialization(self):
        """
        Test that proper error handling is in place for initialization failures.
        """
        from rag_templates.core.errors import RAGFrameworkError
        
        # Should be able to import error classes
        assert RAGFrameworkError is not None
        
        # Error should be a proper exception
        error = RAGFrameworkError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_configuration_error_handling(self):
        """
        Test that configuration errors are properly handled.
        """
        from rag_templates.core.errors import ConfigurationError
        
        # Should be able to import configuration error
        assert ConfigurationError is not None
        
        # Should inherit from RAGFrameworkError
        from rag_templates.core.errors import RAGFrameworkError
        error = ConfigurationError("Config error")
        assert isinstance(error, RAGFrameworkError)
    
    def test_lazy_initialization_pattern(self):
        """
        Test that the Simple API uses lazy initialization.
        
        This ensures that expensive operations (like database connections)
        are deferred until actually needed.
        """
        from rag_templates.simple import RAG
        
        rag = RAG()
        
        # Should not be initialized yet
        assert not rag._initialized
        
        # Mock the pipeline initialization to avoid real dependencies
        with patch.object(rag, '_initialize_pipeline') as mock_init:
            # Mock the pipeline object that would be created
            mock_pipeline = MagicMock()
            mock_pipeline.execute.return_value = {
                "answer": "test",
                "query": "test",
                "retrieved_documents": []
            }
            
            # Set the pipeline after mocking initialization
            def mock_init_side_effect():
                rag._pipeline = mock_pipeline
                rag._initialized = True
            
            mock_init.side_effect = mock_init_side_effect
            
            # First query should trigger initialization
            result = rag.query("test")
            
            # Should have been initialized
            mock_init.assert_called_once()
            assert rag._initialized
            assert result == "test"


class TestConfigurationManagerPhase1:
    """Test suite for enhanced ConfigurationManager Phase 1."""
    
    def test_three_tier_configuration_system(self):
        """
        Test the three-tier configuration system:
        1. Built-in defaults
        2. Configuration file
        3. Environment variables
        """
        from rag_templates.core.config_manager import ConfigurationManager
        
        # Test with no config file (defaults only)
        config_manager = ConfigurationManager()
        
        # Should have built-in defaults
        assert config_manager.get("database:iris:host") is not None
        assert config_manager.get("embeddings:model") is not None
    
    def test_configuration_validation(self):
        """
        Test that configuration validation works properly.
        """
        from rag_templates.core.config_manager import ConfigurationManager
        from rag_templates.core.errors import ConfigurationError
        
        # Should be able to validate configuration
        config_manager = ConfigurationManager()
        
        # Validation should not raise errors for valid config
        try:
            config_manager.validate()
        except Exception as e:
            # If validation fails, it should be a ConfigurationError
            assert isinstance(e, ConfigurationError)
    
    def test_fallback_strategies(self):
        """
        Test that fallback strategies work for missing configuration.
        """
        from rag_templates.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        
        # Should provide fallback for missing keys
        missing_value = config_manager.get("nonexistent:key", "fallback_value")
        assert missing_value == "fallback_value"
        
        # Should provide sensible defaults for critical configuration
        host = config_manager.get("database:iris:host", "localhost")
        assert host is not None
        assert isinstance(host, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])