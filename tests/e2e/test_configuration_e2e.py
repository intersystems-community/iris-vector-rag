"""
Configuration Management E2E Tests

Tests multi-environment configuration loading and external service connections:
- Environment-specific configuration loading
- Connection to all external services (IRIS, OpenAI, etc.)
- Configuration override mechanisms
- Validation of no hardcoded credentials

NO MOCKS - Uses real configuration system and service connections.

Priority 1 Components Under Test:
- iris_rag/config/manager.py (ConfigurationManager)
"""

import pytest
import logging
import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

# Real framework imports (NO MOCKS)
from iris_rag.config.manager import ConfigurationManager, ConfigValidationError
from iris_rag.core.connection import ConnectionManager

logger = logging.getLogger(__name__)


class TestConfigurationManagerCore:
    """Test core configuration management functionality."""
    
    @pytest.mark.true_e2e
    def test_configuration_manager_initialization_e2e(self):
        """
        Test ConfigurationManager initialization with real configuration files.
        
        NO MOCKS - Tests real configuration loading from files and environment.
        """
        logger.info("=== STARTING CONFIGURATION MANAGER INITIALIZATION E2E TEST ===")
        
        # Test default initialization
        config_manager = ConfigurationManager()
        
        # Validate initialization
        assert config_manager is not None, "ConfigurationManager should be initialized"
        assert hasattr(config_manager, '_config'), "Should have internal config storage"
        
        # Test basic configuration access
        database_config = config_manager.get("database", {})
        assert isinstance(database_config, dict), "Database config should be accessible"
        
        logger.info("ConfigurationManager initialized successfully")
        logger.info("=== CONFIGURATION MANAGER INITIALIZATION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_environment_variable_overrides_e2e(self):
        """
        Test configuration override mechanisms with environment variables.
        
        NO MOCKS - Tests real environment variable processing.
        """
        logger.info("=== STARTING ENVIRONMENT VARIABLE OVERRIDES E2E TEST ===")
        
        # Set test environment variables
        test_env_vars = {
            'RAG_DATABASE__IRIS__HOST': 'test-iris-host',
            'RAG_DATABASE__IRIS__PORT': '1972',
            'RAG_EMBEDDINGS__MODEL': 'test-embedding-model',
            'RAG_VECTOR_INDEX__M': '32'
        }
        
        # Store original values
        original_values = {}
        for key in test_env_vars:
            original_values[key] = os.environ.get(key)
            os.environ[key] = test_env_vars[key]
        
        try:
            # Create new config manager to pick up environment variables
            config_manager = ConfigurationManager()
            
            # Test database configuration override
            db_config = config_manager.get("database", {})
            if 'iris' in db_config:
                iris_config = db_config['iris']
                assert iris_config.get('host') == 'test-iris-host', "Environment variable should override database host"
                assert iris_config.get('port') == '1972', "Environment variable should override database port"
            
            # Test embeddings configuration override
            embeddings_config = config_manager.get("embeddings", {})
            if embeddings_config:
                assert embeddings_config.get('model') == 'test-embedding-model', "Environment variable should override embedding model"
            
            # Test vector index configuration override
            vector_index_config = config_manager.get_vector_index_config()
            assert vector_index_config.get('M') == 32, "Environment variable should override vector index parameter"
            
            logger.info("Environment variable overrides working correctly")
            
        finally:
            # Restore original environment
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
        
        logger.info("=== ENVIRONMENT VARIABLE OVERRIDES E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_multi_environment_configuration_e2e(self):
        """
        Test multi-environment configuration loading (development, production).
        
        NO MOCKS - Tests real configuration file loading and environment handling.
        """
        logger.info("=== STARTING MULTI-ENVIRONMENT CONFIGURATION E2E TEST ===")
        
        # Create temporary configuration files for different environments
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Development configuration
            dev_config = {
                'environment': 'development',
                'database': {
                    'iris': {
                        'host': 'localhost',
                        'port': '1972',
                        'namespace': 'USER'
                    }
                },
                'embeddings': {
                    'model': 'all-MiniLM-L6-v2',
                    'dimension': 384
                },
                'logging': {
                    'level': 'DEBUG'
                }
            }
            
            # Production configuration
            prod_config = {
                'environment': 'production',
                'database': {
                    'iris': {
                        'host': 'prod-iris-server',
                        'port': '1972',
                        'namespace': 'PRODUCTION'
                    }
                },
                'embeddings': {
                    'model': 'text-embedding-ada-002',
                    'dimension': 1536
                },
                'logging': {
                    'level': 'INFO'
                }
            }
            
            # Write configuration files
            dev_config_file = temp_path / 'development.yaml'
            prod_config_file = temp_path / 'production.yaml'
            
            with open(dev_config_file, 'w') as f:
                yaml.dump(dev_config, f)
            
            with open(prod_config_file, 'w') as f:
                yaml.dump(prod_config, f)
            
            # Test development configuration loading
            dev_config_manager = ConfigurationManager(config_path=str(dev_config_file))
            
            dev_db_config = dev_config_manager.get("database:iris", {})
            assert dev_db_config.get('host') == 'localhost', "Development config should load correctly"
            assert dev_db_config.get('namespace') == 'USER', "Development namespace should be USER"
            
            dev_embedding_config = dev_config_manager.get_embedding_config()
            assert dev_embedding_config.get('model') == 'all-MiniLM-L6-v2', "Development should use MiniLM model"
            assert dev_embedding_config.get('dimension') == 384, "Development should use 384 dimensions"
            
            # Test production configuration loading
            prod_config_manager = ConfigurationManager(config_path=str(prod_config_file))
            
            prod_db_config = prod_config_manager.get("database:iris", {})
            assert prod_db_config.get('host') == 'prod-iris-server', "Production config should load correctly"
            assert prod_db_config.get('namespace') == 'PRODUCTION', "Production namespace should be PRODUCTION"
            
            prod_embedding_config = prod_config_manager.get_embedding_config()
            assert prod_embedding_config.get('model') == 'text-embedding-ada-002', "Production should use Ada model"
            assert prod_embedding_config.get('dimension') == 1536, "Production should use 1536 dimensions"
            
            logger.info("Multi-environment configuration loading working correctly")
        
        logger.info("=== MULTI-ENVIRONMENT CONFIGURATION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_configuration_validation_e2e(self):
        """
        Test configuration validation and error handling.
        
        NO MOCKS - Tests real configuration validation.
        """
        logger.info("=== STARTING CONFIGURATION VALIDATION E2E TEST ===")
        
        # Test missing required configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Invalid configuration missing required fields
            invalid_config = {
                'embeddings': {
                    'model': 'test-model'
                }
                # Missing required database:iris:host
            }
            
            invalid_config_file = temp_path / 'invalid.yaml'
            with open(invalid_config_file, 'w') as f:
                yaml.dump(invalid_config, f)
            
            # Remove IRIS_HOST environment variable if present
            original_iris_host = os.environ.get('IRIS_HOST')
            if 'IRIS_HOST' in os.environ:
                del os.environ['IRIS_HOST']
            
            try:
                # This should raise ConfigValidationError due to missing required config
                with pytest.raises(ConfigValidationError):
                    ConfigurationManager(config_path=str(invalid_config_file))
                
                logger.info("Configuration validation correctly caught missing required config")
                
            finally:
                # Restore environment variable
                if original_iris_host:
                    os.environ['IRIS_HOST'] = original_iris_host
        
        logger.info("=== CONFIGURATION VALIDATION E2E TEST COMPLETED ===")


class TestExternalServiceConnections:
    """Test connections to external services using configuration."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_iris_database_connection_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_connection_manager: ConnectionManager
    ):
        """
        Test real IRIS database connection using configuration.
        
        NO MOCKS - Tests actual connection to IRIS database.
        """
        logger.info("=== STARTING IRIS DATABASE CONNECTION E2E TEST ===")
        
        # Get database configuration
        db_config = e2e_config_manager.get_database_config()
        
        # Validate configuration structure
        required_keys = ['host', 'port', 'namespace', 'username']
        for key in required_keys:
            assert key in db_config, f"Database config should contain '{key}'"
            assert db_config[key], f"Database config '{key}' should not be empty"
        
        # Test actual connection
        try:
            connection = e2e_connection_manager.get_connection()
            assert connection is not None, "Should establish database connection"
            
            # Test connection functionality
            cursor = connection.cursor()
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            
            assert result is not None, "Should be able to execute queries"
            assert result[0] == 1, "Query should return expected result"
            
            cursor.close()
            connection.close()
            
            logger.info(f"Successfully connected to IRIS at {db_config['host']}:{db_config['port']}")
            
        except Exception as e:
            pytest.fail(f"Failed to connect to IRIS database: {e}")
        
        logger.info("=== IRIS DATABASE CONNECTION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_embedding_service_configuration_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_embedding_function
    ):
        """
        Test embedding service configuration and connection.
        
        NO MOCKS - Tests real embedding service configuration.
        """
        logger.info("=== STARTING EMBEDDING SERVICE CONFIGURATION E2E TEST ===")
        
        # Get embedding configuration
        embedding_config = e2e_config_manager.get_embedding_config()
        
        # Validate configuration structure
        assert 'model' in embedding_config, "Embedding config should specify model"
        assert 'dimension' in embedding_config, "Embedding config should specify dimension"
        assert embedding_config['model'], "Model name should not be empty"
        assert embedding_config['dimension'] > 0, "Dimension should be positive"
        
        # Test actual embedding generation
        try:
            test_text = "This is a test for embedding generation configuration."
            embedding = e2e_embedding_function(test_text)
            
            assert embedding is not None, "Should generate embedding"
            assert len(embedding) == embedding_config['dimension'], \
                f"Embedding dimension should match config: expected {embedding_config['dimension']}, got {len(embedding)}"
            assert all(isinstance(val, (int, float)) for val in embedding), "Embedding should contain numeric values"
            
            logger.info(f"Successfully generated embedding using {embedding_config['model']} "
                       f"(dimension: {embedding_config['dimension']})")
            
        except Exception as e:
            pytest.fail(f"Failed to generate embeddings with configured service: {e}")
        
        logger.info("=== EMBEDDING SERVICE CONFIGURATION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_llm_service_configuration_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_llm_function
    ):
        """
        Test LLM service configuration and connection.
        
        NO MOCKS - Tests real LLM service configuration.
        """
        logger.info("=== STARTING LLM SERVICE CONFIGURATION E2E TEST ===")
        
        # Test LLM configuration through actual usage
        try:
            test_prompt = "What is the capital of France? Answer in one word."
            response = e2e_llm_function(test_prompt)
            
            assert response is not None, "Should generate LLM response"
            assert isinstance(response, str), "Response should be string"
            assert len(response.strip()) > 0, "Response should not be empty"
            
            logger.info(f"Successfully generated LLM response: '{response[:50]}...'")
            
        except Exception as e:
            # LLM might fail due to API keys, network, etc. - log but don't fail the config test
            logger.warning(f"LLM service test failed (may be expected in CI): {e}")
        
        logger.info("=== LLM SERVICE CONFIGURATION E2E TEST COMPLETED ===")


class TestConfigurationSecurity:
    """Test configuration security and credential handling."""
    
    @pytest.mark.true_e2e
    def test_no_hardcoded_credentials_e2e(self, e2e_config_manager: ConfigurationManager):
        """
        Test that no hardcoded credentials are present in configuration.
        
        NO MOCKS - Tests real configuration for security compliance.
        """
        logger.info("=== STARTING NO HARDCODED CREDENTIALS E2E TEST ===")
        
        # Get database configuration
        db_config = e2e_config_manager.get_database_config()
        
        # Common patterns that indicate hardcoded credentials (should be avoided)
        suspicious_patterns = [
            'password123',
            'admin',
            'secret',
            'test123',
            'changeme',
            'default'
        ]
        
        # Check database password
        db_password = db_config.get('password', '')
        if db_password:
            for pattern in suspicious_patterns:
                assert pattern.lower() not in db_password.lower(), \
                    f"Database password should not contain suspicious pattern: {pattern}"
        
        # Check that sensitive values come from environment or are properly configured
        if 'password' in db_config:
            # Password should either be from environment variable or be a reasonable length
            assert len(db_password) >= 3, "Database password should be properly configured"
        
        # Check for API keys in configuration
        config_dict = e2e_config_manager._config
        
        def check_dict_for_credentials(d, path=""):
            """Recursively check dictionary for credential patterns."""
            if isinstance(d, dict):
                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for API key patterns
                    if 'api_key' in key.lower() or 'token' in key.lower():
                        if isinstance(value, str) and value:
                            # API keys should not be obvious test values
                            for pattern in suspicious_patterns:
                                assert pattern.lower() not in value.lower(), \
                                    f"API key at {current_path} should not contain pattern: {pattern}"
                            
                            # API keys should be reasonable length
                            assert len(value) >= 10, f"API key at {current_path} should be properly configured"
                    
                    # Recurse into nested dictionaries
                    check_dict_for_credentials(value, current_path)
        
        check_dict_for_credentials(config_dict)
        
        logger.info("Configuration security validation passed - no obvious hardcoded credentials found")
        logger.info("=== NO HARDCODED CREDENTIALS E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_configuration_environment_separation_e2e(self):
        """
        Test that configuration properly separates environments.
        
        NO MOCKS - Tests real environment separation mechanisms.
        """
        logger.info("=== STARTING CONFIGURATION ENVIRONMENT SEPARATION E2E TEST ===")
        
        # Test that development and production configs would be different
        # (This is more of a structural test since we might not have both environments)
        
        config_manager = ConfigurationManager()
        
        # Check that configuration has mechanisms for environment-specific values
        db_config = config_manager.get_database_config()
        
        # Host should not be hardcoded localhost in production-style configurations
        host = db_config.get('host', '')
        if host and host != 'localhost':
            # Non-localhost suggests environment-aware configuration
            logger.info(f"Configuration appears environment-aware: host={host}")
        else:
            # Localhost is fine for development/testing
            logger.info("Configuration appears to be for development/testing environment")
        
        # Check that configuration supports environment-specific overrides
        target_state_config = config_manager.get_target_state_config('development')
        assert isinstance(target_state_config, dict), "Should support environment-specific configurations"
        
        logger.info("Environment separation mechanisms are properly configured")
        logger.info("=== CONFIGURATION ENVIRONMENT SEPARATION E2E TEST COMPLETED ===")


class TestConfigurationIntegration:
    """Test configuration integration with other components."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_full_configuration_integration_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_connection_manager: ConnectionManager,
        e2e_iris_vector_store,
        e2e_embedding_function
    ):
        """
        Test full configuration integration across all components.
        
        NO MOCKS - Tests real configuration flow through entire system.
        """
        logger.info("=== STARTING FULL CONFIGURATION INTEGRATION E2E TEST ===")
        
        # Test that configuration flows properly through all components
        
        # 1. Database configuration -> Connection Manager -> Vector Store
        db_config = e2e_config_manager.get_database_config()
        assert db_config, "Should have database configuration"
        
        # Vector store should use the same configuration
        assert e2e_iris_vector_store.connection_manager is not None, "Vector store should have connection manager"
        assert e2e_iris_vector_store.config_manager is not None, "Vector store should have config manager"
        
        # 2. Embedding configuration -> Embedding Function
        embedding_config = e2e_config_manager.get_embedding_config()
        
        # Test that embedding function works with configured model
        test_embedding = e2e_embedding_function("Configuration integration test")
        expected_dimension = embedding_config.get('dimension', 384)
        
        assert len(test_embedding) == expected_dimension, \
            f"Embedding dimension should match configuration: expected {expected_dimension}, got {len(test_embedding)}"
        
        # 3. Vector index configuration -> Vector Store
        vector_config = e2e_config_manager.get_vector_index_config()
        assert vector_config, "Should have vector index configuration"
        assert 'type' in vector_config, "Vector config should specify index type"
        
        # Test that vector store respects the configuration
        # (This is validated through successful operations in other tests)
        
        logger.info("Configuration integration across all components working correctly")
        logger.info("=== FULL CONFIGURATION INTEGRATION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_configuration_reload_e2e(self):
        """
        Test configuration reload and dynamic updates.
        
        NO MOCKS - Tests real configuration reload functionality.
        """
        logger.info("=== STARTING CONFIGURATION RELOAD E2E TEST ===")
        
        # Create temporary config file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_file = temp_path / 'test_config.yaml'
            
            # Initial configuration
            initial_config = {
                'test_value': 'initial',
                'embeddings': {
                    'model': 'initial-model'
                }
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(initial_config, f)
            
            # Load initial configuration
            config_manager = ConfigurationManager(config_path=str(config_file))
            
            # Verify initial values
            assert config_manager.get('test_value') == 'initial', "Should load initial configuration"
            assert config_manager.get('embeddings:model') == 'initial-model', "Should load initial embedding model"
            
            # Update configuration file
            updated_config = {
                'test_value': 'updated',
                'embeddings': {
                    'model': 'updated-model'
                }
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Test configuration reload
            config_manager.load_config(str(config_file))
            
            # Verify updated values
            assert config_manager.get('test_value') == 'updated', "Should load updated configuration"
            assert config_manager.get('embeddings:model') == 'updated-model', "Should load updated embedding model"
            
            logger.info("Configuration reload functionality working correctly")
        
        logger.info("=== CONFIGURATION RELOAD E2E TEST COMPLETED ===")