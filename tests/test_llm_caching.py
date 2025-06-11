"""
Test suite for LLM Caching Layer using IRIS backend.

This module tests the lightweight LLM caching implementation that leverages
Langchain capabilities with IRIS as the storage backend.
"""

import pytest
import os
import time
import hashlib
import json
from unittest.mock import Mock, patch, MagicMock # Ensuring AsyncMock is not strictly needed here as we mock sync calls
from typing import Dict, Any, Optional, List # Added List

# Langchain and project-specific imports for new tests
from langchain_core.outputs import Generation
import jaydebeapi # For jaydebeapi.DatabaseError

from common.llm_cache_manager import LangchainIRISCacheWrapper
from common.llm_cache_iris import IRISCacheBackend

# Import the modules we'll be implementing
from common.utils import get_llm_func
from tests.mocks.db import MockIRISConnector


class TestLLMCacheConfiguration:
    """Test cache configuration loading and validation."""
    
    def test_cache_config_from_env_variables(self):
        """Test loading cache configuration from environment variables."""
        # This test will fail initially - we need to implement the config loading
        with patch.dict(os.environ, {
            'LLM_CACHE_ENABLED': 'true',
            'LLM_CACHE_BACKEND': 'iris',
            'LLM_CACHE_TTL': '3600',
            'LLM_CACHE_TABLE': 'llm_cache'
        }):
            from common.llm_cache_config import CacheConfig
            config = CacheConfig.from_env()
            
            assert config.enabled is True
            assert config.backend == 'iris'
            assert config.ttl_seconds == 3600
            assert config.table_name == 'llm_cache'
    
    def test_cache_config_from_yaml_file(self):
        """Test loading cache configuration from YAML file."""
        from common.llm_cache_config import CacheConfig
        
        # Test with default config file
        config = CacheConfig.from_yaml('config/cache_config.yaml')
        
        assert isinstance(config.enabled, bool)
        assert config.backend in ['memory', 'iris', 'redis']
        assert isinstance(config.ttl_seconds, int)
        assert config.ttl_seconds > 0
    
    def test_cache_config_env_overrides_yaml(self):
        """Test that environment variables override YAML settings."""
        with patch.dict(os.environ, {
            'LLM_CACHE_ENABLED': 'false',
            'LLM_CACHE_TTL': '7200'
        }):
            from common.llm_cache_config import CacheConfig
            config = CacheConfig.from_yaml('config/cache_config.yaml')
            
            assert config.enabled is False
            assert config.ttl_seconds == 7200


class TestIRISCacheBackend:
    """Test IRIS-based cache backend implementation."""
    
    @pytest.fixture
    def mock_iris_connector(self):
        """Provide a mock IRIS connector for testing."""
        return MockIRISConnector()
    
    def test_iris_cache_backend_initialization(self, mock_iris_connector):
        """Test IRIS cache backend can be initialized."""
        from common.llm_cache_iris import IRISCacheBackend
        
        cache = IRISCacheBackend(
            iris_connector=mock_iris_connector,
            table_name='llm_cache',
            ttl_seconds=3600
        )
        
        assert cache.iris_connector == mock_iris_connector
        assert cache.table_name == 'llm_cache'
        assert cache.ttl_seconds == 3600
    
    def test_iris_cache_backend_create_table(self, mock_iris_connector):
        """Test that cache backend creates the necessary table."""
        from common.llm_cache_iris import IRISCacheBackend
        
        cache = IRISCacheBackend(
            iris_connector=mock_iris_connector,
            table_name='llm_cache'
        )
        cache.setup_table()
        
        # Verify table creation SQL was called
        cursor_mock = mock_iris_connector.cursor()
        cursor_mock.execute.assert_called()
        
        # Check that the SQL contains the expected table structure
        sql_calls = [call[0][0] for call in cursor_mock.execute.call_args_list]
        create_table_sql = next((sql for sql in sql_calls if 'CREATE TABLE' in sql.upper()), None)
        assert create_table_sql is not None
        assert 'llm_cache' in create_table_sql
        assert 'cache_key' in create_table_sql
        assert 'cache_value' in create_table_sql
        assert 'expires_at' in create_table_sql
    
    def test_iris_cache_set_and_get(self, mock_iris_connector):
        """Test setting and getting values from IRIS cache."""
        from common.llm_cache_iris import IRISCacheBackend
        
        cache = IRISCacheBackend(
            iris_connector=mock_iris_connector,
            table_name='llm_cache'
        )
        
        # Mock the cursor to return a value for get
        cursor_mock = mock_iris_connector.cursor()
        cursor_mock.fetchone.return_value = ('{"response": "test response"}',)
        
        # Test set operation
        cache.set('test_key', 'test_value', ttl=3600)
        
        # Verify INSERT was called
        cursor_mock.execute.assert_called()
        insert_calls = [call for call in cursor_mock.execute.call_args_list 
                       if 'INSERT' in str(call[0][0]).upper()]
        assert len(insert_calls) > 0
        
        # Test get operation
        result = cache.get('test_key')
        
        # Verify SELECT was called
        select_calls = [call for call in cursor_mock.execute.call_args_list 
                       if 'SELECT' in str(call[0][0]).upper()]
        assert len(select_calls) > 0
        assert result == {"response": "test response"}
    
    def test_iris_cache_ttl_expiration(self, mock_iris_connector):
        """Test that expired cache entries are not returned."""
        from common.llm_cache_iris import IRISCacheBackend
        
        cache = IRISCacheBackend(
            iris_connector=mock_iris_connector,
            table_name='llm_cache'
        )
        
        # Mock cursor to return no results (expired)
        cursor_mock = mock_iris_connector.cursor()
        cursor_mock.fetchone.return_value = None
        
        result = cache.get('expired_key')
        assert result is None
    
    def test_iris_cache_clear(self, mock_iris_connector):
        """Test clearing all cache entries."""
        from common.llm_cache_iris import IRISCacheBackend
        
        cache = IRISCacheBackend(
            iris_connector=mock_iris_connector,
            table_name='llm_cache'
        )
        
        cache.clear()
        
        # Verify DELETE was called
        cursor_mock = mock_iris_connector.cursor()
        delete_calls = [call for call in cursor_mock.execute.call_args_list 
                       if 'DELETE' in str(call[0][0]).upper()]
        assert len(delete_calls) > 0


class TestLangchainCacheIntegration:
    """Test integration with Langchain's caching system."""
    
    def test_langchain_cache_setup_with_iris(self):
        """Test setting up Langchain cache with IRIS backend."""
        from common.llm_cache_manager import setup_langchain_cache
        from common.llm_cache_config import CacheConfig
        
        config = CacheConfig(
            enabled=True,
            backend='iris',
            ttl_seconds=3600,
            table_name='llm_cache'
        )
        
        with patch('common.utils.get_iris_connector') as mock_get_connector:
            mock_connector = MockIRISConnector()
            mock_get_connector.return_value = mock_connector
            
            cache_instance = setup_langchain_cache(config)
            
            assert cache_instance is not None
            # Verify that langchain.llm_cache was set
            import langchain
            assert langchain.llm_cache is not None
    
    def test_langchain_cache_disabled(self):
        """Test that cache setup is skipped when disabled."""
        from common.llm_cache_manager import setup_langchain_cache
        from common.llm_cache_config import CacheConfig
        
        config = CacheConfig(enabled=False)
        
        result = setup_langchain_cache(config)
        assert result is None
        
        # Verify langchain.llm_cache is not set
        import langchain
        assert not hasattr(langchain, 'llm_cache') or langchain.llm_cache is None


class TestEnhancedLLMFunction:
    """Test enhanced get_llm_func with caching support."""
    
    def test_get_llm_func_with_cache_enabled(self):
        """Test that get_llm_func works with caching enabled."""
        with patch('common.llm_cache_manager.setup_langchain_cache') as mock_setup:
            mock_setup.return_value = Mock()
            
            # Test with cache enabled
            llm_func = get_llm_func(
                provider="stub", 
                model_name="test-model",
                enable_cache=True
            )
            
            assert callable(llm_func)
            
            # Test that the function works
            response = llm_func("test prompt")
            assert isinstance(response, str)
            assert "test prompt" in response
    
    def test_get_llm_func_cache_hit_miss(self):
        """Test cache hit and miss behavior."""
        with patch('common.llm_cache_manager.setup_langchain_cache') as mock_setup:
            # Mock the cache to simulate hits and misses
            mock_cache = Mock()
            mock_setup.return_value = mock_cache
            
            # First call should be a cache miss
            llm_func = get_llm_func(
                provider="stub",
                model_name="test-model", 
                enable_cache=True
            )
            
            response1 = llm_func("test prompt")
            response2 = llm_func("test prompt")  # Should be same due to caching
            
            assert response1 == response2
    
    def test_get_llm_func_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        from common.llm_cache_manager import generate_cache_key
        
        # Test that same inputs generate same key
        key1 = generate_cache_key("test prompt", "gpt-3.5-turbo", temperature=0.0)
        key2 = generate_cache_key("test prompt", "gpt-3.5-turbo", temperature=0.0)
        assert key1 == key2
        
        # Test that different inputs generate different keys
        key3 = generate_cache_key("different prompt", "gpt-3.5-turbo", temperature=0.0)
        assert key1 != key3
        
        key4 = generate_cache_key("test prompt", "gpt-4", temperature=0.0)
        assert key1 != key4
        
        key5 = generate_cache_key("test prompt", "gpt-3.5-turbo", temperature=0.5)
        assert key1 != key5


class TestCacheMetrics:
    """Test cache performance metrics and monitoring."""
    
    def test_cache_metrics_tracking(self):
        """Test that cache metrics are tracked correctly."""
        from common.llm_cache_manager import CacheMetrics
        
        metrics = CacheMetrics()
        
        # Test initial state
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total_requests == 0
        assert metrics.hit_rate == 0.0
        
        # Test recording hits and misses
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_hit()
        
        assert metrics.hits == 2
        assert metrics.misses == 1
        assert metrics.total_requests == 3
        assert metrics.hit_rate == 2/3
    
    def test_cache_stats_integration(self):
        """Test cache statistics integration with IRIS backend."""
        from common.llm_cache_iris import IRISCacheBackend
        
        mock_connector = MockIRISConnector()
        cache = IRISCacheBackend(
            iris_connector=mock_connector,
            table_name='llm_cache'
        )
        
        # Mock some cache operations
        cursor_mock = mock_connector.cursor()
        cursor_mock.fetchone.return_value = ('{"response": "cached"}',)
        
        # Simulate cache hit
        result = cache.get('test_key')
        assert result == {"response": "cached"}
        
        # Check that metrics were updated
        stats = cache.get_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats


class TestCacheConfigurationFile:
    """Test cache configuration file creation and loading."""
    
    def test_cache_config_yaml_creation(self):
        """Test that cache configuration YAML file is created correctly."""
        # This test will verify the config file exists and has correct structure
        config_path = 'config/cache_config.yaml'
        
        # The file should exist after implementation
        assert os.path.exists(config_path), f"Cache config file should exist at {config_path}"
        
        # Load and verify structure
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert 'llm_cache' in config_data
        cache_config = config_data['llm_cache']
        
        # Verify required fields
        assert 'enabled' in cache_config
        assert 'backend' in cache_config
        assert 'ttl_seconds' in cache_config
        assert 'iris' in cache_config  # IRIS-specific settings
        
        # Verify IRIS backend is configured
        iris_config = cache_config['iris']
        assert 'table_name' in iris_config
        assert 'connection_timeout' in iris_config


class TestDeprecationOfCustomCache:
    """Test deprecation handling of existing custom cache."""
    
    def test_custom_cache_deprecation_warning(self):
        """Test that using the old custom cache shows deprecation warning."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import the old cache module
            from iris_rag.llm.cache import get_global_cache
            
            # Should trigger deprecation warning
            cache = get_global_cache()
            
            # Check that a deprecation warning was issued
            deprecation_warnings = [warning for warning in w 
                                  if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert "deprecated" in str(deprecation_warnings[0].message).lower()


class TestEndToEndCaching:
    """End-to-end tests for the complete caching system."""
    
    @pytest.mark.integration
    def test_e2e_llm_caching_with_iris(self):
        """Test complete end-to-end LLM caching with IRIS backend."""
        # This test requires a real or well-mocked IRIS connection
        with patch('common.utils.get_iris_connector') as mock_get_connector:
            mock_connector = MockIRISConnector()
            mock_get_connector.return_value = mock_connector
            
            # Configure cache
            with patch.dict(os.environ, {
                'LLM_CACHE_ENABLED': 'true',
                'LLM_CACHE_BACKEND': 'iris',
                'LLM_CACHE_TTL': '3600'
            }):
                # Get LLM function with caching
                llm_func = get_llm_func(
                    provider="stub",
                    model_name="test-model",
                    enable_cache=True
                )
                
                # First call - should be cache miss
                response1 = llm_func("What is machine learning?")
                
                # Second call - should be cache hit
                response2 = llm_func("What is machine learning?")
                
                # Responses should be identical due to caching
                assert response1 == response2
                
                # Different prompt should generate different response
                response3 = llm_func("What is deep learning?")
                assert response3 != response1
    
    @pytest.mark.integration
    def test_e2e_cache_persistence(self):
        """Test that cache persists across different LLM function instances."""
        with patch('common.utils.get_iris_connector') as mock_get_connector:
            mock_connector = MockIRISConnector()
            mock_get_connector.return_value = mock_connector
            
            # Mock cursor to simulate persistent storage
            cursor_mock = mock_connector.cursor()
            stored_data = {}
            
            def mock_execute(sql, params=None):
                if 'INSERT' in sql.upper():
                    # Simulate storing data
                    if params:
                        stored_data[params[0]] = params[1]  # key, value
                elif 'SELECT' in sql.upper():
                    # Simulate retrieving data
                    if params and params[0] in stored_data:
                        cursor_mock.fetchone.return_value = (stored_data[params[0]],)
                    else:
                        cursor_mock.fetchone.return_value = None
            
            cursor_mock.execute.side_effect = mock_execute
            
            with patch.dict(os.environ, {
                'LLM_CACHE_ENABLED': 'true',
                'LLM_CACHE_BACKEND': 'iris'
            }):
                # First LLM function instance
                llm_func1 = get_llm_func(
                    provider="stub",
                    model_name="test-model",
                    enable_cache=True
                )
                response1 = llm_func1("test prompt")
                
                # Second LLM function instance (simulating restart)
                llm_func2 = get_llm_func(
                    provider="stub", 
                    model_name="test-model",
                    enable_cache=True
                )
                response2 = llm_func2("test prompt")
                
                # Should get same response from cache
                assert response1 == response2


@pytest.mark.asyncio
class TestAsyncLangchainIRISCacheWrapper:
    """Tests for the asynchronous alookup method of LangchainIRISCacheWrapper."""

    PROMPT_HIT = "test_prompt_hit"
    PROMPT_MISS = "test_prompt_miss"
    PROMPT_DB_ERROR = "test_prompt_db_error"
    PROMPT_JSON_ERROR = "test_prompt_json_error"
    LLM_STRING = "test_llm_string"

    @pytest.fixture
    def mock_iris_cache_backend(self):
        """Provides a MagicMock for the IRISCacheBackend."""
        backend = MagicMock(spec=IRISCacheBackend)
        # backend.get is the synchronous method that alookup will call via asyncio.to_thread
        backend.get = MagicMock()
        return backend

    @pytest.fixture
    def cache_instance(self, mock_iris_cache_backend):
        """Provides an instance of LangchainIRISCacheWrapper with a mocked backend."""
        return LangchainIRISCacheWrapper(iris_cache_backend=mock_iris_cache_backend)

    async def test_alookup_cache_hit(self, cache_instance, mock_iris_cache_backend):
        """Test alookup when a valid entry is found in the cache."""
        prompt = self.PROMPT_HIT
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        cached_generation_obj = Generation(text="cached_response")
        # Langchain stores generations as a list of their JSON dict representations
        cached_generations_list_json_str = json.dumps([cached_generation_obj.to_json()])
        
        # IRISCacheBackend.get() returns the parsed JSON from the DB.
        # LangchainIRISCacheWrapper.update() stores a dict like:
        # {'response': <json_str_of_generations>, 'llm_string': ..., 'timestamp': ...}
        # So, mock_iris_cache_backend.get should return this dict.
        mock_iris_cache_backend.get.return_value = {
            'response': cached_generations_list_json_str,
            'llm_string': llm_string,
            'timestamp': time.time()
        }

        result = await cache_instance.alookup(prompt, llm_string)

        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], Generation)
        assert result[0].text == "cached_response"

    async def test_alookup_cache_miss(self, cache_instance, mock_iris_cache_backend):
        """Test alookup when no entry is found in the cache."""
        prompt = self.PROMPT_MISS
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        mock_iris_cache_backend.get.return_value = None

        result = await cache_instance.alookup(prompt, llm_string)

        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is None

    @patch('common.llm_cache_manager.logger')
    async def test_alookup_database_error(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when the backend raises a database error."""
        prompt = self.PROMPT_DB_ERROR
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        mock_iris_cache_backend.get.side_effect = jaydebeapi.DatabaseError("Simulated DB error")

        result = await cache_instance.alookup(prompt, llm_string)

        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is None
        mock_logger.error.assert_called_once()
        # Check that the error message indicates a problem during lookup or with the database
        args, _ = mock_logger.error.call_args
        assert "Error during async cache lookup" in args[0]
        assert "Simulated DB error" in str(args[1])


    @patch('common.llm_cache_manager.logger')
    async def test_alookup_deserialization_error_invalid_json(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when cached data is malformed and causes a JSON deserialization error."""
        prompt = self.PROMPT_JSON_ERROR
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        # Simulate malformed JSON string for the 'response' field
        mock_iris_cache_backend.get.return_value = {
            'response': "this is not valid json",
            'llm_string': llm_string,
            'timestamp': time.time()
        }

        result = await cache_instance.alookup(prompt, llm_string)

        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is None
        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args
        assert "Error during async cache lookup" in args[0]
        # Check for json.JSONDecodeError or similar in the logged exception
        assert isinstance(args[1], json.JSONDecodeError)


    @patch('common.llm_cache_manager.logger')
    async def test_alookup_deserialization_error_not_list(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when cached data is valid JSON but not a list for generations."""
        prompt = "test_prompt_json_not_list"
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        # Simulate valid JSON, but not a list as expected for generations
        mock_iris_cache_backend.get.return_value = {
            'response': json.dumps({"text": "not a list"}), # a dict, not list of dicts
            'llm_string': llm_string,
            'timestamp': time.time()
        }

        result = await cache_instance.alookup(prompt, llm_string)
        
        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is None
        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args
        assert "Error during async cache lookup" in args[0]
        # Expect TypeError or similar if trying to iterate over a dict like a list for Generations
        assert isinstance(args[1], (TypeError, AttributeError))


    @patch('common.llm_cache_manager.logger')
    async def test_alookup_deserialization_error_bad_generation_data(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when cached data for a generation is missing 'text' field."""
        prompt = "test_prompt_json_bad_gen_data"
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        # Simulate list of dicts, but a dict is missing the 'text' key for Generation
        mock_iris_cache_backend.get.return_value = {
            'response': json.dumps([{"not_text": "some_value"}]),
            'llm_string': llm_string,
            'timestamp': time.time()
        }

        result = await cache_instance.alookup(prompt, llm_string)

        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is None
        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args
        assert "Error during async cache lookup" in args[0]
        # Expect ValidationError when Generation(**g) fails due to missing required fields
        # Langchain's Generation class uses Pydantic validation
        # The exact type can vary between pydantic versions, so check for the error characteristics
        exception = args[1]
        assert hasattr(exception, 'errors') or 'ValidationError' in str(type(exception))