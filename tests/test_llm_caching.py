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
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List

# Langchain and project-specific imports for new tests
from langchain_core.outputs import Generation
import jaydebeapi

from common.llm_cache_manager import LangchainIRISCacheWrapper
from common.llm_cache_iris import IRISCacheBackend
from common.utils import get_llm_func
from tests.mocks.mock_iris_connector import MockIRISConnector


class TestLLMCacheConfiguration:
    """Test cache configuration loading and validation."""

    def test_cache_config_from_env_variables(self):
        """Test loading cache configuration from environment variables."""
        with patch.dict(os.environ, {
            'RAG_CACHE__ENABLED': 'true',
            'RAG_CACHE__BACKEND': 'iris',
            'RAG_CACHE__TTL_SECONDS': '3600',
            'RAG_CACHE__IRIS__TABLE_NAME': 'llm_cache'
        }):
            from common.llm_cache_config import CacheConfig
            config = CacheConfig()
            
            assert config.enabled is True
            assert config.backend == 'iris'
            assert config.ttl_seconds == 3600
            assert config.table_name == 'llm_cache'

    def test_cache_config_from_yaml_file(self):
        """Test loading cache configuration from YAML file."""
        from common.llm_cache_config import CacheConfig
        
        config = CacheConfig()
        
        assert isinstance(config.enabled, bool)
        assert config.backend in ['memory', 'iris', 'redis']
        assert isinstance(config.ttl_seconds, int)
        assert config.ttl_seconds > 0

    def test_cache_config_env_overrides_yaml(self):
        """Test that environment variables override YAML settings."""
        with patch.dict(os.environ, {
            'RAG_CACHE__ENABLED': 'false',
            'RAG_CACHE__TTL_SECONDS': '7200'
        }):
            from common.llm_cache_config import CacheConfig
            config = CacheConfig()
            
            assert config.enabled is False
            assert config.ttl_seconds == 7200


class TestIRISCacheBackend:
    """Test IRIS-based cache backend implementation."""

    def test_iris_cache_backend_initialization(self, mock_connection_manager):
        """Test IRIS cache backend can be initialized."""
        with patch('common.llm_cache_iris.IRISCacheBackend.setup_table'):
            cache = IRISCacheBackend(
                connection_manager=mock_connection_manager,
                table_name='llm_cache',
                ttl_seconds=3600
            )
        
        assert cache.table_name == 'llm_cache'
        assert cache.ttl_seconds == 3600
        assert cache.connection_manager is not None

    def test_setup_table_execution(self, mock_connection_manager):
        """Test that setup_table executes the correct SQL."""
        
        with patch('common.llm_cache_iris.IRISCacheBackend.setup_table'):
            cache = IRISCacheBackend(
                connection_manager=mock_connection_manager,
                table_name='llm_cache'
            )
        
        mock_cursor = MagicMock()
        with patch.object(cache, '_get_cursor', return_value=mock_cursor):
            # We need to call setup_table manually here because we patched it during init
            cache.setup_table()
        
        # A bit more specific
        assert 'CREATE TABLE' in mock_cursor.execute.call_args_list[0].args[0]


    def test_iris_cache_set_and_get(self, mock_connection_manager):
        """Test setting and getting values from IRIS cache."""
        with patch('common.llm_cache_iris.IRISCacheBackend.setup_table'):
            cache = IRISCacheBackend(
                connection_manager=mock_connection_manager,
                table_name='llm_cache'
            )
        
        mock_cursor = MagicMock()
        with patch.object(cache, '_get_cursor', return_value=mock_cursor):
            # Test set
            cache.set('test_key', {"response": "test response"}, ttl=3600)
            assert mock_cursor.execute.call_count == 2 # DELETE and INSERT
            
            # Test get
            mock_cursor.fetchone.return_value = (json.dumps({"response": "test response"}),)
            result = cache.get('test_key')
            
            assert result == {"response": "test response"}
            assert mock_cursor.execute.call_count == 3 # Previous 2 + SELECT

    def test_iris_cache_ttl_expiration(self, mock_connection_manager):
        """Test that expired cache entries are not returned."""
        with patch('common.llm_cache_iris.IRISCacheBackend.setup_table'):
            cache = IRISCacheBackend(
                connection_manager=mock_connection_manager,
                table_name='llm_cache'
            )
        
        mock_cursor = mock_connection_manager.get_connection.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = None
        
        result = cache.get('expired_key')
        assert result is None

    def test_iris_cache_clear(self, mock_connection_manager):
        """Test clearing all cache entries."""
        with patch('common.llm_cache_iris.IRISCacheBackend.setup_table'):
            cache = IRISCacheBackend(
                connection_manager=mock_connection_manager,
                table_name='llm_cache'
            )
        
        mock_cursor = MagicMock()
        with patch.object(cache, '_get_cursor', return_value=mock_cursor):
            cache.clear()
        
        mock_cursor.execute.assert_called_with("DELETE FROM USER.llm_cache")


class TestLangchainCacheIntegration:
    """Test integration with Langchain's caching system."""

    def test_langchain_cache_disabled(self):
        """Test that cache setup is skipped when disabled."""
        from common.llm_cache_manager import setup_langchain_cache
        from common.llm_cache_config import CacheConfig
        
        config = CacheConfig()
        config.enabled = False
        
        result = setup_langchain_cache(config)
        assert result is None
        
        import langchain
        assert not hasattr(langchain, 'llm_cache') or langchain.llm_cache is None


class TestEnhancedLLMFunction:
    """Test enhanced get_llm_func with caching support."""

    def test_get_llm_func_with_cache_enabled(self, mock_config_manager):
        """Test that get_llm_func works with caching enabled."""
        with patch('common.llm_cache_manager.setup_langchain_cache') as mock_setup:
            mock_setup.return_value = Mock()
            
            llm_func = get_llm_func(
                provider="stub", 
                model_name="test-model",
                config_manager=mock_config_manager
            )
            
            assert callable(llm_func)
            
            response = llm_func("test prompt")
            assert isinstance(response, str)
            assert "test prompt" in response

    def test_get_llm_func_cache_hit_miss(self, mock_config_manager):
        """Test cache hit and miss behavior."""
        with patch('common.llm_cache_manager.setup_langchain_cache') as mock_setup:
            mock_cache = MagicMock()
            mock_setup.return_value = mock_cache
            
            llm_func = get_llm_func(
                provider="stub",
                model_name="test-model", 
                config_manager=mock_config_manager
            )
            
            # Simulate cache miss then hit
            mock_cache.lookup.return_value = None
            response1 = llm_func("test prompt")
            
            mock_cache.lookup.return_value = [Generation(text=response1)]
            response2 = llm_func("test prompt")
            
            assert response1 == response2

    def test_get_llm_func_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        from common.llm_cache_manager import LangchainIRISCacheWrapper
        
        key1 = LangchainIRISCacheWrapper._generate_langchain_key("test prompt", "gpt-3.5-turbo")
        key2 = LangchainIRISCacheWrapper._generate_langchain_key("test prompt", "gpt-3.5-turbo")
        assert key1 == key2
        
        key3 = LangchainIRISCacheWrapper._generate_langchain_key("different prompt", "gpt-3.5-turbo")
        assert key1 != key3
        
        key4 = LangchainIRISCacheWrapper._generate_langchain_key("test prompt", "gpt-4")
        assert key1 != key4


class TestCacheMetrics:
    """Test cache performance metrics and monitoring."""

    def test_cache_metrics_tracking(self):
        """Test that cache metrics are tracked correctly."""
        from common.llm_cache_manager import CacheMetrics
        
        metrics = CacheMetrics()
        
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total_requests == 0
        assert metrics.hit_rate == 0.0
        
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_hit()
        
        assert metrics.hits == 2
        assert metrics.misses == 1
        assert metrics.total_requests == 3
        assert metrics.hit_rate == 2/3

    def test_cache_stats_integration(self, mock_connection_manager):
        """Test cache statistics integration with IRIS backend."""
        with patch('common.llm_cache_iris.IRISCacheBackend.setup_table'):
            cache = IRISCacheBackend(
                connection_manager=mock_connection_manager,
                table_name='llm_cache'
            )
        
        mock_cursor = mock_connection_manager.get_connection.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = (json.dumps([{"text": "cached"}]),)
        
        cache.get('test_key')
        
        stats = cache.get_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'sets' in stats


class TestCacheConfigurationFile:
    """Test cache configuration file creation and loading."""

    def test_cache_config_yaml_creation(self):
        """Test that cache configuration YAML file is created correctly."""
        config_path = 'config/cache_config.yaml'
        
        assert os.path.exists(config_path)
        
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert 'llm_cache' in config_data
        cache_config = config_data['llm_cache']
        
        assert 'enabled' in cache_config
        assert 'backend' in cache_config
        assert 'ttl_seconds' in cache_config
        assert 'iris' in cache_config
        
        iris_config = cache_config['iris']
        assert 'table_name' in iris_config


class TestDeprecationOfCustomCache:
    """Test deprecation handling of existing custom cache."""

    def test_custom_cache_deprecation_warning(self):
        """Test that using the old custom cache shows deprecation warning."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from iris_rag.llm.cache import get_global_cache
            
            cache = get_global_cache()
            
            deprecation_warnings = [warn for warn in w if issubclass(warn.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert "deprecated" in str(deprecation_warnings[0].message).lower()


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
        cached_generations_list_json_str = json.dumps([cached_generation_obj.to_json()])
        
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
        args, _ = mock_logger.error.call_args
        assert "Error during async cache lookup" in args[0]
        assert "Simulated DB error" in str(args[1])

    @patch('common.llm_cache_manager.logger')
    async def test_alookup_deserialization_error_invalid_json(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when cached data is malformed and causes a JSON deserialization error."""
        prompt = self.PROMPT_JSON_ERROR
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

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
        assert "Error during cache lookup JSON decode" in args[0]
        assert isinstance(args[1], json.JSONDecodeError)

    @patch('common.llm_cache_manager.logger')
    async def test_alookup_deserialization_error_not_list(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when cached data is valid JSON but not a list for generations."""
        prompt = "test_prompt_json_not_list"
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        mock_iris_cache_backend.get.return_value = {
            'response': json.dumps({"text": "not a list"}),
            'llm_string': llm_string,
            'timestamp': time.time()
        }

        result = await cache_instance.alookup(prompt, llm_string)
        
        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result is None
        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args
        assert "Error during cache lookup data processing" in args[0]
        assert isinstance(args[1], TypeError)

    @patch('common.llm_cache_manager.logger')
    async def test_alookup_deserialization_error_bad_generation_data(self, mock_logger, cache_instance, mock_iris_cache_backend):
        """Test alookup when cached data for a generation is missing 'text' field."""
        prompt = "test_prompt_json_bad_gen_data"
        llm_string = self.LLM_STRING
        expected_key = cache_instance._generate_langchain_key(prompt, llm_string)

        mock_iris_cache_backend.get.return_value = {
            'response': json.dumps([{"not_text": "some_value"}]),
            'llm_string': llm_string,
            'timestamp': time.time()
        }

        result = await cache_instance.alookup(prompt, llm_string)

        mock_iris_cache_backend.get.assert_called_once_with(expected_key)
        assert result == []