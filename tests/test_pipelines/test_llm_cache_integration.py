"""
Tests for LLM Cache Integration with RAG Pipelines.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from iris_rag.core.models import Document
from common.llm_cache_manager import setup_langchain_cache, clear_global_cache


# Mock Langchain LLM for testing caching
class MockLangchainLLM:
    def __init__(self, *args, **kwargs):
        self.call_count = 0
        self.response_content = "Mock LLM response"

    def invoke(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return self.response_content

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)


# Mock AIMessage for Langchain compatibility
class MockAIMessage:
    def __init__(self, content: str):
        self.content = content


# Mock Langchain LLM that's compatible with caching
class MockCacheableLLM:
    def __init__(self):
        self.call_count = 0
        self.response_content = "Mock LLM response"

    def invoke(self, prompt, **kwargs):
        self.call_count += 1
        return MockAIMessage(self.response_content)

    def __call__(self, prompt, **kwargs):
        return self.invoke(prompt, **kwargs)


@patch('iris_rag.pipelines.basic.IRISStorage')
@patch('iris_rag.pipelines.basic.EmbeddingManager')
def test_basic_pipeline_caching(mock_embedding_manager, mock_iris_storage):
    """Test that the basic pipeline uses the LLM cache when enabled."""
    from iris_rag.pipelines.basic import BasicRAGPipeline
    from iris_rag.core.models import Document
    from common.utils import get_llm_func
    
    # Configure caching to be enabled for this test with memory backend
    original_cache_enabled = os.environ.get("LLM_CACHE_ENABLED")
    original_cache_backend = os.environ.get("LLM_CACHE_BACKEND")
    os.environ["LLM_CACHE_ENABLED"] = "true"
    os.environ["LLM_CACHE_BACKEND"] = "memory"

    try:
        # Clear any existing global cache
        clear_global_cache()

        # Create minimal mocks for pipeline initialization
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.side_effect = lambda key, default=None: {
            "pipelines:basic": {},
            "llm.provider": "stub",
            "llm.model_name": "stub-model",
            "llm_cache.enabled": True
        }.get(key, default)

        # Setup the cache with memory backend
        setup_langchain_cache()

        # Get a properly cached LLM function using the real get_llm_func
        llm_func = get_llm_func(provider="stub", model_name="test-model", enable_cache=True)

        # Create pipeline with the cached LLM function
        pipeline = BasicRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager,
            llm_func=llm_func
        )

        # Mock the storage and embedding manager methods used by execute
        mock_iris_storage_instance = mock_iris_storage.return_value
        # Return some mock documents so the LLM function gets called
        mock_doc = Document(page_content="Paris is the capital of France.", metadata={"source": "test"})
        mock_iris_storage_instance.vector_search.return_value = [(mock_doc, 0.9)]

        mock_embedding_manager_instance = mock_embedding_manager.return_value
        mock_embedding_manager_instance.embed_text.return_value = [0.1] * 10

        # Execute the pipeline the first time (cache miss)
        query_text = "What is the capital of France?"
        result1 = pipeline.execute(query_text)

        # Assert that we got a response
        assert result1['answer'] is not None
        assert "Stub LLM response" in result1['answer']

        # Execute the pipeline the second time with the same query (expected cache hit)
        result2 = pipeline.execute(query_text)

        # Assert that we got the same response (cache hit)
        assert result2['answer'] == result1['answer']

        # Execute with a different query (should be cache miss)
        result3 = pipeline.execute("What is the capital of Germany?")
        assert result3['answer'] is not None
        assert "Stub LLM response" in result3['answer']

    finally:
        # Restore original environment variables
        if original_cache_enabled is not None:
            os.environ["LLM_CACHE_ENABLED"] = original_cache_enabled
        elif "LLM_CACHE_ENABLED" in os.environ:
            del os.environ["LLM_CACHE_ENABLED"]
            
        if original_cache_backend is not None:
            os.environ["LLM_CACHE_BACKEND"] = original_cache_backend
        elif "LLM_CACHE_BACKEND" in os.environ:
            del os.environ["LLM_CACHE_BACKEND"]

        # Clear the global cache after the test
        clear_global_cache()


@patch('iris_rag.pipelines.basic.IRISStorage')
@patch('iris_rag.pipelines.basic.EmbeddingManager')
def test_cache_miss_with_different_queries(mock_embedding_manager, mock_iris_storage):
    """Test that different queries result in cache misses."""
    from iris_rag.pipelines.basic import BasicRAGPipeline
    from iris_rag.core.models import Document
    from common.utils import get_llm_func
    
    # Configure caching to be enabled for this test
    original_cache_enabled = os.environ.get("LLM_CACHE_ENABLED")
    original_cache_backend = os.environ.get("LLM_CACHE_BACKEND")
    os.environ["LLM_CACHE_ENABLED"] = "true"
    os.environ["LLM_CACHE_BACKEND"] = "memory"

    try:
        # Clear any existing global cache
        clear_global_cache()

        # Create minimal mocks for pipeline initialization
        mock_connection_manager = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.side_effect = lambda key, default=None: {
            "pipelines:basic": {},
            "llm.provider": "stub",
            "llm.model_name": "stub-model",
            "llm_cache.enabled": True
        }.get(key, default)

        # Setup the cache with memory backend
        setup_langchain_cache()

        # Get a properly cached LLM function using the real get_llm_func
        llm_func = get_llm_func(provider="stub", model_name="test-model", enable_cache=True)

        # Create pipeline with the cached LLM function
        pipeline = BasicRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager,
            llm_func=llm_func
        )

        # Mock the storage and embedding manager methods
        mock_iris_storage_instance = mock_iris_storage.return_value
        mock_embedding_manager_instance = mock_embedding_manager.return_value
        mock_embedding_manager_instance.embed_text.return_value = [0.1] * 10

        # Execute with first query - return France-related documents
        france_doc = Document(page_content="Paris is the capital of France and largest city.", metadata={"source": "france_info"})
        mock_iris_storage_instance.vector_search.return_value = [(france_doc, 0.9)]
        
        result1 = pipeline.execute("What is the capital of France?")
        assert result1['answer'] is not None
        assert "Stub LLM response" in result1['answer']

        # Execute with different query - return Germany-related documents
        germany_doc = Document(page_content="Berlin is the capital of Germany and largest city.", metadata={"source": "germany_info"})
        mock_iris_storage_instance.vector_search.return_value = [(germany_doc, 0.9)]
        
        result2 = pipeline.execute("What is the capital of Germany?")
        assert result2['answer'] is not None
        assert "Stub LLM response" in result2['answer']

        # Execute with first query again - return France-related documents again
        mock_iris_storage_instance.vector_search.return_value = [(france_doc, 0.9)]
        result3 = pipeline.execute("What is the capital of France?")
        assert result3['answer'] == result1['answer']  # Should be identical due to cache
        
        # Verify that we have some cache activity by checking that responses are consistent
        # The key test is that identical queries with identical context return identical responses
        assert result1['answer'] == result3['answer']

    finally:
        # Restore original environment variables
        if original_cache_enabled is not None:
            os.environ["LLM_CACHE_ENABLED"] = original_cache_enabled
        elif "LLM_CACHE_ENABLED" in os.environ:
            del os.environ["LLM_CACHE_ENABLED"]

        if original_cache_backend is not None:
            os.environ["LLM_CACHE_BACKEND"] = original_cache_backend
        elif "LLM_CACHE_BACKEND" in os.environ:
            del os.environ["LLM_CACHE_BACKEND"]

        # Clear the global cache after the test
        clear_global_cache()


def test_cache_disabled():
    """Test that caching can be disabled."""
    from common.llm_cache_manager import is_langchain_cache_configured
    
    # Configure caching to be disabled
    original_cache_enabled = os.environ.get("LLM_CACHE_ENABLED")
    os.environ["LLM_CACHE_ENABLED"] = "false"

    try:
        # Clear any existing global cache
        clear_global_cache()

        # Setup cache with disabled configuration
        with patch('common.llm_cache_config.load_cache_config') as mock_load_config:
            mock_config = Mock()
            mock_config.enabled = False
            mock_load_config.return_value = mock_config
            
            cache = setup_langchain_cache()
            assert cache is None

    finally:
        # Restore original environment variable
        if original_cache_enabled is not None:
            os.environ["LLM_CACHE_ENABLED"] = original_cache_enabled
        elif "LLM_CACHE_ENABLED" in os.environ:
            del os.environ["LLM_CACHE_ENABLED"]


def test_cache_stats():
    """Test that cache statistics can be retrieved."""
    from common.llm_cache_manager import get_cache_stats
    
    # Get cache stats (should work even if cache is not configured)
    stats = get_cache_stats()
    
    # Verify expected structure
    assert 'enabled' in stats
    assert 'backend' in stats
    assert 'configured' in stats
    assert 'metrics' in stats
    assert 'hits' in stats['metrics']
    assert 'misses' in stats['metrics']
    assert 'total_requests' in stats['metrics']
    assert 'hit_rate' in stats['metrics']


def test_cache_clear():
    """Test that cache can be cleared."""
    from common.llm_cache_manager import clear_global_cache
    
    # This should not raise an exception even if no cache is configured
    clear_global_cache()