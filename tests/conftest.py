"""Global pytest configuration and fixtures for the RAG templates framework.

This module contains shared pytest fixtures and configuration that are used
across all test modules. It provides common setup and teardown functionality,
test data, and mock objects.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Generator
from unittest.mock import Mock, AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import framework modules
from iris_rag.config.manager import ConfigurationManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide mock configuration data for tests."""
    return {
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False,
        },
        "redis": {
            "url": "redis://localhost:6379/0",
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-api-key",
        },
        "vector_store": {
            "provider": "iris",
            "connection_string": "localhost:1972/USER",
        },
        "memory": {
            "provider": "mem0",
            "config": {
                "vector_store": {
                    "provider": "iris",
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4",
                    }
                }
            }
        }
    }


@pytest.fixture
def config_manager(temp_dir: Path, mock_config: Dict[str, Any]) -> ConfigurationManager:
    """Create a ConfigurationManager instance for testing."""
    config_file = temp_dir / "test_config.yaml"
    
    # Write config to temporary file
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(mock_config, f)
    
    # Set environment variable to point to test config
    os.environ["RAG_TEMPLATES_CONFIG"] = str(config_file)
    
    try:
        manager = ConfigurationManager()
        yield manager
    finally:
        # Clean up environment variable
        if "RAG_TEMPLATES_CONFIG" in os.environ:
            del os.environ["RAG_TEMPLATES_CONFIG"]


@pytest.fixture
def iris_config_manager(temp_dir: Path, mock_config: Dict[str, Any]) -> ConfigurationManager:
    """Create an IRIS ConfigManager instance for testing."""
    config_file = temp_dir / "iris_config.yaml"
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(mock_config, f)
    
    manager = ConfigurationManager(config_path=str(config_file))
    return manager


@pytest.fixture
def mock_database_session():
    """Create a mock database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get = Mock(return_value=None)
    mock_redis.set = Mock(return_value=True)
    mock_redis.delete = Mock(return_value=1)
    mock_redis.exists = Mock(return_value=False)
    mock_redis.expire = Mock(return_value=True)
    return mock_redis


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Generated response")
    mock_llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock_llm


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    mock_store = Mock()
    mock_store.add_documents = Mock(return_value=["doc1", "doc2"])
    mock_store.similarity_search = Mock(return_value=[])
    mock_store.similarity_search_with_score = Mock(return_value=[])
    return mock_store


@pytest.fixture
def mock_mem0_client():
    """Create a mock Mem0 client for testing."""
    mock_mem0 = AsyncMock()
    mock_mem0.add = AsyncMock(return_value={"id": "memory-123"})
    mock_mem0.search = AsyncMock(return_value=[])
    mock_mem0.get = AsyncMock(return_value=None)
    mock_mem0.delete = AsyncMock(return_value=True)
    return mock_mem0


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "This is a sample document about artificial intelligence.",
            "metadata": {"source": "test", "type": "article"}
        },
        {
            "id": "doc2", 
            "content": "This document discusses machine learning algorithms.",
            "metadata": {"source": "test", "type": "research"}
        },
        {
            "id": "doc3",
            "content": "A comprehensive guide to retrieval-augmented generation.",
            "metadata": {"source": "test", "type": "guide"}
        }
    ]


@pytest.fixture
def sample_queries():
    """Provide sample queries for testing."""
    return [
        "What is artificial intelligence?",
        "How do machine learning algorithms work?",
        "Explain retrieval-augmented generation",
        "What are the benefits of RAG systems?"
    ]


@pytest.fixture
def mock_pipeline():
    """Create a mock RAG pipeline for testing."""
    mock_pipeline = Mock()
    mock_pipeline.process = Mock(return_value={
        "query": "test query",
        "response": "test response",
        "sources": []
    })
    return mock_pipeline


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Cleanup environment variables after each test."""
    original_env = os.environ.copy()
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def docker_services():
    """Wait for Docker services to be ready."""
    # This fixture can be used with pytest-docker to wait for services
    # Implementation would depend on the specific services needed
    pass


# Async fixtures for testing async code
@pytest_asyncio.fixture
async def async_mock_llm_client():
    """Create an async mock LLM client for testing."""
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Async generated response")
    mock_llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock_llm


@pytest_asyncio.fixture
async def async_mock_mem0_client():
    """Create an async mock Mem0 client for testing."""
    mock_mem0 = AsyncMock()
    mock_mem0.add = AsyncMock(return_value={"id": "async-memory-123"})
    mock_mem0.search = AsyncMock(return_value=[])
    mock_mem0.get = AsyncMock(return_value=None)
    mock_mem0.delete = AsyncMock(return_value=True)
    return mock_mem0


# Markers for different test categories
pytest_plugins = ["pytest_asyncio"]

# Configure test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_docker: mark test as requiring Docker"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)