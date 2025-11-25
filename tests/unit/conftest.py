"""Unit test configuration and fixtures.

This module provides mock configurations and fixtures specifically for unit tests.
Unit tests should use mocks for external dependencies (database, LLM, embeddings).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
from typing import Dict, Any


@pytest.fixture
def mock_iris_connection():
    """Mock IRIS database connection for unit tests."""
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[]), rowcount=0))
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    mock_conn.close = Mock()
    return mock_conn


@pytest.fixture
def mock_vector_store():
    """Mock vector store for unit tests."""
    mock_store = MagicMock()
    mock_store.add_documents = Mock(return_value=["doc1", "doc2"])
    mock_store.similarity_search = Mock(return_value=[])
    mock_store.similarity_search_with_score = Mock(return_value=[])
    mock_store.delete = Mock(return_value=True)
    mock_store.get_document_count = Mock(return_value=0)

    # Make it subscriptable for tests that use store[key] syntax
    mock_store.__getitem__ = Mock(return_value=Mock())
    mock_store.__setitem__ = Mock()

    return mock_store


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for unit tests."""
    mock_manager = Mock()
    mock_manager.generate_embeddings = Mock(return_value=[[0.1, 0.2, 0.3]])
    mock_manager.get_embedding_dimension = Mock(return_value=3)
    mock_manager.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock_manager.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
    return mock_manager


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for unit tests."""
    mock_llm = Mock()
    mock_llm.generate = Mock(return_value="Generated response from LLM")
    mock_llm.chat = Mock(return_value="Chat response from LLM")
    mock_llm.__call__ = Mock(return_value="LLM response")
    return mock_llm


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for unit tests."""
    mock_config = Mock()
    mock_config.get = Mock(side_effect=lambda key, default=None: {
        "database.iris.host": "localhost",
        "database.iris.port": 1972,
        "database.iris.username": "_SYSTEM",
        "database.iris.password": "SYS",
        "database.iris.namespace": "USER",
        "llm.provider": "openai",
        "llm.model": "gpt-4",
        "embeddings.model": "all-MiniLM-L6-v2",
        "embeddings.dimension": 384,
    }.get(key, default))
    mock_config.set = Mock()
    mock_config.get_config = Mock(return_value={
        "database": {
            "iris": {
                "host": "localhost",
                "port": 1972,
                "username": "_SYSTEM",
                "password": "SYS",
                "namespace": "USER"
            }
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4"
        },
        "embeddings": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384
        }
    })
    return mock_config


@pytest.fixture
def mock_schema_manager():
    """Mock schema manager for unit tests."""
    mock_manager = Mock()
    mock_manager.create_vector_table = Mock(return_value=True)
    mock_manager.table_exists = Mock(return_value=True)
    mock_manager.validate_table_schema = Mock(return_value=True)
    mock_manager.drop_table = Mock(return_value=True)
    mock_manager.get_table_info = Mock(return_value={
        "name": "test_table",
        "columns": ["id", "content", "embedding"],
        "vector_dimension": 384
    })
    return mock_manager


@pytest.fixture
def mock_entity_extraction_service():
    """Mock entity extraction service for unit tests."""
    mock_service = Mock()
    mock_service.extract_entities = Mock(return_value=[
        {"text": "entity1", "type": "PERSON", "confidence": 0.9},
        {"text": "entity2", "type": "ORG", "confidence": 0.8}
    ])
    mock_service.extract_relationships = Mock(return_value=[
        {"source": "entity1", "target": "entity2", "type": "WORKS_AT", "confidence": 0.85}
    ])
    return mock_service


@pytest.fixture
def mock_storage_service():
    """Mock storage service for unit tests."""
    mock_service = Mock()
    mock_service.store_documents = Mock(return_value=["doc1", "doc2"])
    mock_service.retrieve_documents = Mock(return_value=[])
    mock_service.get_storage_info = Mock(return_value={
        "total_documents": 100,
        "total_size_bytes": 1024000
    })
    mock_service.cleanup_storage = Mock(return_value=True)
    return mock_service


@pytest.fixture
def mock_pipeline():
    """Mock RAG pipeline for unit tests."""
    mock_pipeline = Mock()
    mock_pipeline.process = Mock(return_value={
        "query": "test query",
        "response": "test response",
        "sources": [],
        "metadata": {}
    })
    mock_pipeline.load_documents = Mock(return_value=True)
    mock_pipeline.setup = Mock(return_value=True)
    return mock_pipeline


@pytest.fixture
def mock_validation_orchestrator():
    """Mock validation orchestrator for unit tests."""
    mock_orchestrator = Mock()
    mock_orchestrator.validate = Mock(return_value={
        "valid": True,
        "issues": [],
        "warnings": []
    })
    mock_orchestrator.check_requirements = Mock(return_value=True)
    return mock_orchestrator


@pytest.fixture
def mock_pipeline_factory():
    """Mock pipeline factory for unit tests."""
    mock_factory = Mock()
    mock_factory.create = Mock(return_value=Mock())
    mock_factory.get_available_pipelines = Mock(return_value=["basic", "crag", "graphrag"])
    return mock_factory


@pytest.fixture
def unit_test_config() -> Dict[str, Any]:
    """Provide configuration dictionary for unit tests."""
    return {
        "database": {
            "iris": {
                "host": "localhost",
                "port": 1972,
                "username": "_SYSTEM",
                "password": "SYS",
                "namespace": "USER"
            }
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key"
        },
        "embeddings": {
            "provider": "sentence-transformers",
            "model": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "vector_store": {
            "provider": "iris",
            "table_name": "test_vectors"
        }
    }


@pytest.fixture
def temp_config_file(unit_test_config):
    """Create temporary config file for unit tests."""
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(unit_test_config, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "This is a test document about artificial intelligence.",
            "metadata": {"source": "test", "type": "article"}
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of AI.",
            "metadata": {"source": "test", "type": "tutorial"}
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0]
    ]


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        {"text": "OpenAI", "type": "ORG", "confidence": 0.95},
        {"text": "GPT-4", "type": "PRODUCT", "confidence": 0.90},
        {"text": "artificial intelligence", "type": "CONCEPT", "confidence": 0.85}
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationships for testing."""
    return [
        {
            "source": "OpenAI",
            "target": "GPT-4",
            "type": "DEVELOPS",
            "confidence": 0.92
        },
        {
            "source": "GPT-4",
            "target": "artificial intelligence",
            "type": "IS_TYPE_OF",
            "confidence": 0.88
        }
    ]


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks after each test."""
    yield
    # Cleanup happens automatically with pytest fixtures


# Patch common external dependencies for all unit tests
@pytest.fixture(autouse=True)
def patch_external_dependencies(monkeypatch):
    """Automatically patch external dependencies for unit tests."""
    # Patch sentence transformers to avoid downloading models
    mock_sentence_transformer = Mock()
    mock_sentence_transformer.encode = Mock(return_value=[[0.1, 0.2, 0.3]])

    # Mock the module imports BEFORE they're used to avoid transformer errors
    import sys

    # Create mock modules
    mock_transformers = Mock()
    mock_sentence_transformers_module = Mock()
    mock_sentence_transformers_module.SentenceTransformer = Mock(return_value=mock_sentence_transformer)

    # Inject mocks into sys.modules to prevent actual imports
    if 'transformers' not in sys.modules:
        sys.modules['transformers'] = mock_transformers
    if 'sentence_transformers' not in sys.modules:
        sys.modules['sentence_transformers'] = mock_sentence_transformers_module

    try:
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            Mock(return_value=mock_sentence_transformer)
        )
    except (AttributeError, KeyError):
        pass  # Module not available, skip patching

    # Patch OpenAI client
    try:
        monkeypatch.setattr(
            "openai.OpenAI",
            Mock(return_value=Mock())
        )
    except AttributeError:
        pass

    yield
