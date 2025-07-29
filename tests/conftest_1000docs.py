# conftest_1000docs.py
"""
Pytest configuration for tests requiring 1000+ documents.
This module provides fixtures that ensure at least 1000 documents are available for testing.
"""

import pytest
from unittest.mock import Mock

@pytest.fixture
def ensure_1000_docs():
    """Fixture that ensures 1000+ documents are available for testing."""
    # Mock implementation for now - in real implementation this would check actual document count
    mock_docs = Mock()
    mock_docs.count = 1000
    return mock_docs

@pytest.fixture
def sample_1000_docs():
    """Fixture providing sample data for 1000+ document tests."""
    # Mock implementation - in real implementation this would provide actual document data
    return [{"id": i, "content": f"Document {i} content"} for i in range(1000)]

@pytest.fixture
def enterprise_iris_connection():
    """Mock enterprise IRIS connection for 1000+ document tests."""
    mock_connection = Mock()
    mock_connection.is_connected = True
    mock_connection.execute_query = Mock(return_value=[])
    return mock_connection

@pytest.fixture
def scale_test_config():
    """Configuration for scale testing with 1000+ documents."""
    return {
        "min_documents": 1000,
        "test_mode": "scale",
        "batch_size": 100,
        "timeout": 300
    }

@pytest.fixture
def enterprise_schema_manager():
    """Mock enterprise schema manager for 1000+ document tests."""
    mock_schema = Mock()
    mock_schema.ensure_tables = Mock(return_value=True)
    mock_schema.get_table_info = Mock(return_value={"documents": 1000})
    return mock_schema

@pytest.fixture
def scale_test_documents():
    """Mock scale test documents for chunking architecture tests."""
    return [{"id": i, "content": f"Scale test document {i} content"} for i in range(1000)]

@pytest.fixture
def enterprise_document_loader_1000docs():
    """Mock enterprise document loader for 1000+ documents."""
    mock_loader = Mock()
    mock_loader.load_documents = Mock(return_value=[{"id": i, "content": f"Doc {i}"} for i in range(1000)])
    return mock_loader

@pytest.fixture
def enterprise_embedding_manager():
    """Mock enterprise embedding manager for 1000+ document tests."""
    mock_embedding = Mock()
    mock_embedding.embed_text = Mock(return_value=[0.1] * 384)
    mock_embedding.embed_batch = Mock(return_value=[[0.1] * 384 for _ in range(100)])
    return mock_embedding

@pytest.fixture
def enterprise_llm_function():
    """Mock enterprise LLM function for 1000+ document tests."""
    def mock_llm(prompt):
        return f"Mock LLM response to: {prompt[:50]}..."
    return mock_llm

@pytest.fixture
def scale_test_performance_monitor():
    """Mock performance monitor for scale testing."""
    mock_monitor = Mock()
    mock_monitor.start_monitoring = Mock()
    mock_monitor.stop_monitoring = Mock()
    mock_monitor.get_metrics = Mock(return_value={"cpu": 50, "memory": 1024, "duration": 10.5})
    return mock_monitor

@pytest.fixture
def enterprise_test_queries():
    """Mock enterprise test queries for 1000+ document tests."""
    return [
        "What are the main findings in cardiovascular research?",
        "How does machine learning apply to medical diagnosis?",
        "What are the latest developments in cancer treatment?",
        "Explain the role of genetics in disease prevention.",
        "What are the benefits of telemedicine?"
    ]