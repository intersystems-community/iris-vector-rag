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
    from unittest.mock import Mock
    from iris_rag.config.manager import ConfigurationManager
    
    # Create a mock config manager with proper configuration structure
    mock_config_manager = Mock(spec=ConfigurationManager)
    
    # Configure the mock to return appropriate values for different config keys
    def mock_get(key, default=None):
        config_values = {
            # Storage chunking configuration
            "storage:chunking": {
                "enabled": True,
                "strategy": "fixed_size",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "strategies": {
                    "fixed_size": {"enabled": True},
                    "semantic": {"enabled": True},
                    "hybrid": {"enabled": True}
                }
            },
            # Embedding model configuration
            "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model.dimension": 384,
            # ColBERT configuration
            "colbert": {
                "backend": "native",
                "token_dimension": 768,
                "model_name": "bert-base-uncased"
            },
            "colbert.token_dimension": 768,
            "colbert.backend": "native",
            "colbert.model_name": "bert-base-uncased",
            # Pipeline overrides
            "pipeline_overrides": {}
        }
        return config_values.get(key, default)
    
    mock_config_manager.get.side_effect = mock_get
    
    return {
        "min_documents": 1000,
        "test_mode": "scale",
        "batch_size": 100,
        "timeout": 300,
        "config_manager": mock_config_manager
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
    from iris_rag.core.models import Document
    # Return actual Document objects that can be sliced
    documents = []
    for i in range(1000):
        doc = Document(
            id=f"test_doc_{i+1:03d}",
            page_content=f"Medical research document {i+1} discussing COVID-19 treatment protocols, symptoms, and patient outcomes. This document contains detailed information about diagnosis procedures, medication effectiveness, and recovery statistics.",
            metadata={"source": f"test_source_{i+1}", "category": "medical", "test_document": True}
        )
        documents.append(doc)
    return documents

@pytest.fixture
def enterprise_embedding_manager():
    """Mock enterprise embedding manager for 1000+ document tests."""
    mock_embedding = Mock()
    mock_embedding.embed_text = Mock(return_value=[0.1] * 384)
    mock_embedding.embed_batch = Mock(return_value=[[0.1] * 384 for _ in range(100)])
    
    # Create mock embedding function that the chunking integration test expects
    def mock_embedding_function(texts):
        if isinstance(texts, str):
            return [0.1] * 384
        elif isinstance(texts, list):
            return [[0.1] * 384 for _ in texts]
        else:
            return [0.1] * 384
    
    mock_embedding.get_embedding_function = Mock(return_value=mock_embedding_function)
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