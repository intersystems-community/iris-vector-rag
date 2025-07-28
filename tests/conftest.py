import sys
import os
import pytest
from tests.mocks.mock_iris_connector import MockIRISConnector

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_iris_connector():
    """A pytest fixture that provides a mock IRIS connector."""
    return MockIRISConnector()

@pytest.fixture
def iris_testcontainer_connection():
    """
    Provide IRIS testcontainer connection for E2E tests.
    This fixture provides a real IRIS database connection for testing.
    """
    from common.iris_connection_manager import get_iris_connection
    
    # Get a real IRIS connection
    connection = get_iris_connection()
    if connection is None:
        pytest.skip("IRIS connection not available for testcontainer tests")
    
    yield connection
    
    # Cleanup: close connection if needed
    try:
        if hasattr(connection, 'close'):
            connection.close()
    except Exception:
        pass  # Ignore cleanup errors

@pytest.fixture
def embedding_model_fixture():
    """Provide embedding model for tests."""
    from common.utils import get_embedding_func
    return get_embedding_func()

@pytest.fixture
def llm_client_fixture():
    """Provide LLM client for tests."""
    from common.utils import get_llm_func
    return get_llm_func()

@pytest.fixture
def iris_connection():
    """
    Provide IRIS connection for tests. This is an alias for iris_testcontainer_connection
    to maintain compatibility with existing test files.
    """
    from common.iris_connection_manager import get_iris_connection
    
    # Get a real IRIS connection
    connection = get_iris_connection()
    if connection is None:
        pytest.skip("IRIS connection not available for tests")
    
    yield connection
    
    # Cleanup: close connection if needed
    try:
        if hasattr(connection, 'close'):
            connection.close()
    except Exception:
        pass  # Ignore cleanup errors

@pytest.fixture
def real_config_manager():
    """Provide a real configuration manager for tests."""
    from iris_rag.config.manager import ConfigurationManager
    return ConfigurationManager()

@pytest.fixture
def colbert_query_encoder():
    """Provide ColBERT query encoder for tests."""
    from common.utils import get_colbert_query_encoder
    try:
        return get_colbert_query_encoder()
    except (ImportError, AttributeError):
        # Fallback to a mock encoder if real one is not available
        from unittest.mock import MagicMock
        import numpy as np
        mock_encoder = MagicMock()
        mock_encoder.return_value = np.random.rand(5, 128).astype(np.float32)
        return mock_encoder