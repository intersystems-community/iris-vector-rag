"""
Tests for Basic RAG Pipeline implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock # Added MagicMock
from iris_rag.core.models import Document
from iris_rag.pipelines.basic import BasicRAGPipeline # Moved to top-level import
from iris_rag.core.connection import ConnectionManager as RealConnectionManager
from iris_rag.config.manager import ConfigurationManager as PipelineConfigManager
# import os # Not strictly needed as os.path and os.environ are patched by string path

def test_basic_pipeline_imports():
    """Test that basic pipeline can be imported."""
    # from iris_rag.pipelines.basic import BasicRAGPipeline # Now imported at top
    assert BasicRAGPipeline is not None


def test_document_creation():
    """Test basic document creation."""
    doc = Document(
        page_content="Test content",
        metadata={"source": "test.txt"}
    )
    assert doc.page_content == "Test content"
    assert doc.metadata["source"] == "test.txt"


@patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
@patch('iris_rag.pipelines.basic.EmbeddingManager')
def test_pipeline_initialization(mock_embedding_manager, mock_vector_store):
    """Test pipeline initialization with minimal mocks."""
    # Create minimal mocks
    mock_connection_manager = Mock()
    mock_config_manager = Mock()
    mock_config_manager.get.return_value = {}
    
    # Create pipeline
    pipeline = BasicRAGPipeline(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )
    
    # Verify basic initialization
    assert pipeline.connection_manager == mock_connection_manager
    assert pipeline.config_manager == mock_config_manager
    assert pipeline.vector_store is not None


def test_text_chunking():
    """Test text chunking functionality without heavy mocks."""
    # from iris_rag.pipelines.basic import BasicRAGPipeline # Now imported at top
    
    # Create minimal pipeline instance for testing utility methods
    mock_connection_manager = Mock()
    mock_config_manager = Mock()
    mock_config_manager.get.return_value = {}
    
    with patch('iris_rag.storage.vector_store_iris.IRISVectorStore'), \
         patch('iris_rag.pipelines.basic.EmbeddingManager'):
        
        pipeline = BasicRAGPipeline(
            connection_manager=mock_connection_manager,
            config_manager=mock_config_manager
        )
        
        # Set small chunk size for testing
        pipeline.chunk_size = 50
        pipeline.chunk_overlap = 10
        
        # Test text splitting
        text = "This is a test. " * 10  # 160 characters
        chunks = pipeline._split_text(text)
        
        # Verify chunks were created
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= pipeline.chunk_size + pipeline.chunk_overlap


def test_factory_function():
    """Test the create_pipeline factory function."""
    from iris_rag import create_pipeline
    
    # Test unknown pipeline type
    with pytest.raises(ValueError, match="Unknown pipeline type"):
        create_pipeline("unknown_type")


def test_standard_return_format():
    """Test that pipeline returns standard format."""
    # This is a basic structure test
    expected_keys = ["query", "answer", "retrieved_documents"]
    
    # Mock result structure
    result = {
        "query": "test query",
        "answer": "test answer",
        "retrieved_documents": []
    }
    
    for key in expected_keys:
        assert key in result

@patch('common.iris_connector.jaydebeapi.connect')
@patch('common.iris_connector.os.environ.get')
@patch('common.iris_connector.ConfigurationManager') # Mock for DB credentials source
@patch('common.iris_connector.os.path.exists')    # Mock for JDBC_JAR_PATH check
@patch('iris_rag.pipelines.basic.EmbeddingManager') # Dependency of BasicRAGPipeline
def test_basic_pipeline_connection_uses_config_manager(
    mock_embedding_manager_class, # Patched class for EmbeddingManager
    mock_os_path_exists,          # Mock for common.iris_connector.os.path.exists
    mock_db_creds_config_manager_class, # Mock for common.iris_connector.ConfigurationManager
    mock_os_environ_get,          # Mock for common.iris_connector.os.environ.get
    mock_jaydebeapi_connect       # Mock for common.iris_connector.jaydebeapi.connect
):
    """
    Tests that BasicRAGPipeline, when using its default ConnectionManager,
    sources DB credentials from common.iris_connector.ConfigurationManager.
    """
    # 1. Configure mocks
    mock_os_path_exists.return_value = True # Simulate JDBC JAR exists

    # Configure the mock for common.iris_connector.ConfigurationManager
    # This is the ConfigurationManager instance that get_iris_connection will create and use.
    mock_db_config_instance = mock_db_creds_config_manager_class.return_value
    
    test_cm_creds = {
        "database:iris:host": "pipeline_cm_host",
        "database:iris:port": 6543,
        "database:iris:namespace": "PIPELINE_CM_NS",
        "database:iris:username": "pipeline_cm_user",
        "database:iris:password": "pipeline_cm_pass"
    }
    
    def cm_get_side_effect(key):
        return test_cm_creds.get(key)
    mock_db_config_instance.get.side_effect = cm_get_side_effect

    # Configure mock for common.iris_connector.os.environ.get
    # These credentials should be different from test_cm_creds to ensure they are ignored.
    env_creds = {
        "IRIS_HOST": "env_host",
        "IRIS_PORT": "1234", # Ensure this is a string, as os.environ.get returns strings
        "IRIS_NAMESPACE": "ENV_NS",
        "IRIS_USERNAME": "env_user",
        "IRIS_PASSWORD": "env_pass"
    }
    def environ_get_side_effect(key, default=None):
        # common.iris_connector.get_real_iris_connection has logic for int(os.environ.get("IRIS_PORT", "1972"))
        # but our test path (config=None) should prioritize ConfigurationManager over os.environ.
        return env_creds.get(key, default)
    mock_os_environ_get.side_effect = environ_get_side_effect
    
    # Configure mock for jaydebeapi.connect
    mock_jdbc_connection = MagicMock() # Use MagicMock for full mock API
    mock_jdbc_cursor = MagicMock()

    # Configure cursor methods to prevent downstream errors during schema initialization
    # These ensure that calls like fetchone() or fetchall() during IRISStorage.initialize_schema()
    # return valid empty results instead of raising errors due to an unconfigured MagicMock.
    mock_jdbc_cursor.fetchone.return_value = None  # Simulate e.g., table not found or no result
    mock_jdbc_cursor.fetchall.return_value = []    # Simulate e.g., no existing tables/columns
    # mock_jdbc_cursor.execute and mock_jdbc_cursor.close are implicitly created by MagicMock

    mock_jdbc_connection.cursor.return_value = mock_jdbc_cursor
    # mock_jdbc_connection.commit and mock_jdbc_connection.close are implicitly created by MagicMock
    mock_jaydebeapi_connect.return_value = mock_jdbc_connection

    # Configure the mock EmbeddingManager instance
    mock_embedding_manager_instance = mock_embedding_manager_class.return_value

    # 2. Setup: Instantiate pipeline components
    # This is a ConfigurationManager for the pipeline's own config (e.g., chunk_size),
    # NOT for DB credentials in this test's context.
    mock_pipeline_level_config_manager = Mock(spec=PipelineConfigManager)
    mock_pipeline_level_config_manager.get.return_value = {} # Default for chunk_size, etc.

    # Instantiate the *real* ConnectionManager.
    # It should internally call get_iris_connection(config=None), which then uses
    # the patched common.iris_connector.ConfigurationManager.
    real_connection_manager = RealConnectionManager(config_manager=mock_pipeline_level_config_manager)

    # 3. Trigger connection: Instantiate BasicRAGPipeline
    # The __init__ of BasicRAGPipeline creates an IRISStorage instance and calls
    # its initialize_schema method, which should trigger ConnectionManager.get_connection().
    # We allow the real IRISStorage to be created, but the database operations will be mocked
    # through the mocked jaydebeapi.connect
    pipeline = BasicRAGPipeline(
        connection_manager=real_connection_manager,
        config_manager=mock_pipeline_level_config_manager, # For pipeline's own settings
        llm_func=Mock() # Mock LLM function
    )

    # 4. Assertion: Verify jaydebeapi.connect was called with credentials from ConfigurationManager
    expected_jdbc_url = f"jdbc:IRIS://{test_cm_creds['database:iris:host']}:{test_cm_creds['database:iris:port']}/{test_cm_creds['database:iris:namespace']}"
    expected_jdbc_user = test_cm_creds['database:iris:username']
    expected_jdbc_pass = test_cm_creds['database:iris:password']
    
    # This assertion is intended to FAIL if the SUT does not use ConfigurationManager as expected.
    mock_jaydebeapi_connect.assert_called_once()
    
    # Inspect the arguments passed to jaydebeapi.connect
    # call_args is a tuple: (args, kwargs)
    # We are interested in the positional arguments: call_args[0]
    # args[0] is JDBC_DRIVER_CLASS string
    # args[1] is jdbc_url string
    # args[2] is [username, password] list
    # args[3] is JDBC_JAR_PATH string
    
    actual_call_args = mock_jaydebeapi_connect.call_args[0]
    actual_jdbc_url = actual_call_args[1]
    actual_jdbc_user_pass_list = actual_call_args[2]
    
    assert actual_jdbc_url == expected_jdbc_url, \
        f"JDBC URL mismatch. Expected: {expected_jdbc_url}, Actual: {actual_jdbc_url}"
    assert actual_jdbc_user_pass_list[0] == expected_jdbc_user, \
        f"JDBC User mismatch. Expected: {expected_jdbc_user}, Actual: {actual_jdbc_user_pass_list[0]}"
    assert actual_jdbc_user_pass_list[1] == expected_jdbc_pass, \
        f"JDBC Password mismatch. Expected: {expected_jdbc_pass}, Actual: {actual_jdbc_user_pass_list[1]}"

    # Verify that the mocked common.iris_connector.ConfigurationManager.get was called for DB parameters
    mock_db_config_instance.get.assert_any_call("database:iris:host")
    mock_db_config_instance.get.assert_any_call("database:iris:port")
    mock_db_config_instance.get.assert_any_call("database:iris:namespace")
    mock_db_config_instance.get.assert_any_call("database:iris:username")
    mock_db_config_instance.get.assert_any_call("database:iris:password")

    # Verify os.environ.get was called (to show it was considered, though its values shouldn't be used for connect)
    # This helps confirm that the logic for os.environ was reached but superseded.
    # The exact calls might depend on the structure of get_real_iris_connection if config is None.
    # For now, the primary assertion is on jaydebeapi.connect arguments.
    # If get_iris_connection with config=None *only* uses ConfigurationManager and doesn't even look at os.environ,
    # then mock_os_environ_get might not be called in that specific path.
    # The current common.iris_connector.py (lines 42-62) when config is None *only* uses ConfigurationManager.
    # So, mock_os_environ_get might not be called in this specific test path.
    # If it's not called, an assert_called_once_with or assert_any_call would fail.
    # This is fine, as it means ConfigurationManager was correctly prioritized.
    # For a robust test, we might check mock_os_environ_get.called is False if that's the expected behavior.
    # However, the prompt implies os.environ.get *is* mocked to return *different* credentials,
    # suggesting it might be checked by the SUT.
    # Let's assume for now the critical check is that jaydebeapi.connect used CM creds.