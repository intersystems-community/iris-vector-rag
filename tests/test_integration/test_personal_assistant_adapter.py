"""
Integration tests for the PersonalAssistantAdapter.
"""
import pytest
import json
import logging
from unittest.mock import patch, MagicMock

from iris_rag.adapters.personal_assistant import PersonalAssistantAdapter
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.basic import BasicRAGPipeline # Target pipeline
from iris_rag.core.connection import ConnectionManager

# Configure basic logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_basic_rag_pipeline():
    """Fixture for a mocked BasicRAGPipeline."""
    pipeline = MagicMock(spec=BasicRAGPipeline)
    pipeline.query.return_value = {"answer": "mocked answer", "retrieved_documents": []}
    return pipeline

@pytest.fixture
def mock_connection_manager():
    """Fixture for a mocked ConnectionManager."""
    cm = MagicMock(spec=ConnectionManager)
    cm.get_iris_connection.return_value = MagicMock() # Simulate a successful connection
    return cm

@pytest.fixture
def base_pa_config():
    """Base Personal Assistant-like configuration."""
    return {
        "iris_host": "localhost",
        "iris_port": 1972,
        "iris_namespace": "TESTNS",
        "iris_user": "testuser",
        "iris_password": "testpassword",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model_name": "test-llm",
        "log_level": "DEBUG"
    }

@pytest.fixture
def temp_config_file(tmp_path, base_pa_config):
    """Creates a temporary config file and returns its path."""
    config_file = tmp_path / "pa_config.json"
    with open(config_file, "w") as f:
        json.dump(base_pa_config, f)
    return str(config_file)

def test_adapter_initialization_default():
    """Test PersonalAssistantAdapter initializes with default config manager."""
    adapter = PersonalAssistantAdapter()
    assert isinstance(adapter.config_manager, ConfigurationManager)
    assert isinstance(adapter.connection_manager, ConnectionManager)
    assert adapter.rag_pipeline is None

def test_adapter_initialization_with_config_dict(base_pa_config):
    """Test PersonalAssistantAdapter initializes with a config dictionary."""
    adapter = PersonalAssistantAdapter(config=base_pa_config)
    assert adapter.config_manager.get_config("iris_host") == "localhost"

@patch('iris_rag.adapters.personal_assistant.BasicRAGPipeline')
def test_initialize_iris_rag_pipeline_with_pa_config(
    MockedBasicRAGPipeline, base_pa_config, mock_connection_manager
):
    """Test pipeline initialization with PA-specific config dictionary."""
    mock_pipeline_instance = MockedBasicRAGPipeline.return_value
    mock_pipeline_instance.query.return_value = {"answer": "success"}

    adapter = PersonalAssistantAdapter()
    # Patch the adapter's connection_manager to use the mock
    adapter.connection_manager = mock_connection_manager
    
    # Simulate PA-specific keys that need translation (if any defined in _translate_config)
    pa_specific_config = base_pa_config.copy()
    pa_specific_config["pa_db_host"] = "pa_specific_host" # Example PA key

    # Expected translated key if _translate_config maps it
    # For this test, assume _translate_config is simple or we test its effect indirectly
    
    pipeline = adapter.initialize_iris_rag_pipeline(pa_specific_config=pa_specific_config)

    assert pipeline is mock_pipeline_instance
    MockedBasicRAGPipeline.assert_called_once()
    # Check that config_manager within BasicRAGPipeline received translated/merged config
    called_config_manager = MockedBasicRAGPipeline.call_args[1]['config_manager']
    assert called_config_manager.get_config("iris_host") == "localhost" # From base
    # If "pa_db_host" was translated to "iris_host", this would be "pa_specific_host"
    # Current _translate_config in adapter is a placeholder, so direct check is tricky.
    # We rely on the fact that config_manager was updated.
    
    # Verify that the connection_manager passed to BasicRAGPipeline is the adapter's (mocked) one
    assert MockedBasicRAGPipeline.call_args[1]['connection_manager'] is mock_connection_manager


@patch('iris_rag.adapters.personal_assistant.BasicRAGPipeline')
def test_initialize_iris_rag_pipeline_with_config_file(
    MockedBasicRAGPipeline, temp_config_file, base_pa_config, mock_connection_manager
):
    """Test pipeline initialization with a config file path."""
    mock_pipeline_instance = MockedBasicRAGPipeline.return_value
    mock_pipeline_instance.query.return_value = {"answer": "success"}

    adapter = PersonalAssistantAdapter()
    adapter.connection_manager = mock_connection_manager # Use mocked CM

    pipeline = adapter.initialize_iris_rag_pipeline(config_path=temp_config_file)

    assert pipeline is mock_pipeline_instance
    MockedBasicRAGPipeline.assert_called_once()
    called_config_manager = MockedBasicRAGPipeline.call_args[1]['config_manager']
    assert called_config_manager.get_config("iris_host") == base_pa_config["iris_host"]
    assert MockedBasicRAGPipeline.call_args[1]['connection_manager'] is mock_connection_manager


@patch('iris_rag.adapters.personal_assistant.BasicRAGPipeline')
def test_initialize_iris_rag_pipeline_failure(MockedBasicRAGPipeline, base_pa_config, mock_connection_manager):
    """Test pipeline initialization failure is handled."""
    MockedBasicRAGPipeline.side_effect = Exception("Initialization failed")

    adapter = PersonalAssistantAdapter()
    adapter.connection_manager = mock_connection_manager

    with pytest.raises(Exception, match="Initialization failed"):
        adapter.initialize_iris_rag_pipeline(pa_specific_config=base_pa_config)

def test_query_before_initialization():
    """Test query method raises error if pipeline is not initialized."""
    adapter = PersonalAssistantAdapter()
    with pytest.raises(RuntimeError, match="RAG pipeline not initialized"):
        adapter.query("test query")

@patch('iris_rag.adapters.personal_assistant.BasicRAGPipeline')
def test_query_success(MockedBasicRAGPipeline, base_pa_config, mock_connection_manager):
    """Test successful query after pipeline initialization."""
    mock_pipeline_instance = MockedBasicRAGPipeline.return_value
    expected_response = {"answer": "mocked answer from pipeline", "retrieved_documents": ["doc1"]}
    mock_pipeline_instance.query.return_value = expected_response

    adapter = PersonalAssistantAdapter()
    adapter.connection_manager = mock_connection_manager
    adapter.initialize_iris_rag_pipeline(pa_specific_config=base_pa_config)

    response = adapter.query("test query", top_k=5)
    
    assert response == expected_response
    mock_pipeline_instance.query.assert_called_once_with("test query", top_k=5)


@patch('iris_rag.adapters.personal_assistant.BasicRAGPipeline')
def test_query_pipeline_error_handling(MockedBasicRAGPipeline, base_pa_config, mock_connection_manager):
    """Test query method handles errors from the RAG pipeline."""
    mock_pipeline_instance = MockedBasicRAGPipeline.return_value
    mock_pipeline_instance.query.side_effect = Exception("Pipeline query error")

    adapter = PersonalAssistantAdapter()
    adapter.connection_manager = mock_connection_manager
    adapter.initialize_iris_rag_pipeline(pa_specific_config=base_pa_config)

    response = adapter.query("test query")

    assert "error" in response
    assert response["error"] == "Pipeline query error"
    assert response["answer"] == "Sorry, I encountered an error processing your request."


def test_config_translation_logic(base_pa_config):
    """Test the _translate_config logic if it were more complex."""
    adapter = PersonalAssistantAdapter()
    
    # Example: PA config has different key names
    pa_specific_conf = {
        "pa_db_host": "pa_host",
        "pa_db_port": 1234,
        "pa_namespace": "PA_NS",
        "pa_user": "pa_u",
        "pa_pass": "pa_p",
        "embedding_config": {"model": "text-embed-001"},
        "llm_config": {"provider": "some_llm", "key": "secret"}
    }
    
    # This test depends on the actual implementation of _translate_config.
    # The current _translate_config is a simple copy.
    # If it were more complex, e.g.:
    # def _translate_config(self, pa_config: Dict[str, Any]) -> Dict[str, Any]:
    #     rag_config = {}
    #     rag_config["iris_host"] = pa_config.get("pa_db_host")
    #     rag_config["embedding_model_name"] = pa_config.get("embedding_config", {}).get("model")
    #     # ... and so on
    #     return rag_config
    # Then this test would verify those specific translations.
    
    # For the current placeholder _translate_config:
    translated = adapter._translate_config(pa_specific_conf)
    assert translated["pa_db_host"] == "pa_host" # It's a copy

    # If we had actual translation rules in the adapter:
    # For example, if PersonalAssistantAdapter._translate_config was:
    # def _translate_config(self, pa_config: Dict[str, Any]) -> Dict[str, Any]:
    #     rag_config = {}
    #     if "pa_db_host" in pa_config:
    #         rag_config["iris_host"] = pa_config["pa_db_host"]
    #     return rag_config
    #
    # Then the test would be:
    # adapter_with_rules = PersonalAssistantAdapter()
    # adapter_with_rules._translate_config = MethodType(lambda self, conf: {"iris_host": conf.get("pa_db_host")}, adapter_with_rules) # Monkey patch for test
    # translated_with_rules = adapter_with_rules._translate_config({"pa_db_host": "specific_pa_host"})
    # assert translated_with_rules.get("iris_host") == "specific_pa_host"
    
    # Since the current _translate_config is basic, this test mainly serves as a placeholder
    # for when more complex translation logic is added.
    assert "pa_db_host" in translated # Verifies the copy behavior for now.


# To run these tests: pytest tests/test_integration/test_personal_assistant_adapter.py