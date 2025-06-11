"""
Integration tests for the SurvivalModeRAGService.
"""
import pytest
import logging
from unittest.mock import MagicMock, patch, PropertyMock

from iris_rag.services.survival_mode import SurvivalModeRAGService
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

# Configure basic logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_config_manager():
    """Fixture for a mocked ConfigurationManager."""
    cm = MagicMock(spec=ConfigurationManager)
    cm.get_config.side_effect = lambda key, default=None: {"embedding_model_name": "mock_embed", "llm_model_name": "mock_llm"}.get(key, default)
    return cm

@pytest.fixture
def mock_connection_manager(mock_config_manager):
    """Fixture for a mocked ConnectionManager."""
    conn_mgr = MagicMock(spec=ConnectionManager)
    conn_mgr.config_manager = mock_config_manager
    mock_db_conn = MagicMock() # Mock for the database connection object
    conn_mgr.get_iris_connection.return_value = mock_db_conn
    return conn_mgr

@pytest.fixture
def mock_successful_basic_rag_pipeline(mock_connection_manager, mock_config_manager):
    """Fixture for a BasicRAGPipeline that queries successfully."""
    pipeline = MagicMock(spec=BasicRAGPipeline)
    pipeline.query.return_value = {"answer": "Primary answer", "retrieved_documents": [], "source": "PrimaryRAG"}
    # Mock attributes that might be accessed during init or health check
    pipeline.connection_manager = mock_connection_manager
    pipeline.config_manager = mock_config_manager
    pipeline.embedding_model = MagicMock()
    pipeline.llm = MagicMock()
    pipeline.iris_connector = mock_connection_manager.get_iris_connection()
    return pipeline

@pytest.fixture
def mock_failing_basic_rag_pipeline(mock_connection_manager, mock_config_manager):
    """Fixture for a BasicRAGPipeline that fails on query."""
    pipeline = MagicMock(spec=BasicRAGPipeline)
    pipeline.query.side_effect = Exception("Pipeline query failed")
    pipeline.connection_manager = mock_connection_manager
    pipeline.config_manager = mock_config_manager
    pipeline.embedding_model = MagicMock()
    pipeline.llm = MagicMock()
    pipeline.iris_connector = mock_connection_manager.get_iris_connection()
    return pipeline


# Patch BasicRAGPipeline for tests where its instantiation is part of the test
@patch('iris_rag.services.survival_mode.BasicRAGPipeline')
def test_service_initialization_with_successful_primary_pipeline(
    MockedBasicRAGPipeline, mock_successful_basic_rag_pipeline, mock_connection_manager, mock_config_manager
):
    """Test service initializes and uses the primary pipeline successfully."""
    MockedBasicRAGPipeline.return_value = mock_successful_basic_rag_pipeline
    
    service = SurvivalModeRAGService(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
        # primary_pipeline will be set by the mocked constructor
    )
    assert service.primary_pipeline is mock_successful_basic_rag_pipeline
    assert service.is_primary_pipeline_healthy is True # Initial assumption after successful init

    response = service.query("test query")
    assert response["answer"] == "Primary answer"
    assert response["source"] == "PrimaryRAG"
    mock_successful_basic_rag_pipeline.query.assert_called_once_with("test query")


@patch('iris_rag.services.survival_mode.BasicRAGPipeline')
def test_service_initialization_failure_of_primary_pipeline(
    MockedBasicRAGPipeline, mock_connection_manager, mock_config_manager
):
    """Test service falls back if primary pipeline fails to initialize."""
    MockedBasicRAGPipeline.side_effect = Exception("Primary pipeline init failed")

    service = SurvivalModeRAGService(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )
    assert service.primary_pipeline is None
    assert service.is_primary_pipeline_healthy is False

    response = service.query("test query")
    assert response["source"] == "SurvivalModeFallback"
    assert "Primary RAG pipeline unavailable" in response["error"]
    assert "advanced information retrieval system is temporarily unavailable" in response["answer"]


def test_query_with_healthy_primary_pipeline(
    mock_successful_basic_rag_pipeline, mock_connection_manager, mock_config_manager
):
    """Test query uses primary pipeline when healthy."""
    service = SurvivalModeRAGService(
        primary_pipeline=mock_successful_basic_rag_pipeline,
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )
    # Manually set healthy state if constructor logic is complex
    service.is_primary_pipeline_healthy = True 
    
    response = service.query("another query")
    assert response["answer"] == "Primary answer"
    mock_successful_basic_rag_pipeline.query.assert_called_once_with("another query")


def test_query_fallback_when_primary_pipeline_fails_on_query(
    mock_failing_basic_rag_pipeline, mock_connection_manager, mock_config_manager
):
    """Test service falls back when a healthy pipeline fails during a query."""
    service = SurvivalModeRAGService(
        primary_pipeline=mock_failing_basic_rag_pipeline,
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )
    service.is_primary_pipeline_healthy = True # Start as healthy

    response = service.query("failing query")
    assert response["source"] == "SurvivalModeFallback"
    assert "Pipeline query failed" in response["error"]
    assert service.is_primary_pipeline_healthy is False # Should be marked unhealthy

    # Subsequent query should also use fallback
    mock_failing_basic_rag_pipeline.query.reset_mock() # Reset for the next call
    response_after_failure = service.query("query after fail")
    assert response_after_failure["source"] == "SurvivalModeFallback"
    mock_failing_basic_rag_pipeline.query.assert_not_called() # Should not be called again


def test_query_fallback_when_primary_pipeline_is_None(mock_connection_manager, mock_config_manager):
    """Test service falls back if primary_pipeline is None (e.g. init failed)."""
    # This scenario is partially covered by test_service_initialization_failure_of_primary_pipeline
    # Here, we explicitly set primary_pipeline to None after service init.
    service = SurvivalModeRAGService(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager,
        primary_pipeline=None # Explicitly None
    )
    service.is_primary_pipeline_healthy = False # Reflecting that it's None

    response = service.query("test query")
    assert response["source"] == "SurvivalModeFallback"
    assert "Primary RAG pipeline unavailable" in response.get("error", "")


@patch('iris_rag.services.survival_mode.BasicRAGPipeline')
def test_reinitialize_primary_pipeline_success(
    MockedBasicRAGPipeline, mock_successful_basic_rag_pipeline, mock_connection_manager, mock_config_manager
):
    """Test re-initializing the primary pipeline successfully."""
    # Start with a service where primary pipeline init failed initially
    MockedBasicRAGPipeline.side_effect = Exception("Initial init fail")
    service = SurvivalModeRAGService(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )
    assert service.primary_pipeline is None
    assert not service.is_primary_pipeline_healthy

    # Now, make the mock succeed for reinitialization
    MockedBasicRAGPipeline.reset_mock(side_effect=None) # Clear side_effect
    MockedBasicRAGPipeline.return_value = mock_successful_basic_rag_pipeline
    
    reinit_success = service.reinitialize_primary_pipeline()
    assert reinit_success is True
    assert service.primary_pipeline is mock_successful_basic_rag_pipeline
    assert service.is_primary_pipeline_healthy is True

    # Query should now use the re-initialized primary pipeline
    response = service.query("query after reinit")
    assert response["answer"] == "Primary answer"
    mock_successful_basic_rag_pipeline.query.assert_called_once_with("query after reinit")


@patch('iris_rag.services.survival_mode.BasicRAGPipeline')
def test_reinitialize_primary_pipeline_failure(
    MockedBasicRAGPipeline, mock_connection_manager, mock_config_manager
):
    """Test re-initializing the primary pipeline fails again."""
    # Start with a service where primary pipeline init failed initially
    MockedBasicRAGPipeline.side_effect = Exception("Initial init fail")
    service = SurvivalModeRAGService(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager
    )
    assert service.primary_pipeline is None
    assert not service.is_primary_pipeline_healthy

    # Mock reinitialization to fail again
    MockedBasicRAGPipeline.reset_mock(side_effect=Exception("Reinit fail"))
    
    reinit_success = service.reinitialize_primary_pipeline()
    assert reinit_success is False
    assert service.primary_pipeline is None
    assert service.is_primary_pipeline_healthy is False

    # Query should still use fallback
    response = service.query("query after failed reinit")
    assert response["source"] == "SurvivalModeFallback"


def test_fallback_query_structure(mock_connection_manager, mock_config_manager):
    """Test the structure of the fallback query response."""
    service = SurvivalModeRAGService(primary_pipeline=None, connection_manager=mock_connection_manager, config_manager=mock_config_manager)
    service.is_primary_pipeline_healthy = False
    
    response = service._fallback_query("test", original_error="Test error")
    assert "query" in response and response["query"] == "test"
    assert "answer" in response
    assert "retrieved_documents" in response and response["retrieved_documents"] == []
    assert "source" in response and response["source"] == "SurvivalModeFallback"
    assert "error" in response and response["error"] == "Test error"
    assert "status" in response and response["status"] == "degraded"


@patch.object(ConnectionManager, 'get_iris_connection')
def test_health_check_no_db_connection(mock_get_iris_connection, mock_successful_basic_rag_pipeline, mock_config_manager):
    """Test health check fails if DB connection cannot be established."""
    mock_get_iris_connection.return_value = None # Simulate DB connection failure
    
    # Need a real CM instance for this, not a full mock, to test its method
    cm_instance = ConnectionManager(config_manager=mock_config_manager)

    service = SurvivalModeRAGService(
        primary_pipeline=mock_successful_basic_rag_pipeline, # Provide a pipeline
        connection_manager=cm_instance, # Use the CM that will fail connection
        config_manager=mock_config_manager
    )
    # The _check_primary_pipeline_health is called internally by query if needed,
    # or we can call it directly for testing its logic.
    # For this test, let's assume the pipeline was initially "healthy" (instantiated)
    # but the runtime check of its dependencies (like DB conn) fails.
    
    # The current _check_primary_pipeline_health is simple.
    # Let's refine the test to reflect its current behavior or assume it's called by query.
    
    # If query calls _check_primary_pipeline_health, and it fails:
    mock_successful_basic_rag_pipeline.query.reset_mock() # Ensure it's not called if health check fails first
    
    # To make _check_primary_pipeline_health fail due to DB, we need to ensure it's called.
    # The current query logic calls it if is_primary_pipeline_healthy is True.
    service.is_primary_pipeline_healthy = True # Assume it was healthy
    
    # The health check is currently very basic in the provided code.
    # It doesn't explicitly re-check the DB connection *within* _check_primary_pipeline_health
    # if the pipeline object itself exists.
    # The example in the code for _check_primary_pipeline_health is:
    # if self.connection_manager.get_iris_connection() is None:
    # This part needs to be triggered.

    # Let's directly test _check_primary_pipeline_health
    is_healthy = service._check_primary_pipeline_health()
    assert not is_healthy
    assert not service.is_primary_pipeline_healthy # State should be updated

    # Now, a query should use fallback
    response = service.query("query with no db")
    assert response["source"] == "SurvivalModeFallback"
    mock_successful_basic_rag_pipeline.query.assert_not_called()


# To run these tests: pytest tests/test_integration/test_survival_mode_service.py