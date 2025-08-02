"""
Tests for the Pipeline Registry Service.

This module contains unit tests for the PipelineRegistry class,
following TDD principles to ensure robust pipeline management.
"""

import pytest
from unittest.mock import Mock

from iris_rag.pipelines.registry import PipelineRegistry
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.core.base import RAGPipeline


class MockPipeline(RAGPipeline):
    """Mock pipeline class for testing that doesn't call super().__init__."""
    
    def __init__(self, connection_manager, config_manager, llm_func=None, **kwargs):
        # Don't call super().__init__ to avoid IRISVectorStore creation
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.llm_func = llm_func
        self.init_kwargs = kwargs
        self.vector_store = None  # Set to None to avoid issues
    
    def execute(self, query_text: str, **kwargs) -> dict:
        return {"query": query_text, "answer": "mock answer", "retrieved_documents": []}
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        pass
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> list:
        return []


class TestPipelineRegistry:
    """Test cases for PipelineRegistry."""

    @pytest.fixture
    def mock_pipeline_factory(self) -> Mock:
        """Create a mock PipelineFactory."""
        mock_factory = Mock(spec=PipelineFactory)
        
        # Configure factory to return mock pipelines
        mock_pipelines = {
            'TestRAG': MockPipeline(Mock(), Mock()),
            'AnotherRAG': MockPipeline(Mock(), Mock())
        }
        mock_factory.create_all_pipelines.return_value = mock_pipelines
        
        # Configure individual pipeline creation
        def create_pipeline_side_effect(name):
            if name in mock_pipelines:
                return mock_pipelines[name]
            return None
        
        mock_factory.create_pipeline.side_effect = create_pipeline_side_effect
        
        return mock_factory

    @pytest.fixture
    def pipeline_registry(self, mock_pipeline_factory) -> PipelineRegistry:
        """Create a PipelineRegistry instance for testing."""
        return PipelineRegistry(pipeline_factory=mock_pipeline_factory)

    def test_init_creates_registry_instance(self, mock_pipeline_factory):
        """Test that PipelineRegistry can be instantiated."""
        registry = PipelineRegistry(pipeline_factory=mock_pipeline_factory)
        assert registry is not None

    def test_register_pipelines_calls_factory(self, pipeline_registry: PipelineRegistry, mock_pipeline_factory):
        """Test that register_pipelines calls the factory to create all pipelines."""
        pipeline_registry.register_pipelines()
        
        # Verify factory was called
        mock_pipeline_factory.create_all_pipelines.assert_called_once()

    def test_register_pipelines_stores_pipelines(self, pipeline_registry: PipelineRegistry):
        """Test that register_pipelines stores the created pipelines."""
        pipeline_registry.register_pipelines()
        
        # Check that pipelines are stored
        assert len(pipeline_registry._pipelines) == 2
        assert 'TestRAG' in pipeline_registry._pipelines
        assert 'AnotherRAG' in pipeline_registry._pipelines

    def test_get_pipeline_returns_registered_pipeline(self, pipeline_registry: PipelineRegistry):
        """Test getting a registered pipeline by name."""
        pipeline_registry.register_pipelines()
        
        pipeline = pipeline_registry.get_pipeline('TestRAG')
        
        assert pipeline is not None
        assert isinstance(pipeline, MockPipeline)

    def test_get_pipeline_returns_none_for_unregistered(self, pipeline_registry: PipelineRegistry):
        """Test getting an unregistered pipeline returns None."""
        pipeline_registry.register_pipelines()
        
        pipeline = pipeline_registry.get_pipeline('NonExistentRAG')
        
        assert pipeline is None

    def test_get_pipeline_before_registration_returns_none(self, pipeline_registry: PipelineRegistry):
        """Test getting a pipeline before registration returns None."""
        pipeline = pipeline_registry.get_pipeline('TestRAG')
        
        assert pipeline is None

    def test_list_pipeline_names_returns_registered_names(self, pipeline_registry: PipelineRegistry):
        """Test listing pipeline names returns all registered pipeline names."""
        pipeline_registry.register_pipelines()
        
        names = pipeline_registry.list_pipeline_names()
        
        assert isinstance(names, list)
        assert len(names) == 2
        assert 'TestRAG' in names
        assert 'AnotherRAG' in names

    def test_list_pipeline_names_before_registration_returns_empty(self, pipeline_registry: PipelineRegistry):
        """Test listing pipeline names before registration returns empty list."""
        names = pipeline_registry.list_pipeline_names()
        
        assert isinstance(names, list)
        assert len(names) == 0

    def test_is_pipeline_registered_returns_true_for_registered(self, pipeline_registry: PipelineRegistry):
        """Test checking if a registered pipeline is registered returns True."""
        pipeline_registry.register_pipelines()
        
        assert pipeline_registry.is_pipeline_registered('TestRAG') is True
        assert pipeline_registry.is_pipeline_registered('AnotherRAG') is True

    def test_is_pipeline_registered_returns_false_for_unregistered(self, pipeline_registry: PipelineRegistry):
        """Test checking if an unregistered pipeline is registered returns False."""
        pipeline_registry.register_pipelines()
        
        assert pipeline_registry.is_pipeline_registered('NonExistentRAG') is False

    def test_is_pipeline_registered_before_registration_returns_false(self, pipeline_registry: PipelineRegistry):
        """Test checking if a pipeline is registered before registration returns False."""
        assert pipeline_registry.is_pipeline_registered('TestRAG') is False

    def test_register_pipelines_handles_empty_factory_result(self, mock_pipeline_factory):
        """Test that register_pipelines handles when factory returns no pipelines."""
        mock_pipeline_factory.create_all_pipelines.return_value = {}
        
        registry = PipelineRegistry(pipeline_factory=mock_pipeline_factory)
        registry.register_pipelines()
        
        assert len(registry._pipelines) == 0
        assert registry.list_pipeline_names() == []

    def test_register_pipelines_can_be_called_multiple_times(self, pipeline_registry: PipelineRegistry, mock_pipeline_factory):
        """Test that register_pipelines can be called multiple times safely."""
        # First registration
        pipeline_registry.register_pipelines()
        first_count = len(pipeline_registry._pipelines)
        
        # Second registration should replace pipelines
        pipeline_registry.register_pipelines()
        second_count = len(pipeline_registry._pipelines)
        
        assert first_count == second_count
        assert mock_pipeline_factory.create_all_pipelines.call_count == 2

    def test_registry_maintains_pipeline_references(self, pipeline_registry: PipelineRegistry):
        """Test that registry maintains proper references to pipeline instances."""
        pipeline_registry.register_pipelines()
        
        # Get the same pipeline twice
        pipeline1 = pipeline_registry.get_pipeline('TestRAG')
        pipeline2 = pipeline_registry.get_pipeline('TestRAG')
        
        # Should be the same instance
        assert pipeline1 is pipeline2

    def test_register_pipelines_with_factory_exception(self, mock_pipeline_factory):
        """Test handling when factory raises an exception."""
        mock_pipeline_factory.create_all_pipelines.side_effect = Exception("Factory error")
        
        registry = PipelineRegistry(pipeline_factory=mock_pipeline_factory)
        
        # Should not raise exception, but handle gracefully
        registry.register_pipelines()
        
        # Registry should be empty
        assert len(registry._pipelines) == 0
        assert registry.list_pipeline_names() == []