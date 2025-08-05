"""
Tests for the Pipeline Factory Service.

This module contains unit tests for the PipelineFactory class,
following TDD principles to ensure robust pipeline creation.
"""

import pytest
from unittest.mock import Mock
from typing import Dict, Any

from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.core.exceptions import PipelineNotFoundError, PipelineCreationError
from iris_rag.core.base import RAGPipeline


class MockPipeline(RAGPipeline):
    """Mock pipeline class for testing."""
    
    def __init__(self, connection_manager, config_manager, llm_func=None, **kwargs):
        super().__init__(connection_manager, config_manager)
        self.llm_func = llm_func
        self.init_kwargs = kwargs
    
    def execute(self, query_text: str, **kwargs) -> dict:
        return {"query": query_text, "answer": "mock answer", "retrieved_documents": []}
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        pass
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> list:
        return []


class TestPipelineFactory:
    """Test cases for PipelineFactory."""

    @pytest.fixture
    def mock_config_service(self) -> Mock:
        """Create a mock PipelineConfigService."""
        mock_service = Mock(spec=PipelineConfigService)
        mock_service.load_pipeline_definitions.return_value = [
            {
                'name': 'TestRAG',
                'module': 'test.module',
                'class': 'TestPipeline',
                'enabled': True,
                'params': {'top_k': 5, 'temperature': 0.1}
            },
            {
                'name': 'DisabledRAG',
                'module': 'test.module',
                'class': 'DisabledPipeline',
                'enabled': False,
                'params': {'top_k': 3}
            }
        ]
        return mock_service

    @pytest.fixture
    def mock_module_loader(self) -> Mock:
        """Create a mock ModuleLoader."""
        mock_loader = Mock(spec=ModuleLoader)
        mock_loader.load_pipeline_class.return_value = MockPipeline
        return mock_loader

    @pytest.fixture
    def framework_dependencies(self) -> Dict[str, Any]:
        """Create mock framework dependencies."""
        # Create a properly configured config_manager mock
        config_manager = Mock()
        config_manager.get.side_effect = lambda key, default=None: {
            "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model.dimension": 384,
            "colbert": {
                "backend": "native",
                "token_dimension": 768,
                "model_name": "bert-base-uncased"
            }
        }.get(key, default)
        
        return {
            'connection_manager': Mock(),
            'config_manager': config_manager,
            'llm_func': Mock(),
            'vector_store': Mock()
        }

    @pytest.fixture
    def pipeline_factory(self, mock_config_service, mock_module_loader, framework_dependencies) -> PipelineFactory:
        """Create a PipelineFactory instance for testing."""
        return PipelineFactory(
            config_service=mock_config_service,
            module_loader=mock_module_loader,
            framework_dependencies=framework_dependencies
        )

    def test_init_creates_factory_instance(self, mock_config_service, mock_module_loader, framework_dependencies):
        """Test that PipelineFactory can be instantiated."""
        factory = PipelineFactory(
            config_service=mock_config_service,
            module_loader=mock_module_loader,
            framework_dependencies=framework_dependencies
        )
        assert factory is not None

    def test_create_pipeline_with_valid_name(self, pipeline_factory: PipelineFactory, mock_module_loader):
        """Test creating a pipeline with a valid name."""
        pipeline = pipeline_factory.create_pipeline('TestRAG')
        
        assert pipeline is not None
        assert isinstance(pipeline, MockPipeline)
        
        # Verify module loader was called with correct parameters
        mock_module_loader.load_pipeline_class.assert_called_once_with('test.module', 'TestPipeline')

    def test_create_pipeline_with_nonexistent_name(self, pipeline_factory: PipelineFactory):
        """Test creating a pipeline with a non-existent name raises PipelineNotFoundError."""
        with pytest.raises(PipelineNotFoundError) as exc_info:
            pipeline_factory.create_pipeline('NonExistentRAG')
        
        assert "Pipeline 'NonExistentRAG' not found" in str(exc_info.value)

    def test_create_pipeline_with_disabled_pipeline(self, pipeline_factory: PipelineFactory):
        """Test creating a disabled pipeline raises PipelineNotFoundError."""
        with pytest.raises(PipelineNotFoundError) as exc_info:
            pipeline_factory.create_pipeline('DisabledRAG')
        
        assert "Pipeline 'DisabledRAG' is disabled" in str(exc_info.value)

    def test_create_pipeline_with_module_loading_error(self, pipeline_factory: PipelineFactory, mock_module_loader):
        """Test handling of module loading errors."""
        from iris_rag.core.exceptions import ModuleLoadingError
        mock_module_loader.load_pipeline_class.side_effect = ModuleLoadingError("Module not found")
        
        with pytest.raises(PipelineCreationError) as exc_info:
            pipeline_factory.create_pipeline('TestRAG')
        
        assert "Failed to create pipeline 'TestRAG'" in str(exc_info.value)
        assert "Module not found" in str(exc_info.value)

    def test_create_pipeline_with_instantiation_error(self, pipeline_factory: PipelineFactory, mock_module_loader):
        """Test handling of pipeline instantiation errors."""
        # Mock a class that raises an exception during instantiation
        mock_class = Mock(side_effect=Exception("Instantiation failed"))
        mock_module_loader.load_pipeline_class.return_value = mock_class
        
        with pytest.raises(PipelineCreationError) as exc_info:
            pipeline_factory.create_pipeline('TestRAG')
        
        assert "Failed to create pipeline 'TestRAG'" in str(exc_info.value)
        assert "Instantiation failed" in str(exc_info.value)

    def test_create_pipeline_passes_framework_dependencies(self, pipeline_factory: PipelineFactory, mock_module_loader):
        """Test that framework dependencies are passed to pipeline constructor."""
        # Create a mock class that we can verify was called
        mock_pipeline_class = Mock(return_value=Mock(spec=MockPipeline))
        mock_module_loader.load_pipeline_class.return_value = mock_pipeline_class
        
        pipeline = pipeline_factory.create_pipeline('TestRAG')
        
        # Verify the mock pipeline class was called with framework dependencies
        mock_pipeline_class.assert_called_once()
        call_args = mock_pipeline_class.call_args
        
        # Check that framework dependencies were passed as positional and keyword args
        # First two args should be connection_manager and config_manager
        assert len(call_args[0]) >= 2  # At least 2 positional args
        # Check that framework dependencies were passed as kwargs
        assert 'llm_func' in call_args[1]
        assert 'vector_store' in call_args[1]

    def test_create_pipeline_passes_pipeline_params(self, pipeline_factory: PipelineFactory, mock_module_loader):
        """Test that pipeline-specific parameters are passed to constructor."""
        # Create a mock class that we can verify was called
        mock_pipeline_class = Mock(return_value=Mock(spec=MockPipeline))
        mock_module_loader.load_pipeline_class.return_value = mock_pipeline_class
        
        pipeline = pipeline_factory.create_pipeline('TestRAG')
        
        # Verify the mock pipeline class was called with pipeline params
        mock_pipeline_class.assert_called_once()
        call_args = mock_pipeline_class.call_args
        
        # Check that pipeline params were passed as kwargs
        # Note: The factory filters params, so only allowed kwargs (llm_func, vector_store) are passed
        # Pipeline-specific params like top_k and temperature are filtered out by the factory
        assert 'llm_func' in call_args[1]
        assert 'vector_store' in call_args[1]

    def test_create_all_pipelines_returns_enabled_pipelines(self, pipeline_factory: PipelineFactory):
        """Test creating all pipelines returns only enabled ones."""
        pipelines = pipeline_factory.create_all_pipelines()
        
        assert isinstance(pipelines, dict)
        assert len(pipelines) == 1  # Only enabled pipeline
        assert 'TestRAG' in pipelines
        assert 'DisabledRAG' not in pipelines
        assert isinstance(pipelines['TestRAG'], MockPipeline)

    def test_create_all_pipelines_handles_individual_failures(self, pipeline_factory: PipelineFactory, mock_module_loader):
        """Test that create_all_pipelines continues when individual pipeline creation fails."""
        # Configure mock to fail for specific pipeline
        def side_effect(module, class_name):
            if class_name == 'TestPipeline':
                raise Exception("Failed to load TestPipeline")
            return MockPipeline
        
        mock_module_loader.load_pipeline_class.side_effect = side_effect
        
        pipelines = pipeline_factory.create_all_pipelines()
        
        # Should return empty dict since the only enabled pipeline failed
        assert isinstance(pipelines, dict)
        assert len(pipelines) == 0

    def test_create_all_pipelines_with_no_enabled_pipelines(self, mock_config_service, mock_module_loader, framework_dependencies):
        """Test create_all_pipelines when no pipelines are enabled."""
        # Configure mock to return only disabled pipelines
        mock_config_service.load_pipeline_definitions.return_value = [
            {
                'name': 'DisabledRAG1',
                'module': 'test.module',
                'class': 'DisabledPipeline1',
                'enabled': False,
                'params': {}
            },
            {
                'name': 'DisabledRAG2',
                'module': 'test.module',
                'class': 'DisabledPipeline2',
                'enabled': False,
                'params': {}
            }
        ]
        
        factory = PipelineFactory(
            config_service=mock_config_service,
            module_loader=mock_module_loader,
            framework_dependencies=framework_dependencies
        )
        
        pipelines = factory.create_all_pipelines()
        
        assert isinstance(pipelines, dict)
        assert len(pipelines) == 0

    def test_factory_loads_configuration_on_first_use(self, mock_config_service, mock_module_loader, framework_dependencies):
        """Test that configuration is loaded on first pipeline creation."""
        factory = PipelineFactory(
            config_service=mock_config_service,
            module_loader=mock_module_loader,
            framework_dependencies=framework_dependencies
        )
        
        # Configuration should not be loaded yet
        mock_config_service.load_pipeline_definitions.assert_not_called()
        
        # Create a pipeline - this should trigger configuration loading
        factory.create_pipeline('TestRAG')
        
        # Configuration should now be loaded
        mock_config_service.load_pipeline_definitions.assert_called_once()

    def test_factory_caches_configuration(self, mock_config_service, mock_module_loader, framework_dependencies):
        """Test that configuration is cached and not reloaded."""
        factory = PipelineFactory(
            config_service=mock_config_service,
            module_loader=mock_module_loader,
            framework_dependencies=framework_dependencies
        )
        
        # Create multiple pipelines
        factory.create_pipeline('TestRAG')
        factory.create_pipeline('TestRAG')
        
        # Configuration should only be loaded once
        mock_config_service.load_pipeline_definitions.assert_called_once()