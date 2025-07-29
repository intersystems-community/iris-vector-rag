"""
Tests for the Module Loader Service.

This module contains unit tests for the ModuleLoader class,
following TDD principles to ensure robust dynamic module loading.
"""

import pytest
from unittest.mock import Mock, patch

from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.core.exceptions import ModuleLoadingError
from iris_rag.core.base import RAGPipeline


class TestModuleLoader:
    """Test cases for ModuleLoader."""

    @pytest.fixture
    def module_loader(self) -> ModuleLoader:
        """Create a ModuleLoader instance for testing."""
        return ModuleLoader()

    def test_init_creates_loader_instance(self):
        """Test that ModuleLoader can be instantiated."""
        loader = ModuleLoader()
        assert loader is not None

    def test_load_pipeline_class_with_valid_module_and_class(self, module_loader: ModuleLoader):
        """Test loading a valid pipeline class from an existing module."""
        # Use an actual existing pipeline class for this test
        pipeline_class = module_loader.load_pipeline_class(
            "iris_rag.pipelines.basic", 
            "BasicRAGPipeline"
        )
        
        assert pipeline_class is not None
        assert hasattr(pipeline_class, '__name__')
        assert pipeline_class.__name__ == 'BasicRAGPipeline'
        # Verify it's a subclass of RAGPipeline
        assert issubclass(pipeline_class, RAGPipeline)

    def test_load_pipeline_class_with_nonexistent_module(self, module_loader: ModuleLoader):
        """Test loading from a non-existent module raises ModuleLoadingError."""
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class(
                "nonexistent.module.path", 
                "SomeClass"
            )
        
        assert "Failed to import module" in str(exc_info.value)
        assert "nonexistent.module.path" in str(exc_info.value)

    def test_load_pipeline_class_with_nonexistent_class(self, module_loader: ModuleLoader):
        """Test loading a non-existent class from valid module raises ModuleLoadingError."""
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class(
                "iris_rag.pipelines.basic", 
                "NonExistentClass"
            )
        
        assert "Class 'NonExistentClass' not found" in str(exc_info.value)

    def test_load_pipeline_class_with_invalid_class_type(self, module_loader: ModuleLoader):
        """Test loading a class that is not a RAGPipeline subclass raises ModuleLoadingError."""
        # Try to load a class that exists but is not a RAGPipeline
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class(
                "builtins", 
                "dict"  # dict is not a RAGPipeline subclass
            )
        
        assert "is not a subclass of RAGPipeline" in str(exc_info.value)

    @patch('iris_rag.utils.module_loader.importlib.import_module')
    def test_load_pipeline_class_with_import_error(self, mock_import, module_loader: ModuleLoader):
        """Test handling of import errors during module loading."""
        mock_import.side_effect = ImportError("Mocked import error")
        
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class("some.module", "SomeClass")
        
        assert "Failed to import module" in str(exc_info.value)
        assert "Mocked import error" in str(exc_info.value)

    def test_load_pipeline_class_caching_behavior(self, module_loader: ModuleLoader):
        """Test that modules are cached to avoid repeated imports."""
        # Load the same class twice
        class1 = module_loader.load_pipeline_class(
            "iris_rag.pipelines.basic", 
            "BasicRAGPipeline"
        )
        class2 = module_loader.load_pipeline_class(
            "iris_rag.pipelines.basic", 
            "BasicRAGPipeline"
        )
        
        # Should be the same class object (cached)
        assert class1 is class2

    def test_load_pipeline_class_with_empty_module_path(self, module_loader: ModuleLoader):
        """Test loading with empty module path raises ModuleLoadingError."""
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class("", "SomeClass")
        
        assert "Module path cannot be empty" in str(exc_info.value)

    def test_load_pipeline_class_with_empty_class_name(self, module_loader: ModuleLoader):
        """Test loading with empty class name raises ModuleLoadingError."""
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class("some.module", "")
        
        assert "Class name cannot be empty" in str(exc_info.value)

    def test_load_pipeline_class_with_none_values(self, module_loader: ModuleLoader):
        """Test loading with None values raises ModuleLoadingError."""
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class(None, "SomeClass")
        
        assert "Module path cannot be None" in str(exc_info.value)

        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class("some.module", None)
        
        assert "Class name cannot be None" in str(exc_info.value)

    @patch('iris_rag.utils.module_loader.importlib.import_module')
    def test_load_pipeline_class_with_attribute_error(self, mock_import, module_loader: ModuleLoader):
        """Test handling when getattr raises AttributeError."""
        mock_module = Mock()
        mock_import.return_value = mock_module
        
        # Configure mock to raise AttributeError when accessing the class
        del mock_module.NonExistentClass  # Ensure attribute doesn't exist
        
        with pytest.raises(ModuleLoadingError) as exc_info:
            module_loader.load_pipeline_class("some.module", "NonExistentClass")
        
        assert "Class 'NonExistentClass' not found" in str(exc_info.value)

    def test_multiple_different_classes_from_same_module(self, module_loader: ModuleLoader):
        """Test loading multiple different classes from the same module."""
        # This test assumes there are multiple pipeline classes in the basic module
        # If not, we can use different modules
        class1 = module_loader.load_pipeline_class(
            "iris_rag.pipelines.basic", 
            "BasicRAGPipeline"
        )
        
        # Try to load from a different module
        class2 = module_loader.load_pipeline_class(
            "iris_rag.pipelines.hyde", 
            "HyDERAGPipeline"
        )
        
        assert class1 is not class2
        assert class1.__name__ == 'BasicRAGPipeline'
        assert class2.__name__ == 'HyDERAGPipeline'