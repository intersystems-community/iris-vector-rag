"""
Tests for pipeline configuration loading robustness to current working directory changes.

This module tests that pipeline configuration loading works correctly regardless
of the current working directory when scripts are run from different locations.
"""

import os
import tempfile
import pytest
from pathlib import Path

from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.core.exceptions import PipelineConfigurationError


class TestPipelineConfigCWDRobustness:
    """Test pipeline configuration loading robustness to CWD changes."""

    def test_load_config_from_project_root(self):
        """Test loading config when running from project root."""
        service = PipelineConfigService()
        
        # This should work regardless of current directory
        definitions = service.load_pipeline_definitions("config/pipelines.yaml")
        
        assert isinstance(definitions, list)
        assert len(definitions) > 0
        
        # Verify we got actual pipeline definitions
        first_pipeline = definitions[0]
        assert 'name' in first_pipeline
        assert 'module' in first_pipeline
        assert 'class' in first_pipeline

    def test_load_config_from_subdirectory(self):
        """Test loading config when running from a subdirectory (simulating eval/ scenario)."""
        service = PipelineConfigService()
        
        # Change to a subdirectory to simulate running from eval/
        original_cwd = os.getcwd()
        try:
            # Create a temporary subdirectory and change to it
            with tempfile.TemporaryDirectory() as temp_dir:
                subdir = Path(temp_dir) / "subdir"
                subdir.mkdir()
                os.chdir(str(subdir))
                
                # This should still work because we resolve relative to project root
                definitions = service.load_pipeline_definitions("config/pipelines.yaml")
                
                assert isinstance(definitions, list)
                assert len(definitions) > 0
                
        finally:
            os.chdir(original_cwd)

    def test_load_config_with_absolute_path(self):
        """Test that absolute paths still work."""
        service = PipelineConfigService()
        
        # Get the absolute path to the config file
        from iris_rag.utils.project_root import get_project_root
        project_root = get_project_root()
        absolute_config_path = str(project_root / "config" / "pipelines.yaml")
        
        # This should work with absolute paths too
        definitions = service.load_pipeline_definitions(absolute_config_path)
        
        assert isinstance(definitions, list)
        assert len(definitions) > 0

    def test_nonexistent_config_file_error(self):
        """Test that appropriate error is raised for nonexistent config files."""
        service = PipelineConfigService()
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.load_pipeline_definitions("config/nonexistent.yaml")
        
        assert "Configuration file not found" in str(exc_info.value)
        assert "nonexistent.yaml" in str(exc_info.value)

    def test_project_root_detection_robustness(self):
        """Test that project root detection works from various directories."""
        from iris_rag.utils.project_root import get_project_root
        
        original_cwd = os.getcwd()
        try:
            # Test from project root
            project_root = get_project_root()
            assert project_root.exists()
            assert (project_root / "iris_rag").exists()
            assert (project_root / "config").exists()
            
            # Test from a subdirectory
            eval_dir = project_root / "eval"
            if eval_dir.exists():
                os.chdir(str(eval_dir))
                root_from_subdir = get_project_root()
                assert root_from_subdir == project_root
                
        finally:
            os.chdir(original_cwd)

    def test_resolve_project_relative_path(self):
        """Test the resolve_project_relative_path utility function."""
        from iris_rag.utils.project_root import resolve_project_relative_path
        
        # Test resolving a known path
        config_path = resolve_project_relative_path("config/pipelines.yaml")
        assert config_path.exists()
        assert config_path.name == "pipelines.yaml"
        assert config_path.parent.name == "config"

    def test_pipeline_factory_integration(self):
        """Test that PipelineFactory works with the robust configuration loading."""
        from iris_rag.config.pipeline_config_service import PipelineConfigService
        from iris_rag.utils.module_loader import ModuleLoader
        from iris_rag.pipelines.factory import PipelineFactory
        
        # Create minimal dependencies for testing
        config_service = PipelineConfigService()
        module_loader = ModuleLoader()
        framework_dependencies = {
            'connection_manager': None,
            'config_manager': None
        }
        
        # This should work regardless of CWD
        factory = PipelineFactory(config_service, module_loader, framework_dependencies)
        
        # The factory should be able to load pipeline definitions
        # (We can't test actual pipeline creation without full setup)
        assert factory is not None
        assert factory.config_service is not None