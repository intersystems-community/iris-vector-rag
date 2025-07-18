"""
Tests for the Pipeline Configuration Service.

This module contains unit tests for the PipelineConfigService class,
following TDD principles to ensure robust configuration handling.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any

from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.core.exceptions import PipelineConfigurationError


class TestPipelineConfigService:
    """Test cases for PipelineConfigService."""

    @pytest.fixture
    def valid_config_content(self) -> str:
        """Valid YAML configuration content for testing."""
        return """
pipelines:
  - name: "TestRAG"
    module: "test.module"
    class: "TestPipeline"
    enabled: true
    params:
      top_k: 5
      temperature: 0.1
  - name: "DisabledRAG"
    module: "test.module"
    class: "DisabledPipeline"
    enabled: false
    params:
      top_k: 3

framework:
  llm:
    model: "gpt-4o-mini"
    temperature: 0
"""

    @pytest.fixture
    def invalid_config_content(self) -> str:
        """Invalid YAML configuration content for testing."""
        return """
pipelines:
  - name: "InvalidRAG"
    # Missing required fields: module, class
    enabled: true
    params:
      top_k: 5
"""

    @pytest.fixture
    def malformed_yaml_content(self) -> str:
        """Malformed YAML content for testing."""
        return """
pipelines:
  - name: "TestRAG"
    module: "test.module"
    class: "TestPipeline"
    enabled: true
    params:
      top_k: 5
      invalid_yaml: [unclosed list
"""

    @pytest.fixture
    def temp_config_file(self, valid_config_content: str) -> str:
        """Create a temporary config file with valid content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_config_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_invalid_config_file(self, invalid_config_content: str) -> str:
        """Create a temporary config file with invalid content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_config_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_malformed_config_file(self, malformed_yaml_content: str) -> str:
        """Create a temporary config file with malformed YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_init_creates_service_instance(self):
        """Test that PipelineConfigService can be instantiated."""
        service = PipelineConfigService()
        assert service is not None

    def test_load_pipeline_definitions_with_valid_file(self, temp_config_file: str):
        """Test loading pipeline definitions from a valid YAML file."""
        service = PipelineConfigService()
        definitions = service.load_pipeline_definitions(temp_config_file)
        
        assert isinstance(definitions, list)
        assert len(definitions) == 2
        
        # Check first pipeline
        first_pipeline = definitions[0]
        assert first_pipeline['name'] == 'TestRAG'
        assert first_pipeline['module'] == 'test.module'
        assert first_pipeline['class'] == 'TestPipeline'
        assert first_pipeline['enabled'] is True
        assert first_pipeline['params']['top_k'] == 5

    def test_load_pipeline_definitions_with_nonexistent_file(self):
        """Test loading pipeline definitions from a non-existent file."""
        service = PipelineConfigService()
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.load_pipeline_definitions("/nonexistent/path/config.yaml")
        
        assert "Configuration file not found" in str(exc_info.value)

    def test_load_pipeline_definitions_with_malformed_yaml(self, temp_malformed_config_file: str):
        """Test loading pipeline definitions from malformed YAML."""
        service = PipelineConfigService()
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.load_pipeline_definitions(temp_malformed_config_file)
        
        assert "Failed to parse YAML" in str(exc_info.value)

    def test_validate_pipeline_definition_with_valid_definition(self):
        """Test validation of a valid pipeline definition."""
        service = PipelineConfigService()
        valid_definition = {
            'name': 'TestRAG',
            'module': 'test.module',
            'class': 'TestPipeline',
            'enabled': True,
            'params': {'top_k': 5}
        }
        
        # Should not raise an exception
        result = service.validate_pipeline_definition(valid_definition)
        assert result is True

    def test_validate_pipeline_definition_missing_name(self):
        """Test validation fails when name is missing."""
        service = PipelineConfigService()
        invalid_definition = {
            'module': 'test.module',
            'class': 'TestPipeline',
            'enabled': True
        }
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.validate_pipeline_definition(invalid_definition)
        
        assert "Missing required field: name" in str(exc_info.value)

    def test_validate_pipeline_definition_missing_module(self):
        """Test validation fails when module is missing."""
        service = PipelineConfigService()
        invalid_definition = {
            'name': 'TestRAG',
            'class': 'TestPipeline',
            'enabled': True
        }
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.validate_pipeline_definition(invalid_definition)
        
        assert "Missing required field: module" in str(exc_info.value)

    def test_validate_pipeline_definition_missing_class(self):
        """Test validation fails when class is missing."""
        service = PipelineConfigService()
        invalid_definition = {
            'name': 'TestRAG',
            'module': 'test.module',
            'enabled': True
        }
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.validate_pipeline_definition(invalid_definition)
        
        assert "Missing required field: class" in str(exc_info.value)

    def test_validate_pipeline_definition_invalid_enabled_type(self):
        """Test validation fails when enabled field has wrong type."""
        service = PipelineConfigService()
        invalid_definition = {
            'name': 'TestRAG',
            'module': 'test.module',
            'class': 'TestPipeline',
            'enabled': 'yes'  # Should be boolean
        }
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.validate_pipeline_definition(invalid_definition)
        
        assert "Field 'enabled' must be a boolean" in str(exc_info.value)

    def test_validate_pipeline_definition_invalid_params_type(self):
        """Test validation fails when params field has wrong type."""
        service = PipelineConfigService()
        invalid_definition = {
            'name': 'TestRAG',
            'module': 'test.module',
            'class': 'TestPipeline',
            'enabled': True,
            'params': 'invalid'  # Should be dict
        }
        
        with pytest.raises(PipelineConfigurationError) as exc_info:
            service.validate_pipeline_definition(invalid_definition)
        
        assert "Field 'params' must be a dictionary" in str(exc_info.value)

    def test_load_and_validate_integration(self, temp_invalid_config_file: str):
        """Test integration of loading and validation with invalid config."""
        service = PipelineConfigService()
        
        with pytest.raises(PipelineConfigurationError):
            service.load_pipeline_definitions(temp_invalid_config_file)