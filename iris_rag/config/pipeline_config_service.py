"""
Pipeline Configuration Service.

This module provides the PipelineConfigService class for loading and validating
pipeline configurations from YAML files.
"""

import logging
import yaml
from typing import Dict, List

from ..core.exceptions import PipelineConfigurationError
from ..utils.project_root import resolve_project_relative_path


class PipelineConfigService:
    """
    Service for loading and validating pipeline configurations.
    
    This service handles:
    - Loading pipeline definitions from YAML configuration files
    - Validating pipeline configuration schema
    - Providing structured access to pipeline definitions
    """

    def __init__(self):
        """Initialize the pipeline configuration service."""
        self.logger = logging.getLogger(__name__)

    def load_pipeline_definitions(self, config_file_path: str) -> List[Dict]:
        """
        Load pipeline definitions from a YAML configuration file.
        
        Args:
            config_file_path: Path to the YAML configuration file (relative to project root)
            
        Returns:
            List of pipeline definition dictionaries
            
        Raises:
            PipelineConfigurationError: If file cannot be loaded or parsed
        """
        try:
            # Resolve path relative to project root, making it robust to cwd changes
            config_path = resolve_project_relative_path(config_file_path)
            self.logger.debug(f"Resolved config path: {config_path}")
        except Exception as e:
            raise PipelineConfigurationError(
                f"Failed to resolve configuration path '{config_file_path}': {str(e)}"
            )
        
        # Check if file exists
        if not config_path.exists():
            raise PipelineConfigurationError(
                f"Configuration file not found: {config_path} "
                f"(resolved from '{config_file_path}')"
            )
        
        try:
            # Load and parse YAML
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                
            if not config_data:
                raise PipelineConfigurationError("Configuration file is empty")
                
            # Extract pipeline definitions
            pipelines = config_data.get('pipelines', [])
            if not isinstance(pipelines, list):
                raise PipelineConfigurationError(
                    "Configuration must contain a 'pipelines' list"
                )
            
            # Validate each pipeline definition
            validated_pipelines = []
            for pipeline_def in pipelines:
                if self.validate_pipeline_definition(pipeline_def):
                    validated_pipelines.append(pipeline_def)
            
            self.logger.info(f"Loaded {len(validated_pipelines)} pipeline definitions")
            return validated_pipelines
            
        except yaml.YAMLError as e:
            raise PipelineConfigurationError(f"Failed to parse YAML: {str(e)}")
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to load configuration: {str(e)}")

    def validate_pipeline_definition(self, definition: Dict) -> bool:
        """
        Validate a single pipeline definition.
        
        Args:
            definition: Pipeline definition dictionary to validate
            
        Returns:
            True if validation passes
            
        Raises:
            PipelineConfigurationError: If validation fails
        """
        if not isinstance(definition, dict):
            raise PipelineConfigurationError("Pipeline definition must be a dictionary")
        
        # Check required fields
        required_fields = ['name', 'module', 'class']
        for field in required_fields:
            if field not in definition:
                raise PipelineConfigurationError(f"Missing required field: {field}")
            if not isinstance(definition[field], str):
                raise PipelineConfigurationError(f"Field '{field}' must be a string")
        
        # Check optional fields with type validation
        if 'enabled' in definition:
            if not isinstance(definition['enabled'], bool):
                raise PipelineConfigurationError("Field 'enabled' must be a boolean")
        
        if 'params' in definition:
            if not isinstance(definition['params'], dict):
                raise PipelineConfigurationError("Field 'params' must be a dictionary")
        
        # Set defaults for optional fields
        if 'enabled' not in definition:
            definition['enabled'] = True
        if 'params' not in definition:
            definition['params'] = {}
        
        self.logger.debug(f"Validated pipeline definition: {definition['name']}")
        return True