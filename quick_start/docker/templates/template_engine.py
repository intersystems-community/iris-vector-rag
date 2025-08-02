"""
Docker template engine for generating docker-compose configurations.

This module provides the DockerTemplateEngine class that loads and processes
Docker-compose templates with variable substitution and inheritance.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DockerTemplateEngine:
    """
    Template engine for Docker-compose configurations.
    
    Provides template loading, variable substitution, and inheritance
    for generating docker-compose files from templates.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the Docker template engine.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir or Path(__file__).parent
        self._template_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a Docker-compose template.
        
        Args:
            template_name: Name of the template file (without .yml extension)
            
        Returns:
            Dictionary containing template data
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name].copy()
        
        template_file = self.template_dir / f"{template_name}.yml"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template not found: {template_file}")
        
        try:
            with open(template_file, 'r') as f:
                template_data = yaml.safe_load(f)
            
            self._template_cache[template_name] = template_data
            return template_data.copy()
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    def process_template(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a template with variable substitution.
        
        Args:
            template_name: Name of the template
            variables: Variables for substitution
            
        Returns:
            Processed template data
        """
        template_data = self.load_template(template_name)
        return self._substitute_variables(template_data, variables)
    
    def _substitute_variables(self, data: Any, variables: Dict[str, Any]) -> Any:
        """
        Recursively substitute variables in template data.
        
        Args:
            data: Template data (can be dict, list, or string)
            variables: Variables for substitution
            
        Returns:
            Data with variables substituted
        """
        if isinstance(data, dict):
            return {key: self._substitute_variables(value, variables) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_variables(item, variables) for item in data]
        elif isinstance(data, str):
            return self._substitute_string(data, variables)
        else:
            return data
    
    def _substitute_string(self, text: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in a string.
        
        Args:
            text: Text with variable placeholders
            variables: Variables for substitution
            
        Returns:
            Text with variables substituted
        """
        # Simple variable substitution using ${VAR_NAME} format
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            else:
                # Check for default value syntax: ${VAR_NAME:-default}
                if ':-' in var_name:
                    var_name, default = var_name.split(':-', 1)
                    return str(variables.get(var_name, default))
                return match.group(0)  # Return original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_var, text)
    
    def merge_templates(self, base_template: str, override_template: str, 
                       variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two templates with the override taking precedence.
        
        Args:
            base_template: Name of the base template
            override_template: Name of the override template
            variables: Variables for substitution
            
        Returns:
            Merged template data
        """
        base_data = self.process_template(base_template, variables)
        override_data = self.process_template(override_template, variables)
        
        return self._deep_merge(base_data, override_data)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_available_templates(self) -> list[str]:
        """
        Get list of available template names.
        
        Returns:
            List of template names (without .yml extension)
        """
        templates = []
        for file in self.template_dir.glob("*.yml"):
            templates.append(file.stem)
        return sorted(templates)
    
    def validate_template(self, template_name: str) -> bool:
        """
        Validate that a template is properly formatted.
        
        Args:
            template_name: Name of the template to validate
            
        Returns:
            True if template is valid, False otherwise
        """
        try:
            template_data = self.load_template(template_name)
            
            # Basic validation - check for required docker-compose structure
            required_keys = ['version', 'services']
            for key in required_keys:
                if key not in template_data:
                    logger.error(f"Template {template_name} missing required key: {key}")
                    return False
            
            # Validate services section
            if not isinstance(template_data['services'], dict):
                logger.error(f"Template {template_name} services section must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating template {template_name}: {e}")
            return False