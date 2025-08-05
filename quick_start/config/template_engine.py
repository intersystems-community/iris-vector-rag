"""
Configuration template engine for the Quick Start system.

This module provides template-based configuration management with inheritance,
validation, and environment variable injection capabilities.
"""

import yaml
import re
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from quick_start.config.interfaces import (
    IConfigurationTemplate,
    ConfigurationContext,
    ConfigurationError,
    TemplateNotFoundError,
    InheritanceError,
    ValidationError,
    EnvironmentVariableError,
)

logger = logging.getLogger(__name__)


class ConfigurationTemplateEngine(IConfigurationTemplate):
    """
    Template engine for configuration management with inheritance and validation.
    
    Provides a flexible system for managing configuration templates that can
    inherit from base templates, inject environment variables, and validate
    configuration values.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the configuration template engine.
        
        Args:
            template_dir: Directory containing configuration templates
        """
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self._template_cache: Dict[str, Dict[str, Any]] = {}
        self._inheritance_cache: Dict[str, List[str]] = {}
        
        # Pattern for environment variable substitution: ${VAR_NAME:-default_value}
        self.env_var_pattern = re.compile(r'\$\{([^}]+)\}')
        
        # Schema validation support
        self.enable_schema_validation = False
        self._schema_validator = None
    
    def generate_configuration(self, profile: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate configuration for the specified profile.
        
        Args:
            profile: Profile name to generate configuration for
            context: Additional context for configuration generation
            
        Returns:
            Generated configuration dictionary
        """
        # Default configuration based on profile
        base_config = {
            "database": {"host": "localhost", "port": 1972},
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"model": "text-embedding-ada-002"}
        }
        
        # Profile-specific adjustments
        if profile == "minimal":
            base_config["llm"]["model"] = "gpt-3.5-turbo"
        elif profile == "extended":
            base_config["database"]["pool_size"] = 20
            base_config["llm"]["model"] = "gpt-4-turbo"
            
        return base_config
    
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a configuration template by name.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Dictionary containing the loaded template configuration
            
        Raises:
            TemplateNotFoundError: If template file doesn't exist
            ConfigurationError: If template file is invalid
        """
        # Check cache first
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        template_path = self.template_dir / f"{template_name}.yaml"
        
        if not template_path.exists():
            raise TemplateNotFoundError(f"Template not found: {template_name}")
        
        try:
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
            
            if template_data is None:
                template_data = {}
            
            # Cache the loaded template
            self._template_cache[template_name] = template_data
            return template_data
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in template {template_name}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load template {template_name}: {e}")
    
    def resolve_template(self, context: ConfigurationContext) -> Dict[str, Any]:
        """
        Resolve a template with the given context and environment variables.
        
        Args:
            context: Configuration context with profile, environment, and overrides
            
        Returns:
            Dictionary containing the resolved configuration
            
        Raises:
            ConfigurationError: If template resolution fails
        """
        try:
            # Build inheritance chain
            inheritance_chain = self._build_inheritance_chain(context.profile)
            
            # Load and merge configurations
            merged_config = {}
            for template_name in inheritance_chain:
                template_config = self.load_template(template_name)
                # Remove 'extends' directive from merged config
                template_config = {k: v for k, v in template_config.items() if k != 'extends'}
                merged_config = self._deep_merge(merged_config, template_config)
            
            # Apply context overrides
            if context.overrides:
                merged_config = self._deep_merge(merged_config, context.overrides)
            
            # Inject environment variables
            resolved_config = self._inject_environment_variables(
                merged_config,
                context.environment_variables
            )
            
            # Perform schema validation if enabled
            if self.enable_schema_validation:
                self._validate_configuration(resolved_config, context.profile)
            
            return resolved_config
            
        except Exception as e:
            if isinstance(e, (TemplateNotFoundError, InheritanceError, ConfigurationError)):
                raise
            raise ConfigurationError(f"Failed to resolve template {context.profile}: {e}")
    
    def _build_inheritance_chain(self, profile: str) -> List[str]:
        """
        Build inheritance chain for a profile.
        
        Args:
            profile: Profile name to build chain for
            
        Returns:
            List of template names in inheritance order (base first)
            
        Raises:
            InheritanceError: If circular inheritance is detected
        """
        # Check cache first
        if profile in self._inheritance_cache:
            return self._inheritance_cache[profile]
        
        chain = []
        current_profile = profile
        visited = set()
        
        while current_profile:
            if current_profile in visited:
                raise InheritanceError(f"Circular inheritance detected: {current_profile}")
            
            visited.add(current_profile)
            chain.insert(0, current_profile)
            
            # Load template to check for 'extends' directive
            template_data = self.load_template(current_profile)
            current_profile = template_data.get('extends')
        
        # Cache the inheritance chain
        self._inheritance_cache[profile] = chain
        return chain
    
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
    
    def _inject_environment_variables(
        self, 
        config: Dict[str, Any], 
        env_vars: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Inject environment variables into configuration values.
        
        Args:
            config: Configuration dictionary to process
            env_vars: Environment variables to use (defaults to os.environ)
            
        Returns:
            Configuration with environment variables injected
        """
        if env_vars is None:
            env_vars = dict(os.environ)
        
        return self._process_value(config, env_vars)
    
    def _process_value(self, value: Any, env_vars: Dict[str, str]) -> Any:
        """
        Process a configuration value for environment variable substitution.
        
        Args:
            value: Value to process
            env_vars: Environment variables
            
        Returns:
            Processed value with environment variables substituted
        """
        if isinstance(value, str):
            return self._substitute_env_vars(value, env_vars)
        elif isinstance(value, dict):
            return {k: self._process_value(v, env_vars) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._process_value(item, env_vars) for item in value]
        else:
            return value
    
    def _substitute_env_vars(self, text: str, env_vars: Dict[str, str]) -> Any:
        """
        Substitute environment variables in text.
        
        Args:
            text: Text to process
            env_vars: Environment variables
            
        Returns:
            Text with environment variables substituted and type converted
        """
        def replace_var(match):
            var_expr = match.group(1)
            
            # Handle default values: VAR_NAME:-default_value
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                value = env_vars.get(var_name, default_value)
            else:
                var_name = var_expr
                value = env_vars.get(var_name, f"${{{var_expr}}}")
            
            return value
        
        # Substitute all environment variables
        result = self.env_var_pattern.sub(replace_var, text)
        
        # Try to convert to appropriate type
        return self._convert_type(result)
    
    def _convert_type(self, value: str) -> Any:
        """
        Convert string value to appropriate Python type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Don't convert version-like strings (e.g., "2024.1", "1.0.0")
        # These should remain as strings for schema validation
        if self._is_version_string(value):
            return value
        
        # Try to convert to int
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _is_version_string(self, value: str) -> bool:
        """
        Check if a string looks like a version number that should remain a string.
        
        Args:
            value: String to check
            
        Returns:
            True if it looks like a version string
        """
        import re
        # Match patterns like "2024.1", "1.0.0", "v1.2.3", etc.
        version_patterns = [
            r'^\d{4}\.\d+$',  # Year.version like "2024.1"
            r'^\d+\.\d+\.\d+$',  # Semantic version like "1.0.0"
            r'^v\d+\.\d+(\.\d+)?$',  # Version with v prefix like "v1.2" or "v1.2.3"
        ]
        
        for pattern in version_patterns:
            if re.match(pattern, value):
                return True
        return False
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration against a schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        # Basic validation - can be extended with JSON schema validation
        errors = []
        
        # Check for required metadata
        if 'metadata' not in config:
            errors.append("Missing required 'metadata' section")
        
        return errors
    
    def get_available_profiles(self) -> List[str]:
        """
        Get a list of available configuration templates.
        
        Returns:
            List of template names
        """
        if not self.template_dir.exists():
            return []
        
        profiles = []
        for yaml_file in self.template_dir.glob("*.yaml"):
            profiles.append(yaml_file.stem)
        
        return profiles
    
    def _get_schema_validator(self):
        """
        Get or create the schema validator instance.
        
        Returns:
            ConfigurationSchemaValidator instance
        """
        if self._schema_validator is None:
            from quick_start.config.schema_validator import ConfigurationSchemaValidator
            self._schema_validator = ConfigurationSchemaValidator()
        return self._schema_validator
    
    def _validate_configuration(self, config: Dict[str, Any], profile: str) -> None:
        """
        Validate configuration using JSON schema validation.
        
        Args:
            config: Configuration dictionary to validate
            profile: Profile name for profile-specific validation
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            validator = self._get_schema_validator()
            validator.validate_configuration(config, "base_config", profile)
            logger.debug(f"Configuration validation passed for profile: {profile}")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Configuration validation failed: {e}")
    
    def render_template(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render a template with the given context and environment variables.
        
        Args:
            template_name: Name of the template to render
            context: Additional context variables for template rendering
            
        Returns:
            Dictionary containing the rendered configuration
        """
        # For now, this is equivalent to resolve_template with a simple context
        if context is None:
            context = {}
        
        config_context = ConfigurationContext(
            profile=template_name,
            environment="default",
            overrides=context,
            template_path=self.template_dir,
            environment_variables=dict(os.environ)
        )
        
        return self.resolve_template(config_context)
    
    def inject_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject environment variables into configuration values.
        
        Args:
            config: Configuration dictionary to process
            
        Returns:
            Configuration with environment variables injected
        """
        return self._inject_environment_variables(config)