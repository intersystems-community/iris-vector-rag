"""
Interfaces and data classes for the Quick Start configuration system.

This module defines the core interfaces, data classes, and exceptions
used throughout the configuration template system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class ConfigurationContext:
    """Context for configuration resolution."""
    profile: str
    environment: str
    overrides: Dict[str, Any]
    template_path: Path
    environment_variables: Dict[str, str]


class IConfigurationTemplate(ABC):
    """Interface for configuration template operations."""
    
    @abstractmethod
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a configuration template by name."""
        pass
    
    @abstractmethod
    def resolve_template(self, context: ConfigurationContext) -> Dict[str, Any]:
        """Resolve configuration template with context."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        pass
    
    @abstractmethod
    def get_available_profiles(self) -> List[str]:
        """Get list of available configuration profiles."""
        pass


class IEnvironmentVariableInjector(ABC):
    """Interface for environment variable injection."""
    
    @abstractmethod
    def inject_variables(
        self, 
        config: Dict[str, Any], 
        env_vars: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Inject environment variables into configuration."""
        pass


class IConfigurationValidator(ABC):
    """Interface for configuration validation."""
    
    @abstractmethod
    def validate_schema(
        self, 
        config: Dict[str, Any], 
        schema_name: str
    ) -> List[str]:
        """Validate configuration against a schema."""
        pass


# Exception classes
class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class TemplateNotFoundError(ConfigurationError):
    """Raised when a template file cannot be found."""
    pass


class InheritanceError(ConfigurationError):
    """Raised when there are issues with template inheritance."""
    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class EnvironmentVariableError(ConfigurationError):
    """Raised when environment variable processing fails."""
    pass