"""
Quick Start configuration management system.

This module provides template-based configuration management with inheritance,
validation, and environment variable injection capabilities.
"""

from quick_start.config.template_engine import ConfigurationTemplateEngine
from quick_start.config.interfaces import (
    IConfigurationTemplate,
    IEnvironmentVariableInjector,
    IConfigurationValidator,
    ConfigurationContext,
    ConfigurationError,
    TemplateNotFoundError,
    InheritanceError,
    ValidationError,
    EnvironmentVariableError,
)

__all__ = [
    "ConfigurationTemplateEngine",
    "IConfigurationTemplate",
    "IEnvironmentVariableInjector", 
    "IConfigurationValidator",
    "ConfigurationContext",
    "ConfigurationError",
    "TemplateNotFoundError",
    "InheritanceError",
    "ValidationError",
    "EnvironmentVariableError",
]