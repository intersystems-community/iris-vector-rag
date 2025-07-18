"""
RAG Templates Library Consumption Framework.

This package provides a comprehensive framework for implementing and using
various RAG (Retrieval Augmented Generation) techniques with InterSystems IRIS.

The framework offers multiple API layers:
1. Simple API: Zero-configuration RAG for immediate use
2. Standard API: Advanced configuration with technique selection
3. Expert API: Full control and customization (Future Phase)

Phase 1: Foundation Layer (Simple API and Core Services)
- Simple API RAG Class: Zero-config RAG() class with immediate usability
- Configuration Manager: Three-tier configuration system with environment variable support
- Basic Error Handling: Clear error messages and fallback strategies

Phase 2: Standard API Layer (Advanced Configuration and Pipeline Factory)
- Standard API ConfigurableRAG Class: Advanced RAG with technique selection
- Pipeline Factory: Dynamic pipeline creation with dependency injection
- Technique Registry: Dynamic registration and discovery of RAG techniques
- Advanced Configuration: Support for complex configurations and technique-specific settings
"""

__version__ = "2.0.0"
__author__ = "RAG Templates Project"

# Import main classes for easy access
from .simple import RAG
from .standard import ConfigurableRAG
from .core.config_manager import ConfigurationManager
from .core.errors import RAGFrameworkError, ConfigurationError

__all__ = [
    "RAG",
    "ConfigurableRAG",
    "ConfigurationManager",
    "RAGFrameworkError",
    "ConfigurationError"
]