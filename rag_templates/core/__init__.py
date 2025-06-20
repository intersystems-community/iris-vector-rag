"""
Core module for RAG Templates Library Consumption Framework.

This module contains the foundational components including configuration management,
error handling, and base classes for the Simple API.
"""

from .config_manager import ConfigurationManager
from .errors import RAGFrameworkError, ConfigurationError

__all__ = [
    "ConfigurationManager",
    "RAGFrameworkError", 
    "ConfigurationError"
]