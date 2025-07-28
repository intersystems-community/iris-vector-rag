"""
Quick Start System for RAG Templates.

This package provides a comprehensive quick start system for the RAG Templates project,
enabling users to experience all 8 RAG techniques with minimal setup.

Key Features:
- Zero-configuration start with sample data
- Progressive complexity from quick start to production
- Community Edition compatible
- Modular architecture with clean separation of concerns
"""

__version__ = "1.0.0"
__author__ = "RAG Templates Team"

from quick_start.core.orchestrator import QuickStartOrchestrator
from quick_start.data.sample_manager import SampleDataManager
from quick_start.config.template_engine import ConfigurationTemplateEngine

__all__ = [
    "QuickStartOrchestrator",
    "SampleDataManager", 
    "ConfigurationTemplateEngine",
]