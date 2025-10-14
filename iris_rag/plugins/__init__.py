"""
Plugin System Interface.

This module provides the base interface and utilities for creating RAG pipeline plugins.
"""

from .interface import PluginManifest, RAGPlugin

__all__ = ["RAGPlugin", "PluginManifest"]
