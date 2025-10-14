"""
Plugin Interface Definition.

This module defines the interface that all RAG pipeline plugins must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from ..core.base import RAGPipeline
from ..storage.schema_manager import SchemaManager


@dataclass
class PluginManifest:
    """
    Plugin manifest containing metadata and requirements.

    Attributes:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        dependencies: External package dependencies
        rag_templates_min_version: Minimum required rag-templates version
        required_features: Required rag-templates features
        provides_pipelines: Pipeline names provided by this plugin
        provides_schema_managers: Schema manager names provided by this plugin
    """

    name: str
    version: str
    description: str
    dependencies: List[str] = None
    rag_templates_min_version: str = "1.0.0"
    required_features: List[str] = None
    provides_pipelines: List[str] = None
    provides_schema_managers: List[str] = None

    def __post_init__(self):
        """Initialize default empty lists."""
        if self.dependencies is None:
            self.dependencies = []
        if self.required_features is None:
            self.required_features = []
        if self.provides_pipelines is None:
            self.provides_pipelines = []
        if self.provides_schema_managers is None:
            self.provides_schema_managers = []


class RAGPlugin(ABC):
    """
    Abstract base class for all RAG pipeline plugins.

    All plugins must implement this interface to be discoverable
    and loadable by the rag-templates framework.
    """

    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """
        Return plugin manifest with capabilities and requirements.

        Returns:
            PluginManifest containing plugin metadata
        """
        pass

    @abstractmethod
    def get_pipeline_classes(self) -> Dict[str, Type[RAGPipeline]]:
        """
        Return mapping of pipeline names to implementation classes.

        Returns:
            Dictionary mapping pipeline names to RAGPipeline subclass types
        """
        pass

    @abstractmethod
    def get_schema_managers(self) -> Dict[str, Type[SchemaManager]]:
        """
        Return specialized schema managers provided by plugin.

        Returns:
            Dictionary mapping schema manager names to SchemaManager subclass types
        """
        pass

    @abstractmethod
    def validate_environment(self) -> bool:
        """
        Validate that plugin dependencies and environment are satisfied.

        Returns:
            True if environment is valid, False otherwise
        """
        pass

    def get_default_configuration(self) -> Dict[str, Any]:
        """
        Return default configuration parameters for plugin pipelines.

        Returns:
            Dictionary of default configuration parameters
        """
        return {}

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure plugin with user-provided settings.

        Args:
            config: Configuration dictionary
        """
        pass

    def initialize(self) -> None:
        """
        Initialize plugin after configuration.

        Called once after plugin is loaded and configured.
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup plugin resources.

        Called when plugin is being unloaded or system is shutting down.
        """
        pass
