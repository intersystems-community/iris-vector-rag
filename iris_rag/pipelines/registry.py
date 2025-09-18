"""
Pipeline Registry Service.

This module provides the PipelineRegistry class for managing
registered pipeline instances.
"""

import logging
from typing import Dict, List, Optional

from .factory import PipelineFactory
from ..core.base import RAGPipeline


class PipelineRegistry:
    """
    Registry for managing RAG pipeline instances.

    This registry handles:
    - Registration of pipeline instances
    - Retrieval of pipelines by name
    - Listing available pipelines
    - Pipeline lifecycle management
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        """
        Initialize the pipeline registry.

        Args:
            pipeline_factory: Factory for creating pipeline instances
        """
        self.pipeline_factory = pipeline_factory
        self.logger = logging.getLogger(__name__)

        # Storage for registered pipelines
        self._pipelines: Dict[str, RAGPipeline] = {}

    def register_pipelines(self) -> None:
        """
        Register all available pipelines from the factory.

        This method creates all enabled pipelines using the factory
        and stores them in the registry for quick access.
        """
        try:
            # Create all pipelines using the factory
            pipelines = self.pipeline_factory.create_all_pipelines()

            # Store the pipelines in the registry
            self._pipelines = pipelines

            self.logger.info(
                f"Registered {len(pipelines)} pipelines: {list(pipelines.keys())}"
            )

        except Exception as e:
            self.logger.error(f"Failed to register pipelines: {str(e)}")
            # Ensure registry is in a clean state
            self._pipelines = {}

    def get_pipeline(self, name: str) -> Optional[RAGPipeline]:
        """
        Get a registered pipeline by name.

        Args:
            name: Name of the pipeline to retrieve

        Returns:
            Pipeline instance if found, None otherwise
        """
        pipeline = self._pipelines.get(name)

        if pipeline:
            self.logger.debug(f"Retrieved pipeline: {name}")
        else:
            self.logger.debug(f"Pipeline not found: {name}")

        return pipeline

    def list_pipeline_names(self) -> List[str]:
        """
        Get a list of all registered pipeline names.

        Returns:
            List of pipeline names
        """
        names = list(self._pipelines.keys())
        self.logger.debug(f"Listed {len(names)} pipeline names")
        return names

    def is_pipeline_registered(self, name: str) -> bool:
        """
        Check if a pipeline is registered.

        Args:
            name: Name of the pipeline to check

        Returns:
            True if pipeline is registered, False otherwise
        """
        is_registered = name in self._pipelines
        self.logger.debug(f"Pipeline '{name}' registered: {is_registered}")
        return is_registered
