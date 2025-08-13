"""
Pipeline Factory Service.

This module provides the PipelineFactory class for creating pipeline instances
from configuration definitions.
"""

import logging
from typing import Dict, Any, Optional

from ..config.pipeline_config_service import PipelineConfigService
from ..utils.module_loader import ModuleLoader
from ..core.exceptions import PipelineNotFoundError, PipelineCreationError, ModuleLoadingError
from ..core.base import RAGPipeline


class PipelineFactory:
    """
    Factory for creating RAG pipeline instances from configuration.

    This factory handles:
    - Loading pipeline configurations
    - Creating pipeline instances with dependency injection
    - Managing pipeline lifecycle
    - Error handling and logging
    """

    def __init__(
        self, config_service: PipelineConfigService, module_loader: ModuleLoader, framework_dependencies: Dict[str, Any]
    ):
        """
        Initialize the pipeline factory.

        Args:
            config_service: Service for loading pipeline configurations
            module_loader: Service for dynamically loading pipeline classes
            framework_dependencies: Framework-level dependencies to inject into pipelines
        """
        self.config_service = config_service
        self.module_loader = module_loader
        self.framework_dependencies = framework_dependencies
        self.logger = logging.getLogger(__name__)

        # Cache for loaded pipeline definitions
        self._pipeline_definitions: Optional[Dict[str, Dict[str, Any]]] = None

    def create_pipeline(self, pipeline_name: str) -> Optional[RAGPipeline]:
        """
        Create a specific pipeline by name.

        Args:
            pipeline_name: Name of the pipeline to create

        Returns:
            Created pipeline instance

        Raises:
            PipelineNotFoundError: If pipeline is not found or disabled
            PipelineCreationError: If pipeline creation fails
        """
        # Load pipeline definitions if not already cached
        if self._pipeline_definitions is None:
            self._load_pipeline_definitions()

        # Find the pipeline definition
        if pipeline_name not in self._pipeline_definitions:
            raise PipelineNotFoundError(f"Pipeline '{pipeline_name}' not found in configuration")

        pipeline_def = self._pipeline_definitions[pipeline_name]

        # Check if pipeline is enabled
        if not pipeline_def.get("enabled", True):
            raise PipelineNotFoundError(f"Pipeline '{pipeline_name}' is disabled")

        try:
            # Load the pipeline class
            pipeline_class = self.module_loader.load_pipeline_class(pipeline_def["module"], pipeline_def["class"])

            # Prepare constructor arguments
            # Pipelines expect (connection_manager, config_manager) as positional args
            # and other arguments as keyword arguments
            framework_kwargs = {
                **self.framework_dependencies,  # Framework dependencies (llm_func, vector_store)
                **pipeline_def.get("params", {}),  # Pipeline-specific parameters (filtered to avoid conflicts)
            }

            # Filter out parameters that might conflict with constructor signature
            # Most pipelines don't accept arbitrary config parameters in constructor
            allowed_kwargs = {"llm_func", "vector_store"}
            filtered_kwargs = {k: v for k, v in framework_kwargs.items() if k in allowed_kwargs}

            # Create the pipeline instance with required positional args and filtered kwargs
            pipeline_instance = pipeline_class(
                self.framework_dependencies.get("connection_manager"),
                self.framework_dependencies.get("config_manager"),
                **filtered_kwargs,
            )

            self.logger.info(f"Successfully created pipeline: {pipeline_name}")
            return pipeline_instance

        except ModuleLoadingError as e:
            error_msg = f"Failed to create pipeline '{pipeline_name}': {str(e)}"
            self.logger.error(error_msg)
            raise PipelineCreationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to create pipeline '{pipeline_name}': {str(e)}"
            self.logger.error(error_msg)
            raise PipelineCreationError(error_msg)

    def create_all_pipelines(self) -> Dict[str, RAGPipeline]:
        """
        Create all enabled pipelines.

        Returns:
            Dictionary mapping pipeline names to pipeline instances
        """
        # Load pipeline definitions if not already cached
        if self._pipeline_definitions is None:
            self._load_pipeline_definitions()

        pipelines = {}

        for pipeline_name, pipeline_def in self._pipeline_definitions.items():
            # Skip disabled pipelines
            if not pipeline_def.get("enabled", True):
                self.logger.debug(f"Skipping disabled pipeline: {pipeline_name}")
                continue

            try:
                pipeline = self.create_pipeline(pipeline_name)
                pipelines[pipeline_name] = pipeline

            except (PipelineNotFoundError, PipelineCreationError) as e:
                self.logger.error(f"Failed to create pipeline '{pipeline_name}': {str(e)}")
                # Continue with other pipelines
                continue

        self.logger.info(f"Successfully created {len(pipelines)} pipelines")
        return pipelines

    def _load_pipeline_definitions(self) -> None:
        """Load and cache pipeline definitions from configuration service."""
        try:
            # This should be configurable, but for now use a default path
            config_path = "config/pipelines.yaml"
            definitions_list = self.config_service.load_pipeline_definitions(config_path)

            # Convert list to dictionary for easier lookup
            self._pipeline_definitions = {definition["name"]: definition for definition in definitions_list}

            self.logger.debug(f"Loaded {len(self._pipeline_definitions)} pipeline definitions")

        except Exception as e:
            error_msg = f"Failed to load pipeline definitions: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineCreationError(error_msg)
