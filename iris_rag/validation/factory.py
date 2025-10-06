"""
Validated pipeline factory with pre-condition checking.

This module provides an enhanced factory that validates requirements
before creating pipeline instances.
"""

import logging
from typing import Any, Callable, Dict, Optional

from ..config.manager import ConfigurationManager
from ..core.base import RAGPipeline
from ..core.connection import ConnectionManager
from ..embeddings.manager import EmbeddingManager
from ..pipelines.basic import BasicRAGPipeline
from ..pipelines.basic_rerank import BasicRAGRerankingPipeline
from ..pipelines.crag import CRAGPipeline
from ..pipelines.graphrag import GraphRAGPipeline
from ..pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from .orchestrator import SetupOrchestrator
from .requirements import get_pipeline_requirements
from .validator import PreConditionValidator

logger = logging.getLogger(__name__)


class PipelineValidationError(Exception):
    """Raised when pipeline validation fails."""


class ValidatedPipelineFactory:
    """
    Enhanced pipeline factory with pre-condition validation.

    This factory ensures that pipelines have all required data and dependencies
    before creating instances. It can automatically set up missing requirements
    or provide clear error messages with setup suggestions.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config_manager: ConfigurationManager,
    ):
        """
        Initialize the validated factory.

        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.embedding_manager = EmbeddingManager(
            config_manager
        )  # Initialize EmbeddingManager
        self.validator = PreConditionValidator(connection_manager)
        self.orchestrator = SetupOrchestrator(
            connection_manager, config_manager
        )  # SetupOrchestrator creates its own EmbeddingManager
        self.logger = logging.getLogger(__name__)

    def create_pipeline(
        self,
        pipeline_type: str,
        llm_func: Optional[Callable[[str], str]] = None,
        auto_setup: bool = False,
        validate_requirements: bool = True,
        **kwargs,
    ) -> RAGPipeline:
        """
        Create a validated pipeline instance.

        Args:
            pipeline_type: Type of pipeline to create
            llm_func: Optional LLM function for answer generation
            auto_setup: Whether to automatically set up missing requirements
            validate_requirements: Whether to validate requirements before creation
            **kwargs: Additional pipeline-specific arguments

        Returns:
            Validated pipeline instance

        Raises:
            PipelineValidationError: If validation fails and auto_setup is False
            ValueError: If pipeline type is unknown
        """
        self.logger.info(f"Creating {pipeline_type} pipeline with validation")

        # Validate requirements if requested
        if validate_requirements:
            validation_report = self._validate_and_setup(pipeline_type, auto_setup)

            if not validation_report.overall_valid:
                error_msg = f"Pipeline {pipeline_type} validation failed: {validation_report.summary}"
                if validation_report.setup_suggestions:
                    error_msg += f"\nSuggestions: {'; '.join(validation_report.setup_suggestions)}"

                if not auto_setup:
                    raise PipelineValidationError(error_msg)
                else:
                    self.logger.warning(
                        f"Pipeline created despite validation issues: {error_msg}"
                    )

        # Create pipeline instance
        return self._create_pipeline_instance(pipeline_type, llm_func, **kwargs)

    def _validate_and_setup(self, pipeline_type: str, auto_setup: bool):
        """Validate requirements and optionally set up missing components."""
        requirements = get_pipeline_requirements(pipeline_type)
        validation_report = self.validator.validate_pipeline_requirements(requirements)

        if not validation_report.overall_valid and auto_setup:
            self.logger.info(
                f"Auto-setup enabled, attempting to fix issues for {pipeline_type}"
            )
            validation_report = self.orchestrator.setup_pipeline(
                pipeline_type, auto_fix=True
            )

        return validation_report

    def _create_pipeline_instance(
        self, pipeline_type: str, llm_func: Optional[Callable[[str], str]], **kwargs
    ) -> RAGPipeline:
        """Create the actual pipeline instance with proper schema manager."""
        # Create appropriate schema manager for the pipeline type
        from ..storage.schema_manager import SchemaManager

        schema_manager = SchemaManager.create_schema_manager(
            pipeline_type, self.connection_manager, self.config_manager
        )

        # Ensure schema requirements are met
        if not schema_manager.ensure_pipeline_schema(pipeline_type):
            logger.warning(
                f"Some schema requirements for {pipeline_type} could not be ensured"
            )

        # Create pipeline instance with schema manager
        pipeline_kwargs = {
            "connection_manager": self.connection_manager,
            "config_manager": self.config_manager,
            "llm_func": llm_func,
            **kwargs,
        }

        # For pipelines that accept schema_manager, pass it
        # Currently only HybridGraphRAG explicitly accepts schema_manager
        if pipeline_type == "hybrid_graphrag":
            pipeline_kwargs["schema_manager"] = schema_manager

        if pipeline_type == "basic":
            return BasicRAGPipeline(**pipeline_kwargs)
        elif pipeline_type == "crag":
            return CRAGPipeline(**pipeline_kwargs)
        elif pipeline_type == "basic_rerank":
            return BasicRAGRerankingPipeline(**pipeline_kwargs)
        elif pipeline_type == "graphrag":
            return GraphRAGPipeline(**pipeline_kwargs)
        elif pipeline_type == "hybrid_graphrag":
            return HybridGraphRAGPipeline(**pipeline_kwargs)
        elif pipeline_type == "pylate_colbert":
            from iris_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
            return PyLateColBERTPipeline(**pipeline_kwargs)
        else:
            available_types = [
                "basic",
                "basic_rerank",
                "crag",
                "graphrag",
                "hybrid_graphrag",
                "pylate_colbert",
            ]
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}. Available: {available_types}"
            )

    def validate_pipeline_type(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Validate a pipeline type without creating an instance.

        Args:
            pipeline_type: Type of pipeline to validate

        Returns:
            Validation results dictionary
        """
        try:
            requirements = get_pipeline_requirements(pipeline_type)
            validation_report = self.validator.validate_pipeline_requirements(
                requirements
            )

            return {
                "pipeline_type": pipeline_type,
                "valid": validation_report.overall_valid,
                "summary": validation_report.summary,
                "table_issues": [
                    name
                    for name, result in validation_report.table_validations.items()
                    if not result.is_valid
                ],
                "embedding_issues": [
                    name
                    for name, result in validation_report.embedding_validations.items()
                    if not result.is_valid
                ],
                "suggestions": validation_report.setup_suggestions,
            }

        except Exception as e:
            return {
                "pipeline_type": pipeline_type,
                "valid": False,
                "summary": f"Validation error: {e}",
                "table_issues": [],
                "embedding_issues": [],
                "suggestions": ["Check pipeline type and database connection"],
            }

    def get_pipeline_status(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Get detailed status information for a pipeline type.

        Args:
            pipeline_type: Type of pipeline to check

        Returns:
            Detailed status information
        """
        try:
            requirements = get_pipeline_requirements(pipeline_type)
            validation_report = self.validator.validate_pipeline_requirements(
                requirements
            )

            # Collect detailed information
            table_details = {}
            for name, result in validation_report.table_validations.items():
                table_details[name] = {
                    "valid": result.is_valid,
                    "message": result.message,
                    "details": result.details,
                }

            embedding_details = {}
            for name, result in validation_report.embedding_validations.items():
                embedding_details[name] = {
                    "valid": result.is_valid,
                    "message": result.message,
                    "details": result.details,
                }

            return {
                "pipeline_type": pipeline_type,
                "overall_valid": validation_report.overall_valid,
                "summary": validation_report.summary,
                "tables": table_details,
                "embeddings": embedding_details,
                "setup_suggestions": validation_report.setup_suggestions,
                "requirements": {
                    "required_tables": [
                        {
                            "name": req.name,
                            "schema": req.schema,
                            "description": req.description,
                        }
                        for req in requirements.required_tables
                    ],
                    "required_embeddings": [
                        {
                            "name": req.name,
                            "table": req.table,
                            "column": req.column,
                            "description": req.description,
                        }
                        for req in requirements.required_embeddings
                    ],
                },
            }

        except Exception as e:
            return {
                "pipeline_type": pipeline_type,
                "overall_valid": False,
                "summary": f"Status check failed: {e}",
                "tables": {},
                "embeddings": {},
                "setup_suggestions": ["Check pipeline type and database connection"],
                "requirements": {},
            }

    def setup_pipeline_requirements(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Set up all requirements for a pipeline type.

        Args:
            pipeline_type: Type of pipeline to set up

        Returns:
            Setup results
        """
        try:
            self.logger.info(f"Setting up requirements for {pipeline_type}")
            validation_report = self.orchestrator.setup_pipeline(
                pipeline_type, auto_fix=True
            )

            return {
                "pipeline_type": pipeline_type,
                "success": validation_report.overall_valid,
                "summary": validation_report.summary,
                "setup_completed": validation_report.overall_valid,
                "remaining_issues": [
                    name
                    for name, result in {
                        **validation_report.table_validations,
                        **validation_report.embedding_validations,
                    }.items()
                    if not result.is_valid
                ],
            }

        except Exception as e:
            self.logger.error(f"Setup failed for {pipeline_type}: {e}")
            return {
                "pipeline_type": pipeline_type,
                "success": False,
                "summary": f"Setup failed: {e}",
                "setup_completed": False,
                "remaining_issues": ["Setup process failed"],
            }

    def list_available_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available pipeline types with their validation status.

        Returns:
            Dictionary of pipeline types and their status
        """
        pipeline_types = [
            "basic",
            "crag",
        ]
        results = {}

        for pipeline_type in pipeline_types:
            try:
                status = self.validate_pipeline_type(pipeline_type)
                results[pipeline_type] = status
            except Exception as e:
                results[pipeline_type] = {
                    "pipeline_type": pipeline_type,
                    "valid": False,
                    "summary": f"Error checking {pipeline_type}: {e}",
                    "table_issues": [],
                    "embedding_issues": [],
                    "suggestions": [],
                }

        return results


# Convenience function for backward compatibility
def create_validated_pipeline(
    pipeline_type: str,
    connection_manager: ConnectionManager,
    config_manager: ConfigurationManager,
    llm_func: Optional[Callable[[str], str]] = None,
    auto_setup: bool = False,
    **kwargs,
) -> RAGPipeline:
    """
    Create a validated pipeline instance (convenience function).

    Args:
        pipeline_type: Type of pipeline to create
        connection_manager: Database connection manager
        config_manager: Configuration manager
        llm_func: Optional LLM function
        auto_setup: Whether to automatically set up missing requirements
        **kwargs: Additional pipeline arguments

    Returns:
        Validated pipeline instance
    """
    factory = ValidatedPipelineFactory(connection_manager, config_manager)
    return factory.create_pipeline(
        pipeline_type=pipeline_type, llm_func=llm_func, auto_setup=auto_setup, **kwargs
    )
