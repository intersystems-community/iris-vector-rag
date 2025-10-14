"""
Pipeline Configuration Service.

This module provides the PipelineConfigService class for loading and validating
pipeline configurations from YAML files.
"""

import logging
from typing import Dict, List

import yaml

from ..core.exceptions import PipelineConfigurationError
from ..utils.project_root import resolve_project_relative_path


class PipelineConfigService:
    """
    Service for loading and validating pipeline configurations.

    This service handles:
    - Loading pipeline definitions from YAML configuration files
    - Validating pipeline configuration schema
    - Providing structured access to pipeline definitions
    """

    def __init__(self):
        """Initialize the pipeline configuration service."""
        self.logger = logging.getLogger(__name__)

    def load_pipeline_definitions(self, config_file_path: str) -> List[Dict]:
        """
        Load pipeline definitions from a YAML configuration file and discover plugin pipelines.

        Args:
            config_file_path: Path to the YAML configuration file (relative to project root)

        Returns:
            List of pipeline definition dictionaries (static + plugin-provided)

        Raises:
            PipelineConfigurationError: If file cannot be loaded or parsed
        """
        # Load static definitions from YAML (existing functionality)
        static_pipelines = self._load_static_definitions(config_file_path)

        # Discover plugin pipelines (NEW)
        plugin_pipelines = self._discover_plugin_pipelines()

        # Merge and return
        all_pipelines = static_pipelines + plugin_pipelines
        self.logger.info(
            f"Loaded {len(static_pipelines)} static + {len(plugin_pipelines)} plugin pipeline definitions"
        )
        return all_pipelines

    def _load_static_definitions(self, config_file_path: str) -> List[Dict]:
        """
        Load static pipeline definitions from YAML configuration file.

        Args:
            config_file_path: Path to the YAML configuration file (relative to project root)

        Returns:
            List of static pipeline definition dictionaries

        Raises:
            PipelineConfigurationError: If file cannot be loaded or parsed
        """
        try:
            # Resolve path relative to project root, making it robust to cwd changes
            config_path = resolve_project_relative_path(config_file_path)
            self.logger.debug(f"Resolved config path: {config_path}")
        except Exception as e:
            raise PipelineConfigurationError(
                f"Failed to resolve configuration path '{config_file_path}': {str(e)}"
            )

        # Check if file exists
        if not config_path.exists():
            raise PipelineConfigurationError(
                f"Configuration file not found: {config_path} "
                f"(resolved from '{config_file_path}')"
            )

        try:
            # Load and parse YAML
            with open(config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

            if not config_data:
                raise PipelineConfigurationError("Configuration file is empty")

            # Extract pipeline definitions
            pipelines = config_data.get("pipelines", [])
            if not isinstance(pipelines, list):
                raise PipelineConfigurationError(
                    "Configuration must contain a 'pipelines' list"
                )

            # Validate each pipeline definition
            validated_pipelines = []
            for pipeline_def in pipelines:
                if self.validate_pipeline_definition(pipeline_def):
                    validated_pipelines.append(pipeline_def)

            return validated_pipelines

        except yaml.YAMLError as e:
            raise PipelineConfigurationError(f"Failed to parse YAML: {str(e)}")
        except Exception as e:
            raise PipelineConfigurationError(f"Failed to load configuration: {str(e)}")

    def _discover_plugin_pipelines(self) -> List[Dict]:
        """
        Discover pipelines from installed plugin packages.

        Returns:
            List of plugin-provided pipeline definitions

        Note:
            Uses Python entry points to discover packages with 'rag_templates_plugins' entry points.
            Each plugin must implement the plugin interface with get_pipeline_classes() method.
        """
        plugin_pipelines = []

        try:
            import pkg_resources

            for entry_point in pkg_resources.iter_entry_points("rag_templates_plugins"):
                try:
                    # Load plugin class
                    plugin_class = entry_point.load()
                    plugin = plugin_class()

                    # Get pipeline definitions from plugin
                    pipeline_classes = plugin.get_pipeline_classes()
                    schema_managers = plugin.get_schema_managers()

                    for pipeline_name, pipeline_class in pipeline_classes.items():
                        plugin_def = {
                            "name": pipeline_name,
                            "type": "plugin",
                            "plugin_package": entry_point.name,
                            "module": pipeline_class.__module__,
                            "class": pipeline_class.__name__,
                            "enabled": True,
                            "params": (
                                plugin.get_default_configuration()
                                if hasattr(plugin, "get_default_configuration")
                                else {}
                            ),
                        }

                        # Add schema requirements if plugin provides schema manager
                        if pipeline_name in schema_managers:
                            plugin_def["schema_requirements"] = {
                                "schema_manager": schema_managers[
                                    pipeline_name
                                ].__name__,
                                "tables": getattr(plugin, "REQUIRED_TABLES", []),
                                "dependencies": getattr(
                                    plugin, "EXTERNAL_DEPENDENCIES", []
                                ),
                            }

                        plugin_pipelines.append(plugin_def)
                        self.logger.debug(
                            f"Discovered plugin pipeline: {pipeline_name} from {entry_point.name}"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to load plugin {entry_point.name}: {e}"
                    )

        except ImportError:
            # pkg_resources not available - no plugin support
            self.logger.debug("pkg_resources not available - plugin discovery disabled")

        return plugin_pipelines

    def validate_pipeline_definition(self, definition: Dict) -> bool:
        """
        Validate a single pipeline definition (supports both core and plugin types).

        Args:
            definition: Pipeline definition dictionary to validate

        Returns:
            True if validation passes

        Raises:
            PipelineConfigurationError: If validation fails
        """
        if not isinstance(definition, dict):
            raise PipelineConfigurationError("Pipeline definition must be a dictionary")

        # Check required fields - name is always required
        if "name" not in definition:
            raise PipelineConfigurationError("Missing required field: name")
        if not isinstance(definition["name"], str):
            raise PipelineConfigurationError("Field 'name' must be a string")

        # Check type-specific required fields
        pipeline_type = definition.get("type", "core")

        if pipeline_type == "plugin":
            # Plugin pipelines require different fields
            plugin_required = ["plugin_package", "module", "class"]
            for field in plugin_required:
                if field not in definition:
                    raise PipelineConfigurationError(
                        f"Plugin pipeline missing required field: {field}"
                    )
                if not isinstance(definition[field], str):
                    raise PipelineConfigurationError(
                        f"Field '{field}' must be a string"
                    )
        else:
            # Core pipelines require module and class
            core_required = ["module", "class"]
            for field in core_required:
                if field not in definition:
                    raise PipelineConfigurationError(
                        f"Core pipeline missing required field: {field}"
                    )
                if not isinstance(definition[field], str):
                    raise PipelineConfigurationError(
                        f"Field '{field}' must be a string"
                    )

        # Check optional fields with type validation
        if "enabled" in definition:
            if not isinstance(definition["enabled"], bool):
                raise PipelineConfigurationError("Field 'enabled' must be a boolean")

        if "params" in definition:
            if not isinstance(definition["params"], dict):
                raise PipelineConfigurationError("Field 'params' must be a dictionary")

        if "type" in definition:
            if definition["type"] not in ["core", "plugin"]:
                raise PipelineConfigurationError(
                    "Field 'type' must be 'core' or 'plugin'"
                )

        # Set defaults for optional fields
        if "enabled" not in definition:
            definition["enabled"] = True
        if "params" not in definition:
            definition["params"] = {}
        if "type" not in definition:
            definition["type"] = "core"

        self.logger.debug(
            f"Validated {pipeline_type} pipeline definition: {definition['name']}"
        )
        return True
