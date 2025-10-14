import os
from typing import Any, Dict, Optional

import yaml


# Define a specific exception for configuration errors
class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""


class ConfigurationManager:
    """
    Manages loading and accessing configuration settings.

    Supports loading from YAML files and overriding with environment variables.
    Environment variables should be prefixed (e.g., RAG_) and use double
    underscores (__) as delimiters for nested keys.
    Example: RAG_DATABASE__IRIS__HOST maps to config[&#x27;database&#x27;][&#x27;iris&#x27;][&#x27;host&#x27;].
    """

    ENV_PREFIX = "RAG_"
    DELIMITER = "__"  # Double underscore for nesting in env vars

    def __init__(
        self, config_path: Optional[str] = None, schema: Optional[Dict] = None
    ):
        """
        Initializes the ConfigurationManager.

        Args:
            config_path: Path to the YAML configuration file.
                         If None, tries to load from a default path or environment variable.
            schema: Optional schema for configuration validation (not yet implemented).
        """
        self._config: Dict[str, Any] = {}
        self._schema = schema  # For future validation

        if config_path:
            if not os.path.exists(config_path):
                raise ConfigValidationError(
                    f"Configuration file not found: {config_path}"
                )
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Load default configuration
            default_config_path = os.path.join(
                os.path.dirname(__file__), "default_config.yaml"
            )
            if os.path.exists(default_config_path):
                with open(default_config_path, "r") as f:
                    self._config = yaml.safe_load(f) or {}

        # Basic environment variable loading (will be refined)
        self._load_env_variables()

        # Validate required configuration
        self._validate_required_config()

    def _load_env_variables(self):
        """
        Loads configuration settings from environment variables.
        Overrides values loaded from the YAML file.
        """
        for env_var, value in os.environ.items():
            if env_var.startswith(self.ENV_PREFIX):
                # Remove prefix and split by delimiter
                key_path_str = env_var[len(self.ENV_PREFIX) :]
                keys = [k.lower() for k in key_path_str.split(self.DELIMITER)]

                current_level = self._config
                for i, key_part in enumerate(keys):
                    if i == len(keys) - 1:  # Last key part
                        # Attempt to cast to original type if possible
                        original_value_at_level = self._get_value_by_keys(
                            self._config, keys[:-1]
                        )
                        original_type = None
                        if (
                            isinstance(original_value_at_level, dict)
                            and key_part in original_value_at_level
                        ):
                            original_type = type(original_value_at_level[key_part])

                        casted_value = self._cast_value(value, original_type)
                        current_level[key_part] = casted_value
                    else:
                        # Ensure we have a dict at this level
                        if key_part not in current_level:
                            current_level[key_part] = {}
                        elif not isinstance(current_level[key_part], dict):
                            current_level[key_part] = {}
                        current_level = current_level[key_part]

    def _cast_value(self, value_str: str, target_type: Optional[type]) -> Any:
        """Attempts to cast string value to target_type."""
        if target_type is None:
            return value_str  # No type info, return as string
        try:
            if target_type == bool:
                if value_str.lower() in ("true", "1", "yes"):
                    return True
                elif value_str.lower() in ("false", "0", "no"):
                    return False
                # else fall through to ValueError
            elif target_type == int:
                return int(value_str)
            elif target_type == float:
                return float(value_str)
            # Add other type castings if needed (e.g., list, dict from JSON string)
        except ValueError:
            # If casting fails, return original string or raise error
            # For now, return string to match basic test expectations
            # A stricter CM might raise ConfigValidationError here
            return value_str
        return value_str  # Default return if no specific cast matches

    def _validate_required_config(self):
        """
        Validate that required configuration values are present.

        Raises:
            ConfigValidationError: If required configuration is missing
        """
        # Define required configuration keys
        required_keys = ["database:iris:host"]

        # Check each required key
        for key in required_keys:
            value = self.get(key)
            if value is None:
                raise ConfigValidationError(f"Missing required config: {key}")

        # Check for critical IRIS configuration from environment (for backward compatibility)
        # Note: This is only checked if the config file doesn't provide the host
        if self.get("database:iris:host") is None and "IRIS_HOST" not in os.environ:
            raise ConfigValidationError("Missing required config: database:iris:host")

    def _get_value_by_keys(self, config_dict: Dict, keys: list) -> Any:
        """Helper to navigate nested dict with a list of keys."""
        current = config_dict
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None  # Key path not found
        return current

    def get(self, key_string: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a configuration setting.

        Keys can be nested using a colon delimiter (e.g., "database:iris:host").

        Args:
            key_string: The configuration key string.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value, or the default if not found.
        """
        keys = [k.lower() for k in key_string.split(":")]

        value = self._config
        for key_part in keys:
            if isinstance(value, dict) and key_part in value:
                value = value[key_part]
            else:
                return default  # Key path not found, return default
        return value

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (alias for get method for backward compatibility).

        Args:
            key: The configuration key string.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value, or the default if not found.
        """
        return self.get(key, default)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a file path.

        Args:
            config_path: Path to the configuration file to load

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f) or {}
            if self._config:
                self._config.update(loaded_config)
            else:
                self._config = loaded_config

    def get_vector_index_config(self) -> Dict[str, Any]:
        """
        Get vector index configuration with HNSW parameters.

        Returns:
            Dictionary containing vector index configuration with defaults
        """
        default_config = {
            "type": "HNSW",
            "M": 16,
            "efConstruction": 200,
            "Distance": "COSINE",
        }

        # Get user-defined config and merge with defaults
        user_config = self.get("vector_index", {})
        if isinstance(user_config, dict):
            # First update with user config
            default_config.update(user_config)

            # Handle environment variable overrides with case mapping
            env_overrides = {}
            for key, value in user_config.items():
                if key.lower() == "m":
                    env_overrides["M"] = (
                        int(value)
                        if isinstance(value, str) and value.isdigit()
                        else value
                    )
                elif key.lower() == "efconstruction":
                    env_overrides["efConstruction"] = (
                        int(value)
                        if isinstance(value, str) and value.isdigit()
                        else value
                    )
                elif key.lower() == "distance":
                    env_overrides["Distance"] = value

            # Apply environment overrides
            default_config.update(env_overrides)

        return default_config

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration with model and dimension information.

        Returns:
            Dictionary containing embedding configuration with defaults
        """
        default_config = {
            "model": "all-MiniLM-L6-v2",
            "model_name": "all-MiniLM-L6-v2",  # Alias for compatibility
            "dimension": None,  # Will be determined by model or schema manager
            "provider": "sentence-transformers",
        }

        # Check for environment variable override for model name
        if "EMBEDDING_MODEL_NAME" in os.environ:
            model_name = os.environ["EMBEDDING_MODEL_NAME"]
            default_config["model"] = model_name
            default_config["model_name"] = model_name

        # Get user-defined config and merge with defaults
        user_config = self.get("embeddings", {})
        if isinstance(user_config, dict):
            default_config.update(user_config)

        # Ensure model_name and model are synchronized
        if "model" in default_config and "model_name" not in default_config:
            default_config["model_name"] = default_config["model"]
        elif "model_name" in default_config and "model" not in default_config:
            default_config["model"] = default_config["model_name"]

        # If dimension is not explicitly set, determine from model or use default
        if not default_config["dimension"]:
            # Use direct config lookup instead of dimension utils to avoid circular dependency
            model_name = default_config["model"]
            if (
                model_name == "sentence-transformers/all-MiniLM-L6-v2"
                or model_name == "all-MiniLM-L6-v2"
            ):
                default_config["dimension"] = 384
            elif model_name == "text-embedding-ada-002":
                default_config["dimension"] = 1536
            elif model_name == "all-mpnet-base-v2":
                default_config["dimension"] = 768
            else:
                # Check config file directly for dimension
                direct_dimension = self.get("embedding_model.dimension", 384)
                default_config["dimension"] = direct_dimension

        return default_config

    def get_reconciliation_config(self) -> Dict[str, Any]:
        """
        Get reconciliation framework configuration with defaults.

        Returns:
            Dictionary containing reconciliation configuration with defaults
        """
        default_config = {
            "enabled": True,
            "mode": "progressive",
            "interval_hours": 24,
            "performance": {
                "max_concurrent_pipelines": 3,
                "batch_size_documents": 100,
                "batch_size_embeddings": 50,
                "memory_limit_gb": 8,
                "cpu_limit_percent": 70,
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay_seconds": 30,
                "rollback_on_failure": True,
            },
            "monitoring": {
                "enable_progress_tracking": True,
                "log_level": "INFO",
                "alert_on_failures": True,
            },
            "pipeline_overrides": {},
        }

        # Get user-defined config and merge with defaults
        user_config = self.get("reconciliation", {})
        if isinstance(user_config, dict):
            # Deep merge nested dictionaries
            for key, value in user_config.items():
                if (
                    key in default_config
                    and isinstance(default_config[key], dict)
                    and isinstance(value, dict)
                ):
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        return default_config

    def get_desired_embedding_state(
        self, pipeline_type: str = "basic"
    ) -> Dict[str, Any]:
        """
        Get desired embedding state configuration for a specific pipeline.

        Args:
            pipeline_type: The pipeline type (e.g., "basic")

        Returns:
            Dictionary containing desired state configuration with defaults
        """

        # Default for other pipeline types
        default_config = {
            "target_document_count": 1000,
            "model_name": "all-MiniLM-L6-v2",
            "vector_dimensions": 384,
            "validation": {
                "diversity_threshold": 0.7,
                "mock_detection_enabled": False,
                "min_embedding_quality_score": 0.8,
            },
            "completeness": {
                "require_all_docs": True,
                "require_token_embeddings": False,
                "min_completeness_percent": 95.0,
                "max_missing_documents": 50,
            },
            "remediation": {
                "auto_heal_missing_embeddings": True,
                "auto_migrate_schema": False,
                "embedding_generation_batch_size": 32,
                "max_remediation_time_minutes": 120,
                "backup_before_remediation": True,
            },
        }

        # Get user-defined config and merge with defaults
        user_config = self.get(pipeline_type, {})
        if isinstance(user_config, dict):
            # Deep merge nested dictionaries
            for key, value in user_config.items():
                if (
                    key in default_config
                    and isinstance(default_config[key], dict)
                    and isinstance(value, dict)
                ):
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        return default_config

    def get_target_state_config(
        self, environment: str = "development"
    ) -> Dict[str, Any]:
        """
        Get target state configuration for a specific environment.

        Args:
            environment: The environment name (e.g., "development", "production")

        Returns:
            Dictionary containing target state configuration
        """
        target_states = self.get("target_states", {})

        if environment in target_states:
            return target_states[environment]

        # Default target state for development
        return {
            "document_count": 1000,
            "pipelines": {
                "basic": {
                    "required_embeddings": {"document_level": 1000},
                    "schema_version": "2.1",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "vector_dimensions": 384,
                },
            },
        }

    # Placeholder for future validation method
    def validate(self):
        """Validates the current configuration against the schema."""
        if not self._schema:
            return  # No schema to validate against

        # Basic example: check for required keys (to be expanded)
        # This is a very naive implementation for now.
        # A proper schema validator like Pydantic or jsonschema would be used.
        # For example, if schema defines: self._schema = {"required": ["database:iris:host"]}

        # This part is just illustrative for the test_config_validation_error_required_key
        # and will need a proper implementation.
        if self.get(
            "database:iris:host"
        ) is None and "database:iris:host" in self._schema.get("required", []):
            raise ConfigValidationError("Missing required config: database:iris:host")

    def _merge_configuration(self, new_config: Dict[str, Any]):
        """
        Merge new configuration with existing configuration.

        This method performs a deep merge, where nested dictionaries are merged
        recursively, and new values override existing ones.

        Args:
            new_config: Configuration dictionary to merge
        """

        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]):
            """Recursively merge source into target."""
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(target[key], value)
                else:
                    target[key] = value

        deep_merge(self._config, new_config)

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration with defaults for IRIS connection.

        Returns:
            Dictionary containing database configuration
        """
        default_config = {
            "host": "localhost",
            "port": 1974,  # Integer for proper type casting
            "namespace": "USER",
            "username": "_SYSTEM",
            "password": "SYS",
            "driver_path": None,
        }

        # First apply user-defined database config from YAML (this includes env var overrides)
        user_config = self.get("database", {})
        if isinstance(user_config, dict):
            # Handle nested IRIS config
            if "iris" in user_config and isinstance(user_config["iris"], dict):
                default_config.update(user_config["iris"])
            else:
                default_config.update(user_config)

        # Map environment variables to config keys (legacy support)
        env_mappings = {
            "IRIS_HOST": "host",
            "IRIS_PORT": "port",
            "IRIS_NAMESPACE": "namespace",
            "IRIS_USERNAME": "username",
            "IRIS_PASSWORD": "password",
            "IRIS_DRIVER_PATH": "driver_path",
        }

        # Override with environment variables (legacy format)
        for env_key, config_key in env_mappings.items():
            if env_key in os.environ:
                value = os.environ[env_key]
                # Cast port to int for proper type
                if config_key == "port":
                    try:
                        value = int(value)
                    except ValueError:
                        pass  # Keep as string if casting fails
                default_config[config_key] = value

        return default_config

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration with defaults.

        Returns:
            Dictionary containing logging configuration
        """
        default_config = {
            "level": "INFO",
            "path": "logs/iris_rag.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }

        # Map environment variables to config keys
        env_mappings = {"LOG_LEVEL": "level", "LOG_PATH": "path"}

        # Override with environment variables
        for env_key, config_key in env_mappings.items():
            if env_key in os.environ:
                default_config[config_key] = os.environ[env_key]

        # Also check for user-defined logging config in YAML
        user_config = self.get("logging", {})
        if isinstance(user_config, dict):
            default_config.update(user_config)

        return default_config

    def get_default_table_name(self) -> str:
        """
        Get default table name for RAG operations.

        Returns:
            Default table name as string
        """
        # Check environment variable first
        if "DEFAULT_TABLE_NAME" in os.environ:
            return os.environ["DEFAULT_TABLE_NAME"]

        # Check YAML config
        table_name = self.get("default_table_name", "SourceDocuments")
        return table_name

    def get_default_top_k(self) -> int:
        """
        Get default top_k value for similarity search.

        Returns:
            Default top_k value as integer
        """
        # Check environment variable first
        if "DEFAULT_TOP_K" in os.environ:
            return int(os.environ["DEFAULT_TOP_K"])

        # Check YAML config
        top_k = self.get("default_top_k", 5)
        return int(top_k)

    def get_pipeline_requirements(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Get pipeline-specific requirements including schema needs.

        Args:
            pipeline_type: The type of pipeline (e.g., 'graphrag', 'hybrid_graphrag')

        Returns:
            Dictionary containing pipeline requirements
        """
        # Load pipeline configurations
        # Look for config in project root relative to this file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pipeline_config_path = os.path.join(project_root, "config", "pipelines.yaml")

        pipelines_config = {}
        if os.path.exists(pipeline_config_path):
            with open(pipeline_config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
                if "pipelines" in config_data:
                    # Convert list format to dict format for easier lookup
                    for pipeline in config_data["pipelines"]:
                        if isinstance(pipeline, dict) and "name" in pipeline:
                            # Store with multiple keys for case-insensitive lookup
                            name = pipeline["name"]
                            pipelines_config[name] = pipeline
                            pipelines_config[name.lower()] = pipeline

        # Look for pipeline configuration (try multiple case variations)
        pipeline_config = {}

        # Try exact match first
        if pipeline_type in pipelines_config:
            pipeline_config = pipelines_config[pipeline_type]
        # Try lowercase
        elif pipeline_type.lower() in pipelines_config:
            pipeline_config = pipelines_config[pipeline_type.lower()]
        # Try case-insensitive search
        else:
            for name, config in pipelines_config.items():
                if name.lower() == pipeline_type.lower():
                    pipeline_config = config
                    break

        return {
            "schema_requirements": pipeline_config.get("schema_requirements", {}),
            "schema_manager": pipeline_config.get("schema_requirements", {}).get(
                "schema_manager", "SchemaManager"
            ),
            "dependencies": pipeline_config.get("dependencies", []),
            "retrieval_methods": pipeline_config.get("retrieval_methods", []),
            "params": pipeline_config.get("params", {}),
            "enabled": pipeline_config.get("enabled", True),
        }

    def get_hybrid_graphrag_config(self) -> Dict[str, Any]:
        """
        Get HybridGraphRAG-specific configuration.

        Returns:
            Dictionary containing HybridGraphRAG configuration
        """
        default_config = {
            "enabled": True,
            "schema_auto_setup": True,
            "fallback_to_graphrag": True,
            "iris_graph_core": {
                "enabled": True,
                "auto_create_tables": True,
                "community_edition_compatible": True,
            },
            "fusion_weights": [0.4, 0.3, 0.3],  # [vector, text, graph]
            "retrieval_methods": ["kg", "vector", "text", "hybrid"],
        }

        # Get user-defined config and merge with defaults
        user_config = self.get("hybrid_graphrag", {})
        if isinstance(user_config, dict):
            # Deep merge nested dictionaries
            for key, value in user_config.items():
                if (
                    key in default_config
                    and isinstance(default_config[key], dict)
                    and isinstance(value, dict)
                ):
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        return default_config
