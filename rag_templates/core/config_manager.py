"""
Enhanced Configuration Manager for RAG Templates Library Consumption Framework.

This module provides a three-tier configuration system:
1. Built-in defaults for zero-config operation
2. Configuration file support
3. Environment variable integration

The configuration manager ensures no hard-coded secrets and provides
sensible defaults for immediate usability.
"""

import os
import yaml
import logging
from typing import Any, Optional, Dict, Union
from .errors import ConfigurationError, handle_configuration_fallback

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Enhanced Configuration Manager with three-tier configuration system.
    
    This manager provides:
    1. Built-in defaults for zero-configuration operation
    2. Configuration file loading and parsing
    3. Environment variable override support
    4. Configuration validation and error handling
    5. Fallback strategies for missing configuration
    """
    
    ENV_PREFIX = "RAG_"
    DELIMITER = "__"  # Double underscore for nesting in env vars
    
    def __init__(self, config_path: Optional[str] = None, 
                 schema: Optional[Dict] = None):
        """
        Initialize the Enhanced Configuration Manager.
        
        Args:
            config_path: Optional path to configuration file
            schema: Optional schema for configuration validation
        """
        self._config: Dict[str, Any] = {}
        self._schema = schema
        self._defaults_loaded = False
        
        # Load configuration in order of precedence
        self._load_builtin_defaults()
        
        if config_path:
            self._load_config_file(config_path)
        else:
            self._load_default_config_file()
        
        self._load_env_variables()
        
        logger.info("Configuration manager initialized successfully")
    
    def _load_builtin_defaults(self) -> None:
        """
        Load built-in default configuration for zero-config operation.
        
        These defaults ensure the system can operate without any configuration
        while avoiding hard-coded secrets.
        """
        self._config = {
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": 1972,
                    "namespace": "USER",
                    "username": None,  # No default username - must be provided
                    "password": None,  # No default password - must be provided
                    "timeout": 30,
                    "pool_size": 5
                }
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "provider": "sentence-transformers",
                "batch_size": 32,
                "normalize": True
            },
            "llm": {
                "provider": None,  # No default LLM provider
                "model": None,     # No default model
                "api_key": None,   # No default API key
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "pipelines": {
                "basic": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "default_top_k": 5,
                    "embedding_batch_size": 32
                }
            },
            "vector_index": {
                "type": "HNSW",
                "M": 16,
                "efConstruction": 200,
                "Distance": "COSINE"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        self._defaults_loaded = True
        logger.debug("Built-in defaults loaded successfully")
    
    def _load_config_file(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            if not os.path.exists(config_path):
                raise ConfigurationError(
                    f"Configuration file not found: {config_path}",
                    details={"config_path": config_path}
                )
            
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Deep merge with existing configuration
            self._deep_merge(self._config, file_config)
            logger.info(f"Configuration loaded from: {config_path}")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration file: {config_path}",
                details={"yaml_error": str(e), "config_path": config_path}
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {config_path}",
                details={"error": str(e), "config_path": config_path}
            ) from e
    
    def _load_default_config_file(self) -> None:
        """
        Attempt to load default configuration files.
        
        Looks for configuration files in standard locations without
        raising errors if they don't exist.
        """
        default_paths = [
            "config.yaml",
            "config/config.yaml",
            os.path.join(os.path.expanduser("~"), ".rag_templates", "config.yaml")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    self._load_config_file(path)
                    logger.info(f"Loaded default configuration from: {path}")
                    break
                except ConfigurationError:
                    logger.debug(f"Failed to load default config from: {path}")
                    continue
    
    def _load_env_variables(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables override both defaults and file configuration.
        Variables should be prefixed with RAG_ and use double underscores
        for nesting (e.g., RAG_DATABASE__IRIS__HOST).
        """
        env_overrides = {}
        
        for env_var, value in os.environ.items():
            if env_var.startswith(self.ENV_PREFIX):
                # Remove prefix and split by delimiter
                key_path_str = env_var[len(self.ENV_PREFIX):]
                keys = [k.lower() for k in key_path_str.split(self.DELIMITER)]
                
                # Set the value in the nested structure
                self._set_nested_value(env_overrides, keys, value)
        
        if env_overrides:
            # Deep merge environment overrides
            self._deep_merge(self._config, env_overrides)
            logger.debug(f"Applied {len(env_overrides)} environment variable overrides")
    
    def _set_nested_value(self, config_dict: Dict, keys: list, value: str) -> None:
        """
        Set a value in a nested dictionary structure.
        
        Args:
            config_dict: The dictionary to modify
            keys: List of keys representing the path
            value: The value to set
        """
        current_level = config_dict
        
        for i, key_part in enumerate(keys):
            if i == len(keys) - 1:  # Last key part
                # Try to cast to appropriate type
                casted_value = self._cast_env_value(value, keys)
                current_level[key_part] = casted_value
            else:
                # Ensure we have a dict at this level
                if key_part not in current_level:
                    current_level[key_part] = {}
                elif not isinstance(current_level[key_part], dict):
                    current_level[key_part] = {}
                current_level = current_level[key_part]
    
    def _cast_env_value(self, value_str: str, keys: list) -> Any:
        """
        Cast environment variable string to appropriate type.
        
        Args:
            value_str: The string value from environment variable
            keys: The key path for context
            
        Returns:
            The value cast to appropriate type
        """
        # Get the original value type for reference
        original_value = self._get_nested_value(self._config, keys)
        target_type = type(original_value) if original_value is not None else None
        
        try:
            if target_type == bool:
                return value_str.lower() in ("true", "1", "yes", "on")
            elif target_type == int:
                return int(value_str)
            elif target_type == float:
                return float(value_str)
            elif target_type == list:
                # Simple comma-separated list support
                return [item.strip() for item in value_str.split(",")]
            else:
                # Return as string for unknown types or None target type
                return value_str
        except (ValueError, TypeError):
            logger.warning(f"Failed to cast environment variable value: {value_str}")
            return value_str
    
    def _get_nested_value(self, config_dict: Dict, keys: list) -> Any:
        """
        Get a value from a nested dictionary structure.
        
        Args:
            config_dict: The dictionary to search
            keys: List of keys representing the path
            
        Returns:
            The value if found, None otherwise
        """
        current = config_dict
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Deep merge two dictionaries, modifying base_dict in place.
        
        Args:
            base_dict: The base dictionary to merge into
            update_dict: The dictionary to merge from
        """
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_string: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a configuration setting using dot notation.
        
        Args:
            key_string: The configuration key (e.g., "database:iris:host")
            default: The default value if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = [k.lower() for k in key_string.split(':')]
        value = self._get_nested_value(self._config, keys)
        
        if value is None:
            logger.debug(f"Configuration key not found: {key_string}, using default: {default}")
            return default
        
        return value
    
    def set(self, key_string: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_string: The configuration key (e.g., "database:iris:host")
            value: The value to set
        """
        keys = [k.lower() for k in key_string.split(':')]
        self._set_nested_value(self._config, keys, str(value))
        logger.debug(f"Configuration set: {key_string} = {value}")
    
    def validate(self) -> None:
        """
        Validate the current configuration.
        
        Raises:
            ConfigurationError: If validation fails
        """
        if not self._defaults_loaded:
            raise ConfigurationError("Configuration defaults not loaded")
        
        # Basic validation - ensure critical paths exist
        critical_paths = [
            "database:iris:host",
            "database:iris:port",
            "embeddings:model"
        ]
        
        for path in critical_paths:
            value = self.get(path)
            if value is None:
                raise ConfigurationError(
                    f"Missing required configuration: {path}",
                    config_key=path
                )
        
        # Validate data types
        self._validate_types()
        
        logger.info("Configuration validation passed")
    
    def _validate_types(self) -> None:
        """Validate configuration value types."""
        type_validations = [
            ("database:iris:port", int),
            ("database:iris:timeout", (int, float)),
            ("embeddings:dimension", int),
            ("embeddings:batch_size", int),
            ("pipelines:basic:chunk_size", int),
            ("pipelines:basic:chunk_overlap", int)
        ]
        
        for key, expected_type in type_validations:
            value = self.get(key)
            if value is not None and not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Invalid type for {key}: expected {expected_type}, got {type(value)}",
                    config_key=key,
                    details={"expected_type": str(expected_type), "actual_type": str(type(value))}
                )
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration with validation."""
        config = {
            "host": self.get("database:iris:host"),
            "port": self.get("database:iris:port"),
            "namespace": self.get("database:iris:namespace"),
            "username": self.get("database:iris:username"),
            "password": self.get("database:iris:password"),
            "timeout": self.get("database:iris:timeout"),
            "pool_size": self.get("database:iris:pool_size")
        }
        
        # Validate required fields
        if not config["host"]:
            raise ConfigurationError("Database host is required", config_key="database:iris:host")
        if not config["port"]:
            raise ConfigurationError("Database port is required", config_key="database:iris:port")
        
        return config
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return {
            "model": self.get("embeddings:model"),
            "dimension": self.get("embeddings:dimension"),
            "provider": self.get("embeddings:provider"),
            "batch_size": self.get("embeddings:batch_size"),
            "normalize": self.get("embeddings:normalize")
        }
    
    def get_pipeline_config(self, pipeline_name: str = "basic") -> Dict[str, Any]:
        """Get pipeline-specific configuration."""
        base_key = f"pipelines:{pipeline_name}"
        return {
            "chunk_size": self.get(f"{base_key}:chunk_size"),
            "chunk_overlap": self.get(f"{base_key}:chunk_overlap"),
            "default_top_k": self.get(f"{base_key}:default_top_k"),
            "embedding_batch_size": self.get(f"{base_key}:embedding_batch_size")
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the complete configuration as a dictionary."""
        return self._config.copy()