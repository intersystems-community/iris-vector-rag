"""
LLM Cache Configuration Module

This module provides configuration management for the LLM caching layer,
supporting both YAML file configuration and environment variable overrides.
"""

import os
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration model for LLM caching."""

    # Core settings
    enabled: bool = True
    backend: str = "iris"  # memory, iris, redis
    ttl_seconds: int = 3600
    normalize_prompts: bool = False
    max_cache_size: int = 1000

    # IRIS-specific settings
    table_name: str = "llm_cache"
    iris_schema: str = "RAG"
    connection_timeout: int = 30
    cleanup_batch_size: int = 1000
    auto_cleanup: bool = True
    cleanup_interval: int = 86400

    # Key generation settings
    include_temperature: bool = True
    include_max_tokens: bool = True
    include_model_name: bool = True
    hash_algorithm: str = "sha256"
    normalize_whitespace: bool = True
    normalize_case: bool = False

    # Monitoring settings
    monitoring_enabled: bool = True
    log_operations: bool = False
    track_stats: bool = True
    metrics_interval: int = 300

    # Error handling settings
    graceful_fallback: bool = True
    max_retries: int = 3
    retry_delay: int = 1
    operation_timeout: int = 10

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if "LLM_CACHE_ENABLED" in os.environ:
            config.enabled = os.environ["LLM_CACHE_ENABLED"].lower() in (
                "true",
                "1",
                "yes",
            )

        if "LLM_CACHE_BACKEND" in os.environ:
            config.backend = os.environ["LLM_CACHE_BACKEND"]

        if "LLM_CACHE_TTL" in os.environ:
            config.ttl_seconds = int(os.environ["LLM_CACHE_TTL"])

        if "LLM_CACHE_TABLE" in os.environ:
            config.table_name = os.environ["LLM_CACHE_TABLE"]

        if "LLM_CACHE_NORMALIZE_PROMPTS" in os.environ:
            config.normalize_prompts = os.environ[
                "LLM_CACHE_NORMALIZE_PROMPTS"
            ].lower() in ("true", "1", "yes")

        if "LLM_CACHE_MAX_SIZE" in os.environ:
            config.max_cache_size = int(os.environ["LLM_CACHE_MAX_SIZE"])

        if "LLM_CACHE_IRIS_SCHEMA" in os.environ:
            config.iris_schema = os.environ["LLM_CACHE_IRIS_SCHEMA"]

        logger.info(
            f"Cache configuration loaded from environment: backend={config.backend}, enabled={config.enabled}"
        )
        return config

    @classmethod
    def from_yaml(cls, config_path: str = "config/cache_config.yaml") -> "CacheConfig":
        """Create configuration from YAML file with environment variable overrides."""
        config = cls()

        # Load from YAML file if it exists
        yaml_path = Path(config_path)
        if yaml_path.exists():
            try:
                with open(yaml_path, "r") as f:
                    yaml_data = yaml.safe_load(f)

                if "llm_cache" in yaml_data:
                    cache_data = yaml_data["llm_cache"]

                    # Core settings
                    config.enabled = cache_data.get("enabled", config.enabled)
                    config.backend = cache_data.get("backend", config.backend)
                    config.ttl_seconds = cache_data.get(
                        "ttl_seconds", config.ttl_seconds
                    )
                    config.normalize_prompts = cache_data.get(
                        "normalize_prompts", config.normalize_prompts
                    )
                    config.max_cache_size = cache_data.get(
                        "max_cache_size", config.max_cache_size
                    )

                    # IRIS settings
                    if "iris" in cache_data:
                        iris_data = cache_data["iris"]
                        config.table_name = iris_data.get(
                            "table_name", config.table_name
                        )
                        config.iris_schema = iris_data.get("schema", config.iris_schema)
                        config.connection_timeout = iris_data.get(
                            "connection_timeout", config.connection_timeout
                        )
                        config.cleanup_batch_size = iris_data.get(
                            "cleanup_batch_size", config.cleanup_batch_size
                        )
                        config.auto_cleanup = iris_data.get(
                            "auto_cleanup", config.auto_cleanup
                        )
                        config.cleanup_interval = iris_data.get(
                            "cleanup_interval", config.cleanup_interval
                        )

                    # Key generation settings
                    if "key_generation" in cache_data:
                        key_data = cache_data["key_generation"]
                        config.include_temperature = key_data.get(
                            "include_temperature", config.include_temperature
                        )
                        config.include_max_tokens = key_data.get(
                            "include_max_tokens", config.include_max_tokens
                        )
                        config.include_model_name = key_data.get(
                            "include_model_name", config.include_model_name
                        )
                        config.hash_algorithm = key_data.get(
                            "hash_algorithm", config.hash_algorithm
                        )
                        config.normalize_whitespace = key_data.get(
                            "normalize_whitespace", config.normalize_whitespace
                        )
                        config.normalize_case = key_data.get(
                            "normalize_case", config.normalize_case
                        )

                    # Monitoring settings
                    if "monitoring" in cache_data:
                        monitor_data = cache_data["monitoring"]
                        config.monitoring_enabled = monitor_data.get(
                            "enabled", config.monitoring_enabled
                        )
                        config.log_operations = monitor_data.get(
                            "log_operations", config.log_operations
                        )
                        config.track_stats = monitor_data.get(
                            "track_stats", config.track_stats
                        )
                        config.metrics_interval = monitor_data.get(
                            "metrics_interval", config.metrics_interval
                        )

                    # Error handling settings
                    if "error_handling" in cache_data:
                        error_data = cache_data["error_handling"]
                        config.graceful_fallback = error_data.get(
                            "graceful_fallback", config.graceful_fallback
                        )
                        config.max_retries = error_data.get(
                            "max_retries", config.max_retries
                        )
                        config.retry_delay = error_data.get(
                            "retry_delay", config.retry_delay
                        )
                        config.operation_timeout = error_data.get(
                            "operation_timeout", config.operation_timeout
                        )

                logger.info(f"Cache configuration loaded from YAML: {config_path}")

            except Exception as e:
                logger.warning(
                    f"Failed to load cache config from {config_path}: {e}. Using defaults."
                )
        else:
            logger.info(
                f"Cache config file not found at {config_path}. Using defaults."
            )

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config)

        return config

    @classmethod
    def _apply_env_overrides(cls, config: "CacheConfig") -> "CacheConfig":
        """Apply environment variable overrides to configuration."""
        # Override with environment variables if present
        if "LLM_CACHE_ENABLED" in os.environ:
            config.enabled = os.environ["LLM_CACHE_ENABLED"].lower() in (
                "true",
                "1",
                "yes",
            )

        if "LLM_CACHE_BACKEND" in os.environ:
            config.backend = os.environ["LLM_CACHE_BACKEND"]

        if "LLM_CACHE_TTL" in os.environ:
            config.ttl_seconds = int(os.environ["LLM_CACHE_TTL"])

        if "LLM_CACHE_TABLE" in os.environ:
            config.table_name = os.environ["LLM_CACHE_TABLE"]

        if "LLM_CACHE_NORMALIZE_PROMPTS" in os.environ:
            config.normalize_prompts = os.environ[
                "LLM_CACHE_NORMALIZE_PROMPTS"
            ].lower() in ("true", "1", "yes")

        if "LLM_CACHE_MAX_SIZE" in os.environ:
            config.max_cache_size = int(os.environ["LLM_CACHE_MAX_SIZE"])

        if "LLM_CACHE_REDIS_URL" in os.environ:
            config.redis_url = os.environ["LLM_CACHE_REDIS_URL"]

        if "LLM_CACHE_IRIS_SCHEMA" in os.environ:
            config.iris_schema = os.environ["LLM_CACHE_IRIS_SCHEMA"]

        return config

    def validate(self) -> bool:
        """Validate the configuration settings."""
        if self.backend not in ["memory", "iris"]:
            logger.error(
                f"Invalid cache backend: {self.backend}. Supported backends: memory, iris"
            )
            return False

        if self.ttl_seconds <= 0:
            logger.error(f"Invalid TTL: {self.ttl_seconds}")
            return False

        if self.max_cache_size <= 0:
            logger.error(f"Invalid max cache size: {self.max_cache_size}")
            return False

        if self.hash_algorithm not in ["sha256", "md5", "sha1"]:
            logger.error(f"Invalid hash algorithm: {self.hash_algorithm}")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "ttl_seconds": self.ttl_seconds,
            "normalize_prompts": self.normalize_prompts,
            "max_cache_size": self.max_cache_size,
            "table_name": self.table_name,
            "iris_schema": self.iris_schema,
            "connection_timeout": self.connection_timeout,
            "redis_url": self.redis_url,
            "redis_prefix": self.redis_prefix,
            "monitoring_enabled": self.monitoring_enabled,
            "graceful_fallback": self.graceful_fallback,
        }


def load_cache_config(config_path: Optional[str] = None) -> CacheConfig:
    """
    Load cache configuration from YAML file with environment overrides.

    Args:
        config_path: Path to YAML config file. Defaults to 'config/cache_config.yaml'

    Returns:
        CacheConfig instance
    """
    if config_path is None:
        config_path = "config/cache_config.yaml"

    config = CacheConfig.from_yaml(config_path)

    if not config.validate():
        logger.warning("Cache configuration validation failed. Using safe defaults.")
        config = CacheConfig()  # Use defaults

    return config
