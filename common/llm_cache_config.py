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
    
    def __init__(self, config_path: Optional[str] = None):
        """Initializes the CacheConfig, loading from YAML and overriding with environment variables."""
        if config_path is None:
            config_path = 'config/cache_config.yaml'
        
        # Load from YAML file if it exists
        yaml_path = Path(config_path)
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                if 'cache' in yaml_data:
                    cache_data = yaml_data['cache']
                    
                    # Core settings
                    self.enabled = cache_data.get('enabled', self.enabled)
                    self.backend = cache_data.get('backend', self.backend)
                    self.ttl_seconds = cache_data.get('ttl_seconds', self.ttl_seconds)
                    
                    # IRIS settings
                    if 'iris' in cache_data:
                        iris_data = cache_data['iris']
                        self.table_name = iris_data.get('table_name', self.table_name)

                logger.info(f"Cache configuration loaded from YAML: {config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load cache config from {config_path}: {e}. Using defaults.")
        else:
            logger.info(f"Cache config file not found at {config_path}. Using defaults.")
        
        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        self.enabled = os.environ.get('RAG_CACHE__ENABLED', str(self.enabled)).lower() in ('true', '1', 'yes')
        self.backend = os.environ.get('RAG_CACHE__BACKEND', self.backend)
        self.ttl_seconds = int(os.environ.get('RAG_CACHE__TTL_SECONDS', self.ttl_seconds))
        self.table_name = os.environ.get('RAG_CACHE__IRIS__TABLE_NAME', self.table_name)
    
    def validate(self) -> bool:
        """Validate the configuration settings."""
        if self.backend not in ['memory', 'iris']:
            logger.error(f"Invalid cache backend: {self.backend}. Supported backends: memory, iris")
            return False
        
        if self.ttl_seconds <= 0:
            logger.error(f"Invalid TTL: {self.ttl_seconds}")
            return False
        
        if self.max_cache_size <= 0:
            logger.error(f"Invalid max cache size: {self.max_cache_size}")
            return False
        
        if self.hash_algorithm not in ['sha256', 'md5', 'sha1']:
            logger.error(f"Invalid hash algorithm: {self.hash_algorithm}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'ttl_seconds': self.ttl_seconds,
            'normalize_prompts': self.normalize_prompts,
            'max_cache_size': self.max_cache_size,
            'table_name': self.table_name,
            'iris_schema': self.iris_schema,
            'connection_timeout': self.connection_timeout,
            'redis_url': self.redis_url,
            'redis_prefix': self.redis_prefix,
            'monitoring_enabled': self.monitoring_enabled,
            'graceful_fallback': self.graceful_fallback
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
        config_path = 'config/cache_config.yaml'
    
    config = CacheConfig.from_yaml(config_path)
    
    if not config.validate():
        logger.warning("Cache configuration validation failed. Using safe defaults.")
        config = CacheConfig()  # Use defaults
    
    return config