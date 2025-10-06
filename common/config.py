"""
Common configuration module for IRIS RAG system
Provides unified access to configuration settings
"""

import os
import sys
from pathlib import Path

# Add config path to sys.path
config_dir = Path(__file__).parent.parent / "config"
sys.path.insert(0, str(config_dir))

from loader import get_config


def get_iris_config():
    """Get IRIS database configuration"""
    config = get_config()
    if not config:
        raise ValueError("Failed to load configuration")

    db_config = config.get("database", {})
    return {
        "host": db_config.get("db_host", "localhost"),
        "port": db_config.get("db_port", 1974),
        "username": db_config.get("db_user", "SuperUser"),
        "password": db_config.get("db_password", "SYS"),
        "namespace": db_config.get("db_namespace", "USER"),
        "connection_type": "dbapi",  # Default to DBAPI
    }


def get_embedding_config():
    """Get embedding model configuration"""
    config = get_config()
    if not config:
        raise ValueError("Failed to load configuration")

    return config.get("embedding_model", {})


def get_storage_config():
    """Get storage configuration"""
    config = get_config()
    if not config:
        raise ValueError("Failed to load configuration")

    return config.get("storage", {})


def get_pipeline_config():
    """Get pipeline configuration"""
    config = get_config()
    if not config:
        raise ValueError("Failed to load configuration")

    return config.get("pipelines", {})
