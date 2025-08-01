"""
Environment detection utilities for RAG Templates.

This module provides utilities to detect the current execution environment
(test, development, production) and configure appropriate defaults.
"""

import os
import sys
from typing import Literal

EnvironmentType = Literal["test", "development", "production"]


def detect_environment() -> EnvironmentType:
    """
    Detect the current execution environment.
    
    Returns:
        EnvironmentType: The detected environment type
        
    Detection logic:
    1. If pytest is running -> "test"
    2. If APP_ENV environment variable is set -> use that value
    3. If CI environment variables are set -> "test" 
    4. If DEBUG_MODE is true -> "development"
    5. Default -> "production"
    """
    # Check if we're running under pytest
    if _is_pytest_running():
        return "test"
    
    # Check explicit APP_ENV setting
    app_env = os.getenv("APP_ENV", "").lower()
    if app_env in ["test", "testing"]:
        return "test"
    elif app_env in ["dev", "development"]:
        return "development"
    elif app_env in ["prod", "production"]:
        return "production"
    
    # Check CI environment indicators
    if _is_ci_environment():
        return "test"
    
    # Check debug mode
    if os.getenv("DEBUG_MODE", "false").lower() in ["true", "1", "yes"]:
        return "development"
    
    # Default to production for safety
    return "production"


def _is_pytest_running() -> bool:
    """Check if code is running under pytest."""
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def _is_ci_environment() -> bool:
    """Check if code is running in a CI environment."""
    ci_indicators = [
        "CI", "CONTINUOUS_INTEGRATION", 
        "GITLAB_CI", "GITHUB_ACTIONS", 
        "JENKINS_URL", "TRAVIS", "CIRCLECI"
    ]
    return any(os.getenv(indicator) for indicator in ci_indicators)


def get_environment_config(environment: EnvironmentType) -> dict:
    """
    Get environment-specific configuration defaults.
    
    Args:
        environment: The environment type
        
    Returns:
        dict: Configuration defaults for the environment
    """
    configs = {
        "test": {
            "daemon_error_retry_seconds": 1,
            "daemon_default_interval_seconds": 1,
            "log_level": "DEBUG",
            "enable_health_monitoring": False,
            "strict_validation": False
        },
        "development": {
            "daemon_error_retry_seconds": 30,
            "daemon_default_interval_seconds": 300,  # 5 minutes
            "log_level": "DEBUG",
            "enable_health_monitoring": True,
            "strict_validation": False
        },
        "production": {
            "daemon_error_retry_seconds": 300,  # 5 minutes
            "daemon_default_interval_seconds": 3600,  # 1 hour
            "log_level": "INFO",
            "enable_health_monitoring": True,
            "strict_validation": True
        }
    }
    
    return configs.get(environment, configs["production"])


def get_daemon_retry_interval(override_seconds: int = None) -> int:
    """
    Get the appropriate daemon error retry interval for the current environment.
    
    Args:
        override_seconds: Optional explicit override value
        
    Returns:
        int: Retry interval in seconds
    """
    if override_seconds is not None:
        return override_seconds
    
    # Check environment variable first
    env_override = os.getenv("DAEMON_ERROR_RETRY_SECONDS")
    if env_override:
        try:
            return int(env_override)
        except ValueError:
            pass
    
    # Use environment-specific default
    environment = detect_environment()
    config = get_environment_config(environment)
    return config["daemon_error_retry_seconds"]


def get_daemon_default_interval(override_seconds: int = None) -> int:
    """
    Get the appropriate daemon default interval for the current environment.
    
    Args:
        override_seconds: Optional explicit override value
        
    Returns:
        int: Default interval in seconds
    """
    if override_seconds is not None:
        return override_seconds
    
    # Check environment variable first
    env_override = os.getenv("DAEMON_DEFAULT_INTERVAL_SECONDS")
    if env_override:
        try:
            return int(env_override)
        except ValueError:
            pass
    
    # Use environment-specific default
    environment = detect_environment()
    config = get_environment_config(environment)
    return config["daemon_default_interval_seconds"]