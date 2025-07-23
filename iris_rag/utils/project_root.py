"""
Project Root Detection Utility.

This module provides utilities for reliably determining the project root directory
regardless of the current working directory.
"""

from pathlib import Path
from typing import Optional


def get_project_root(marker_files: Optional[list] = None) -> Path:
    """
    Determine the project root directory by looking for marker files.
    
    This function searches up the directory tree from the current file location
    to find the project root, identified by the presence of marker files.
    
    Args:
        marker_files: List of files/directories that indicate project root.
                     Defaults to common project markers.
    
    Returns:
        Path object pointing to the project root directory
        
    Raises:
        RuntimeError: If project root cannot be determined
    """
    if marker_files is None:
        # Use markers that are unique to the project root, not subdirectories
        marker_files = [
            '.git',
            'pyproject.toml',
            'setup.py',
            'requirements.txt'
        ]
    
    # Start from the directory containing this file
    current_path = Path(__file__).resolve().parent
    
    # Walk up the directory tree
    for parent in [current_path] + list(current_path.parents):
        # Check if any marker files exist in this directory
        for marker in marker_files:
            marker_path = parent / marker
            if marker_path.exists():
                return parent
    
    # If we can't find standard markers, use a more specific approach
    # This file is in iris_rag/utils/, so project root should be 2 levels up
    fallback_root = Path(__file__).resolve().parent.parent.parent
    
    # Verify the fallback makes sense by checking for both iris_rag and config directories
    if (fallback_root / 'iris_rag').exists() and (fallback_root / 'config').exists():
        return fallback_root
    
    raise RuntimeError(
        f"Could not determine project root. Searched for markers: {marker_files}. "
        f"Current file location: {Path(__file__).resolve()}"
    )


def get_config_path(config_filename: str) -> Path:
    """
    Get the absolute path to a configuration file in the project's config directory.
    
    Args:
        config_filename: Name of the configuration file (e.g., 'pipelines.yaml')
        
    Returns:
        Absolute path to the configuration file
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    project_root = get_project_root()
    config_path = project_root / 'config' / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Project root detected as: {project_root}"
        )
    
    return config_path


def resolve_project_relative_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root, regardless of current working directory.
    
    Args:
        relative_path: Path relative to project root (e.g., 'config/pipelines.yaml')
                      or absolute path (which will be returned as-is)
        
    Returns:
        Absolute path resolved from project root
    """
    path_obj = Path(relative_path)
    
    # If it's already an absolute path, return it as-is
    if path_obj.is_absolute():
        return path_obj
    
    # Otherwise, resolve relative to project root
    project_root = get_project_root()
    return project_root / relative_path