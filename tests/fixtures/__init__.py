"""
Unified test fixture infrastructure.

This package provides a unified interface for managing test fixtures across
.DAT, JSON, and programmatic sources.
"""

from .manager import (
    FixtureManager,
    FixtureError,
    FixtureNotFoundError,
    ChecksumMismatchError,
    IncompatibleVersionError,
    FixtureLoadError,
)
from .models import FixtureMetadata, FixtureManifest, FixtureLoadResult, FixtureSourceType
from .embedding_generator import EmbeddingGenerator, ModelLoadError, DimensionMismatchError

__all__ = [
    # Manager
    "FixtureManager",
    # Models
    "FixtureMetadata",
    "FixtureManifest",
    "FixtureLoadResult",
    "FixtureSourceType",
    # Embedding Generator
    "EmbeddingGenerator",
    # Exceptions
    "FixtureError",
    "FixtureNotFoundError",
    "ChecksumMismatchError",
    "IncompatibleVersionError",
    "FixtureLoadError",
    "ModelLoadError",
    "DimensionMismatchError",
]