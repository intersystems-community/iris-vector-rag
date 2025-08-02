"""
Sample data management for the Quick Start system.

This module provides automated management of sample PMC documents,
including downloading, validation, caching, and ingestion into IRIS database.
"""

from quick_start.data.sample_manager import SampleDataManager
from quick_start.data.interfaces import (
    ISampleDataManager,
    IDataSource,
    SampleDataConfig,
    DocumentMetadata,
    DownloadProgress,
    ValidationResult,
    IngestionResult,
)

__all__ = [
    "SampleDataManager",
    "ISampleDataManager",
    "IDataSource", 
    "SampleDataConfig",
    "DocumentMetadata",
    "DownloadProgress",
    "ValidationResult",
    "IngestionResult",
]