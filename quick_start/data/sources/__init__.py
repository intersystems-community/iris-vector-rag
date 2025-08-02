"""
Data source implementations for the Quick Start system.

This module contains various data source implementations for downloading
sample documents from different sources.
"""

from quick_start.data.sources.pmc_api import PMCAPIDataSource
from quick_start.data.sources.local_cache import LocalCacheDataSource
from quick_start.data.sources.custom_set import CustomSetDataSource

__all__ = [
    "PMCAPIDataSource",
    "LocalCacheDataSource", 
    "CustomSetDataSource",
]