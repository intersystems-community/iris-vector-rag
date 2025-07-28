"""
PMC Document Downloader Package

Enterprise-scale PMC document downloader for RAG Templates system.
Provides comprehensive downloading, validation, and integration capabilities
for real medical/scientific documents from the PMC Open Access Subset.

Key Components:
- PMCAPIClient: Interface to PMC APIs and FTP services
- PMCBatchDownloader: Batch downloading with progress tracking
- PMCEnterpriseLoader: Integration with UnifiedDocumentLoader
"""

from .api_client import PMCAPIClient
from .batch_downloader import PMCBatchDownloader, DownloadProgress, ValidationResult
from .integration import PMCEnterpriseLoader, create_enterprise_loader_config, load_enterprise_pmc_dataset

__all__ = [
    'PMCAPIClient',
    'PMCBatchDownloader', 
    'DownloadProgress',
    'ValidationResult',
    'PMCEnterpriseLoader',
    'create_enterprise_loader_config',
    'load_enterprise_pmc_dataset'
]

__version__ = '1.0.0'