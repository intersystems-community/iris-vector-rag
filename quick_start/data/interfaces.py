"""
Interfaces and data models for the Sample Data Manager.

This module defines the core interfaces and data structures used throughout
the sample data management system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DataSourceType(Enum):
    """Types of data sources available for sample data."""
    PMC_API = "pmc_api"
    LOCAL_CACHE = "local_cache"
    CUSTOM_SET = "custom_set"


@dataclass
class SampleDataConfig:
    """Configuration for sample data operations."""
    source_type: DataSourceType
    document_count: int
    categories: List[str]
    storage_path: Path
    cache_enabled: bool = True
    parallel_downloads: int = 4
    batch_size: int = 10
    cleanup_on_success: bool = False
    iris_edition: str = "community"


@dataclass
class DocumentMetadata:
    """Metadata for a sample document."""
    pmc_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    file_size: int
    download_url: str
    local_path: Optional[Path] = None


@dataclass
class DownloadProgress:
    """Progress tracking for download operations."""
    total_documents: int
    downloaded: int
    failed: int
    current_document: Optional[str] = None
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    estimated_time_remaining: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    document_count: int
    total_size: int


@dataclass
class IngestionResult:
    """Result of database ingestion."""
    success: bool
    documents_processed: int
    documents_ingested: int
    errors: List[str]
    processing_time: float
    database_size: int


class ISampleDataManager(ABC):
    """Primary interface for sample data management."""
    
    @abstractmethod
    async def download_samples(
        self, 
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> List[DocumentMetadata]:
        """
        Download sample documents according to configuration.
        
        Args:
            config: Download configuration
            progress_callback: Optional progress tracking callback
            
        Returns:
            List of downloaded document metadata
            
        Raises:
            DownloadError: If download fails
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def validate_samples(
        self, 
        storage_path: Path,
        strict_mode: bool = False
    ) -> ValidationResult:
        """
        Validate downloaded sample documents.
        
        Args:
            storage_path: Path to downloaded documents
            strict_mode: Enable strict validation rules
            
        Returns:
            Validation result with details
            
        Raises:
            ValidationError: If validation fails critically
        """
        pass
    
    @abstractmethod
    async def ingest_samples(
        self, 
        storage_path: Path,
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IngestionResult:
        """
        Ingest samples into IRIS database.
        
        Args:
            storage_path: Path to validated documents
            config: Ingestion configuration
            progress_callback: Optional progress tracking callback
            
        Returns:
            Ingestion result with statistics
            
        Raises:
            IngestionError: If ingestion fails
            DatabaseError: If database operations fail
        """
        pass
    
    @abstractmethod
    async def cleanup_samples(
        self, 
        storage_path: Path,
        keep_cache: bool = True
    ) -> None:
        """
        Clean up temporary sample files.
        
        Args:
            storage_path: Path to clean up
            keep_cache: Whether to preserve cache files
            
        Raises:
            CleanupError: If cleanup fails
        """
        pass
    
    @abstractmethod
    async def get_available_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of available data sources.
        
        Returns:
            List of available data source configurations
        """
        pass
    
    @abstractmethod
    async def estimate_requirements(
        self, 
        config: SampleDataConfig
    ) -> Dict[str, Any]:
        """
        Estimate resource requirements for configuration.
        
        Args:
            config: Sample data configuration
            
        Returns:
            Dictionary with estimated disk space, memory, time requirements
        """
        pass


class IDataSource(ABC):
    """Interface for data source implementations."""
    
    @abstractmethod
    async def list_available_documents(
        self, 
        categories: List[str],
        limit: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """List available documents for download."""
        pass
    
    @abstractmethod
    async def download_document(
        self, 
        metadata: DocumentMetadata,
        storage_path: Path
    ) -> Path:
        """Download a single document."""
        pass
    
    @abstractmethod
    async def verify_document(
        self, 
        metadata: DocumentMetadata,
        local_path: Path
    ) -> bool:
        """Verify downloaded document integrity."""
        pass


# Exception classes for error handling
class SampleDataError(Exception):
    """Base exception for sample data operations."""
    pass


class ConfigurationError(SampleDataError):
    """Configuration validation errors."""
    pass


class DownloadError(SampleDataError):
    """Download operation errors."""
    
    def __init__(self, message: str, failed_documents: List[str] = None):
        super().__init__(message)
        self.failed_documents = failed_documents or []


class ValidationError(SampleDataError):
    """Document validation errors."""
    
    def __init__(self, message: str, validation_details: Dict[str, Any] = None):
        super().__init__(message)
        self.validation_details = validation_details or {}


class IngestionError(SampleDataError):
    """Database ingestion errors."""
    
    def __init__(self, message: str, processed_count: int = 0):
        super().__init__(message)
        self.processed_count = processed_count


class StorageError(SampleDataError):
    """File system storage errors."""
    pass


class CleanupError(SampleDataError):
    """Cleanup operation errors."""
    pass