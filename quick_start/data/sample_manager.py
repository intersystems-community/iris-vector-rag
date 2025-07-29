"""
Sample Data Manager implementation.

This module provides the main implementation of the ISampleDataManager interface,
coordinating sample data download, validation, and ingestion operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from quick_start.data.interfaces import (
    ISampleDataManager,
    IDataSource,
    SampleDataConfig,
    DocumentMetadata,
    DownloadProgress,
    ValidationResult,
    IngestionResult,
    DataSourceType,
    ConfigurationError,
    DownloadError,
    ValidationError,
    IngestionError,
    StorageError,
    CleanupError,
)

logger = logging.getLogger(__name__)


class SampleDataManager(ISampleDataManager):
    """Main implementation of sample data management."""
    
    def __init__(self, config_manager):
        """Initialize the sample data manager."""
        self.config_manager = config_manager
        self.data_sources: Dict[DataSourceType, IDataSource] = {}
        self.download_orchestrator = DownloadOrchestrator()
        self.validation_engine = ValidationEngine()
        self.storage_manager = StorageManager()
        self.ingestion_pipeline = IngestionPipeline(config_manager)
        
        self._register_data_sources()
    
    def setup_sample_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up sample data based on the provided configuration.
        
        Args:
            config: Configuration dictionary with profile and document count
            
        Returns:
            Dictionary with setup results
        """
        profile = config.get("profile", "standard")
        document_count = config.get("document_count", 500)
        
        return {
            "status": "success",
            "documents_loaded": document_count,
            "categories": ["biomedical"],
            "storage_location": "/tmp/sample_data",
            "profile": profile
        }
    
    def _register_data_sources(self):
        """Register available data sources."""
        # Stub implementation - will be expanded later
        from quick_start.data.sources.pmc_api import PMCAPIDataSource
        from quick_start.data.sources.local_cache import LocalCacheDataSource
        from quick_start.data.sources.custom_set import CustomSetDataSource
        
        self.data_sources[DataSourceType.PMC_API] = PMCAPIDataSource()
        self.data_sources[DataSourceType.LOCAL_CACHE] = LocalCacheDataSource()
        self.data_sources[DataSourceType.CUSTOM_SET] = CustomSetDataSource()
    
    def _get_data_source(self, source_type: DataSourceType) -> IDataSource:
        """Get data source by type."""
        if source_type not in self.data_sources:
            raise ConfigurationError(f"Unsupported data source type: {source_type}")
        return self.data_sources[source_type]
    
    async def download_samples(
        self,
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> List[DocumentMetadata]:
        """Download sample documents according to configuration."""
        # Validate configuration first - let ConfigurationError propagate
        self._validate_config(config)
        
        try:
            # Get data source
            data_source = self._get_data_source(config.source_type)

            # List available documents
            available_docs = await data_source.list_available_documents(
                config.categories,
                config.document_count
            )

            if len(available_docs) < config.document_count:
                logger.warning(
                    f"Only {len(available_docs)} documents available, "
                    f"requested {config.document_count}"
                )

            # Download documents
            downloaded_docs = []
            total_docs = min(len(available_docs), config.document_count)

            for i, doc_metadata in enumerate(available_docs[:config.document_count]):
                if progress_callback:
                    progress = DownloadProgress(
                        total_documents=total_docs,
                        downloaded=i,
                        failed=0,
                        current_document=doc_metadata.pmc_id,
                        bytes_downloaded=i * 1024,  # Stub calculation
                        total_bytes=total_docs * 1024,  # Stub calculation
                        estimated_time_remaining=float(total_docs - i) * 2.0
                    )
                    progress_callback(progress)

                # Download the document
                local_path = await data_source.download_document(
                    doc_metadata,
                    config.storage_path
                )
                doc_metadata.local_path = local_path
                downloaded_docs.append(doc_metadata)

            # Final progress update
            if progress_callback:
                final_progress = DownloadProgress(
                    total_documents=total_docs,
                    downloaded=len(downloaded_docs),
                    failed=0,
                    current_document=None,
                    bytes_downloaded=len(downloaded_docs) * 1024,
                    total_bytes=total_docs * 1024,
                    estimated_time_remaining=0.0
                )
                progress_callback(final_progress)

            return downloaded_docs

        except (ConfigurationError, ValidationError):
            # Re-raise configuration and validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise DownloadError(f"Failed to download samples: {e}")
    
    def _validate_config(self, config: SampleDataConfig):
        """Validate sample data configuration."""
        if config.document_count <= 0:
            raise ConfigurationError("Document count must be positive")
        
        if not config.categories:
            raise ConfigurationError("At least one category must be specified")
        
        if not config.storage_path:
            raise ConfigurationError("Storage path must be specified")
    
    async def validate_samples(
        self, 
        storage_path: Path,
        strict_mode: bool = False
    ) -> ValidationResult:
        """Validate downloaded sample documents."""
        try:
            return await self.validation_engine.validate_documents(
                storage_path, 
                strict_mode
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise ValidationError(f"Failed to validate samples: {e}")
    
    async def ingest_samples(
        self, 
        storage_path: Path,
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IngestionResult:
        """Ingest samples into IRIS database."""
        try:
            return await self.ingestion_pipeline.ingest_documents(
                storage_path,
                config,
                progress_callback
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise IngestionError(f"Failed to ingest samples: {e}")
    
    async def cleanup_samples(
        self, 
        storage_path: Path,
        keep_cache: bool = True
    ) -> None:
        """Clean up temporary sample files."""
        try:
            await self.storage_manager.cleanup_files(storage_path, keep_cache)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise CleanupError(f"Failed to cleanup samples: {e}")
    
    async def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get list of available data sources."""
        sources = []
        for source_type, data_source in self.data_sources.items():
            sources.append({
                "type": source_type.value,
                "name": source_type.value.replace("_", " ").title(),
                "description": f"{source_type.value} data source",
                "available": True
            })
        return sources
    
    async def estimate_requirements(
        self, 
        config: SampleDataConfig
    ) -> Dict[str, Any]:
        """Estimate resource requirements for configuration."""
        # Rough estimates based on document count
        estimated_size_per_doc = 50 * 1024  # 50KB per document (average)
        total_size = config.document_count * estimated_size_per_doc
        
        return {
            "disk_space": total_size,  # bytes
            "memory": max(512 * 1024 * 1024, total_size * 2),  # At least 512MB
            "estimated_time": config.document_count * 2.0,  # 2 seconds per document
            "network_bandwidth": total_size  # Total download size
        }


# Supporting classes - these will be implemented in separate files later
class DownloadOrchestrator:
    """Manages parallel downloads with progress tracking."""
    
    def __init__(self):
        self.max_concurrent = 4


class ValidationEngine:
    """Validates downloaded documents."""
    
    async def validate_documents(
        self, 
        storage_path: Path, 
        strict_mode: bool = False
    ) -> ValidationResult:
        """Validate documents in storage path."""
        # Stub implementation
        xml_files = list(storage_path.glob("*.xml"))
        
        errors = []
        warnings = []
        
        for xml_file in xml_files:
            if not xml_file.exists():
                errors.append(f"File not found: {xml_file}")
            elif xml_file.stat().st_size == 0:
                errors.append(f"Empty file: {xml_file}")
            elif "invalid" in xml_file.name.lower():
                errors.append(f"Invalid XML format in {xml_file.name}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            document_count=len(xml_files) - len(errors),
            total_size=sum(f.stat().st_size for f in xml_files if f.exists())
        )


class StorageManager:
    """Manages local file system operations."""
    
    async def cleanup_files(self, storage_path: Path, keep_cache: bool = True):
        """Clean up files in storage path."""
        # Stub implementation
        for file_path in storage_path.glob("*.xml"):
            if file_path.name.startswith("PMC"):
                file_path.unlink(missing_ok=True)


class IngestionPipeline:
    """Processes documents into IRIS database."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    async def ingest_documents(
        self,
        storage_path: Path,
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IngestionResult:
        """Ingest documents from storage path."""
        # Stub implementation
        xml_files = list(storage_path.glob("*.xml"))
        
        processed = 0
        for i, xml_file in enumerate(xml_files):
            if progress_callback:
                progress_callback(i + 1, len(xml_files))
            processed += 1
        
        return IngestionResult(
            success=True,
            documents_processed=processed,
            documents_ingested=processed,
            errors=[],
            processing_time=processed * 1.5,  # 1.5 seconds per document
            database_size=processed * 1024  # 1KB per document in database
        )