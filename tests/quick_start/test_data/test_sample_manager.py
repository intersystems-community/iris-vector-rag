"""
Tests for the Sample Data Manager.

This module contains comprehensive tests for the SampleDataManager class,
following TDD principles and testing all core functionality.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from quick_start.data.interfaces import (
    SampleDataConfig,
    DataSourceType,
    DocumentMetadata,
    DownloadProgress,
    ValidationResult,
    IngestionResult,
    ConfigurationError,
    DownloadError,
    ValidationError,
    IngestionError,
)


class TestSampleDataManager:
    """Test cases for SampleDataManager class."""

    @pytest.mark.asyncio
    async def test_sample_data_manager_initialization(self, mock_config_manager):
        """Test that SampleDataManager can be initialized properly."""
        # This test will fail initially since we haven't implemented SampleDataManager yet
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        assert manager is not None
        assert manager.config_manager == mock_config_manager
        assert hasattr(manager, 'data_sources')
        assert hasattr(manager, 'download_orchestrator')
        assert hasattr(manager, 'validation_engine')
        assert hasattr(manager, 'storage_manager')
        assert hasattr(manager, 'ingestion_pipeline')

    @pytest.mark.asyncio
    async def test_download_samples_with_valid_config(
        self, 
        mock_config_manager, 
        sample_config_minimal,
        mock_document_metadata
    ):
        """Test downloading samples with valid configuration."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        # Mock the data source to return our test metadata
        with patch.object(manager, '_get_data_source') as mock_get_source:
            mock_source = AsyncMock()
            mock_source.list_available_documents.return_value = mock_document_metadata
            mock_source.download_document.return_value = Path("/tmp/test_doc.xml")
            mock_get_source.return_value = mock_source
            
            result = await manager.download_samples(sample_config_minimal)
            
            assert isinstance(result, list)
            assert len(result) == len(mock_document_metadata)
            assert all(isinstance(doc, DocumentMetadata) for doc in result)

    @pytest.mark.asyncio
    async def test_download_samples_with_invalid_config(self, mock_config_manager):
        """Test downloading samples with invalid configuration."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        # Create invalid config with negative document count
        invalid_config = SampleDataConfig(
            source_type=DataSourceType.PMC_API,
            document_count=-1,  # Invalid
            categories=["medical"],
            storage_path=Path("/tmp/test"),
            cache_enabled=True,
            parallel_downloads=2,
            batch_size=5,
            cleanup_on_success=False,
            iris_edition="community"
        )
        
        with pytest.raises(ConfigurationError):
            await manager.download_samples(invalid_config)

    @pytest.mark.asyncio
    async def test_download_samples_with_progress_callback(
        self, 
        mock_config_manager, 
        sample_config_minimal,
        mock_document_metadata
    ):
        """Test downloading samples with progress callback."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        progress_calls = []
        
        def progress_callback(progress: DownloadProgress):
            progress_calls.append(progress)
        
        with patch.object(manager, '_get_data_source') as mock_get_source:
            mock_source = AsyncMock()
            mock_source.list_available_documents.return_value = mock_document_metadata
            mock_source.download_document.return_value = Path("/tmp/test_doc.xml")
            mock_get_source.return_value = mock_source
            
            result = await manager.download_samples(
                sample_config_minimal, 
                progress_callback=progress_callback
            )
            
            assert len(progress_calls) > 0
            assert all(isinstance(p, DownloadProgress) for p in progress_calls)

    @pytest.mark.asyncio
    async def test_validate_samples_success(
        self, 
        mock_config_manager, 
        temp_storage_path,
        sample_xml_content
    ):
        """Test successful validation of sample documents."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        # Create test XML files
        test_file = temp_storage_path / "PMC000001.xml"
        test_file.write_text(sample_xml_content)
        
        result = await manager.validate_samples(temp_storage_path)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.document_count > 0

    @pytest.mark.asyncio
    async def test_validate_samples_with_errors(
        self, 
        mock_config_manager, 
        temp_storage_path
    ):
        """Test validation with invalid documents."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        # Create invalid XML file
        invalid_file = temp_storage_path / "invalid.xml"
        invalid_file.write_text("This is not valid XML content")
        
        result = await manager.validate_samples(temp_storage_path, strict_mode=True)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_ingest_samples_success(
        self, 
        mock_config_manager, 
        sample_config_minimal,
        temp_storage_path,
        sample_xml_content
    ):
        """Test successful ingestion of sample documents."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        # Create test XML files
        test_file = temp_storage_path / "PMC000001.xml"
        test_file.write_text(sample_xml_content)
        
        with patch.object(manager, 'ingestion_pipeline') as mock_pipeline:
            # Create an async mock for the ingest_documents method
            async def mock_ingest(storage_path, config, progress_callback=None):
                return IngestionResult(
                    success=True,
                    documents_processed=1,
                    documents_ingested=1,
                    errors=[],
                    processing_time=10.5,
                    database_size=1024
                )
            mock_pipeline.ingest_documents = mock_ingest
            
            result = await manager.ingest_samples(temp_storage_path, sample_config_minimal)
            
            assert isinstance(result, IngestionResult)
            assert result.success is True
            assert result.documents_ingested > 0

    @pytest.mark.asyncio
    async def test_ingest_samples_with_progress_callback(
        self, 
        mock_config_manager, 
        sample_config_minimal,
        temp_storage_path,
        sample_xml_content
    ):
        """Test ingestion with progress callback."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        progress_calls = []
        
        def progress_callback(processed: int, total: int):
            progress_calls.append((processed, total))
        
        # Create test XML files
        test_file = temp_storage_path / "PMC000001.xml"
        test_file.write_text(sample_xml_content)
        
        with patch.object(manager, 'ingestion_pipeline') as mock_pipeline:
            # Create an async mock for the ingest_documents method
            async def mock_ingest_with_progress(storage_path, config, progress_callback=None):
                if progress_callback:
                    progress_callback(1, 1)  # Simulate progress
                return IngestionResult(
                    success=True,
                    documents_processed=1,
                    documents_ingested=1,
                    errors=[],
                    processing_time=10.5,
                    database_size=1024
                )
            mock_pipeline.ingest_documents = mock_ingest_with_progress
            
            result = await manager.ingest_samples(
                temp_storage_path,
                sample_config_minimal,
                progress_callback=progress_callback
            )
            
            assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_cleanup_samples(
        self, 
        mock_config_manager, 
        temp_storage_path,
        sample_xml_content
    ):
        """Test cleanup of sample files."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        # Create test files
        test_file = temp_storage_path / "PMC000001.xml"
        test_file.write_text(sample_xml_content)
        cache_file = temp_storage_path / "cache" / "cached_doc.xml"
        cache_file.parent.mkdir(exist_ok=True)
        cache_file.write_text(sample_xml_content)
        
        assert test_file.exists()
        assert cache_file.exists()
        
        await manager.cleanup_samples(temp_storage_path, keep_cache=True)
        
        # Test file should be removed, cache should remain
        assert not test_file.exists()
        assert cache_file.exists()

    @pytest.mark.asyncio
    async def test_get_available_sources(self, mock_config_manager):
        """Test getting available data sources."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        sources = await manager.get_available_sources()
        
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert all(isinstance(source, dict) for source in sources)
        assert all('type' in source for source in sources)

    @pytest.mark.asyncio
    async def test_estimate_requirements(
        self, 
        mock_config_manager, 
        sample_config_minimal
    ):
        """Test estimation of resource requirements."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        requirements = await manager.estimate_requirements(sample_config_minimal)
        
        assert isinstance(requirements, dict)
        assert 'disk_space' in requirements
        assert 'memory' in requirements
        assert 'estimated_time' in requirements
        assert all(isinstance(v, (int, float)) for v in requirements.values())

    @pytest.mark.asyncio
    async def test_download_failure_handling(
        self, 
        mock_config_manager, 
        sample_config_minimal
    ):
        """Test handling of download failures."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        with patch.object(manager, '_get_data_source') as mock_get_source:
            mock_source = AsyncMock()
            mock_source.list_available_documents.side_effect = Exception("Network error")
            mock_get_source.return_value = mock_source
            
            with pytest.raises(DownloadError):
                await manager.download_samples(sample_config_minimal)

    @pytest.mark.asyncio
    async def test_ingestion_failure_handling(
        self, 
        mock_config_manager, 
        sample_config_minimal,
        temp_storage_path
    ):
        """Test handling of ingestion failures."""
        from quick_start.data.sample_manager import SampleDataManager
        
        manager = SampleDataManager(mock_config_manager)
        
        with patch.object(manager, 'ingestion_pipeline') as mock_pipeline:
            mock_pipeline.ingest_documents.side_effect = Exception("Database error")
            
            with pytest.raises(IngestionError):
                await manager.ingest_samples(temp_storage_path, sample_config_minimal)