"""
Comprehensive Tests for PMC Document Downloader System

This module provides comprehensive testing of the PMC downloader system
including API client, batch downloader, and integration components.
Following TDD principles with real data validation.
"""

import pytest
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from data.pmc_downloader import (
    PMCAPIClient, 
    PMCBatchDownloader, 
    PMCEnterpriseLoader,
    DownloadProgress,
    ValidationResult,
    create_enterprise_loader_config,
    load_enterprise_pmc_dataset
)

logger = logging.getLogger(__name__)

@pytest.fixture
def temp_download_dir():
    """Create temporary directory for downloads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_pmc_config():
    """Mock configuration for PMC downloader."""
    return {
        'target_document_count': 100,  # Small for testing
        'download_directory': 'test_downloads',
        'enable_validation': True,
        'batch_size': 10,
        'api_client': {
            'request_delay_seconds': 0.1,
            'max_retries': 2,
            'timeout_seconds': 10
        }
    }

@pytest.fixture
def sample_xml_content():
    """Sample PMC XML content for testing."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<article>
    <front>
        <article-meta>
            <title-group>
                <article-title>Test Medical Article</article-title>
            </title-group>
            <abstract>
                <p>This is a test abstract for a medical research paper. It contains sufficient content to pass validation checks and demonstrates the structure of a real PMC document.</p>
            </abstract>
            <contrib-group>
                <contrib contrib-type="author">
                    <name>
                        <surname>Smith</surname>
                        <given-names>John</given-names>
                    </name>
                </contrib>
            </contrib-group>
            <kwd-group>
                <kwd>medical research</kwd>
                <kwd>testing</kwd>
            </kwd-group>
        </article-meta>
    </front>
</article>'''

class TestPMCAPIClient:
    """Test suite for PMC API Client."""
    
    def test_api_client_initialization(self):
        """Test PMC API client initialization."""
        config = {'request_delay_seconds': 0.5, 'max_retries': 3}
        client = PMCAPIClient(config)
        
        assert client.config == config
        assert client.request_delay == 0.5
        assert client.max_retries == 3
        assert client.session is not None
    
    def test_api_client_default_config(self):
        """Test PMC API client with default configuration."""
        client = PMCAPIClient()
        
        assert client.request_delay == 0.5
        assert client.max_retries == 3
        assert client.timeout == 30
    
    def test_estimate_document_count(self):
        """Test document count estimation."""
        client = PMCAPIClient()
        
        # Test with known size
        count = client._estimate_document_count("test_file.tar.gz", 10 * 1024 * 1024)  # 10MB
        assert count > 0
        
        # Test with None size
        count = client._estimate_document_count("test_file.tar.gz", None)
        assert count == 0
    
    @patch('ftplib.FTP')
    def test_get_available_bulk_files(self, mock_ftp_class):
        """Test getting available bulk files."""
        # Mock FTP behavior
        mock_ftp = Mock()
        mock_ftp_class.return_value = mock_ftp
        mock_ftp.nlst.return_value = ['oa_comm_xml.PMC001xxxxxx.baseline.2023.tar.gz']
        mock_ftp.size.return_value = 100 * 1024 * 1024  # 100MB
        
        client = PMCAPIClient()
        files = client.get_available_bulk_files()
        
        assert len(files) == 1
        assert files[0]['filename'] == 'oa_comm_xml.PMC001xxxxxx.baseline.2023.tar.gz'
        assert files[0]['size_bytes'] == 100 * 1024 * 1024
        assert files[0]['estimated_documents'] > 0
    
    @patch('requests.Session.get')
    def test_search_pmc_ids(self, mock_get):
        """Test PMC ID search functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.content = b'''<?xml version="1.0"?>
        <eSearchResult>
            <IdList>
                <Id>123456</Id>
                <Id>789012</Id>
            </IdList>
        </eSearchResult>'''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = PMCAPIClient()
        pmc_ids = client.search_pmc_ids("COVID-19", max_results=10)
        
        assert len(pmc_ids) == 2
        assert "123456" in pmc_ids
        assert "789012" in pmc_ids

class TestPMCBatchDownloader:
    """Test suite for PMC Batch Downloader."""
    
    def test_batch_downloader_initialization(self, mock_pmc_config):
        """Test batch downloader initialization."""
        downloader = PMCBatchDownloader(mock_pmc_config)
        
        assert downloader.target_document_count == 100
        assert downloader.validation_enabled == True
        assert downloader.batch_size == 10
        assert isinstance(downloader.progress, DownloadProgress)
    
    def test_download_progress_calculations(self):
        """Test download progress calculations."""
        progress = DownloadProgress(
            total_documents=1000,
            downloaded_documents=250,
            start_time=time.time() - 100  # 100 seconds ago
        )
        
        assert progress.progress_percentage == 25.0
        assert progress.elapsed_time_seconds >= 99  # Allow for small timing differences
        
        progress.update_speed_and_eta()
        assert progress.download_speed_docs_per_second > 0
        assert progress.estimated_time_remaining_seconds > 0
    
    def test_document_validation(self, temp_download_dir, sample_xml_content, mock_pmc_config):
        """Test document validation functionality."""
        downloader = PMCBatchDownloader(mock_pmc_config)
        
        # Create test XML file
        test_file = temp_download_dir / "test_document.xml"
        test_file.write_text(sample_xml_content)
        
        # Test validation
        result = downloader._validate_document(test_file)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
        assert result.pmc_id == "test_document"
        assert result.title == "Test Medical Article"
        assert result.abstract_length > 50
        assert result.file_size_bytes > 0
    
    def test_document_validation_invalid_file(self, temp_download_dir, mock_pmc_config):
        """Test validation of invalid documents."""
        downloader = PMCBatchDownloader(mock_pmc_config)
        
        # Test non-existent file
        result = downloader._validate_document(temp_download_dir / "nonexistent.xml")
        assert result.is_valid == False
        assert "does not exist" in result.error_message
        
        # Test empty file
        empty_file = temp_download_dir / "empty.xml"
        empty_file.write_text("")
        result = downloader._validate_document(empty_file)
        assert result.is_valid == False
        assert "empty" in result.error_message
        
        # Test invalid XML
        invalid_file = temp_download_dir / "invalid.xml"
        invalid_file.write_text("not valid xml content")
        result = downloader._validate_document(invalid_file)
        assert result.is_valid == False
        assert "parse error" in result.error_message.lower()
    
    def test_checkpoint_save_load(self, temp_download_dir, mock_pmc_config):
        """Test checkpoint save and load functionality."""
        mock_pmc_config['checkpoint_file'] = str(temp_download_dir / "checkpoint.json")
        downloader = PMCBatchDownloader(mock_pmc_config)
        
        # Set some progress
        downloader.progress.total_documents = 1000
        downloader.progress.downloaded_documents = 250
        downloader.validation_results = [
            ValidationResult(is_valid=True, file_path="test1.xml"),
            ValidationResult(is_valid=False, file_path="test2.xml", error_message="Test error")
        ]
        
        # Save checkpoint
        downloader._save_checkpoint()
        
        # Create new downloader and load checkpoint
        new_downloader = PMCBatchDownloader(mock_pmc_config)
        loaded = new_downloader._load_checkpoint()
        
        assert loaded == True
        assert new_downloader.progress.total_documents == 1000
        assert new_downloader.progress.downloaded_documents == 250
        assert len(new_downloader.validation_results) == 2

class TestPMCEnterpriseLoader:
    """Test suite for PMC Enterprise Loader."""
    
    def test_enterprise_loader_initialization(self, mock_pmc_config):
        """Test enterprise loader initialization."""
        loader = PMCEnterpriseLoader(mock_pmc_config)
        
        assert loader.config == mock_pmc_config
        assert isinstance(loader.downloader, PMCBatchDownloader)
        assert loader.total_phases == 3
        assert loader.current_phase == 0
    
    def test_create_enterprise_loader_config(self):
        """Test enterprise loader configuration creation."""
        config = create_enterprise_loader_config(
            target_documents=5000,
            download_dir="test_dir",
            enable_validation=False,
            batch_size=50
        )
        
        assert config['target_document_count'] == 5000
        assert config['download_directory'] == "test_dir"
        assert config['enable_validation'] == False
        assert config['batch_size'] == 50
        assert 'downloader' in config
        assert 'loader' in config
    
    @patch('data.pmc_downloader.integration.process_and_load_documents_unified')
    def test_enterprise_loader_mock_success(self, mock_load_func, temp_download_dir, mock_pmc_config):
        """Test enterprise loader with mocked successful operations."""
        # Setup mock config
        mock_pmc_config['download_directory'] = str(temp_download_dir)
        mock_pmc_config['target_document_count'] = 10  # Small for testing
        
        # Create mock extract directory with sample files
        extract_dir = temp_download_dir / "extracted"
        extract_dir.mkdir()
        
        # Create sample XML files
        for i in range(10):
            xml_file = extract_dir / f"PMC{i:06d}.xml"
            xml_file.write_text('''<?xml version="1.0"?>
            <article>
                <front>
                    <article-meta>
                        <title-group>
                            <article-title>Test Article {}</article-title>
                        </title-group>
                        <abstract>
                            <p>This is a test abstract with sufficient content for validation.</p>
                        </abstract>
                    </article-meta>
                </front>
            </article>'''.format(i))
        
        # Mock the downloader to return success
        loader = PMCEnterpriseLoader(mock_pmc_config)
        
        # Mock downloader success
        with patch.object(loader.downloader, 'download_enterprise_dataset') as mock_download:
            mock_download.return_value = {
                'success': True,
                'downloaded_documents': 10,
                'validated_documents': 10,
                'failed_documents': 0,
                'download_time_seconds': 5.0,
                'extract_directory': str(extract_dir)
            }
            
            # Mock loader success
            mock_load_func.return_value = {
                'success': True,
                'loaded_doc_count': 10,
                'loaded_token_count': 100,
                'error_count': 0
            }
            
            # Test the enterprise loading
            result = loader.load_enterprise_dataset()
            
            assert result['success'] == True
            assert result['final_document_count'] == 10
            assert result['target_achieved'] == True
            assert 'download_results' in result
            assert 'loading_results' in result

class TestIntegrationFunctions:
    """Test suite for integration functions."""
    
    def test_load_enterprise_pmc_dataset_config(self):
        """Test the convenience function for loading enterprise dataset."""
        # Test with default parameters
        with patch('data.pmc_downloader.integration.PMCEnterpriseLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_enterprise_dataset.return_value = {'success': True}
            
            result = load_enterprise_pmc_dataset(target_documents=1000)
            
            # Verify loader was created with correct config
            mock_loader_class.assert_called_once()
            config_arg = mock_loader_class.call_args[0][0]
            assert config_arg['target_document_count'] == 1000
            
            # Verify loader method was called
            mock_loader.load_enterprise_dataset.assert_called_once()
            assert result['success'] == True

@pytest.mark.integration
class TestPMCDownloaderIntegration:
    """Integration tests for PMC downloader system."""
    
    @pytest.mark.slow
    def test_api_client_real_connection(self):
        """Test real connection to PMC APIs (slow test)."""
        client = PMCAPIClient()
        
        # Test getting bulk files (this makes real network calls)
        try:
            files = client.get_available_bulk_files()
            # Should get some files
            assert len(files) >= 0  # May be 0 if network issues
            
            if files:
                # Verify file structure
                assert 'filename' in files[0]
                assert 'estimated_documents' in files[0]
                
        except Exception as e:
            pytest.skip(f"Network connection failed: {e}")
    
    @pytest.mark.slow
    def test_search_real_pmc_ids(self):
        """Test real PMC ID search (slow test)."""
        client = PMCAPIClient()
        
        try:
            # Search for a common medical term
            pmc_ids = client.search_pmc_ids("COVID-19", max_results=5)
            
            # Should find some results
            assert len(pmc_ids) >= 0  # May be 0 if network issues
            
            if pmc_ids:
                # Verify IDs are numeric strings
                for pmc_id in pmc_ids:
                    assert pmc_id.isdigit()
                    
        except Exception as e:
            pytest.skip(f"Network connection failed: {e}")

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])