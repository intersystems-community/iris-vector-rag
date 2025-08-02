"""
Test configuration and fixtures for Quick Start system tests.

This module provides shared fixtures and configuration for testing
the Quick Start system components.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from quick_start.data.interfaces import (
    SampleDataConfig,
    DataSourceType,
    DocumentMetadata,
    DownloadProgress,
    ValidationResult,
    IngestionResult,
)


@pytest.fixture
def temp_storage_path():
    """Provide a temporary directory for test storage."""
    temp_dir = tempfile.mkdtemp(prefix="quick_start_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config_minimal(temp_storage_path):
    """Provide a minimal sample data configuration for testing."""
    return SampleDataConfig(
        source_type=DataSourceType.PMC_API,
        document_count=10,
        categories=["medical"],
        storage_path=temp_storage_path,
        cache_enabled=True,
        parallel_downloads=2,
        batch_size=5,
        cleanup_on_success=False,
        iris_edition="community"
    )


@pytest.fixture
def sample_config_standard(temp_storage_path):
    """Provide a standard sample data configuration for testing."""
    return SampleDataConfig(
        source_type=DataSourceType.PMC_API,
        document_count=50,
        categories=["medical", "research"],
        storage_path=temp_storage_path,
        cache_enabled=True,
        parallel_downloads=4,
        batch_size=10,
        cleanup_on_success=False,
        iris_edition="community"
    )


@pytest.fixture
def mock_document_metadata():
    """Provide mock document metadata for testing."""
    return [
        DocumentMetadata(
            pmc_id="PMC000001",
            title="Test Medical Document 1",
            authors=["Dr. Test Author"],
            abstract="This is a test medical document abstract.",
            categories=["medical"],
            file_size=1024,
            download_url="https://example.com/PMC000001.xml",
            local_path=None
        ),
        DocumentMetadata(
            pmc_id="PMC000002",
            title="Test Research Document 2",
            authors=["Dr. Research Author"],
            abstract="This is a test research document abstract.",
            categories=["research"],
            file_size=2048,
            download_url="https://example.com/PMC000002.xml",
            local_path=None
        ),
    ]


@pytest.fixture
def mock_download_progress():
    """Provide mock download progress for testing."""
    return DownloadProgress(
        total_documents=10,
        downloaded=5,
        failed=0,
        current_document="PMC000005",
        bytes_downloaded=5120,
        total_bytes=10240,
        estimated_time_remaining=30.0
    )


@pytest.fixture
def mock_validation_result_success():
    """Provide successful validation result for testing."""
    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        document_count=10,
        total_size=10240
    )


@pytest.fixture
def mock_validation_result_failure():
    """Provide failed validation result for testing."""
    return ValidationResult(
        is_valid=False,
        errors=["Invalid XML format in PMC000003.xml"],
        warnings=["Missing abstract in PMC000004.xml"],
        document_count=8,
        total_size=8192
    )


@pytest.fixture
def mock_ingestion_result_success():
    """Provide successful ingestion result for testing."""
    return IngestionResult(
        success=True,
        documents_processed=10,
        documents_ingested=10,
        errors=[],
        processing_time=45.5,
        database_size=1048576
    )


@pytest.fixture
def mock_data_source():
    """Provide a mock data source for testing."""
    mock_source = AsyncMock()
    mock_source.list_available_documents.return_value = []
    mock_source.download_document.return_value = Path("/tmp/test_doc.xml")
    mock_source.verify_document.return_value = True
    return mock_source


@pytest.fixture
def mock_config_manager():
    """Provide a mock configuration manager for testing."""
    mock_manager = Mock()
    mock_manager.get_config.return_value = {
        "database": {
            "iris": {
                "host": "localhost",
                "port": 1972,
                "namespace": "USER",
                "username": "_SYSTEM",
                "password": "SYS"
            }
        },
        "storage": {
            "iris": {
                "table_name": "QuickStart.Documents",
                "vector_dimension": 384
            }
        }
    }
    return mock_manager


@pytest.fixture
def sample_xml_content():
    """Provide sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<article>
    <front>
        <article-meta>
            <title-group>
                <article-title>Test Medical Document</article-title>
            </title-group>
            <contrib-group>
                <contrib contrib-type="author">
                    <name>
                        <surname>Author</surname>
                        <given-names>Test</given-names>
                    </name>
                </contrib>
            </contrib-group>
            <abstract>
                <p>This is a test medical document abstract for testing purposes.</p>
            </abstract>
        </article-meta>
    </front>
    <body>
        <sec>
            <title>Introduction</title>
            <p>This is the introduction section of the test document.</p>
        </sec>
        <sec>
            <title>Methods</title>
            <p>This section describes the methods used in the study.</p>
        </sec>
        <sec>
            <title>Results</title>
            <p>This section presents the results of the study.</p>
        </sec>
        <sec>
            <title>Conclusion</title>
            <p>This section provides the conclusions of the study.</p>
        </sec>
    </body>
</article>"""


@pytest.fixture
def mock_iris_connection():
    """Provide a mock IRIS database connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    return mock_conn


@pytest.fixture(scope="session")
def test_environment_config():
    """Provide test environment configuration."""
    return {
        "test_data_path": "tests/quick_start/test_data",
        "mock_pmc_api_url": "http://localhost:8080/mock-pmc-api",
        "test_iris_namespace": "QUICKSTARTTEST",
        "cleanup_after_tests": True,
        "enable_real_downloads": False,  # Set to True for integration tests
    }