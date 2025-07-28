#!/usr/bin/env python3
"""
Enterprise-Scale Testing Fixtures for 1000+ Document RAG Testing

This module provides pytest fixtures for large-scale RAG testing with real PMC data.
Designed for production enterprise capability validation.

Features:
- Ensures minimum 1000 documents for meaningful testing
- Supports scaling up to 92K+ documents for enterprise testing
- Automatic data loading and validation
- Performance monitoring and metrics collection
- IRIS Enterprise edition requirement enforcement
"""

import pytest
import logging
import os
import time
from typing import Dict, Any, List
from pathlib import Path

from common.iris_connection_manager import get_iris_connection
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.schema_manager import SchemaManager
from data.unified_loader import UnifiedDocumentLoader

logger = logging.getLogger(__name__)

# Enterprise testing configuration
MINIMUM_DOCS_FOR_SCALE_TESTING = 1000
ENTERPRISE_SCALE_DOCS = 92000
DEFAULT_SCALE_DOCS = 1000

@pytest.fixture(scope="session")
def enterprise_iris_connection():
    """
    Enterprise IRIS connection with validation for large-scale testing.
    
    Ensures IRIS Enterprise edition is being used for scale testing.
    """
    # Check for Enterprise edition requirement
    iris_image = os.getenv('IRIS_DOCKER_IMAGE', '')
    if 'community' in iris_image.lower():
        pytest.skip("Enterprise scale testing requires IRIS Enterprise edition (10GB+ data limit)")
    
    connection = get_iris_connection()
    
    # Validate connection can handle enterprise workloads
    cursor = connection.cursor()
    try:
        # Test enterprise features
        cursor.execute("SELECT $SYSTEM.License.GetFeature('IRIS_ENTERPRISE')")
        result = cursor.fetchone()
        if not result or not result[0]:
            logger.warning("IRIS Enterprise features may not be available")
    except Exception as e:
        logger.warning(f"Could not verify IRIS Enterprise features: {e}")
    finally:
        cursor.close()
    
    return connection

@pytest.fixture(scope="session")
def scale_test_config():
    """Configuration for scale testing with document count parameters."""
    config = ConfigurationManager()
    
    # Determine scale based on environment
    scale_mode = os.getenv('RAG_SCALE_TEST_MODE', 'standard')
    
    if scale_mode == 'enterprise':
        target_docs = ENTERPRISE_SCALE_DOCS
    elif scale_mode == 'large':
        target_docs = int(os.getenv('RAG_SCALE_TEST_DOCS', '10000'))
    else:
        target_docs = DEFAULT_SCALE_DOCS
    
    return {
        'target_document_count': target_docs,
        'minimum_document_count': MINIMUM_DOCS_FOR_SCALE_TESTING,
        'scale_mode': scale_mode,
        'config_manager': config
    }

@pytest.fixture(scope="session")
def enterprise_schema_manager(enterprise_iris_connection, scale_test_config):
    """Schema manager configured for enterprise-scale testing."""
    config_manager = scale_test_config['config_manager']
    
    # Create connection manager wrapper
    connection_manager = type('ConnectionManager', (), {
        'get_connection': lambda self: enterprise_iris_connection
    })()
    
    schema_manager = SchemaManager(connection_manager, config_manager)
    
    # Ensure core tables for chunking and ColBERT testing
    core_tables = [
        "SourceDocuments",
        "DocumentTokenEmbeddings",
        "DocumentChunks"
    ]
    
    for table in core_tables:
        try:
            success = schema_manager.ensure_table_schema(table)
            if not success:
                logger.warning(f"Could not ensure schema for {table} - some tests may be skipped")
        except Exception as e:
            logger.warning(f"Schema setup failed for {table}: {e} - some tests may be skipped")
    
    return schema_manager

@pytest.fixture(scope="session")
def scale_test_documents(enterprise_iris_connection, scale_test_config, enterprise_schema_manager):
    """
    Ensures minimum document count for scale testing.
    
    Loads documents if needed and validates count meets requirements.
    """
    target_count = scale_test_config['target_document_count']
    minimum_count = scale_test_config['minimum_document_count']
    
    # Check current document count
    cursor = enterprise_iris_connection.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        current_count = cursor.fetchone()[0]
        
        logger.info(f"Current document count: {current_count}")
        logger.info(f"Target document count: {target_count}")
        logger.info(f"Minimum required: {minimum_count}")
        
        if current_count < minimum_count:
            # Need to load more documents
            logger.warning(f"Insufficient documents for scale testing: {current_count} < {minimum_count}")
            
            # Check if we have sample data available
            sample_data_path = Path("data/test_loader_pmc_sample")
            if sample_data_path.exists():
                logger.info("Loading available sample documents...")
                
                config_manager = scale_test_config['config_manager']
                
                # Create a basic config for the loader
                loader_config = {
                    'embedding_column_type': 'VECTOR',
                    'batch_size': 10,
                    'enable_checkpointing': False,
                    'performance_mode': 'conservative'
                }
                loader = UnifiedDocumentLoader(loader_config)
                
                # Load all available sample documents
                loaded_count = 0
                for pmc_dir in sample_data_path.iterdir():
                    if pmc_dir.is_dir() and pmc_dir.name.startswith('PMC'):
                        try:
                            # Use the loader's load_documents method
                            loader.load_documents([str(pmc_dir)])
                            loaded_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to load {pmc_dir}: {e}")
                
                # Recheck count
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                new_count = cursor.fetchone()[0]
                
                if new_count < minimum_count:
                    pytest.skip(
                        f"Insufficient documents for scale testing: {new_count} < {minimum_count}. "
                        f"Please load more PMC documents for enterprise-scale testing."
                    )
            else:
                pytest.skip(
                    f"No sample data available and insufficient documents: {current_count} < {minimum_count}"
                )
        
        # Return document metadata for testing
        cursor.execute("""
            SELECT COUNT(*) as total_docs,
                   MIN(created_at) as earliest_doc,
                   MAX(created_at) as latest_doc
            FROM RAG.SourceDocuments
        """)
        doc_stats = cursor.fetchone()
        
        return {
            'document_count': doc_stats[0],
            'earliest_document': doc_stats[1],
            'latest_document': doc_stats[2],
            'meets_scale_requirements': doc_stats[0] >= minimum_count,
            'target_count': target_count,
            'scale_mode': scale_test_config['scale_mode']
        }
        
    finally:
        cursor.close()

@pytest.fixture(scope="function")
def scale_test_performance_monitor():
    """Performance monitoring for scale tests."""
    start_time = time.time()
    start_memory = None
    
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        logger.warning("psutil not available - memory monitoring disabled")
    
    metrics = {
        'start_time': start_time,
        'start_memory_mb': start_memory,
        'operations': []
    }
    
    def record_operation(operation_name: str, duration: float, **kwargs):
        """Record a performance operation."""
        metrics['operations'].append({
            'operation': operation_name,
            'duration_seconds': duration,
            'timestamp': time.time(),
            **kwargs
        })
    
    metrics['record_operation'] = record_operation
    
    yield metrics
    
    # Calculate final metrics
    end_time = time.time()
    total_duration = end_time - start_time
    
    end_memory = None
    if start_memory:
        try:
            import psutil
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
    
    metrics.update({
        'end_time': end_time,
        'total_duration_seconds': total_duration,
        'end_memory_mb': end_memory,
        'memory_delta_mb': (end_memory - start_memory) if (end_memory and start_memory) else None
    })
    
    logger.info(f"Scale test completed in {total_duration:.2f}s")
    if metrics['memory_delta_mb']:
        logger.info(f"Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB (Δ{metrics['memory_delta_mb']:+.1f}MB)")

@pytest.fixture(scope="session")
def enterprise_test_queries():
    """Standard test queries for enterprise-scale testing."""
    return [
        {
            'query': 'What are the effects of BRCA1 mutations on breast cancer risk?',
            'category': 'genetics',
            'expected_keywords': ['BRCA1', 'breast cancer', 'mutation', 'risk']
        },
        {
            'query': 'How does p53 protein function in cell cycle regulation?',
            'category': 'cell_biology', 
            'expected_keywords': ['p53', 'cell cycle', 'regulation', 'protein']
        },
        {
            'query': 'What is the role of inflammation in cardiovascular disease?',
            'category': 'cardiovascular',
            'expected_keywords': ['inflammation', 'cardiovascular', 'disease']
        },
        {
            'query': 'Describe the mechanism of action of checkpoint inhibitors in cancer therapy',
            'category': 'immunotherapy',
            'expected_keywords': ['checkpoint', 'inhibitor', 'cancer', 'therapy']
        },
        {
            'query': 'What are the molecular pathways involved in Alzheimer disease progression?',
            'category': 'neurology',
            'expected_keywords': ['molecular', 'pathway', 'Alzheimer', 'progression']
        }
    ]

# Scale testing markers
def pytest_configure(config):
    """Configure pytest markers for scale testing."""
    config.addinivalue_line(
        "markers", "scale_1000: mark test as requiring 1000+ documents"
    )
    config.addinivalue_line(
        "markers", "scale_enterprise: mark test as requiring enterprise-scale (92K+) documents"
    )
    config.addinivalue_line(
        "markers", "performance_benchmark: mark test as performance benchmark"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for scale testing."""
    scale_mode = os.getenv('RAG_SCALE_TEST_MODE', 'standard')
    
    for item in items:
        # Skip enterprise tests if not in enterprise mode
        if item.get_closest_marker("scale_enterprise") and scale_mode != 'enterprise':
            item.add_marker(pytest.mark.skip(reason="Enterprise scale testing not enabled"))
        
        # Skip 1000+ doc tests if insufficient documents
        if item.get_closest_marker("scale_1000"):
            # This will be validated by the scale_test_documents fixture
            pass