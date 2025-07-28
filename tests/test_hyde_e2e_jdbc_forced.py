"""
HyDE E2E Tests with Forced JDBC Driver Usage

This test file demonstrates that the HyDE pipeline works correctly
even when forced to use the JDBC driver instead of the preferred DBAPI driver.
"""

import logging
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connection_manager import DriverType
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from data.unified_loader import process_and_load_documents_unified

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def force_jdbc_setup():
    """
    Setup fixture that forces the system to use JDBC driver
    by mocking the DBAPI connection to fail.
    """
    logger.info("Setting up database for HyDE E2E pipeline tests with FORCED JDBC...")
    
    # Mock DBAPI to force JDBC usage - patch the _get_dbapi_connection method directly
    with patch('common.iris_connection_manager.IRISConnectionManager._get_dbapi_connection') as mock_dbapi:
        # Make DBAPI connection fail
        mock_dbapi.side_effect = Exception("DBAPI forced to fail for testing")
        
        # Initialize schema manager
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda self: get_iris_connection()
        })()
        config_manager = ConfigurationManager()
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Ensure required schemas exist
        schema_manager.ensure_table_schema('SourceDocuments')
        schema_manager.ensure_table_schema('DocumentTokenEmbeddings')
        schema_manager.ensure_table_schema('DocumentEntities')
        
        # Clean up any existing test data
        logger.info("Attempting to delete DOCA and DOCB if they exist to ensure clean test data ingestion for HyDE.")
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        for doc_id in ['DOCA', 'DOCB']:
            try:
                cursor.execute("DELETE FROM SourceDocuments WHERE doc_id = ?", (doc_id,))
                rows_affected = cursor.rowcount
                logger.info(f"HyDE E2E: Executed delete for {doc_id}. Rows affected: {rows_affected}")
            except Exception as e:
                logger.debug(f"Delete failed for {doc_id} (expected if not exists): {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("HyDE E2E: Finished attempting to delete DOCA and DOCB.")
        
        yield
        
        logger.info("HyDE E2E: Closing database connection.")


def test_hyde_e2e_with_forced_jdbc_cellular_energy(force_jdbc_setup):
    """
    Test HyDE E2E pipeline with cellular energy query using FORCED JDBC driver.
    
    This test forces the system to use JDBC by mocking DBAPI connection failures.
    """
    # Ingest test documents first (this will trigger the JDBC fallback)
    logger.info("HyDE E2E: Ingesting E2E test documents from tests/test_data/e2e_docs")
    
    # Use the unified data processing approach
    config = {
        "file_pattern": "*.txt",
        "limit": 10  # Small limit for testing
    }
    process_and_load_documents_unified(
        config=config,
        pmc_directory="tests/test_data/e2e_docs"
    )
    
    logger.info("HyDE E2E: Test documents ingested successfully.")
    
    # Verify we're using JDBC by checking connection type
    from common.iris_connection_manager import get_driver_type
    driver_type = get_driver_type()
    assert driver_type == DriverType.JDBC, f"Expected JDBC driver, got {driver_type}"
    logger.info(f"✓ Confirmed using JDBC driver: {driver_type}")
    
    # Initialize HyDE pipeline components
    llm_func = get_llm_func()
    config_manager = ConfigurationManager()
    
    # Create HyDE pipeline with correct parameters
    hyde_pipeline = HyDERAGPipeline(
        config_manager=config_manager,
        llm_func=llm_func
    )
    
    # Test query about cellular energy
    query = "How do cells produce energy?"
    logger.info(f"Executing HyDE E2E test with abstract query: {query}")
    
    # Execute HyDE pipeline using the correct method
    result = hyde_pipeline.query(query, top_k=5)
    
    # Verify result structure
    assert isinstance(result, dict), "HyDE result should be a dictionary"
    assert "query" in result, "Result should contain 'query' field"
    assert "answer" in result, "Result should contain 'answer' field"
    assert "retrieved_documents" in result, "Result should contain 'retrieved_documents' field"
    assert "hypothetical_document" in result, "Result should contain 'hypothetical_document' field"
    
    # Verify query matches
    assert result["query"] == query, f"Query mismatch: expected '{query}', got '{result['query']}'"
    
    # Verify hypothetical document was generated
    hypothetical_doc = result["hypothetical_document"]
    assert isinstance(hypothetical_doc, str), "Hypothetical document should be a string"
    assert len(hypothetical_doc) > 50, "Hypothetical document should be substantial"
    logger.info(f"HyDE generated hypothetical document (first 100 chars): {hypothetical_doc[:100]}...")
    
    # Verify documents were retrieved
    retrieved_docs = result["retrieved_documents"]
    assert isinstance(retrieved_docs, list), "Retrieved documents should be a list"
    assert len(retrieved_docs) > 0, "Should retrieve at least one document"
    
    # Verify answer was generated
    answer = result["answer"]
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 10, "Answer should be substantial"
    
    # Log results
    doc_ids = [doc["doc_id"] for doc in retrieved_docs]
    logger.info(f"HyDE retrieved doc IDs: {doc_ids}, Answer: {answer[:100]}...")
    
    # Verify we retrieved expected documents
    assert any("DOCA" in doc_id or "DOCB" in doc_id for doc_id in doc_ids), \
        f"Should retrieve test documents, got: {doc_ids}"
    
    logger.info("✅ HyDE E2E test for abstract query (cellular energy) passed successfully with JDBC driver.")


def test_hyde_e2e_with_forced_jdbc_genetic_modification(force_jdbc_setup):
    """
    Test HyDE E2E pipeline with genetic modification query using FORCED JDBC driver.
    """
    # Verify we're using JDBC by checking connection type
    from common.iris_connection_manager import get_driver_type
    driver_type = get_driver_type()
    assert driver_type == DriverType.JDBC, f"Expected JDBC driver, got {driver_type}"
    logger.info(f"✓ Confirmed using JDBC driver: {driver_type}")
    
    # Initialize HyDE pipeline components
    llm_func = get_llm_func()
    config_manager = ConfigurationManager()
    
    # Create HyDE pipeline with correct parameters
    hyde_pipeline = HyDERAGPipeline(
        config_manager=config_manager,
        llm_func=llm_func
    )
    
    # Test query about genetic modification
    query = "What are modern methods for altering genetic code?"
    logger.info(f"Executing HyDE E2E test with abstract query: {query}")
    
    # Execute HyDE pipeline using the correct method
    result = hyde_pipeline.query(query, top_k=5)
    
    # Verify result structure
    assert isinstance(result, dict), "HyDE result should be a dictionary"
    assert "query" in result, "Result should contain 'query' field"
    assert "answer" in result, "Result should contain 'answer' field"
    assert "retrieved_documents" in result, "Result should contain 'retrieved_documents' field"
    assert "hypothetical_document" in result, "Result should contain 'hypothetical_document' field"
    
    # Verify hypothetical document was generated
    hypothetical_doc = result["hypothetical_document"]
    assert isinstance(hypothetical_doc, str), "Hypothetical document should be a string"
    assert len(hypothetical_doc) > 50, "Hypothetical document should be substantial"
    logger.info(f"HyDE generated hypothetical document for CRISPR query (first 100 chars): {hypothetical_doc[:100]}...")
    
    # Verify documents were retrieved
    retrieved_docs = result["retrieved_documents"]
    assert isinstance(retrieved_docs, list), "Retrieved documents should be a list"
    assert len(retrieved_docs) > 0, "Should retrieve at least one document"
    
    # Verify answer was generated
    answer = result["answer"]
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 10, "Answer should be substantial"
    
    # Log results
    doc_ids = [doc["doc_id"] for doc in retrieved_docs]
    logger.info(f"HyDE retrieved doc IDs for CRISPR query: {doc_ids}, Answer: {answer[:100]}...")
    
    # Verify we retrieved expected documents
    assert any("DOCA" in doc_id or "DOCB" in doc_id for doc_id in doc_ids), \
        f"Should retrieve test documents, got: {doc_ids}"
    
    logger.info("✅ HyDE E2E test for abstract query (genetic modification) passed successfully with JDBC driver.")