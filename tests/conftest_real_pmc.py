"""
Real PMC document conftest for testing with 1000+ documents.

This is a specialized conftest file that ensures tests use real PMC documents,
satisfying the .clinerules requirement that tests must use real PMC documents
and not synthetic data.
"""

import os
import sys
import logging
import pytest
import time
from typing import Dict, List, Any, Callable, Optional

# Add parent directory to Python path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection # Use the standard connection utility
# Removed: setup_docker_test_db, initialize_database, process_and_load_documents, get_embedding_func
# These are now handled by user-run scripts before tests.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conftest_real_pmc")

# Constants
MIN_REQUIRED_DOCUMENTS = 1000 # Reverted to original value, or can be controlled by env var if needed
# PMC_DATA_DIR is not used by this fixture anymore as data loading is external.

# Define test-specific environment variables (if still needed by other parts of the code)
# os.environ["IRIS_USE_REAL_DATA"] = "1" # This might be implicitly true now
# os.environ["IRIS_TEST_PMC_LIMIT"] = str(MIN_REQUIRED_DOCUMENTS) # Data loading limit is now in load_pmc_data.py

@pytest.fixture(scope="session")
def e2e_iris_connection(): # Renamed fixture
    """
    Provides a DB-API connection to an already running and data-loaded IRIS instance
    for End-to-End (E2E) tests.
    Verifies that the database meets prerequisites for E2E tests (e.g., document count).
    """
    logger.info("Attempting to connect to pre-existing IRIS container for E2E tests.")
    logger.info("Prerequisites: IRIS container must be running (e.g., 'make start-iris'), "
                "schema initialized ('make init-db'), and data loaded ('make load-data').")

    conn = None
    try:
        conn = get_iris_connection() # Get connection using the standard utility
        if not conn:
            pytest.fail("Failed to connect to IRIS. Ensure IRIS container is running and accessible "
                        "and connection environment variables (IRIS_HOST, IRIS_PORT, etc.) are set.")
        
        # Verify data presence and count
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM RAG.SourceDocuments") # Schema qualified
            count = cursor.fetchone()[0]
            logger.info(f"Found {count} documents in RAG.SourceDocuments.")
            
            # Determine required docs, potentially from an environment variable for flexibility
            # For now, using the constant.
            required_docs = int(os.environ.get("E2E_MIN_DOCS", MIN_REQUIRED_DOCUMENTS))

            if count < required_docs:
                pytest.fail(
                    f"Prerequisite not met: Expected at least {required_docs} documents "
                    f"in RAG.SourceDocuments, but found {count}. "
                    f"Please ensure data is loaded correctly before running E2E tests (e.g., 'make load-data')."
                )
        logger.info(f"Successfully connected to IRIS and verified document count >= {required_docs}.")
        yield conn # Provide the connection to tests
    
    except Exception as e:
        logger.error(f"Error during e2e_iris_connection fixture setup: {e}") # Updated log message
        if conn and hasattr(conn, 'close'): # Attempt to close connection if opened before failure
            try:
                conn.close()
            except Exception as e_close:
                logger.error(f"Error closing connection during setup failure: {e_close}")
        pytest.fail(f"IRIS E2E fixture setup failed: {e}")
    
    finally:
        # Clean up: just close the connection
        if conn and hasattr(conn, 'close'):
            logger.info("Closing IRIS database connection after E2E test session.")
            try:
                conn.close()
            except Exception as e_close:
                 logger.error(f"Error closing IRIS connection in fixture teardown: {e_close}")
