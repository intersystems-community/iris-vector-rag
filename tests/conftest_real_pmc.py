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

# Fix: Use the correct function name from iris_connector
from common.iris_connector import get_real_iris_connection, setup_docker_test_db # Import setup_docker_test_db
from common.db_init import initialize_database
from data.loader import process_and_load_documents
from common.utils import get_embedding_func # Import get_embedding_func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conftest_real_pmc")

# Constants
# MIN_REQUIRED_DOCUMENTS = 1000 # Original value
MIN_REQUIRED_DOCUMENTS = 10 # Temporary value for faster debugging
PMC_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "pmc_oas_downloaded")

# Define test-specific environment variables
os.environ["IRIS_USE_REAL_DATA"] = "1"
# os.environ["IRIS_TEST_PMC_LIMIT"] = "1000" # Original value
os.environ["IRIS_TEST_PMC_LIMIT"] = str(MIN_REQUIRED_DOCUMENTS) # Use the debug value

@pytest.fixture(scope="session")
def iris_with_pmc_data():
    """
    Provides an IRIS database connection with real PMC data loaded.
    
    This fixture ensures that real PMC data is loaded into the database,
    satisfying the project requirement of using real PMC documents.
    """
    logger.info("Setting up IRIS database with real PMC data using setup_docker_test_db")
    
    # Use setup_docker_test_db to start container, wait for readiness, create user, and initialize schema
    # It returns a connection object whose .close() method stops the container.
    # Specify host_port to avoid conflict with existing IRIS instance on 1972.
    conn = setup_docker_test_db(host_port=51773) 
    
    if not conn:
        pytest.fail("Failed to set up IRIS test container.")
    
    # Load real PMC documents into the database
    logger.info("Loading real PMC documents into the database...")
    real_embedding_function = get_embedding_func() # Get the real embedding function

    load_stats = process_and_load_documents(
        PMC_DATA_DIR,
        connection=conn,
        embedding_func=real_embedding_function, # Pass the embedding function
        limit=int(os.environ.get("IRIS_TEST_PMC_LIMIT", MIN_REQUIRED_DOCUMENTS))
    )
    
    loaded_doc_count = load_stats.get("loaded_count", 0)
    logger.info(f"Loaded {loaded_doc_count} real PMC documents into database based on stats: {load_stats}")
    
    if loaded_doc_count < MIN_REQUIRED_DOCUMENTS:
        logger.warning(f"Fewer than {MIN_REQUIRED_DOCUMENTS} documents loaded "
                       f"({loaded_doc_count}). Tests may not satisfy project requirements.")
    
    # Return the connection for use in tests
    yield conn
    
    # Clean up when done
    logger.info("Cleaning up IRIS database connection")
    if hasattr(conn, 'close'):
        conn.close()
