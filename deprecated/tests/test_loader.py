# tests/test_loader.py

import pytest
import os
import sys
import sqlalchemy
from sqlalchemy.sql import text # For executing text-based SQL queries
from unittest.mock import MagicMock # Added import

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval.loader import DataLoader
from common.utils import get_embedding_func, get_llm_func # Using stub LLM by default
from common.db_init import initialize_database # To ensure schema can be created

# Testcontainers for managing a real IRIS instance for testing
try:
    from testcontainers.iris import IRISContainer
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

# Configuration for the test
# Using a very small number of files for quick testing
TEST_DATASET_CONFIG = {
    "name": "PMCOAS_TestLoader_Sample",
    # "source": "pmc_oas_hf", # Source is no longer used for download
    "data_dir": "data/test_loader_pmc_sample", # Directory with local XML files
    "max_files": 2, # Load only 2 articles for this test
    "batch_size": 1, # Process one by one for easier debugging if needed
    "format": "XML",
    "license_type": "oa_comm" 
}

@pytest.fixture(scope="module") # Module scope to run IRIS once for all tests in this file
def iris_db_connection():
    """
    Spins up an IRIS Docker container using testcontainers and provides a DBAPI connection.
    Yields the connection and ensures the container is stopped afterwards.
    """
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers library not installed or IRISContainer not available. Skipping loader tests.")

    # Determine IRIS image based on environment or default
    is_arm64 = os.uname().machine == 'arm64'
    default_image = "intersystemsdc/iris-community:latest" # Using latest for broader compatibility
    iris_image_tag = os.getenv("IRIS_DOCKER_IMAGE", default_image)
    
    print(f"\nAttempting to start IRIS Testcontainer with image: {iris_image_tag} for test_loader.py")
    
    db_conn = None
    engine = None
    try:
        with IRISContainer(iris_image_tag) as iris_container:
            connection_url = iris_container.get_connection_url()
            print(f"IRIS Testcontainer for test_loader.py started. URL: {connection_url}")
            engine = sqlalchemy.create_engine(connection_url)
            # Get a raw DBAPI connection from the SQLAlchemy engine
            # This is what DataLoader and initialize_database expect
            with engine.connect() as sa_conn:
                db_conn = sa_conn.connection # Get the underlying DBAPI connection
                yield db_conn 
    finally:
        if engine: # SQLAlchemy engine handles connection pool, dispose it
            engine.dispose()
        print("IRIS Testcontainer for test_loader.py stopped.")


def test_data_loader_pipeline(iris_db_connection):
    """
    Tests the full data loading pipeline: schema init, download, parse, process, embed, load.
    """
    if iris_db_connection is None: # Should be skipped by fixture if TC not available
        pytest.skip("IRIS connection not available for test_loader_pipeline.")

    # Data directory is now expected to contain the sample files.
    # No cleanup of XML files needed as we rely on them being present.
    data_dir = TEST_DATASET_CONFIG["data_dir"]
    assert os.path.exists(data_dir), f"Test data directory {data_dir} must exist and contain sample XML files."
    assert os.path.isdir(data_dir), f"Test data path {data_dir} must be a directory."

    # Get embedding and LLM functions (stub LLM is fine for loader tests)
    embedding_fn = get_embedding_func() # Default sentence transformer
    llm_fn = get_llm_func(provider="stub") # Stub LLM for any KG parts
    
    # Mock ColBERT encoder for now as it's complex and not the primary focus of this initial loader test
    # It should return a list of 1D embeddings (List[List[float]])
    mock_colbert_encoder = MagicMock(return_value=[[0.1]*10 for _ in range(5)]) # Simulates 5 token embeddings, each 10-dim.

    loader = DataLoader(
        iris_connector=iris_db_connection,
        embedding_func=embedding_fn,
        colbert_doc_encoder_func=mock_colbert_encoder,
        llm_func=llm_fn
    )

    # Run the full data loading process
    # The load_data method itself calls initialize_database
    print(f"test_loader.py: Calling loader.load_data with config: {TEST_DATASET_CONFIG}")
    loader.load_data(TEST_DATASET_CONFIG, force_recreate=True) # Force recreate schema for clean test

    # --- Assertions ---
    cursor = None
    try:
        cursor = iris_db_connection.cursor()

        # 1. Check if SourceDocuments table was created and has data
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        row_count = cursor.fetchone()[0]
        print(f"test_loader.py: SourceDocuments table has {row_count} rows.")
        # We expect documents to be chunked. 2 articles might produce more than 2 rows.
        # The exact number depends on titles, abstracts, and paragraphs.
        # For now, assert at least some rows were inserted.
        assert row_count > 0, "SourceDocuments table should have rows after loading."

        # 2. Check if some documents have embeddings
        # (The TO_VECTOR in db_init.sql might store NULL if embedding_str is None,
        #  but loader._generate_embeddings should populate doc.embedding)
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments WHERE embedding IS NOT NULL")
        embedded_row_count = cursor.fetchone()[0]
        print(f"test_loader.py: SourceDocuments table has {embedded_row_count} rows with non-NULL embeddings.")
        assert embedded_row_count > 0, "Some documents should have embeddings."
        assert embedded_row_count == row_count, "All loaded documents should have embeddings."


        # 3. Check a sample document's content (optional, more for deeper inspection)
        # cursor.execute("SELECT TOP 1 doc_id, text_content FROM SourceDocuments")
        # sample_doc = cursor.fetchone()
        # if sample_doc:
        #     print(f"test_loader.py: Sample document - ID: {sample_doc[0]}, Content snippet: {sample_doc[1][:100]}...")
        #     assert len(sample_doc[1]) > 0
            
        # 4. Check if DocumentTokenEmbeddings table was created (even if empty if ColBERT part is mocked)
        # The schema initialization should create it.
        table_exists = False
        try:
            cursor.execute("SELECT COUNT(*) FROM DocumentTokenEmbeddings")
            cursor.fetchone() # Consume the result
            table_exists = True
        except Exception: # Catch specific DB error if table doesn't exist
            pass 
        assert table_exists, "DocumentTokenEmbeddings table should exist after schema initialization."
        
        # If ColBERT embeddings were actually loaded (depends on mock_colbert_encoder behavior and loader logic)
        # cursor.execute("SELECT COUNT(*) FROM DocumentTokenEmbeddings")
        # token_embedding_count = cursor.fetchone()[0]
        # print(f"test_loader.py: DocumentTokenEmbeddings table has {token_embedding_count} rows.")
        # If mock_colbert_encoder is sophisticated enough to lead to data, assert token_embedding_count > 0

    finally:
        if cursor:
            cursor.close()

if __name__ == '__main__':
    # This allows running the test file directly with pytest
    # Ensure IRIS_CONNECTION_URL is set if not using testcontainers directly in a fixture
    # For this test, the fixture handles testcontainers.
    pytest.main([__file__, "-s", "-v"]) # -s for stdout, -v for verbose
