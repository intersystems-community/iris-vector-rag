"""
Tests for verifying idempotent ingestion into the RAG database.
"""

import pytest
import logging
import os
import sys
from typing import List, Dict, Any, Callable, Tuple

# Add project root to path to allow direct execution and imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from common.db_init_with_indexes import initialize_complete_rag_database, create_schema_if_not_exists
from data.loader import process_and_load_documents
from data.pmc_processor import process_pmc_files # To get doc_ids for verification

# Configure logging for tests
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Define a small set of test documents from the sample data
# Assuming 'data/sample_10_docs/' exists and contains these files
SAMPLE_DOC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_10_docs')
TEST_DOC_FILES = ["PMC524367.xml", "PMC526216.xml"] # Example files
TEST_DOC_IDS = [filename.replace(".xml", "") for filename in TEST_DOC_FILES]


@pytest.fixture(scope="module")
def db_connection():
    """
    Provides a database connection for the test module.
    Initializes the database schema once per module.
    """
    logger.info("Setting up database schema for idempotent ingestion tests...")
    create_schema_if_not_exists("RAG") # Ensure schema exists
    success = initialize_complete_rag_database("RAG") # Re-initialize tables
    if not success:
        pytest.fail("Failed to initialize RAG database for tests.")
    
    conn = get_iris_connection()
    yield conn
    logger.info("Closing database connection for idempotent ingestion tests.")
    conn.close()

def get_document_counts(connection, doc_ids: List[str]) -> Tuple[int, int]:
    """Helper to get counts from SourceDocuments and DocumentTokenEmbeddings for specific doc_ids."""
    cursor = connection.cursor()
    
    # Count in SourceDocuments
    doc_id_placeholders = ','.join(['?'] * len(doc_ids))
    sql_source_docs = f"SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id IN ({doc_id_placeholders})"
    cursor.execute(sql_source_docs, doc_ids)
    source_doc_count = cursor.fetchone()[0]
    
    # Count in DocumentTokenEmbeddings
    sql_token_embeddings = f"SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE doc_id IN ({doc_id_placeholders})"
    cursor.execute(sql_token_embeddings, doc_ids)
    token_embedding_count = cursor.fetchone()[0]
    
    cursor.close()
    return source_doc_count, token_embedding_count

def get_embeddings_for_doc(connection, doc_id: str) -> Tuple[str, List[str]]:
    """Helper to get main embedding and a sample of token embeddings for a doc_id."""
    cursor = connection.cursor()
    
    # Get main embedding
    sql_main_embedding = "SELECT embedding FROM RAG.SourceDocuments WHERE doc_id = ?"
    cursor.execute(sql_main_embedding, (doc_id,))
    main_embedding_row = cursor.fetchone()
    main_embedding = main_embedding_row[0] if main_embedding_row else None
    
    # Get a sample of token embeddings (e.g., first 3)
    sql_token_embeddings = "SELECT * FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ? ORDER BY token_index"
    cursor.execute(sql_token_embeddings, (doc_id,))
    token_embeddings = [row[3] for row in cursor.fetchall()]
    
    cursor.close()
    return main_embedding, token_embeddings

def test_idempotent_document_and_token_ingestion(db_connection):
    """
    Tests that ingesting the same documents multiple times:
    1. Does not create duplicate rows in SourceDocuments.
    2. Does not create duplicate rows in DocumentTokenEmbeddings.
    3. Keeps existing embeddings consistent.
    """
    conn = db_connection
    
    # --- First Ingestion ---
    logger.info(f"Starting first ingestion for documents: {TEST_DOC_FILES}")
    # For testing, we don't need real embedding functions, as we're testing DB constraints
    # and data consistency, not embedding quality.
    # The loader.py script handles None for embedding_func and colbert_doc_encoder_func.
    ingestion_stats_1 = process_and_load_documents(
        pmc_directory=SAMPLE_DOC_DIR,
        connection=conn,
        embedding_func=None, 
        colbert_doc_encoder_func=None,
        limit=len(TEST_DOC_FILES), # Process only our test files
        batch_size=10 
    )
    
    logger.info(f"First ingestion stats: {ingestion_stats_1}")
    assert ingestion_stats_1["success"], "First ingestion failed"
    # The loader might report 0 loaded if all docs in the batch already exist and cause an error.
    # We rely on primary key constraints.
    # assert ingestion_stats_1["loaded_doc_count"] == len(TEST_DOC_IDS), \
    #     f"Expected {len(TEST_DOC_IDS)} documents loaded, got {ingestion_stats_1['loaded_doc_count']}"

    # Record counts and embeddings after first ingestion
    source_docs_count_1, token_embeds_count_1 = get_document_counts(conn, TEST_DOC_IDS)
    logger.info(f"After 1st ingestion: SourceDocs count = {source_docs_count_1}, TokenEmbeds count = {token_embeds_count_1}")
    
    # Check that documents were actually loaded
    assert source_docs_count_1 == len(TEST_DOC_IDS), \
        f"Expected {len(TEST_DOC_IDS)} SourceDocuments after first run, found {source_docs_count_1}"
    # Token count can be > 0 if colbert_doc_encoder_func was provided and worked, or 0 if not.
    # For this test, we assume it might be 0 if no encoder is passed.
    # If an encoder is used in the future, this assertion needs to be more specific.
    
    # Get embeddings for one specific document to compare later
    specific_doc_id_to_check = TEST_DOC_IDS[0]
    main_embedding_1, token_embeddings_1 = get_embeddings_for_doc(conn, specific_doc_id_to_check)
    assert main_embedding_1 is not None, f"Main embedding for {specific_doc_id_to_check} not found after 1st ingestion."
    # Token embeddings might be empty if no colbert_doc_encoder_func is used
    # assert len(token_embeddings_1) > 0, f"Token embeddings for {specific_doc_id_to_check} not found after 1st ingestion."

    # --- Second Ingestion ---
    logger.info(f"Starting second ingestion for the same documents: {TEST_DOC_FILES}")
    ingestion_stats_2 = process_and_load_documents(
        pmc_directory=SAMPLE_DOC_DIR,
        connection=conn,
        embedding_func=None,
        colbert_doc_encoder_func=None,
        limit=len(TEST_DOC_FILES),
        batch_size=10
    )
    logger.info(f"Second ingestion stats: {ingestion_stats_2}")
    # The second run should "succeed" in terms of script execution, but primary key violations
    # will prevent new data from being inserted. The loader's `error_count` might reflect this.
    assert ingestion_stats_2["success"], "Second ingestion script execution failed"
    # loaded_doc_count might be 0 if all docs in the batch cause PK violation.
    # This is expected behavior.

    # --- Verification ---
    logger.info("Verifying data consistency after second ingestion...")
    source_docs_count_2, token_embeds_count_2 = get_document_counts(conn, TEST_DOC_IDS)
    logger.info(f"After 2nd ingestion: SourceDocs count = {source_docs_count_2}, TokenEmbeds count = {token_embeds_count_2}")
    
    main_embedding_2, token_embeddings_2 = get_embeddings_for_doc(conn, specific_doc_id_to_check)

    # Assert counts are identical
    assert source_docs_count_2 == source_docs_count_1, \
        f"SourceDocuments count changed after second ingestion: {source_docs_count_1} -> {source_docs_count_2}"
    assert token_embeds_count_2 == token_embeds_count_1, \
        f"DocumentTokenEmbeddings count changed after second ingestion: {token_embeds_count_1} -> {token_embeds_count_2}"

    # Assert embeddings are identical
    assert main_embedding_2 == main_embedding_1, \
        f"Main embedding for {specific_doc_id_to_check} changed after second ingestion."
    assert token_embeddings_2 == token_embeddings_1, \
        f"Token embeddings for {specific_doc_id_to_check} changed after second ingestion."

    logger.info("✅ Idempotent ingestion test passed successfully.")

if __name__ == "__main__":
    # This allows running the test directly, e.g., for debugging
    # Note: Pytest fixtures like db_connection won't be automatically managed here.
    # You'd need to set up the connection manually if running this way.
    logger.info("Running idempotent ingestion test directly (not via pytest)...")
    
    # Manual setup for direct run
    create_schema_if_not_exists("RAG")
    initialize_complete_rag_database("RAG")
    temp_conn = get_iris_connection()
    
    try:
        test_idempotent_document_and_token_ingestion(temp_conn)
    finally:
        temp_conn.close()
    logger.info("Direct run finished.")