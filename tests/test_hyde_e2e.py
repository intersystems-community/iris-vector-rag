"""
Tests for verifying the end-to-end HyDE RAG pipeline.
Ensures HyDE correctly handles abstract queries by generating hypothetical documents.
"""

import pytest
import logging
import os
import sys
from typing import List, Dict, Any, Callable, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experimental.hyde.pipeline import HyDEPipeline # Corrected import path and class name
from common.utils import get_embedding_func, get_llm_func # Updated import
from common.iris_connector import get_iris_connection # Updated import
from common.db_init_with_indexes import initialize_complete_rag_database, create_schema_if_not_exists # Updated import
from data.loader import process_and_load_documents # Path remains correct

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Define the same test document directory as used in test_e2e_pipeline.py
TEST_E2E_DOC_DIR = os.path.join(os.path.dirname(__file__), 'test_data', 'e2e_docs')
# DOCA.xml: <article><front><article-meta><article-id pub-id-type="pmc">DOCA</article-id><title-group><article-title>Mitochondrial DNA</article-title></title-group></article-meta></front><body><p>Mitochondrial DNA is crucial for cellular respiration.</p></body></article>
# DOCB.xml: <article><front><article-meta><article-id pub-id-type="pmc">DOCB</article-id><title-group><article-title>CRISPR Gene Editing</article-title></title-group></article-meta></front><body><p>CRISPR allows for precise gene editing.</p></body></article>

@pytest.fixture(scope="module")
def hyde_e2e_db_connection():
    """
    Provides a database connection for the HyDE E2E test module.
    Initializes the database schema and ingests specific test documents (DOCA, DOCB).
    Ensures test-specific documents are cleared before ingestion for idempotency.
    """
    logger.info("Setting up database for HyDE E2E pipeline tests...")
    create_schema_if_not_exists("RAG") # Ensure schema exists
    success_init = initialize_complete_rag_database("RAG") # Initialize tables and HNSW indexes
    if not success_init:
        pytest.fail("Failed to initialize RAG database for HyDE E2E tests.")

    conn = get_iris_connection()

    logger.info("Attempting to delete DOCA and DOCB if they exist to ensure clean test data ingestion for HyDE.")
    try:
        with conn.cursor() as cursor:
            for doc_id_to_delete in ["DOCA", "DOCB"]:
                delete_sql = "DELETE FROM RAG.SourceDocuments WHERE doc_id = ?"
                cursor.execute(delete_sql, [doc_id_to_delete])
                logger.info(f"HyDE E2E: Executed delete for {doc_id_to_delete}. Rows affected: {cursor.rowcount}")
            conn.commit()
            logger.info("HyDE E2E: Finished attempting to delete DOCA and DOCB.")
    except Exception as e:
        logger.warning(f"HyDE E2E: Could not delete pre-existing test documents DOCA/DOCB: {e}. Proceeding.")
        conn.rollback()

    # Ensure the test document directory and files exist (copied from test_e2e_pipeline.py)
    if not os.path.exists(TEST_E2E_DOC_DIR):
        os.makedirs(TEST_E2E_DOC_DIR)
    doc_a_path = os.path.join(TEST_E2E_DOC_DIR, "DOCA.xml")
    doc_b_path = os.path.join(TEST_E2E_DOC_DIR, "DOCB.xml")

    if not os.path.exists(doc_a_path):
        with open(doc_a_path, "w") as f:
            f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCA</article-id><title-group><article-title>Mitochondrial DNA</article-title></title-group></article-meta></front><body><p>Mitochondrial DNA is crucial for cellular respiration.</p></body></article>')
        logger.info(f"Created dummy test file: {doc_a_path}")
    if not os.path.exists(doc_b_path):
        with open(doc_b_path, "w") as f:
            f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCB</article-id><title-group><article-title>CRISPR Gene Editing</article-title></title-group></article-meta></front><body><p>CRISPR allows for precise gene editing.</p></body></article>')
        logger.info(f"Created dummy test file: {doc_b_path}")

    logger.info(f"HyDE E2E: Ingesting E2E test documents from {TEST_E2E_DOC_DIR}")
    e2e_embedding_func = get_embedding_func() # Use real embedding function
    ingestion_stats = process_and_load_documents(
        pmc_directory=TEST_E2E_DOC_DIR,
        connection=conn,
        embedding_func=e2e_embedding_func,
        colbert_doc_encoder_func=None,
        limit=2,
        batch_size=2
    )
    if not ingestion_stats["success"] or ingestion_stats["loaded_doc_count"] != 2:
        pytest.fail(f"HyDE E2E: Failed to ingest E2E test documents. Stats: {ingestion_stats}")
    
    logger.info("HyDE E2E: Test documents ingested successfully.")
    yield conn
    
    logger.info("HyDE E2E: Closing database connection.")
    conn.close()


def test_hyde_e2e_abstract_query_cellular_energy(hyde_e2e_db_connection):
    """
    Tests the HyDE pipeline with an abstract query related to cellular energy,
    expecting to retrieve DOCA (Mitochondrial DNA).
    """
    conn = hyde_e2e_db_connection

    # Initialize the HyDE pipeline
    test_embedding_func = get_embedding_func()
    test_llm_func = get_llm_func() # Use real LLM for hypothetical doc generation
    
    # Ensure LLM function is valid
    if test_llm_func is None:
        pytest.skip("LLM function not available, skipping HyDE test that requires it.")

    pipeline = HyDEPipelineV2(
        iris_connector=conn,
        embedding_func=test_embedding_func,
        llm_func=test_llm_func
    )

    # Abstract query: "How do cells produce energy?"
    # DOCA content: "Mitochondrial DNA is crucial for cellular respiration."
    abstract_query = "How do cells produce energy?"
    logger.info(f"Executing HyDE E2E test with abstract query: {abstract_query}")
    
    results = pipeline.run(abstract_query, top_k=1) # Ask for top 1

    assert "retrieved_documents" in results, "HyDE result missing 'retrieved_documents' key"
    assert "answer" in results, "HyDE result missing 'answer' key"
    assert "hypothetical_document" in results, "HyDE result missing 'hypothetical_document' key"

    hypothetical_doc = results["hypothetical_document"]
    logger.info(f"HyDE generated hypothetical document (first 100 chars): {hypothetical_doc[:100]}...")
    assert len(hypothetical_doc) > 0, "HyDE generated an empty hypothetical document."
    # A simple check, could be more sophisticated
    assert "cell" in hypothetical_doc.lower() or "energy" in hypothetical_doc.lower() or "respiration" in hypothetical_doc.lower(), \
        f"Hypothetical document doesn't seem related to 'cellular energy'. Got: {hypothetical_doc[:200]}"

    retrieved_docs = results["retrieved_documents"]
    assert len(retrieved_docs) > 0, f"HyDE retrieved no documents for abstract query: {abstract_query}"
    
    retrieved_ids = [doc["id"] for doc in retrieved_docs]
    logger.info(f"HyDE retrieved doc IDs: {retrieved_ids}, Answer: {results['answer'][:100]}...")

    assert "DOCA" in retrieved_ids, \
        f"Expected 'DOCA' (Mitochondrial DNA) to be retrieved for abstract query '{abstract_query}', got {retrieved_ids}. Hypothetical doc: {hypothetical_doc[:200]}"
    
    assert len(results["answer"]) > 0, "HyDE generated answer is empty"
    assert "couldn't find any relevant information" not in results["answer"].lower(), \
        "HyDE answer indicates no information found, but DOCA should be relevant via hypothetical document."
    # Check if the answer mentions mitochondria or respiration, which are key concepts from DOCA
    assert "mitochondria" in results["answer"].lower() or "cellular respiration" in results["answer"].lower() or "energy production" in results["answer"].lower(), \
        f"HyDE answer for '{abstract_query}' does not seem to relate to mitochondrial energy. Answer: {results['answer']}"

    logger.info("✅ HyDE E2E test for abstract query (cellular energy) passed successfully.")


def test_hyde_e2e_abstract_query_genetic_modification(hyde_e2e_db_connection):
    """
    Tests the HyDE pipeline with an abstract query related to genetic modification,
    expecting to retrieve DOCB (CRISPR Gene Editing).
    """
    conn = hyde_e2e_db_connection

    test_embedding_func = get_embedding_func()
    test_llm_func = get_llm_func()
    if test_llm_func is None:
        pytest.skip("LLM function not available, skipping HyDE test that requires it.")

    pipeline = HyDEPipelineV2(
        iris_connector=conn,
        embedding_func=test_embedding_func,
        llm_func=test_llm_func
    )

    # Abstract query: "What are modern methods for altering genetic code?"
    # DOCB content: "CRISPR allows for precise gene editing."
    abstract_query_crispr = "What are modern methods for altering genetic code?"
    logger.info(f"Executing HyDE E2E test with abstract query: {abstract_query_crispr}")

    results_crispr = pipeline.run(abstract_query_crispr, top_k=1)

    assert "hypothetical_document" in results_crispr
    hypothetical_doc_crispr = results_crispr["hypothetical_document"]
    logger.info(f"HyDE generated hypothetical document for CRISPR query (first 100 chars): {hypothetical_doc_crispr[:100]}...")
    assert len(hypothetical_doc_crispr) > 0
    assert "gene" in hypothetical_doc_crispr.lower() or "genetic" in hypothetical_doc_crispr.lower() or "dna" in hypothetical_doc_crispr.lower(), \
        f"Hypothetical document for CRISPR query doesn't seem related. Got: {hypothetical_doc_crispr[:200]}"


    retrieved_docs_crispr = results_crispr["retrieved_documents"]
    assert len(retrieved_docs_crispr) > 0, f"HyDE retrieved no documents for abstract query: {abstract_query_crispr}"

    retrieved_ids_crispr = [doc["id"] for doc in retrieved_docs_crispr]
    logger.info(f"HyDE retrieved doc IDs for CRISPR query: {retrieved_ids_crispr}, Answer: {results_crispr['answer'][:100]}...")

    assert "DOCB" in retrieved_ids_crispr, \
        f"Expected 'DOCB' (CRISPR) to be retrieved for abstract query '{abstract_query_crispr}', got {retrieved_ids_crispr}. Hypothetical doc: {hypothetical_doc_crispr[:200]}"
    
    assert len(results_crispr["answer"]) > 0
    assert "couldn't find any relevant information" not in results_crispr["answer"].lower()
    assert "crispr" in results_crispr["answer"].lower() or "gene editing" in results_crispr["answer"].lower(), \
        f"HyDE answer for '{abstract_query_crispr}' does not seem to relate to CRISPR/gene editing. Answer: {results_crispr['answer']}"

    logger.info("✅ HyDE E2E test for abstract query (genetic modification) passed successfully.")

if __name__ == "__main__":
    # This section allows direct execution of the test file, useful for debugging.
    # It will not run through pytest's fixture management in the same way,
    # so it's primarily for quick checks.
    logger.info("Running HyDE E2E tests directly (not via pytest)...")
    
    # Simplified setup for direct run
    # Note: This direct run might not perfectly replicate pytest environment (e.g. module-scoped fixtures)
    # but is useful for quick validation.
    
    # Create a temporary connection for the direct run
    temp_conn = None
    try:
        # Manually call what the fixture would do
        logger.info("Direct run: Setting up database...")
        create_schema_if_not_exists("RAG")
        initialize_complete_rag_database("RAG") # Ensure clean state
        
        temp_conn = get_iris_connection()

        # Clean up specific test documents
        try:
            with temp_conn.cursor() as cursor:
                for doc_id_to_delete in ["DOCA", "DOCB"]:
                    cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", [doc_id_to_delete])
                temp_conn.commit()
        except Exception as e:
            logger.warning(f"Direct run: Could not delete pre-existing test documents: {e}")
            temp_conn.rollback()

        # Ensure test files exist
        if not os.path.exists(TEST_E2E_DOC_DIR): os.makedirs(TEST_E2E_DOC_DIR)
        doc_a_path = os.path.join(TEST_E2E_DOC_DIR, "DOCA.xml")
        doc_b_path = os.path.join(TEST_E2E_DOC_DIR, "DOCB.xml")
        if not os.path.exists(doc_a_path):
            with open(doc_a_path, "w") as f: f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCA</article-id><title-group><article-title>Mitochondrial DNA</article-title></title-group></article-meta></front><body><p>Mitochondrial DNA is crucial for cellular respiration.</p></body></article>')
        if not os.path.exists(doc_b_path):
            with open(doc_b_path, "w") as f: f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCB</article-id><title-group><article-title>CRISPR Gene Editing</article-title></title-group></article-meta></front><body><p>CRISPR allows for precise gene editing.</p></body></article>')

        # Manually ingest documents
        logger.info("Direct run: Ingesting test documents...")
        direct_embedding_func = get_embedding_func()
        ingestion_stats_direct = process_and_load_documents(
            pmc_directory=TEST_E2E_DOC_DIR,
            connection=temp_conn,
            embedding_func=direct_embedding_func,
            limit=2, batch_size=2
        )
        if not ingestion_stats_direct["success"] or ingestion_stats_direct["loaded_doc_count"] != 2:
            logger.error(f"Direct run: Failed to ingest E2E test documents. Stats: {ingestion_stats_direct}")
        else:
            logger.info("Direct run: Test documents ingested. Running tests...")
            # Call test functions directly, passing the connection
            test_hyde_e2e_abstract_query_cellular_energy(temp_conn)
            test_hyde_e2e_abstract_query_genetic_modification(temp_conn)
            logger.info("Direct run: HyDE E2E tests completed.")

    except Exception as e:
        logger.error(f"Direct run: Error during execution: {e}", exc_info=True)
    finally:
        if temp_conn:
            temp_conn.close()
            logger.info("Direct run: Closed temporary database connection.")
    
    logger.info("Direct HyDE E2E run finished.")