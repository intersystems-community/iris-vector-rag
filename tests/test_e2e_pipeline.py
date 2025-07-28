"""
Tests for verifying the end-to-end RAG pipeline.
Ensures data flows correctly from ingestion to search and answer generation.
"""

import pytest
import logging
import os
import sys
from typing import List, Dict, Any, Callable, Tuple

from iris_rag.pipelines.basic import BasicRAGPipeline as BasicRAGPipeline
from common.utils import get_embedding_func, get_llm_func

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import IRISConnectionManager
from data.unified_loader import process_and_load_documents_unified

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Define a small, distinct set of test documents.
# These should be placed in a dedicated test data directory, e.g., 'tests/test_data/e2e_docs/'
# For this example, we'll define their content inline or assume they exist.
# It's better to have actual files.
TEST_E2E_DOC_DIR = os.path.join(os.path.dirname(__file__), 'test_data', 'e2e_docs')
# Example: Create these files manually in tests/test_data/e2e_docs/
# doc_A.xml: <article><front><article-meta><article-id pub-id-type="pmc">DOCA</article-id><title-group><article-title>Mitochondrial DNA</article-title></title-group></article-meta></front><body><p>Mitochondrial DNA is crucial for cellular respiration.</p></body></article>
# doc_B.xml: <article><front><article-meta><article-id pub-id-type="pmc">DOCB</article-id><title-group><article-title>CRISPR Gene Editing</article-title></title-group></article-meta></front><body><p>CRISPR allows for precise gene editing.</p></body></article>


@pytest.fixture(scope="module")
def e2e_db_connection():
    """
    Provides a database connection for the E2E test module.
    Initializes the database schema and ingests specific test documents.
    Ensures test-specific documents are cleared before ingestion for idempotency.
    """
    logger.info("Setting up database for E2E pipeline tests...")
    config_manager = ConfigurationManager()
    connection_manager = IRISConnectionManager(config_manager=config_manager)
    schema_manager = SchemaManager(connection_manager, config_manager)
    schema_manager.ensure_table_schema("SourceDocuments")
    schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
    schema_manager.ensure_table_schema("DocumentEntities")

    conn = connection_manager.get_connection()

    # Clean up specific test documents before ingestion to ensure idempotency of this fixture
    logger.info("Attempting to delete DOCA and DOCB if they exist to ensure clean test data ingestion.")
    try:
        with conn.cursor() as cursor:
            # Using a loop for clarity, could be a single DELETE with OR
            for doc_id_to_delete in ["DOCA", "DOCB"]:
                delete_sql = "DELETE FROM RAG.SourceDocuments WHERE doc_id = ?"
                cursor.execute(delete_sql, [doc_id_to_delete])
                logger.info(f"Executed delete for {doc_id_to_delete}. Rows affected: {cursor.rowcount}")
            conn.commit()
            logger.info("Finished attempting to delete DOCA and DOCB.")
    except Exception as e:
        logger.warning(f"Could not delete pre-existing test documents DOCA/DOCB: {e}. Proceeding with ingestion attempt.")
        # If deletion fails (e.g., table doesn't exist yet, though init should handle it),
        # we still proceed, and ingestion might fail, which is a valid test failure.
        conn.rollback() # Rollback any partial changes from failed delete attempt

    # Ensure the test document directory exists
    if not os.path.exists(TEST_E2E_DOC_DIR):
        os.makedirs(TEST_E2E_DOC_DIR)
        # Create dummy files if they don't exist (for test to run)
        with open(os.path.join(TEST_E2E_DOC_DIR, "DOCA.xml"), "w") as f:
            f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCA</article-id><title-group><article-title>Mitochondrial DNA</article-title></title-group></article-meta></front><body><p>Mitochondrial DNA is crucial for cellular respiration.</p></body></article>')
        with open(os.path.join(TEST_E2E_DOC_DIR, "DOCB.xml"), "w") as f:
            f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCB</article-id><title-group><article-title>CRISPR Gene Editing</article-title></title-group></article-meta></front><body><p>CRISPR allows for precise gene editing.</p></body></article>')
        logger.info(f"Created dummy test files in {TEST_E2E_DOC_DIR}")

    # Ingest the specific test documents
    logger.info(f"Ingesting E2E test documents from {TEST_E2E_DOC_DIR}")
    # Use actual embedding function for ingestion
    e2e_embedding_func = get_embedding_func()
    loader_config = {
        "limit": 2,
        "batch_size": 2,
        "embedding_column_type": "VECTOR"
    }
    ingestion_stats = process_and_load_documents_unified(
        config=loader_config,
        pmc_directory=TEST_E2E_DOC_DIR,
        colbert_doc_encoder_func=None # Not testing Colbert here
    )
    if not ingestion_stats["success"] or ingestion_stats["loaded_doc_count"] != 2:
        pytest.fail(f"Failed to ingest E2E test documents. Stats: {ingestion_stats}")
    
    logger.info("E2E test documents ingested successfully.")
    yield conn, config_manager
    
    logger.info("Closing database connection for E2E pipeline tests.")
    conn.close()


def test_e2e_ingest_search_retrieve_answer(e2e_db_connection):
    """
    Tests the full pipeline: ingest, search, verify document retrieval, and answer generation.
    """
    conn, config_manager = e2e_db_connection

    # Initialize the RAG pipeline
    test_embedding_func = get_embedding_func()
    test_llm_func = get_llm_func()
    pipeline = BasicRAGPipeline(
        config_manager=config_manager,
        llm_func=test_llm_func
    )

    # Test Case 1: Query targeting Doc A ("DOCA") - "Mitochondrial DNA"
    query_doc_a = "What is the role of mitochondrial DNA?"
    logger.info(f"Executing E2E test query 1: {query_doc_a}")
    results_a = pipeline.run(query_doc_a)

    assert "retrieved_documents" in results_a, "Query result missing 'retrieved_documents' key"
    assert "answer" in results_a, "Query result missing 'answer' key"
    
    retrieved_ids_a = [doc["id"] for doc in results_a["retrieved_documents"]] # 'id' not 'doc_id'
    logger.info(f"Query 1 retrieved doc IDs: {retrieved_ids_a}, Answer: {results_a['answer'][:100]}...")

    assert "DOCA" in retrieved_ids_a, \
        f"Expected 'DOCA' to be retrieved for query '{query_doc_a}', got {retrieved_ids_a}"
    assert len(results_a["answer"]) > 0, "Generated answer for Query 1 is empty"
    assert "couldn't find any relevant information" not in results_a["answer"].lower(), \
        "Answer for Query 1 indicates no information found, but DOCA should be relevant."
    # Optionally, check for keywords if the LLM is consistent enough
    # assert "mitochondrial" in results_a["answer"].lower() or "cellular respiration" in results_a["answer"].lower()


    # Test Case 2: Query targeting Doc B ("DOCB") - "CRISPR Gene Editing"
    query_doc_b = "Explain CRISPR gene editing technology." # This is the key query from the task
    logger.info(f"Executing E2E test query 2: {query_doc_b}")
    results_b = pipeline.run(query_doc_b)

    assert "retrieved_documents" in results_b
    assert "answer" in results_b
    
    retrieved_ids_b = [doc["id"] for doc in results_b["retrieved_documents"]] # 'id' not 'doc_id'
    logger.info(f"Query 2 retrieved doc IDs: {retrieved_ids_b}, Answer: {results_b['answer'][:100]}...")

    assert "DOCB" in retrieved_ids_b, \
        f"Expected 'DOCB' to be retrieved for query '{query_doc_b}', got {retrieved_ids_b}"
    assert len(results_b["answer"]) > 0, "Generated answer for Query 2 is empty"
    assert "couldn't find any relevant information" not in results_b["answer"].lower(), \
        "Answer for Query 2 indicates no information found, but DOCB (CRISPR) should be relevant."
    assert "crispr" in results_b["answer"].lower(), \
        f"Answer for Query 2 (CRISPR) does not seem to mention 'crispr'. Answer: {results_b['answer']}"
    # Optionally, check for "gene editing"
    # assert "gene editing" in results_b["answer"].lower()


    # Test Case 3: Query for content not present
    query_not_present = "Latest advancements in underwater basket weaving."
    logger.info(f"Executing E2E test query 3: {query_not_present}")
    results_c = pipeline.run(query_not_present)
    
    assert "retrieved_documents" in results_c
    assert "answer" in results_c

    retrieved_ids_c = [doc["id"] for doc in results_c["retrieved_documents"]] # 'id' not 'doc_id'
    logger.info(f"Query 3 retrieved doc IDs: {retrieved_ids_c}, Answer: {results_c['answer'][:100]}...")
    
    # For a query with no relevant documents, the answer should reflect that.
    # The pipeline might still retrieve some low-similarity documents.
    # The primary check is that the LLM indicates no useful information was found from the (potentially irrelevant) context.
    
    answer_c_lower = results_c["answer"].lower()
    assert "couldn't find any relevant information" in answer_c_lower or \
           "does not contain any information" in answer_c_lower or \
           "no information related to" in answer_c_lower or \
           "unable to provide information" in answer_c_lower or \
           "context does not provide information" in answer_c_lower, \
           f"Answer for irrelevant query C should indicate no relevant info, but was: {results_c['answer']}"

    # Optionally, we can still check if specific highly irrelevant docs are NOT in the top results,
    # but this is less critical than the answer content.
    # For this test, we'll rely on the answer content check above.
    # logger.debug(f"Retrieved IDs for irrelevant query: {retrieved_ids_c}")

    logger.info("✅ End-to-end pipeline test passed successfully with BasicRAGPipeline.")


if __name__ == "__main__":
    # Manual setup for direct run
    logger.info("Running E2E pipeline test directly (not via pytest)...")
    
    config_manager = ConfigurationManager()
    connection_manager = IRISConnectionManager(config_manager=config_manager)
    schema_manager = SchemaManager(connection_manager, config_manager)
    schema_manager.ensure_table_schema("SourceDocuments")
    schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
    schema_manager.ensure_table_schema("DocumentEntities")
    
    # Create dummy test files if they don't exist
    if not os.path.exists(TEST_E2E_DOC_DIR):
        os.makedirs(TEST_E2E_DOC_DIR)
    if not os.path.exists(os.path.join(TEST_E2E_DOC_DIR, "DOCA.xml")):
         with open(os.path.join(TEST_E2E_DOC_DIR, "DOCA.xml"), "w") as f:
            f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCA</article-id><title-group><article-title>Mitochondrial DNA</article-title></title-group></article-meta></front><body><p>Mitochondrial DNA is crucial for cellular respiration.</p></body></article>')
    if not os.path.exists(os.path.join(TEST_E2E_DOC_DIR, "DOCB.xml")):
        with open(os.path.join(TEST_E2E_DOC_DIR, "DOCB.xml"), "w") as f:
            f.write('<article><front><article-meta><article-id pub-id-type="pmc">DOCB</article-id><title-group><article-title>CRISPR Gene Editing</article-title></title-group></article-meta></front><body><p>CRISPR allows for precise gene editing.</p></body></article>')

    temp_conn_e2e = get_iris_connection()
    
    # Manually ingest for direct run, now with embedding function
    direct_embedding_func = get_embedding_func()
    loader_config = {
        "limit": 2,
        "batch_size": 2,
        "embedding_column_type": "VECTOR"
    }
    ingestion_stats_direct = process_and_load_documents_unified(
        config=loader_config,
        pmc_directory=TEST_E2E_DOC_DIR
    )
    if not ingestion_stats_direct["success"] or ingestion_stats_direct["loaded_doc_count"] != 2:
        logger.error(f"Direct run: Failed to ingest E2E test documents. Stats: {ingestion_stats_direct}")
    else:
        # The test_e2e_ingest_search_retrieve_answer function now handles pipeline creation
        # So, we can call it directly.
        try:
            logger.info("Direct run: Executing test_e2e_ingest_search_retrieve_answer...")
            test_e2e_ingest_search_retrieve_answer(temp_conn_e2e)
            logger.info("Direct run: test_e2e_ingest_search_retrieve_answer completed.")
        except Exception as e:
            logger.error(f"Direct run: Error during test execution: {e}", exc_info=True)
        finally:
            temp_conn_e2e.close()
    logger.info("Direct E2E run finished.")