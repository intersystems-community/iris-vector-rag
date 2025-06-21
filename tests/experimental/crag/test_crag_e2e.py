import pytest
import logging
import sys # Added import
import os # Added import
from typing import List, Dict, Any, Callable
from unittest.mock import MagicMock # For spying on the mock web search

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experimental.crag.pipeline import CRAGPipeline # Updated import
from common.utils import Document # Updated import
# Fixtures like iris_testcontainer_connection, embedding_model_fixture,
# llm_client_fixture will be automatically provided by pytest from conftest.py

logger = logging.getLogger(__name__)

# Define a controlled set of document CHUNKS for testing CRAG's corrective mechanism
TEST_CHUNKS_FOR_CRAG = [
    {
        "chunk_id": "crag_chunk_1", "doc_id": "doc_A", 
        "chunk_text": "Solar power is a renewable energy source. It is clean.", 
        "chunk_type": "content", "chunk_index": 0, 
        "expected_score_initial": 0.35 # Designed to be low/ambiguous for "benefits of solar power"
    },
    {
        "chunk_id": "crag_chunk_2", "doc_id": "doc_B", 
        "chunk_text": "Wind turbines can be noisy but are effective for large scale energy production.", 
        "chunk_type": "content", "chunk_index": 0,
        "expected_score_initial": 0.15 # Irrelevant
    },
    {
        "chunk_id": "crag_chunk_3", "doc_id": "doc_C", 
        "chunk_text": "General information about various energy sources including fossil fuels and nuclear power.", 
        "chunk_type": "content", "chunk_index": 0,
        "expected_score_initial": 0.20 # Irrelevant
    }
]

# Mock web search results to be returned by our placeholder
# Helper to create Document objects and assign metadata, as Document dataclass doesn't take it in __init__
def _create_mock_web_doc_with_metadata(id_val: str, content_val: str, score_val: float, metadata_dict: Dict[str, Any]) -> Document:
    doc = Document(id=id_val, content=content_val, score=score_val)
    doc.metadata = metadata_dict # Dynamically assign metadata attribute
    return doc

MOCK_WEB_SEARCH_RESULTS = [
    _create_mock_web_doc_with_metadata(id_val="web_search_doc_1", content_val="Detailed article on the economic benefits of widespread solar power adoption, including job creation and reduced healthcare costs due to less pollution.", score_val=0.9, metadata_dict={"source": "mock_web_search"}),
    _create_mock_web_doc_with_metadata(id_val="web_search_doc_2", content_val="Environmental benefits of solar power: significant reduction in greenhouse gas emissions and water usage compared to traditional power plants.", score_val=0.88, metadata_dict={"source": "mock_web_search"}),
]

def placeholder_web_search_func(query: str) -> List[Document]:
    """A placeholder web search function for testing CRAG."""
    logger.info(f"PlaceholderWebSearch: Simulating web search for '{query}'")
    if "benefits of solar power" in query.lower():
        return MOCK_WEB_SEARCH_RESULTS
    return []

def insert_crag_test_data(iris_conn, embedding_func: Callable, chunks_data: List[Dict[str, Any]]):
    """Helper to insert test chunks into RAG.DocumentChunks and corresponding SourceDocuments."""
    logger.info(f"Inserting {len(chunks_data)} test chunks for CRAG JDBC E2E test.")
    
    source_docs_to_insert = {}
    for chunk_data in chunks_data:
        if chunk_data["doc_id"] not in source_docs_to_insert:
            source_docs_to_insert[chunk_data["doc_id"]] = {
                "id": chunk_data["doc_id"],
                "title": f"Test Source Document {chunk_data['doc_id']}",
                "content": f"Full content for document {chunk_data['doc_id']}.",
                "embedding_str": ','.join([f'{0.1:.10f}'] * 384), # Placeholder, not used by chunk query
                "source": "CRAG_E2E_TEST_DOC"
            }

    with iris_conn.cursor() as cursor:
        # Insert SourceDocuments first
        for doc_id, doc_data in source_docs_to_insert.items():
            try:
                sql_source = "INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, embedding, source) VALUES (?, ?, ?, ?, ?)"
                cursor.execute(sql_source, [doc_data["id"], doc_data["title"], doc_data["content"], doc_data["embedding_str"], doc_data["source"]])
            except Exception as e:
                if "PRIMARY KEY constraint" in str(e) or "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
                    logger.warning(f"SourceDocument {doc_id} already exists. Skipping insertion.")
                else:
                    logger.error(f"Failed to insert SourceDocument {doc_id}: {e}")
                    raise
        
        # Insert DocumentChunks
        chunk_texts_to_embed = [chunk["chunk_text"] for chunk in chunks_data]
        if not chunk_texts_to_embed:
            logger.info("No chunk texts to embed for DocumentChunks.")
            iris_conn.commit()
            return

        embeddings = embedding_func(chunk_texts_to_embed)

        for i, chunk_data in enumerate(chunks_data):
            embedding_vector_str = ','.join([f'{x:.10f}' for x in embeddings[i]])
            metadata_json_str = f'{{"expected_score_initial": {chunk_data.get("expected_score_initial", 0.0)}}}'
            try:
                sql_chunk = """
                INSERT INTO RAG.DocumentChunks 
                    (chunk_id, doc_id, chunk_text, embedding, chunk_type, chunk_index, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(sql_chunk, [
                    chunk_data["chunk_id"], chunk_data["doc_id"], chunk_data["chunk_text"],
                    embedding_vector_str, chunk_data["chunk_type"], 
                    chunk_data["chunk_index"], metadata_json_str
                ])
                logger.debug(f"Inserted DocumentChunk: {chunk_data['chunk_id']}")
            except Exception as e:
                if "PRIMARY KEY constraint" in str(e) or "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
                    logger.warning(f"DocumentChunk {chunk_data['chunk_id']} already exists. Attempting update.")
                    update_sql_chunk = """
                    UPDATE RAG.DocumentChunks 
                    SET doc_id = ?, chunk_text = ?, embedding = ?, chunk_type = ?, chunk_index = ?, metadata_json = ?
                    WHERE chunk_id = ?
                    """
                    cursor.execute(update_sql_chunk, [
                        chunk_data["doc_id"], chunk_data["chunk_text"], embedding_vector_str,
                        chunk_data["chunk_type"], chunk_data["chunk_index"], metadata_json_str,
                        chunk_data["chunk_id"]
                    ])
                    logger.debug(f"Updated DocumentChunk: {chunk_data['chunk_id']}")
                else:
                    logger.error(f"Failed to insert/update DocumentChunk {chunk_data['chunk_id']}: {e}")
                    raise
    iris_conn.commit()
    logger.info("Test data insertion for CRAG (SourceDocuments and DocumentChunks) complete.")


@pytest.mark.usefixtures("iris_testcontainer_connection", "embedding_model_fixture", "llm_client_fixture")
def test_crag_jdbc_e2e_corrective_web_search_triggered(
    iris_testcontainer_connection, 
    embedding_model_fixture,     
    llm_client_fixture,          
    caplog,
    mocker 
):
    """
    Tests CRAGPipeline's corrective mechanism, specifically web search augmentation.
    - Inserts chunks designed for low initial relevance.
    - Uses a placeholder web_search_func.
    - Verifies web search is triggered and results are incorporated.
    """
    caplog.set_level(logging.INFO) 
    
    logger.info("Preparing database for CRAG JDBC E2E corrective web search test.")
    with iris_testcontainer_connection.cursor() as cursor:
        logger.info("Clearing RAG.DocumentChunks and RAG.SourceDocuments for test data.")
        try:
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id LIKE 'crag_chunk_%'")
            cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id LIKE 'doc_A' OR doc_id LIKE 'doc_B' OR doc_id LIKE 'doc_C'")
            iris_testcontainer_connection.commit()
        except Exception as e:
            logger.warning(f"Could not clear tables (may be normal if first run): {e}")
            iris_testcontainer_connection.rollback() # Rollback on error during clear
            from common.db_init import initialize_database
            try:
                initialize_database(iris_testcontainer_connection, force_recreate=False)
                logger.info("Re-ran initialize_database after clear attempt.")
                # Try clearing again after ensuring schema exists
                cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id LIKE 'crag_chunk_%'")
                cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id LIKE 'doc_A' OR doc_id LIKE 'doc_B' OR doc_id LIKE 'doc_C'")
                iris_testcontainer_connection.commit()
            except Exception as e_init:
                 logger.error(f"Failed to initialize_database or clear after init: {e_init}")
                 iris_testcontainer_connection.rollback()
                 raise 
    
    insert_crag_test_data(iris_testcontainer_connection, embedding_model_fixture, TEST_CHUNKS_FOR_CRAG)

    # Spy on the placeholder web search function
    web_search_spy = mocker.MagicMock(side_effect=placeholder_web_search_func)

    pipeline = CRAGPipeline( # Updated class name
        iris_connector=iris_testcontainer_connection,
        embedding_func=embedding_model_fixture,
        llm_func=llm_client_fixture,
        web_search_func=web_search_spy # Pass the spied mock
    )

    query = "benefits of solar power"
    # top_k for retrieve_and_correct, which then uses it for _retrieve_chunks_jdbc_safe
    # and for limiting the final output of retrieve_and_correct.
    test_top_k = 5 

    # The initial similarity_threshold in _retrieve_chunks_jdbc_safe is 0.1 by default.
    # TEST_CHUNKS_FOR_CRAG are designed to have scores like 0.35, 0.15, 0.20 for this query.
    # So, some might be retrieved initially.
    # _evaluate_retrieval: correct if avg > 0.7, ambiguous if > 0.4, else disoriented.
    # (0.35 + 0.15 + 0.20) / 3 = 0.7 / 3 = 0.23 -> "disoriented", should trigger web search.

    logger.info(f"Running CRAG pipeline (run method) with query: '{query}', top_k={test_top_k}")
    
    result_data = pipeline.run(query_text=query, top_k=test_top_k)
    
    final_documents = result_data.get("retrieved_documents", [])
    answer = result_data.get("answer", "")

    logger.info(f"Final retrieved documents count: {len(final_documents)}")
    for i, doc_dict in enumerate(final_documents):
        logger.info(f"  Doc {i}: ID={doc_dict.get('id')}, Score={doc_dict.get('score')}, Source={doc_dict.get('metadata',{}).get('source')}, Content='{doc_dict.get('content','')[:50]}...'")

    # 1. Verify web search was called
    web_search_spy.assert_called_once_with(query)
    assert "CRAG: Augmenting with web search" in caplog.text, "Log for web search augmentation missing."
    
    # 2. Verify web search results are present in the final documents
    # The pipeline._decompose_recompose_filter might filter some, but some should remain.
    mock_web_search_ids = {doc.id for doc in MOCK_WEB_SEARCH_RESULTS}
    final_doc_ids = {doc_dict.get("id") for doc_dict in final_documents}
    
    assert any(web_id in final_doc_ids for web_id in mock_web_search_ids), \
        f"Expected at least one web search result ID in final documents. Web IDs: {mock_web_search_ids}, Final IDs: {final_doc_ids}"

    # 3. Verify that some initial (low-quality) DB results might also be present if their score > 0.3 (after filtering)
    #    or if web search results are fewer than top_k.
    #    The _decompose_recompose_filter keeps docs with score > 0.3.
    #    Our initial DB chunks have expected scores 0.35, 0.15, 0.20. Only crag_chunk_1 (0.35) might pass this.
    initial_db_chunk_ids = {chunk["chunk_id"] for chunk in TEST_CHUNKS_FOR_CRAG}
    
    # Check if at least one of the original DB chunks (that passed filtering) or web docs is present.
    assert len(final_documents) > 0, "No documents were returned after correction and filtering."
    assert len(final_documents) <= test_top_k, f"Returned more documents ({len(final_documents)}) than top_k ({test_top_k})."

    # 4. Verify answer incorporates web search content
    assert "solar" in answer.lower(), "Answer seems unrelated to 'solar'."
    assert "benefits" in answer.lower(), "Answer does not mention 'benefits'."
    # Check for keywords from MOCK_WEB_SEARCH_RESULTS
    assert "environmental benefits" in answer.lower() or "carbon footprint" in answer.lower() or "job creation" in answer.lower(), \
        "Answer does not seem to incorporate content from mock web search results."

    logger.info("CRAG JDBC E2E test for corrective web search completed successfully.")