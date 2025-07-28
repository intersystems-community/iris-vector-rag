import pytest
import logging
import sys # Added import
import os # Added import
from typing import List, Dict, Any, Callable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.core.models import Document
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
# Helper to create Document objects with metadata
def _create_mock_web_doc_with_metadata(id_val: str, content_val: str, score_val: float, metadata_dict: Dict[str, Any]) -> Document:
    doc = Document(page_content=content_val, metadata=metadata_dict, id=id_val, score=score_val)
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
    """Helper to insert test chunks into RAG.DocumentChunks and corresponding SourceDocuments using Vector Store interface."""
    from common.db_vector_utils import insert_vector
    
    logger.info(f"Inserting {len(chunks_data)} test chunks for CRAG JDBC E2E test.")

    source_docs_to_insert = {}
    for chunk_data in chunks_data:
        if chunk_data["doc_id"] not in source_docs_to_insert:
            source_docs_to_insert[chunk_data["doc_id"]] = {
                "id": chunk_data["doc_id"],
                "title": f"Test Source Document {chunk_data['doc_id']}",
                "content": f"Full content for document {chunk_data['doc_id']}.",
                "embedding": [0.1] * 384,  # Placeholder embedding as list, not used by chunk query
                "source": "CRAG_E2E_TEST_DOC"
            }

    with iris_conn.cursor() as cursor:
        # Insert SourceDocuments first using insert_vector utility
        for doc_id, doc_data in source_docs_to_insert.items():
            try:
                # Use insert_vector utility for proper vector handling
                success = insert_vector(
                    cursor=cursor,
                    table_name="RAG.SourceDocuments",
                    vector_column_name="embedding",
                    vector_data=doc_data["embedding"],
                    target_dimension=384,  # Standard embedding dimension
                    key_columns={"doc_id": doc_data["id"]},
                    additional_data={
                        "title": doc_data["title"],
                        "text_content": doc_data["content"],
                        "source": doc_data["source"]
                    }
                )
                if not success:
                    logger.warning(f"SourceDocument {doc_id} insertion returned False - may already exist.")
            except Exception as e:
                logger.error(f"Failed to insert SourceDocument {doc_id}: {e}")
                raise
        
        # Insert DocumentChunks using insert_vector utility
        chunk_texts_to_embed = [chunk["chunk_text"] for chunk in chunks_data]
        if not chunk_texts_to_embed:
            logger.info("No chunk texts to embed for DocumentChunks.")
            iris_conn.commit()
            return

        embeddings = embedding_func(chunk_texts_to_embed)

        for i, chunk_data in enumerate(chunks_data):
            metadata_json_str = f'{{"expected_score_initial": {chunk_data.get("expected_score_initial", 0.0)}}}'
            try:
                # Use insert_vector utility for proper vector handling
                success = insert_vector(
                    cursor=cursor,
                    table_name="RAG.DocumentChunks",
                    vector_column_name="chunk_embedding",
                    vector_data=embeddings[i],
                    target_dimension=384,  # Standard embedding dimension
                    key_columns={"chunk_id": chunk_data["chunk_id"]},
                    additional_data={
                        "doc_id": chunk_data["doc_id"],
                        "chunk_text": chunk_data["chunk_text"],
                        "chunk_type": chunk_data["chunk_type"],
                        "chunk_index": chunk_data["chunk_index"],
                        "metadata": metadata_json_str
                    }
                )
                if success:
                    logger.debug(f"Inserted DocumentChunk: {chunk_data['chunk_id']}")
                else:
                    logger.warning(f"DocumentChunk {chunk_data['chunk_id']} insertion returned False - may already exist.")
            except Exception as e:
                logger.error(f"Failed to insert DocumentChunk {chunk_data['chunk_id']}: {e}")
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
    
    # Always use SchemaManager to ensure proper schema (following Database Schema Management Rules)
    from iris_rag.storage.schema_manager import SchemaManager
    from iris_rag.config.manager import ConfigurationManager
    
    # Initialize components
    connection_manager = type('ConnectionManager', (), {
        'get_connection': lambda self: iris_testcontainer_connection
    })()
    config_manager = ConfigurationManager()
    
    # Create and use schema manager to ensure proper schema
    schema_manager = SchemaManager(connection_manager, config_manager)
    schema_manager.ensure_table_schema('SourceDocuments')
    schema_manager.ensure_table_schema('DocumentChunks')
    logger.info("Ensured proper table schemas using SchemaManager.")
    
    with iris_testcontainer_connection.cursor() as cursor:
        logger.info("Clearing RAG.DocumentChunks and RAG.SourceDocuments for test data.")
        try:
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id LIKE 'crag_chunk_%'")
            cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id LIKE 'doc_A' OR doc_id LIKE 'doc_B' OR doc_id LIKE 'doc_C'")
            iris_testcontainer_connection.commit()
        except Exception as e:
            logger.warning(f"Could not clear tables: {e}")
            iris_testcontainer_connection.rollback()
            raise
    
    insert_crag_test_data(iris_testcontainer_connection, embedding_model_fixture, TEST_CHUNKS_FOR_CRAG)

    # Spy on the placeholder web search function
    web_search_spy = mocker.MagicMock(side_effect=placeholder_web_search_func)

    pipeline = CRAGPipeline( # Updated class name
        config_manager=config_manager,
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

    result_data = pipeline.run(query, top_k=test_top_k)
    
    final_documents = result_data.get("retrieved_documents", [])
    answer = result_data.get("answer", "")

    logger.info(f"Final retrieved documents count: {len(final_documents)}")
    for i, doc in enumerate(final_documents):
        # Handle Document objects properly
        if hasattr(doc, 'page_content'):
            content = doc.page_content[:50] if doc.page_content else ""
            metadata = getattr(doc, 'metadata', {})
            score = metadata.get('similarity_score', 'N/A')
            source = metadata.get('source', 'N/A')
            doc_id = metadata.get('doc_id', 'N/A')
        else:
            # Fallback for dict-like objects
            content = doc.get('content', '')[:50] if hasattr(doc, 'get') else str(doc)[:50]
            metadata = doc.get('metadata', {}) if hasattr(doc, 'get') else {}
            score = doc.get('score', 'N/A') if hasattr(doc, 'get') else 'N/A'
            source = metadata.get('source', 'N/A')
            doc_id = doc.get('id', 'N/A') if hasattr(doc, 'get') else 'N/A'
        
        logger.info(f"  Doc {i}: ID={doc_id}, Score={score}, Source={source}, Content='{content}...'")

    # 1. Verify web search was called
    web_search_spy.assert_called_once_with(query)
    assert "CRAG: Augmenting with web search" in caplog.text, "Log for web search augmentation missing."
    
    # 2. Verify web search results are present in the final documents
    # The pipeline._decompose_recompose_filter might filter some, but some should remain.
    mock_web_search_ids = {doc.id for doc in MOCK_WEB_SEARCH_RESULTS}
    
    # Handle both Document objects and dictionary objects in final_documents
    final_doc_ids = set()
    for doc in final_documents:
        if hasattr(doc, 'id'):
            final_doc_ids.add(doc.id)
        elif hasattr(doc, 'get'):
            final_doc_ids.add(doc.get("id"))
        elif hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            final_doc_ids.add(doc.metadata.get("doc_id"))
    
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