# common/db_vector_search.py
import logging
from typing import List, Any, Tuple

logger = logging.getLogger(__name__)

def search_source_documents_dynamically(
    iris_connector: Any, top_k: int, vector_string: str
) -> List[Tuple[str, str, float]]:
    """
    Performs a vector search on the SourceDocuments table using dynamic SQL.
    Returns a list of tuples, where each tuple is (doc_id, text_content, score).
    """
    # Ensure top_k is an integer to prevent SQL injection via f-string
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # vector_string is already in the format like "[0.1,0.2,...]"
    # We need to ensure it's properly escaped if it were to contain single quotes,
    # but given its generation from floats, it shouldn't.
    # However, for safety in SQL literal, one might double any internal single quotes if they could exist.
    # For now, assuming vector_string is "clean".
    sql = f"""
        SELECT doc_id, text_content,
               VECTOR_COSINE(embedding, TO_VECTOR('{vector_string}', 'DOUBLE', 768)) AS score
        FROM SourceDocuments
        WHERE embedding IS NOT NULL
        ORDER BY score DESC
        FETCH FIRST {top_k} ROWS ONLY
    """
    results: List[Tuple[str, str, float]] = []
    cursor = None
    try:
        logger.debug(f"Executing dynamic SQL for SourceDocuments search with top_k={top_k}, vector_string (preview): {vector_string[:100]}...")
        cursor = iris_connector.cursor()
        cursor.execute(sql) # No parameters passed here as all are interpolated
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            # Ensure rows are tuples and have the expected number of elements
            results = [(str(row[0]), str(row[1]), float(row[2])) for row in fetched_rows if isinstance(row, tuple) and len(row) == 3]
        logger.debug(f"Found {len(results)} documents from SourceDocuments.")
    except Exception as e:
        logger.error(f"Error during dynamic SQL search on SourceDocuments: {e}")
        # Re-raise the exception so the calling pipeline can handle it or log it appropriately.
        raise
    finally:
        if cursor:
            cursor.close()
    return results

def search_knowledge_graph_nodes_dynamically(
    iris_connector: Any, top_k: int, vector_string: str
) -> List[Tuple[str, float]]:
    """
    Performs a vector search on the KnowledgeGraphNodes table using dynamic SQL.
    Returns a list of tuples, where each tuple is (node_id, score).
    """
    # Ensure top_k is an integer to prevent SQL injection via f-string
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # vector_string is already in the format like "[0.1,0.2,...]"
    sql = f"""
        SELECT node_id,
               VECTOR_COSINE(embedding, TO_VECTOR('{vector_string}', 'DOUBLE', 768)) AS score
        FROM KnowledgeGraphNodes
        WHERE embedding IS NOT NULL
        ORDER BY score DESC
        FETCH FIRST {top_k} ROWS ONLY
    """
    results: List[Tuple[str, float]] = []
    cursor = None
    try:
        logger.debug(f"Executing dynamic SQL for KnowledgeGraphNodes search with top_k={top_k}, vector_string (preview): {vector_string[:100]}...")
        cursor = iris_connector.cursor()
        cursor.execute(sql) # No parameters passed here as all are interpolated
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            # Ensure rows are tuples and have the expected number of elements
            results = [(str(row[0]), float(row[1])) for row in fetched_rows if isinstance(row, tuple) and len(row) == 2]
        logger.debug(f"Found {len(results)} nodes from KnowledgeGraphNodes.")
    except Exception as e:
        logger.error(f"Error during dynamic SQL search on KnowledgeGraphNodes: {e}")
        # Re-raise the exception
        raise
    finally:
        if cursor:
            cursor.close()
    return results
