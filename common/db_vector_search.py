# common/db_vector_search.py
import logging
from typing import List, Any, Tuple

from common.vector_sql_utils import (
    format_vector_search_sql,
    execute_vector_search
)

logger = logging.getLogger(__name__)

def search_source_documents_dynamically(
    iris_connector: Any, top_k: int, vector_string: str
) -> List[Tuple[str, str, float]]:
    """
    Performs a vector search on the SourceDocuments table using dynamic SQL.
    Returns a list of tuples, where each tuple is (doc_id, text_content, score).
    
    This implementation uses utility functions from vector_sql_utils.py to safely
    construct and execute the SQL query.
    """
    # Construct the SQL query using the utility function
    sql = format_vector_search_sql(
        table_name="SourceDocuments",
        vector_column="embedding",
        vector_string=vector_string,
        embedding_dim=768,
        top_k=top_k,
        id_column="doc_id",
        content_column="text_content"
    )
    
    # Execute the query using the utility function
    cursor = None
    results: List[Tuple[str, str, float]] = []
    
    try:
        cursor = iris_connector.cursor()
        fetched_rows = execute_vector_search(cursor, sql)
        
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
    
    This implementation uses utility functions from vector_sql_utils.py to safely
    construct and execute the SQL query.
    """
    # Construct the SQL query using the utility function
    sql = format_vector_search_sql(
        table_name="KnowledgeGraphNodes",
        vector_column="embedding",
        vector_string=vector_string,
        embedding_dim=768,
        top_k=top_k,
        id_column="node_id",
        content_column=None  # KnowledgeGraphNodes table doesn't have a content column in the result
    )
    
    # Execute the query using the utility function
    cursor = None
    results: List[Tuple[str, float]] = []
    
    try:
        cursor = iris_connector.cursor()
        fetched_rows = execute_vector_search(cursor, sql)
        
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
