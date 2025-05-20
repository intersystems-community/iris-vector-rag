"""
Test simple retrieval methods for testing purposes.

This module provides simple retrieval methods that don't rely on vector similarity
functions that might not be available in the test environment.
"""

import logging
from typing import List, Any, Dict
from common.utils import Document

logger = logging.getLogger(__name__)

def retrieve_documents_by_content_match(connection, search_text: str, top_k: int = 5) -> List[Document]:
    """
    Retrieve documents by matching terms in the content using SQL LIKE.
    
    This is a simplified retrieval method for testing that doesn't use vector similarity.
    It's meant to be used in test environments where vector functions aren't available.
    
    Args:
        connection: Database connection
        search_text: Text to search for in documents
        top_k: Maximum number of documents to return
        
    Returns:
        List of Document objects
    """
    # Split the search text into terms
    terms = search_text.lower().split()
    
    # Create a SQL LIKE condition for each term
    # Note: We don't use LOWER() here because IRIS doesn't support it on CLOB fields
    conditions = []
    for term in terms:
        if len(term) > 3:  # Only use terms longer than 3 chars to avoid common words
            conditions.append(f"text_content LIKE '%{term}%'")
    
    if not conditions:
        # If no valid terms, use the whole search text
        conditions.append(f"text_content LIKE '%{search_text}%'")
    
    # Combine conditions with OR
    where_clause = " OR ".join(conditions)
    
    # Create the SQL query
    sql = f"""
        SELECT TOP ? doc_id, text_content, 0.9 AS score
        FROM SourceDocuments
        WHERE {where_clause}
        ORDER BY score DESC
    """
    
    retrieved_docs = []
    try:
        cursor = connection.cursor()
        cursor.execute(sql, [top_k])
        rows = cursor.fetchall()
        
        for row in rows:
            doc_id, content, score = row
            retrieved_docs.append(Document(id=doc_id, content=content, score=float(score)))
            
        logger.info(f"Retrieved {len(retrieved_docs)} documents using content match")
        
    except Exception as e:
        logger.error(f"Error in content-based retrieval: {e}")
    
    return retrieved_docs

def retrieve_documents_by_fixed_ids(connection, doc_ids: List[str]) -> List[Document]:
    """
    Retrieve documents by their IDs.
    
    Args:
        connection: Database connection
        doc_ids: List of document IDs to retrieve
        
    Returns:
        List of Document objects
    """
    if not doc_ids:
        return []
    
    # Create a comma-separated string of doc_ids for the IN clause
    id_string = ", ".join([f"'{doc_id}'" for doc_id in doc_ids])
    
    sql = f"""
        SELECT doc_id, text_content, 0.9 AS score
        FROM SourceDocuments
        WHERE doc_id IN ({id_string})
    """
    
    retrieved_docs = []
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        for row in rows:
            doc_id, content, score = row
            retrieved_docs.append(Document(id=doc_id, content=content, score=float(score)))
            
        logger.info(f"Retrieved {len(retrieved_docs)} documents by IDs")
        
    except Exception as e:
        logger.error(f"Error in ID-based retrieval: {e}")
    
    return retrieved_docs
