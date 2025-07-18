"""
JDBC-Safe Chunk Retrieval Module
Handles vector operations without parameter binding issues
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from .utils import Document # Changed to relative import

logger = logging.getLogger(__name__)

def retrieve_chunks_jdbc_safe(connection, query_embedding: List[float], 
                             top_k: int = 20, threshold: float = 0.1,
                             chunk_types: List[str] = None) -> List[Document]:
    """
    Retrieve chunks using JDBC-safe vector operations
    """
    if chunk_types is None:
        chunk_types = ['content', 'mixed']
    
    cursor = None
    chunks = []
    
    try:
        cursor = connection.cursor()
        
        # Convert embedding to string
        vector_str = ','.join(map(str, query_embedding))
        chunk_types_str = ','.join([f"'{ct}'" for ct in chunk_types])
        
        # Use direct SQL without parameter binding
        query = f"""
            SELECT TOP {top_k}
                chunk_id,
                chunk_text,
                doc_id,
                chunk_type,
                chunk_index,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) AS score
            FROM RAG.DocumentChunks
            WHERE embedding IS NOT NULL
              AND chunk_type IN ({chunk_types_str})
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) > {threshold}
            ORDER BY score DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        for chunk_id, chunk_text, doc_id, chunk_type, chunk_index, score in results:
            # Handle potential stream objects
            if hasattr(chunk_text, 'read'):
                chunk_text = chunk_text.read()
            if isinstance(chunk_text, bytes):
                chunk_text = chunk_text.decode('utf-8', errors='ignore')
            
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_id}",
                content=str(chunk_text),
                score=float(score) if score else 0.0,
                metadata={
                    'doc_id': doc_id,
                    'chunk_type': chunk_type,
                    'chunk_index': chunk_index
                }
            ))
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
    finally:
        if cursor:
            cursor.close()
    
    return chunks

def retrieve_documents_jdbc_safe(connection, query_embedding: List[float],
                                top_k: int = 20, threshold: float = 0.1) -> List[Document]:
    """
    Retrieve documents using JDBC-safe vector operations
    """
    cursor = None
    documents = []
    
    try:
        cursor = connection.cursor()
        
        # Convert embedding to string
        vector_str = ','.join(map(str, query_embedding))
        
        # Use direct SQL without parameter binding
        query = f"""
            SELECT TOP {top_k}
                doc_id,
                text_content,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) AS score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
              AND LENGTH(embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) > {threshold}
            ORDER BY score DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        for doc_id, content, score in results:
            # Handle potential stream objects
            if hasattr(content, 'read'):
                content = content.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            
            documents.append(Document(
                id=doc_id,
                content=str(content),
                score=float(score) if score else 0.0
            ))
        
        logger.info(f"Retrieved {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
    finally:
        if cursor:
            cursor.close()
    
    return documents
