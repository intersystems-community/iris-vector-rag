#!/usr/bin/env python3
"""
Fix Embedding Dimension Mismatch

This script re-embeds all documents in the database with the correct
embedding model (sentence-transformers/all-MiniLM-L6-v2) to ensure
384-dimensional embeddings.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func
from common.db_vector_utils import insert_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_embeddings():
    """Re-embed all documents with the correct model."""
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    try:
        # Get embedding function (should use all-MiniLM-L6-v2)
        embed_func = get_embedding_func()
        logger.info("Using embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)")
        
        # Get all documents
        cursor.execute("SELECT doc_id, text_content FROM RAG.SourceDocuments")
        documents = cursor.fetchall()
        logger.info(f"Found {len(documents)} documents to re-embed")
        
        # Re-embed each document
        success_count = 0
        for i, (doc_id, text_content) in enumerate(documents):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(documents)} documents")
            
            try:
                # Generate embedding
                embedding = embed_func(text_content)
                
                # Verify dimension
                if len(embedding) != 384:
                    logger.error(f"Unexpected embedding dimension: {len(embedding)} for doc {doc_id}")
                    continue
                
                # Update embedding in database using string interpolation
                # Convert embedding to string format
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                # Validate the embedding string
                from common.vector_sql_utils import validate_vector_string
                if not validate_vector_string(embedding_str):
                    logger.error(f"Invalid vector string for doc {doc_id}")
                    continue
                
                # Use string interpolation for TO_VECTOR (IRIS limitation)
                update_sql = f"""
                    UPDATE RAG.SourceDocuments 
                    SET embedding = TO_VECTOR('{embedding_str}', 'FLOAT', 384)
                    WHERE doc_id = '{doc_id}'
                """
                cursor.execute(update_sql)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error re-embedding doc {doc_id}: {e}")
        
        # Commit changes
        connection.commit()
        logger.info(f"Successfully re-embedded {success_count}/{len(documents)} documents")
        
        # Verify the fix
        cursor.execute("""
            SELECT doc_id, LENGTH(embedding) as emb_len
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            LIMIT 5
        """)
        
        logger.info("Sample embedding lengths after fix:")
        for row in cursor.fetchall():
            logger.info(f"  Doc {row[0]}: {row[1]} bytes")
            
    except Exception as e:
        logger.error(f"Error fixing embeddings: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    fix_embeddings()