#!/usr/bin/env python3
"""
Debug script to investigate vector search issues.
Checks what data is actually in the database and tests vector search manually.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use direct connection to avoid circular imports
from common.iris_connection_manager import get_iris_connection
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Debug vector search issues."""
    
    # Get connection directly
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Check if SourceDocuments table exists and has data
        logger.info("=== Checking SourceDocuments table ===")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count = cursor.fetchone()[0]
        logger.info(f"Total documents in SourceDocuments: {count}")
        
        if count > 0:
            # Check first few documents
            cursor.execute("SELECT TOP 3 id, title, LENGTH(content), LENGTH(embedding) FROM RAG.SourceDocuments")
            rows = cursor.fetchall()
            for row in rows:
                logger.info(f"Doc ID: {row[0]}, Title: {row[1][:50]}..., Content Length: {row[2]}, Embedding Length: {row[3]}")
        
        # 2. Check embedding dimensions
        if count > 0:
            logger.info("=== Checking embedding dimensions ===")
            cursor.execute("SELECT TOP 1 embedding FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            result = cursor.fetchone()
            if result:
                embedding_blob = result[0]
                logger.info(f"Raw embedding type: {type(embedding_blob)}")
                logger.info(f"Raw embedding length: {len(embedding_blob) if embedding_blob else 'NULL'}")
                
                # Try to parse the embedding
                try:
                    # IRIS stores vectors as binary data, need to convert
                    cursor.execute("SELECT TOP 1 VECTOR_DOT_PRODUCT(embedding, embedding) as self_dot FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                    dot_result = cursor.fetchone()
                    if dot_result:
                        logger.info(f"Self dot product: {dot_result[0]} (should be positive)")
                except Exception as e:
                    logger.error(f"Error computing self dot product: {e}")
        
        # 3. Test a simple vector search manually
        if count > 0:
            logger.info("=== Testing manual vector search ===")
            
            # Create a simple test embedding (384 dimensions for all-MiniLM-L6-v2)
            # This is just for testing - using a dummy embedding
            test_embedding = [0.1] * 384  # Simple test vector
            logger.info(f"Test embedding dimension: {len(test_embedding)}")
            
            # Convert to string format for IRIS
            embedding_str = ','.join(map(str, test_embedding))
            
            # Test vector search with manual SQL
            try:
                sql = f"""
                SELECT TOP 3
                    id,
                    title,
                    VECTOR_COSINE(embedding, TO_VECTOR(?, FLOAT, {len(test_embedding)})) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
                """
                logger.info(f"Executing SQL: {sql}")
                cursor.execute(sql, [embedding_str])
                results = cursor.fetchall()
                logger.info(f"Manual vector search returned {len(results)} results")
                
                for i, row in enumerate(results):
                    logger.info(f"Result {i+1}: ID={row[0]}, Title={row[1][:50]}..., Score={row[2]}")
                    
            except Exception as e:
                logger.error(f"Manual vector search failed: {e}")
                
                # Try alternative approach - check if embeddings are actually stored
                try:
                    cursor.execute("SELECT TOP 1 id, title FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                    result = cursor.fetchone()
                    if result:
                        logger.info(f"Found document with embedding: ID={result[0]}, Title={result[1]}")
                    else:
                        logger.error("No documents found with non-NULL embeddings!")
                except Exception as e2:
                    logger.error(f"Error checking for non-NULL embeddings: {e2}")
        
        # 4. Check DocumentChunks table if it exists
        try:
            logger.info("=== Checking DocumentChunks table ===")
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            chunk_count = cursor.fetchone()[0]
            logger.info(f"Total chunks in DocumentChunks: {chunk_count}")
            
            if chunk_count > 0:
                cursor.execute("SELECT TOP 3 id, source_document_id, LENGTH(content), LENGTH(embedding) FROM RAG.DocumentChunks")
                rows = cursor.fetchall()
                for row in rows:
                    logger.info(f"Chunk ID: {row[0]}, Source Doc: {row[1]}, Content Length: {row[2]}, Embedding Length: {row[3]}")
        except Exception as e:
            logger.info(f"DocumentChunks table not accessible: {e}")
            
    except Exception as e:
        logger.error(f"Debug script failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()

if __name__ == "__main__":
    main()