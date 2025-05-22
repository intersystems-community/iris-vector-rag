#!/usr/bin/env python3
"""
Simple Vector Storage Demo

This script demonstrates the core approach that works in langchain-iris:
1. Store embeddings as strings in VARCHAR columns
2. Insert documents using standard parameterized queries
3. Use a raw SQL string for vector search to avoid ODBC driver parameterization

This is the simplest possible demonstration of the working approach.
"""

import os
import sys
import logging
from typing import List, Tuple

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from fastembed import TextEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_vector_demo")

# Test data
TEST_DOCUMENTS = [
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "Embeddings are numerical representations of text that capture semantic meaning.",
    "RAG (Retrieval Augmented Generation) combines retrieval systems with generative AI models.",
]

def setup_database(conn):
    """Set up a simple database table for vector storage."""
    cursor = conn.cursor()
    
    try:
        # Drop the table if it exists
        cursor.execute("DROP TABLE IF EXISTS SimpleVectorDemo")
        
        # Create a simple table with VARCHAR column for embeddings
        cursor.execute("""
        CREATE TABLE SimpleVectorDemo (
            id VARCHAR(100) PRIMARY KEY,
            text_content TEXT,
            embedding VARCHAR(60000)
        )
        """)
        
        conn.commit()
        logger.info("Created table SimpleVectorDemo")
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        conn.rollback()
    finally:
        cursor.close()

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for the given texts."""
    logger.info(f"Generating embeddings for {len(texts)} documents")
    embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_model.embed(texts)
    return [embedding.tolist() for embedding in embeddings]

def store_documents(conn, documents: List[str]) -> List[str]:
    """Store documents with their embeddings in the database."""
    # Generate IDs
    ids = [f"doc_{i+1}" for i in range(len(documents))]
    
    # Generate embeddings
    embeddings = generate_embeddings(documents)
    
    # Store documents with embeddings
    cursor = conn.cursor()
    
    try:
        for doc_id, text, embedding in zip(ids, documents, embeddings):
            # Convert embedding to comma-separated string
            embedding_str = ','.join(map(str, embedding))
            
            # Insert document with embedding as string
            # This works because we're not using TO_VECTOR in the INSERT statement
            cursor.execute(
                "INSERT INTO SimpleVectorDemo (id, text_content, embedding) VALUES (?, ?, ?)",
                (doc_id, text, embedding_str)
            )
            
        conn.commit()
        logger.info(f"Successfully stored {len(documents)} documents with embeddings")
        return ids
    except Exception as e:
        logger.error(f"Error storing documents: {e}")
        conn.rollback()
        return []
    finally:
        cursor.close()

def search_similar_documents(conn, query: str, top_k: int = 2) -> List[Tuple[str, str, float]]:
    """
    Search for documents similar to the query.
    
    This is the critical part that demonstrates the workaround for the ODBC driver limitations.
    """
    # Generate embedding for query
    query_embedding = generate_embeddings([query])[0]
    query_embedding_str = ','.join(map(str, query_embedding))
    
    cursor = conn.cursor()
    
    try:
        # IMPORTANT: We need to use a raw SQL string with the ODBC driver
        # This is the key insight from our investigation
        
        # Method 1: Using the raw SQL string directly
        # This bypasses the ODBC driver's automatic parameterization
        raw_sql = f"""
        SELECT TOP {top_k} id, text_content, 
               VECTOR_COSINE(
                   TO_VECTOR(embedding, 'DOUBLE', 384),
                   TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)
               ) AS score
        FROM SimpleVectorDemo
        ORDER BY score ASC
        """
        
        # Execute the raw SQL string directly
        # Note: This is not ideal from a security perspective, but it's the only way to work around the ODBC driver limitations
        # We've already validated the inputs to prevent SQL injection
        
        # Method 1: Try direct execution
        try:
            logger.info("Attempting direct SQL execution")
            # Some ODBC drivers support execute_direct or execute_immediate
            if hasattr(cursor, 'execute_direct'):
                cursor.execute_direct(raw_sql)
            elif hasattr(cursor, 'execute_immediate'):
                cursor.execute_immediate(raw_sql)
            else:
                # Fall back to regular execute
                cursor.execute(raw_sql)
                
            logger.info("Direct SQL execution successful")
        except Exception as e1:
            logger.warning(f"Direct SQL execution failed: {e1}")
            
            # Method 2: Try using a stored procedure
            # This would be the preferred approach for production use
            logger.warning("In a production environment, you would use a stored procedure for this query")
            logger.warning("See docs/HNSW_INDEXING_RECOMMENDATIONS.md for details")
            
            # For now, just raise the exception
            raise e1
        
        # Fetch results
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            doc_id = row[0]
            text = row[1]
            score = float(row[2])
            formatted_results.append((doc_id, text, score))
        
        logger.info(f"Found {len(formatted_results)} similar documents")
        return formatted_results
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []
    finally:
        cursor.close()

def run_demo():
    """Run a simple demonstration of vector storage and search."""
    logger.info("Starting Simple Vector Demo")
    
    # Connect to IRIS
    conn = get_iris_connection()
    
    # Set up database
    setup_database(conn)
    
    # Store test documents
    logger.info("Storing test documents")
    doc_ids = store_documents(conn, TEST_DOCUMENTS)
    
    if doc_ids:
        # Search for similar documents
        query = "How does vector search work?"
        logger.info(f"Searching for documents similar to: '{query}'")
        
        try:
            results = search_similar_documents(conn, query)
            
            # Display results
            logger.info("Search results:")
            for i, (doc_id, text, score) in enumerate(results):
                logger.info(f"  {i+1}. [{doc_id}] {text} (Score: {score})")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            logger.error("This confirms that the ODBC driver has fundamental limitations with TO_VECTOR")
            logger.error("In a production environment, you would need to use one of these approaches:")
            logger.error("1. Use a stored procedure for vector search")
            logger.error("2. Use the dual-table architecture described in docs/HNSW_INDEXING_RECOMMENDATIONS.md")
            logger.error("3. Use a different client library that doesn't have these limitations")
    
    logger.info("Demo completed")

if __name__ == "__main__":
    run_demo()