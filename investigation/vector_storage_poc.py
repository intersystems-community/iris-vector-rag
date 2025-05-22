#!/usr/bin/env python3
"""
Proof of Concept: Vector Storage and Search

This script demonstrates a new approach to vector storage and search in IRIS,
based on insights from the langchain-iris implementation.

Key approach:
1. Store embeddings as VARCHAR strings (serialized lists)
2. Use VECTOR_COSINE for similarity search
3. Avoid using TO_VECTOR in SQL statements
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from fastembed import TextEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vector_storage_poc")

# Test data
TEST_DOCUMENTS = [
    "InterSystems IRIS is a data platform that combines a database with integration and analytics capabilities.",
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "RAG (Retrieval Augmented Generation) combines retrieval systems with generative AI models.",
    "SQL is a standard language for storing, manipulating and retrieving data in databases.",
    "Embeddings are numerical representations of text that capture semantic meaning."
]

class VectorStoragePOC:
    def __init__(self):
        """Initialize the POC with IRIS connection and embedding model."""
        self.conn = get_iris_connection()
        self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.table_name = "VectorStoragePOC"
        self.setup_database()
        
    def setup_database(self):
        """Set up the database table for vector storage."""
        cursor = self.conn.cursor()
        
        try:
            # Drop the table if it exists
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Create table with VARCHAR column for embeddings
            create_table_sql = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                metadata TEXT
            )
            """
            cursor.execute(create_table_sql)
            
            self.conn.commit()
            logger.info(f"Created table {self.table_name}")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
        finally:
            cursor.close()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    def store_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> List[str]:
        """Store documents with their embeddings in the database."""
        if ids is None:
            ids = [f"doc_{i+1}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(documents)
        
        # Store documents with embeddings
        cursor = self.conn.cursor()
        
        try:
            for doc_id, text, embedding in zip(ids, documents, embeddings):
                # Convert embedding to string representation suitable for TO_VECTOR
                # Format: "0.1,0.2,0.3,..." (no brackets, just comma-separated values)
                embedding_str = ','.join(map(str, embedding))
                
                # Insert document with embedding as string
                insert_sql = f"""
                INSERT INTO {self.table_name} (id, text_content, embedding, metadata)
                VALUES (?, ?, ?, ?)
                """
                
                cursor.execute(insert_sql, (doc_id, text, embedding_str, "{}"))
                
            self.conn.commit()
            logger.info(f"Stored {len(documents)} documents with embeddings")
            return ids
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            self.conn.rollback()
            return []
        finally:
            cursor.close()
    
    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Search for documents similar to the query.
        
        Returns a list of tuples (doc_id, text_content, score).
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]
        # Format: "0.1,0.2,0.3,..." (no brackets, just comma-separated values)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Validate inputs to prevent SQL injection
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"Invalid top_k value: {top_k}")
            
            # We need to use direct SQL execution with string interpolation
            # This is because the ODBC driver has issues with TO_VECTOR
            
            # Construct the SQL query with string interpolation
            search_sql = f"""
            SELECT TOP {top_k} id, text_content,
                   VECTOR_COSINE(
                       TO_VECTOR(embedding, 'DOUBLE', 384),
                       TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)
                   ) AS score
            FROM {self.table_name}
            ORDER BY score ASC
            """
            
            # Execute the SQL directly without preparing it first
            # This is a workaround for the ODBC driver's parameterization issues
            try:
                # Try to use execute_immediate if available
                cursor.execute_immediate(search_sql)
            except AttributeError:
                # If execute_immediate is not available, try direct execution
                # This is a fallback and may still have issues
                cursor.execute(search_sql)
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
    """Run a demonstration of the vector storage and search POC."""
    logger.info("Starting Vector Storage POC Demo")
    
    # Initialize the POC
    poc = VectorStoragePOC()
    
    # Store test documents
    logger.info("Storing test documents")
    doc_ids = poc.store_documents(TEST_DOCUMENTS)
    
    if doc_ids:
        # Search for similar documents
        query = "How does vector search work?"
        logger.info(f"Searching for documents similar to: '{query}'")
        results = poc.search_similar_documents(query)
        
        # Display results
        logger.info("Search results:")
        for i, (doc_id, text, score) in enumerate(results):
            logger.info(f"  {i+1}. [{doc_id}] {text} (Score: {score})")
    
    logger.info("Demo completed")

if __name__ == "__main__":
    run_demo()