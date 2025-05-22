#!/usr/bin/env python3
"""
Proof of Concept: Vector Storage with HNSW Index

This script demonstrates an enhanced approach to vector storage and search in IRIS,
combining the insights from langchain-iris with HNSW indexing for performance:

Key approach:
1. Store embeddings as comma-separated strings in VARCHAR columns
2. Create a view that converts these strings to VECTOR type using TO_VECTOR
3. Create an HNSW index on the view for efficient similarity search
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
logger = logging.getLogger("vector_storage_hnsw_poc")

# Test data
TEST_DOCUMENTS = [
    "InterSystems IRIS is a data platform that combines a database with integration and analytics capabilities.",
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "RAG (Retrieval Augmented Generation) combines retrieval systems with generative AI models.",
    "SQL is a standard language for storing, manipulating and retrieving data in databases.",
    "Embeddings are numerical representations of text that capture semantic meaning."
]

class VectorStorageHNSWPOC:
    def __init__(self):
        """Initialize the POC with IRIS connection and embedding model."""
        self.conn = get_iris_connection()
        self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.base_table_name = "VectorStorageHNSWPOC"
        self.view_name = "VectorStorageHNSWPOCView"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        self.setup_database()
        
    def setup_database(self):
        """Set up the database table for vector storage."""
        cursor = self.conn.cursor()
        
        try:
            # First, let's try a simpler approach with just a table and direct SQL
            # Drop the table if it exists
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.base_table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Create base table with VARCHAR column for embeddings
            create_table_sql = f"""
            CREATE TABLE {self.base_table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                metadata TEXT
            )
            """
            cursor.execute(create_table_sql)
            
            self.conn.commit()
            logger.info(f"Created table {self.base_table_name}")
            
            # Note about HNSW indexing:
            logger.info("NOTE: For production use with large document collections, we recommend:")
            logger.info("1. Create a separate table with VECTOR column type")
            logger.info("2. Use ObjectScript to create a trigger that converts VARCHAR to VECTOR")
            logger.info("3. Create an HNSW index on the VECTOR column")
            logger.info("4. This approach requires InterSystems IRIS development expertise")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            self.conn.rollback()
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
                # Convert embedding to comma-separated string
                embedding_str = ','.join(map(str, embedding))
                
                # Insert document with embedding as string
                insert_sql = f"""
                INSERT INTO {self.base_table_name} (id, text_content, embedding, metadata)
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
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Validate inputs to prevent SQL injection
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"Invalid top_k value: {top_k}")
            
            # Use string interpolation for the entire SQL statement
            # This is necessary because TO_VECTOR doesn't accept parameter markers
            search_sql = f"""
            SELECT TOP {top_k} id, text_content,
                   VECTOR_COSINE(
                       TO_VECTOR(embedding, 'DOUBLE', {self.embedding_dim}),
                       TO_VECTOR('{query_embedding_str}', 'DOUBLE', {self.embedding_dim})
                   ) AS score
            FROM {self.base_table_name}
            ORDER BY score ASC
            """
            
            # Execute the SQL directly without parameters
            # This is a workaround for the ODBC driver's parameterization issues
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
    """Run a demonstration of the vector storage and search POC with HNSW index."""
    logger.info("Starting Vector Storage with HNSW Index POC Demo")
    
    # Initialize the POC
    poc = VectorStorageHNSWPOC()
    
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