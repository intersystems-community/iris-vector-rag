#!/usr/bin/env python3
"""
Test View-Based HNSW Indexing with IRIS 2025.1

This script tests whether we can create a view that converts VARCHAR embeddings to VECTOR type
using TO_VECTOR, and then create an HNSW index on that view in IRIS 2025.1.

Previous attempts with IRIS 2024.1 failed because HNSW syntax wasn't introduced until 2025.1.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from fastembed import TextEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_view_hnsw_2025")

# Test data
TEST_DOCUMENTS = [
    "InterSystems IRIS is a data platform that combines a database with integration and analytics capabilities.",
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "RAG (Retrieval Augmented Generation) combines retrieval systems with generative AI models.",
    "SQL is a standard language for storing, manipulating and retrieving data in databases.",
    "Embeddings are numerical representations of text that capture semantic meaning."
]

class ViewHNSWTest:
    def __init__(self):
        """Initialize the test with IRIS connection and embedding model."""
        self.conn = get_iris_connection()
        self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.base_table_name = "ViewHNSWTest"
        self.view_name = "ViewHNSWTestView"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        
    def setup_database(self):
        """Set up the database table and view for testing."""
        cursor = self.conn.cursor()
        
        try:
            # Drop the view and table if they exist
            logger.info("Dropping view and table if they exist...")
            drop_view_sql = f"""
            DROP VIEW IF EXISTS {self.view_name}
            """
            cursor.execute(drop_view_sql)
            
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.base_table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Create base table with VARCHAR column for embeddings
            logger.info("Creating base table with VARCHAR column for embeddings...")
            create_table_sql = f"""
            CREATE TABLE {self.base_table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                metadata TEXT
            )
            """
            cursor.execute(create_table_sql)
            
            # Create view that converts VARCHAR to VECTOR using TO_VECTOR
            logger.info("Creating view that converts VARCHAR to VECTOR using TO_VECTOR...")
            create_view_sql = f"""
            CREATE VIEW {self.view_name} AS
            SELECT 
                id,
                text_content,
                TO_VECTOR(embedding, 'double', {self.embedding_dim}) AS vector_embedding,
                metadata
            FROM {self.base_table_name}
            """
            cursor.execute(create_view_sql)
            
            # Try to create HNSW index on the view
            logger.info("Attempting to create HNSW index on the view...")
            try:
                create_index_sql = f"""
                CREATE INDEX idx_{self.view_name}_vector ON {self.view_name} (vector_embedding) USING HNSW
                """
                cursor.execute(create_index_sql)
                logger.info("✅ Successfully created HNSW index on the view!")
            except Exception as e_index:
                logger.error(f"❌ Failed to create HNSW index on the view: {e_index}")
                logger.error("This confirms that HNSW indexes cannot be created on views with TO_VECTOR.")
            
            self.conn.commit()
            logger.info(f"Created table {self.base_table_name} and view {self.view_name}")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def try_alternative_approach(self):
        """Try an alternative approach with a materialized view or computed column."""
        cursor = self.conn.cursor()
        
        try:
            # Try to create a table with a computed column
            logger.info("Attempting to create a table with a computed column...")
            drop_computed_table_sql = f"""
            DROP TABLE IF EXISTS ComputedVectorTest
            """
            cursor.execute(drop_computed_table_sql)
            
            create_computed_table_sql = f"""
            CREATE TABLE ComputedVectorTest (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                vector_embedding AS TO_VECTOR(embedding, 'double', {self.embedding_dim})
            )
            """
            
            try:
                cursor.execute(create_computed_table_sql)
                logger.info("✅ Successfully created table with computed column!")
                
                # Try to create HNSW index on the computed column
                logger.info("Attempting to create HNSW index on the computed column...")
                create_computed_index_sql = f"""
                CREATE INDEX idx_computed_vector ON ComputedVectorTest (vector_embedding) USING HNSW
                """
                cursor.execute(create_computed_index_sql)
                logger.info("✅ Successfully created HNSW index on the computed column!")
            except Exception as e_computed:
                logger.error(f"❌ Failed to create table with computed column: {e_computed}")
            
            # Try to create a materialized view
            logger.info("Attempting to create a materialized view...")
            drop_mat_view_sql = f"""
            DROP TABLE IF EXISTS MaterializedVectorView
            """
            cursor.execute(drop_mat_view_sql)
            
            try:
                create_mat_view_sql = f"""
                CREATE TABLE MaterializedVectorView AS
                SELECT 
                    id,
                    text_content,
                    TO_VECTOR(embedding, 'double', {self.embedding_dim}) AS vector_embedding,
                    metadata
                FROM {self.base_table_name}
                """
                cursor.execute(create_mat_view_sql)
                logger.info("✅ Successfully created materialized view!")
                
                # Try to create HNSW index on the materialized view
                logger.info("Attempting to create HNSW index on the materialized view...")
                create_mat_index_sql = f"""
                CREATE INDEX idx_mat_vector ON MaterializedVectorView (vector_embedding) USING HNSW
                """
                cursor.execute(create_mat_index_sql)
                logger.info("✅ Successfully created HNSW index on the materialized view!")
            except Exception as e_mat:
                logger.error(f"❌ Failed to create materialized view: {e_mat}")
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error trying alternative approaches: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = list(self.embedding_model.embed(texts))
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
        Search for documents similar to the query using the view.
        
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
                       vector_embedding,
                       TO_VECTOR('{query_embedding_str}', 'double', {self.embedding_dim})
                   ) AS score
            FROM {self.view_name}
            ORDER BY score ASC
            """
            
            # Execute the SQL directly without parameters
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

def run_test():
    """Run the view-based HNSW indexing test."""
    logger.info("Starting View-Based HNSW Indexing Test with IRIS 2025.1")
    
    # Initialize the test
    test = ViewHNSWTest()
    
    # Set up database
    test.setup_database()
    
    # Try alternative approaches
    test.try_alternative_approach()
    
    # Store test documents
    logger.info("Storing test documents")
    doc_ids = test.store_documents(TEST_DOCUMENTS)
    
    if doc_ids:
        # Search for similar documents
        query = "How does vector search work?"
        logger.info(f"Searching for documents similar to: '{query}'")
        results = test.search_similar_documents(query)
        
        # Display results
        logger.info("Search results:")
        for i, (doc_id, text, score) in enumerate(results):
            logger.info(f"  {i+1}. [{doc_id}] {text} (Score: {score})")
    
    logger.info("Test completed")

if __name__ == "__main__":
    run_test()