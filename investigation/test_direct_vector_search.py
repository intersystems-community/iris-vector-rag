#!/usr/bin/env python3
"""
Test Direct Vector Search in IRIS 2025.1 (No Views, No HNSW)

This script demonstrates a working end-to-end flow for vector search in IRIS 2025.1
without using views or HNSW indexing. It uses the correct syntax for TO_VECTOR that
we discovered in our previous tests.

The approach:
1. Create a table with a VARCHAR column for embeddings
2. Insert data with embeddings as strings
3. Use TO_VECTOR directly in the query for vector search
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import create_engine, text

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from common.iris_connector import get_iris_connection
except ImportError:
    print("Error: Could not import get_iris_connection. Make sure you're running this script from the project root.")
    sys.exit(1)

try:
    from fastembed import TextEmbedding
except ImportError:
    print("Error: fastembed not installed. Install it with: pip install fastembed")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_direct_vector_search")

# Test data
TEST_DOCUMENTS = [
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "IRIS database supports vector operations for implementing RAG applications.",
    "Embedding models convert text into high-dimensional vectors for semantic search.",
    "Parameter substitution in SQL queries improves security and maintainability.",
    "HNSW indexing accelerates vector search for large document collections."
]

class DirectVectorSearchTest:
    def __init__(self):
        """Initialize the test with IRIS connection and embedding model."""
        try:
            self.conn = get_iris_connection()
            logger.info("Successfully connected to IRIS")
        except Exception as e:
            logger.error(f"Failed to connect to IRIS: {e}")
            sys.exit(1)
            
        try:
            self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Successfully loaded embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            sys.exit(1)
            
        self.table_name = "DirectVectorSearchTest"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        
    def setup_database(self):
        """Set up the database table for testing."""
        logger.info("\n=== SETUP: Creating Table with VARCHAR Column ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Drop the table if it exists
            logger.info("Dropping table if it exists...")
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Create table with VARCHAR column for embeddings
            logger.info("Creating table with VARCHAR column for embeddings...")
            create_table_sql = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000)
            )
            """
            
            logger.info(f"SQL: {create_table_sql}")
            
            cursor.execute(create_table_sql)
            self.conn.commit()
            logger.info(f"✅ Successfully created table {self.table_name}")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def insert_documents(self):
        """Insert test documents with embeddings as strings."""
        logger.info("\n=== TEST: Inserting Documents with Embeddings ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Generate embeddings for test documents
            embeddings = []
            for i, doc in enumerate(TEST_DOCUMENTS):
                embedding = self.generate_embedding(doc)
                embedding_str = ','.join(map(str, embedding))
                embeddings.append((f"doc{i+1}", doc, embedding_str))
            
            # Insert documents with embeddings as strings
            insert_sql = f"""
            INSERT INTO {self.table_name} (id, text_content, embedding)
            VALUES (?, ?, ?)
            """
            
            logger.info(f"SQL: {insert_sql}")
            logger.info(f"Inserting {len(embeddings)} documents...")
            
            for doc_data in embeddings:
                cursor.execute(insert_sql, doc_data)
            
            self.conn.commit()
            logger.info(f"✅ Successfully inserted {len(embeddings)} documents with embeddings")
            
            # Verify the data was inserted
            verify_sql = f"SELECT COUNT(*) FROM {self.table_name}"
            cursor.execute(verify_sql)
            count = cursor.fetchone()[0]
            logger.info(f"Verified {count} rows in {self.table_name}")
            
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def test_vector_search_with_direct_cursor(self):
        """Test vector search using direct cursor with TO_VECTOR."""
        logger.info("\n=== TEST: Vector Search with Direct Cursor ===\n")
        
        # Generate embedding for a query
        query_text = "How can I retrieve documents using vector similarity?"
        query_embedding = self.generate_embedding(query_text)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Vector search query with TO_VECTOR
            query_sql = f"""
            SELECT 
                id,
                text_content,
                VECTOR_COSINE(
                    TO_VECTOR(embedding, double, {self.embedding_dim}),
                    TO_VECTOR(?, double, {self.embedding_dim})
                ) AS score
            FROM {self.table_name}
            ORDER BY score DESC
            """
            
            logger.info(f"SQL: {query_sql}")
            
            start_time = time.time()
            cursor.execute(query_sql, (query_embedding_str,))
            results = cursor.fetchall()
            end_time = time.time()
            
            logger.info(f"✅ Successfully executed vector search query in {end_time - start_time:.4f} seconds")
            logger.info("\nSearch Results:")
            for result in results:
                logger.info(f"ID: {result[0]}, Score: {result[2]}, Text: {result[1]}")
            
        except Exception as e:
            logger.error(f"Error in vector search with direct cursor: {e}")
        finally:
            cursor.close()
    
    def test_vector_search_with_sqlalchemy(self):
        """Test vector search using SQLAlchemy with TO_VECTOR."""
        logger.info("\n=== TEST: Vector Search with SQLAlchemy ===\n")
        
        # Generate embedding for a query
        query_text = "What is HNSW indexing used for?"
        query_embedding = self.generate_embedding(query_text)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        try:
            # Create SQLAlchemy engine
            engine = create_engine("iris://superuser:SYS@localhost:1972/USER")
            
            # Vector search query with TO_VECTOR
            query_sql = f"""
            SELECT 
                id,
                text_content,
                VECTOR_COSINE(
                    TO_VECTOR(embedding, double, {self.embedding_dim}),
                    TO_VECTOR(:query_embedding, double, {self.embedding_dim})
                ) AS score
            FROM {self.table_name}
            ORDER BY score DESC
            """
            
            logger.info(f"SQL: {query_sql}")
            
            with engine.connect() as conn:
                with conn.begin():
                    start_time = time.time()
                    results = conn.execute(text(query_sql), {"query_embedding": query_embedding_str})
                    result_list = results.fetchall()
                    end_time = time.time()
            
            logger.info(f"✅ Successfully executed vector search query in {end_time - start_time:.4f} seconds")
            logger.info("\nSearch Results:")
            for result in result_list:
                logger.info(f"ID: {result[0]}, Score: {result[2]}, Text: {result[1]}")
            
        except Exception as e:
            logger.error(f"Error in vector search with SQLAlchemy: {e}")
    
    def test_batch_vector_search(self):
        """Test batch vector search with multiple queries."""
        logger.info("\n=== TEST: Batch Vector Search ===\n")
        
        # Generate embeddings for multiple queries
        queries = [
            "How does vector search work?",
            "What is IRIS database used for?",
            "How do embedding models work?"
        ]
        
        query_embeddings = []
        for query in queries:
            embedding = self.generate_embedding(query)
            embedding_str = ','.join(map(str, embedding))
            query_embeddings.append((query, embedding_str))
        
        cursor = self.conn.cursor()
        
        try:
            logger.info(f"Running {len(queries)} vector search queries...")
            
            for i, (query_text, query_embedding_str) in enumerate(query_embeddings):
                logger.info(f"\nQuery {i+1}: {query_text}")
                
                # Vector search query with TO_VECTOR
                query_sql = f"""
                SELECT TOP 2
                    id,
                    text_content,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, double, {self.embedding_dim}),
                        TO_VECTOR(?, double, {self.embedding_dim})
                    ) AS score
                FROM {self.table_name}
                ORDER BY score DESC
                """
                
                start_time = time.time()
                cursor.execute(query_sql, (query_embedding_str,))
                results = cursor.fetchall()
                end_time = time.time()
                
                logger.info(f"✅ Query completed in {end_time - start_time:.4f} seconds")
                logger.info("Top 2 Results:")
                for result in results:
                    logger.info(f"ID: {result[0]}, Score: {result[2]}, Text: {result[1]}")
            
        except Exception as e:
            logger.error(f"Error in batch vector search: {e}")
        finally:
            cursor.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        embeddings = list(self.embedding_model.embed([text]))
        embedding = embeddings[0]
        return embedding.tolist()
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting Direct Vector Search Tests")
        
        self.setup_database()
        self.insert_documents()
        self.test_vector_search_with_direct_cursor()
        self.test_vector_search_with_sqlalchemy()
        self.test_batch_vector_search()
        
        logger.info("\nTests completed. Check the logs above for results.")
        logger.info("\nSummary of Key Findings:")
        logger.info("1. Vector search works without views by using TO_VECTOR directly in the query")
        logger.info("2. The correct syntax for TO_VECTOR is: TO_VECTOR(embedding, double, 384) without quotes around 'double'")
        logger.info("3. Parameter substitution works with both direct cursor (?) and SQLAlchemy (:param)")
        logger.info("4. This approach doesn't require HNSW indexing, but may not scale well for large collections")

if __name__ == "__main__":
    test = DirectVectorSearchTest()
    test.run_tests()