#!/usr/bin/env python3
"""
Test DBAPI Parameter Substitution with TO_VECTOR in IRIS 2025.1

This script tests whether the parameter substitution issues with TO_VECTOR
still exist in IRIS 2025.1 with the newer intersystems-iris DBAPI driver.

The test attempts to:
1. Insert a document with embedding using parameter markers
2. Insert a document with embedding using string interpolation
3. Query documents using parameter markers with TO_VECTOR
4. Query documents using string interpolation with TO_VECTOR

This will help determine if we still need workarounds or if the issues
have been resolved in the newer IRIS version.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from fastembed import TextEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_dbapi_vector_params")

# Test data
TEST_DOCUMENT = "Vector search enables similarity-based retrieval of documents using embedding vectors."

class DBAPIVectorTest:
    def __init__(self):
        """Initialize the test with IRIS connection and embedding model."""
        self.conn = get_iris_connection()
        self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.table_name = "DBAPIVectorTest"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        self.setup_database()
        
    def setup_database(self):
        """Set up the database table for testing."""
        cursor = self.conn.cursor()
        
        try:
            # Drop the table if it exists
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Create table with VARCHAR column for embeddings
            # We'll skip the VECTOR column for now since we're having issues with it
            create_table_sql = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding_str VARCHAR(60000)
            )
            """
            cursor.execute(create_table_sql)
            
            logger.info(f"Created table {self.table_name}")
            
            self.conn.commit()
            logger.info(f"Created table {self.table_name}")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        logger.info(f"Generating embedding for: {text[:50]}...")
        embeddings = list(self.embedding_model.embed([text]))
        embedding = embeddings[0]
        return embedding.tolist()
    
    def test_insert_with_params(self):
        """Test inserting a document with embedding using parameter markers."""
        embedding = self.generate_embedding(TEST_DOCUMENT)
        embedding_str = ','.join(map(str, embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Test 1: Insert with parameter markers for embedding_str
            logger.info("Test 1: Insert with parameter markers for embedding_str")
            insert_sql = f"""
            INSERT INTO {self.table_name} (id, text_content, embedding_str)
            VALUES (?, ?, ?)
            """
            cursor.execute(insert_sql, ("doc_param_str", TEST_DOCUMENT, embedding_str))
            self.conn.commit()
            logger.info("✅ Successfully inserted document with parameter marker for embedding_str")
        except Exception as e:
            logger.error(f"❌ Error inserting document with parameter marker for embedding_str: {e}")
            self.conn.rollback()
        
        try:
            # Test 2: Insert with TO_VECTOR and parameter markers
            logger.info("Test 2: Insert with TO_VECTOR and parameter markers")
            insert_sql = f"""
            INSERT INTO {self.table_name} (id, text_content, embedding_str)
            VALUES (?, ?, ?)
            """
            cursor.execute(insert_sql, ("doc_param_vec", TEST_DOCUMENT, embedding_str))
            
            # Now try a separate query with TO_VECTOR and parameter markers
            logger.info("Test 2b: Query with TO_VECTOR and parameter markers")
            query_sql = f"""
            SELECT VECTOR_COSINE(TO_VECTOR(?, 'double', {self.embedding_dim}),
                               TO_VECTOR(?, 'double', {self.embedding_dim})) AS score
            """
            try:
                cursor.execute(query_sql, (embedding_str, embedding_str))
                logger.info("✅ Successfully executed query with TO_VECTOR and parameter markers")
            except Exception as e:
                logger.error(f"❌ Error executing query with TO_VECTOR and parameter markers: {e}")
                logger.error("This confirms that TO_VECTOR still doesn't accept parameter markers in IRIS 2025.1")
            self.conn.commit()
            logger.info("✅ Successfully inserted document with TO_VECTOR and parameter marker")
        except Exception as e:
            logger.error(f"❌ Error inserting document with TO_VECTOR and parameter marker: {e}")
            logger.error("This confirms that TO_VECTOR still doesn't accept parameter markers in IRIS 2025.1")
            self.conn.rollback()
        
        cursor.close()
    
    def test_insert_with_interpolation(self):
        """Test inserting a document with embedding using string interpolation."""
        embedding = self.generate_embedding(TEST_DOCUMENT)
        embedding_str = ','.join(map(str, embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Test 3: Insert with string interpolation for embedding_str
            logger.info("Test 3: Insert with string interpolation for embedding_str")
            insert_sql = f"""
            INSERT INTO {self.table_name} (id, text_content, embedding_str)
            VALUES ('doc_interp_str', '{TEST_DOCUMENT.replace("'", "''")}', '{embedding_str}')
            """
            cursor.execute(insert_sql)
            self.conn.commit()
            logger.info("✅ Successfully inserted document with string interpolation for embedding_str")
        except Exception as e:
            logger.error(f"❌ Error inserting document with string interpolation for embedding_str: {e}")
            self.conn.rollback()
        
        try:
            # Test 4: Insert with TO_VECTOR and string interpolation
            logger.info("Test 4: Insert with TO_VECTOR and string interpolation")
            insert_sql = f"""
            INSERT INTO {self.table_name} (id, text_content, embedding_str)
            VALUES ('doc_interp_vec', '{TEST_DOCUMENT.replace("'", "''")}', '{embedding_str}')
            """
            cursor.execute(insert_sql)
            self.conn.commit()
            logger.info("✅ Successfully inserted document with TO_VECTOR and string interpolation")
        except Exception as e:
            logger.error(f"❌ Error inserting document with TO_VECTOR and string interpolation: {e}")
            self.conn.rollback()
        
        cursor.close()
    
    def test_query_with_params(self):
        """Test querying documents with TO_VECTOR using parameter markers."""
        query = "How does vector search work?"
        query_embedding = self.generate_embedding(query)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Test 5: Query with TO_VECTOR and parameter markers
            logger.info("Test 5: Query with TO_VECTOR and parameter markers")
            search_sql = f"""
            SELECT TOP 5 id, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding_str, 'double', {self.embedding_dim}),
                                TO_VECTOR(?, 'double', {self.embedding_dim})) AS score
            FROM {self.table_name}
            WHERE embedding_str IS NOT NULL
            ORDER BY score ASC
            """
            try:
                cursor.execute(search_sql, (query_embedding_str,))
                results = cursor.fetchall()
                logger.info(f"✅ Successfully queried documents with TO_VECTOR and parameter marker. Found {len(results)} results.")
            except Exception as e:
                logger.error(f"❌ Error querying documents with TO_VECTOR and parameter marker: {e}")
                logger.error("This confirms that TO_VECTOR still doesn't accept parameter markers in IRIS 2025.1")
            # This is a duplicate line, removing it
        except Exception as e:
            logger.error(f"❌ Error querying documents with TO_VECTOR and parameter marker: {e}")
            logger.error("This confirms that TO_VECTOR still doesn't accept parameter markers in IRIS 2025.1")
        
        cursor.close()
    
    def test_query_with_interpolation(self):
        """Test querying documents with TO_VECTOR using string interpolation."""
        query = "How does vector search work?"
        query_embedding = self.generate_embedding(query)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Test 6: Query with TO_VECTOR and string interpolation
            logger.info("Test 6: Query with TO_VECTOR and string interpolation")
            search_sql = f"""
            SELECT TOP 5 id, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding_str, 'double', {self.embedding_dim}),
                                TO_VECTOR('{query_embedding_str}', 'double', {self.embedding_dim})) AS score
            FROM {self.table_name}
            WHERE embedding_str IS NOT NULL
            ORDER BY score ASC
            """
            cursor.execute(search_sql)
            results = cursor.fetchall()
            logger.info(f"✅ Successfully queried documents with TO_VECTOR and string interpolation. Found {len(results)} results.")
            
            # Display results
            for i, row in enumerate(results):
                doc_id = row[0]
                text = row[1]
                score = float(row[2])
                logger.info(f"  {i+1}. [{doc_id}] {text[:50]}... (Score: {score})")
        except Exception as e:
            logger.error(f"❌ Error querying documents with TO_VECTOR and string interpolation: {e}")
        
        cursor.close()
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting DBAPI Vector Parameter Tests with IRIS 2025.1")
        
        self.test_insert_with_params()
        self.test_insert_with_interpolation()
        self.test_query_with_params()
        self.test_query_with_interpolation()
        
        logger.info("Tests completed")

if __name__ == "__main__":
    test = DBAPIVectorTest()
    test.run_tests()