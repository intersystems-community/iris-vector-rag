#!/usr/bin/env python3
"""
Test Materialized View with HNSW Index in IRIS 2025.1

This script tests whether we can create a materialized view (CREATE TABLE AS SELECT)
with TO_VECTOR and then create an HNSW index on it, using the correct syntax we
discovered in our previous tests.

Based on our findings about the correct syntax for TO_VECTOR (using 'double' without quotes),
we want to see if we can create a materialized view with TO_VECTOR and then create an HNSW index on it.
"""

import os
import sys
import logging
import traceback
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
logger = logging.getLogger("test_materialized_view_hnsw")

# Test data
TEST_DOCUMENT = "Vector search enables similarity-based retrieval of documents using embedding vectors."

class MaterializedViewHNSWTest:
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
            
        self.base_table_name = "MaterializedViewHNSWTest"
        self.mat_view_name = "MaterializedViewHNSW"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        
    def setup_base_table(self):
        """Set up the base table with a VARCHAR column for embeddings."""
        logger.info("\n=== SETUP: Creating Base Table with VARCHAR Column ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Drop the tables if they exist
            logger.info("Dropping tables if they exist...")
            drop_mat_view_sql = f"""
            DROP TABLE IF EXISTS {self.mat_view_name}
            """
            cursor.execute(drop_mat_view_sql)
            
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
                embedding VARCHAR(60000)
            )
            """
            
            logger.info(f"SQL: {create_table_sql}")
            
            cursor.execute(create_table_sql)
            self.conn.commit()
            logger.info(f"✅ Successfully created base table {self.base_table_name}")
            
            # Insert test document with embedding as string
            embedding = self.generate_embedding(TEST_DOCUMENT)
            embedding_str = ','.join(map(str, embedding))
            
            insert_sql = f"""
            INSERT INTO {self.base_table_name} (id, text_content, embedding)
            VALUES (?, ?, ?)
            """
            
            cursor.execute(insert_sql, ("doc1", TEST_DOCUMENT, embedding_str))
            self.conn.commit()
            logger.info("✅ Successfully inserted document with embedding as string")
            
        except Exception as e:
            logger.error(f"Error setting up base table: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def test_create_materialized_view(self):
        """Test creating a materialized view with TO_VECTOR."""
        logger.info("\n=== TEST: Creating Materialized View with TO_VECTOR ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Create materialized view with TO_VECTOR (without quotes around 'double')
            logger.info("Creating materialized view with TO_VECTOR (without quotes)")
            create_mat_view_sql = f"""
            CREATE TABLE {self.mat_view_name} AS
            SELECT 
                id,
                text_content,
                TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
            FROM {self.base_table_name}
            """
            
            logger.info(f"SQL: {create_mat_view_sql}")
            
            cursor.execute(create_mat_view_sql)
            self.conn.commit()
            logger.info(f"✅ Successfully created materialized view {self.mat_view_name} with TO_VECTOR (without quotes)")
            
            # Verify the materialized view was created and has data
            verify_sql = f"SELECT COUNT(*) FROM {self.mat_view_name}"
            cursor.execute(verify_sql)
            count = cursor.fetchone()[0]
            logger.info(f"Verified {count} rows in {self.mat_view_name}")
            
            # Check the column type of vector_embedding
            check_type_sql = f"""
            SELECT TOP 1 
                TYPEOF(vector_embedding) AS vector_type
            FROM {self.mat_view_name}
            """
            
            cursor.execute(check_type_sql)
            vector_type = cursor.fetchone()[0]
            logger.info(f"Column vector_embedding has type: {vector_type}")
            
        except Exception as e:
            logger.error(f"Error creating materialized view with TO_VECTOR: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def test_create_hnsw_index_on_mat_view(self):
        """Test creating an HNSW index on the materialized view."""
        logger.info("\n=== TEST: Creating HNSW Index on Materialized View ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Try different HNSW index syntax variations
            
            # Syntax 1: CREATE INDEX ... USING HNSW
            logger.info("\nTrying HNSW Index Syntax 1: CREATE INDEX ... USING HNSW")
            index_sql1 = f"""
            CREATE INDEX idx_{self.mat_view_name}_hnsw_1 ON {self.mat_view_name} (vector_embedding) USING HNSW
            """
            
            logger.info(f"SQL: {index_sql1}")
            
            try:
                cursor.execute(index_sql1)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index on {self.mat_view_name} with USING HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index on {self.mat_view_name} with USING HNSW syntax: {e}")
                self.conn.rollback()
            
            # Syntax 2: CREATE INDEX ... TYPE HNSW
            logger.info("\nTrying HNSW Index Syntax 2: CREATE INDEX ... TYPE HNSW")
            index_sql2 = f"""
            CREATE INDEX idx_{self.mat_view_name}_hnsw_2 ON {self.mat_view_name} (vector_embedding) TYPE HNSW
            """
            
            logger.info(f"SQL: {index_sql2}")
            
            try:
                cursor.execute(index_sql2)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index on {self.mat_view_name} with TYPE HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index on {self.mat_view_name} with TYPE HNSW syntax: {e}")
                self.conn.rollback()
            
            # Syntax 3: CREATE HNSW INDEX ...
            logger.info("\nTrying HNSW Index Syntax 3: CREATE HNSW INDEX ...")
            index_sql3 = f"""
            CREATE HNSW INDEX idx_{self.mat_view_name}_hnsw_3 ON {self.mat_view_name} (vector_embedding)
            """
            
            logger.info(f"SQL: {index_sql3}")
            
            try:
                cursor.execute(index_sql3)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index on {self.mat_view_name} with CREATE HNSW INDEX syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index on {self.mat_view_name} with CREATE HNSW INDEX syntax: {e}")
                self.conn.rollback()
            
            # Verify the index was created
            try:
                verify_sql = f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES WHERE TABLE_NAME = '{self.mat_view_name}'"
                cursor.execute(verify_sql)
                count = cursor.fetchone()[0]
                logger.info(f"Verified {count} indexes on {self.mat_view_name}")
            except Exception as e:
                logger.error(f"Error verifying indexes on {self.mat_view_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error in HNSW index test: {e}")
        finally:
            cursor.close()
    
    def test_vector_search_query(self):
        """Test vector search query using the materialized view and HNSW index."""
        logger.info("\n=== TEST: Vector Search Query ===\n")
        
        # Generate embedding for a similar query
        query_text = "How can I retrieve documents using vector similarity?"
        query_embedding = self.generate_embedding(query_text)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Try different vector search query syntax variations
            
            # Syntax 1: Direct query with TO_VECTOR
            logger.info("\nTrying Vector Search Syntax 1: Direct query with TO_VECTOR")
            query_sql1 = f"""
            SELECT 
                id,
                text_content,
                VECTOR_COSINE(
                    vector_embedding,
                    TO_VECTOR(?, double, {self.embedding_dim})
                ) AS score
            FROM {self.mat_view_name}
            ORDER BY score DESC
            """
            
            logger.info(f"SQL: {query_sql1}")
            
            try:
                cursor.execute(query_sql1, (query_embedding_str,))
                result = cursor.fetchall()
                logger.info(f"✅ Successfully executed vector search query with direct TO_VECTOR: {result}")
            except Exception as e:
                logger.error(f"❌ Failed to execute vector search query with direct TO_VECTOR: {e}")
            
            # Syntax 2: Using SQLAlchemy with named parameters
            logger.info("\nTrying Vector Search Syntax 2: SQLAlchemy with named parameters")
            engine = create_engine("iris://superuser:SYS@localhost:1972/USER")
            query_sql2 = f"""
            SELECT 
                id,
                text_content,
                VECTOR_COSINE(
                    vector_embedding,
                    TO_VECTOR(:query_embedding, double, {self.embedding_dim})
                ) AS score
            FROM {self.mat_view_name}
            ORDER BY score DESC
            """
            
            logger.info(f"SQL: {query_sql2}")
            
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        results = conn.execute(text(query_sql2), {"query_embedding": query_embedding_str})
                        result = results.fetchall()
                        logger.info(f"✅ Successfully executed vector search query with SQLAlchemy and named parameters: {result}")
            except Exception as e:
                logger.error(f"❌ Failed to execute vector search query with SQLAlchemy and named parameters: {e}")
        
        except Exception as e:
            logger.error(f"Error in vector search query test: {e}")
        finally:
            cursor.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        embeddings = list(self.embedding_model.embed([text]))
        embedding = embeddings[0]
        return embedding.tolist()
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting Materialized View with HNSW Index Tests")
        
        self.setup_base_table()
        self.test_create_materialized_view()
        self.test_create_hnsw_index_on_mat_view()
        self.test_vector_search_query()
        
        logger.info("\nTests completed. Check the logs above for results.")

if __name__ == "__main__":
    test = MaterializedViewHNSWTest()
    test.run_tests()