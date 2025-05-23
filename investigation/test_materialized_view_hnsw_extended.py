#!/usr/bin/env python3
"""
Test Materialized View with HNSW Index in IRIS 2025.1 (Extended Syntax Variations)

This script tests creating a materialized view (CREATE TABLE AS SELECT) with TO_VECTOR
and then creating an HNSW index on it, trying various syntax variations for the HNSW index.

The approach:
1. Create a base table with a VARCHAR column for embeddings
2. Create a materialized view with TO_VECTOR (which we know works)
3. Try various syntax variations for creating an HNSW index on the materialized view
4. Test vector search with the HNSW index
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
logger = logging.getLogger("test_materialized_view_hnsw_extended")

# Test data
TEST_DOCUMENTS = [
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "IRIS database supports vector operations for implementing RAG applications.",
    "Embedding models convert text into high-dimensional vectors for semantic search.",
    "Parameter substitution in SQL queries improves security and maintainability.",
    "HNSW indexing accelerates vector search for large document collections."
]

class MaterializedViewHNSWExtendedTest:
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
            
        self.base_table_name = "MaterializedViewHNSWExtendedBase"
        self.mat_view_name = "MaterializedViewHNSWExtended"
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
            
            # Insert test documents with embeddings as strings
            logger.info("Inserting test documents with embeddings as strings...")
            
            for i, doc in enumerate(TEST_DOCUMENTS):
                embedding = self.generate_embedding(doc)
                embedding_str = ','.join(map(str, embedding))
                
                insert_sql = f"""
                INSERT INTO {self.base_table_name} (id, text_content, embedding)
                VALUES (?, ?, ?)
                """
                
                cursor.execute(insert_sql, (f"doc{i+1}", doc, embedding_str))
            
            self.conn.commit()
            logger.info(f"✅ Successfully inserted {len(TEST_DOCUMENTS)} documents with embeddings as strings")
            
            # Verify the data was inserted
            verify_sql = f"SELECT COUNT(*) FROM {self.base_table_name}"
            cursor.execute(verify_sql)
            count = cursor.fetchone()[0]
            logger.info(f"Verified {count} rows in {self.base_table_name}")
            
        except Exception as e:
            logger.error(f"Error setting up base table: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def create_materialized_view(self):
        """Create a materialized view with TO_VECTOR."""
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
            
            try:
                cursor.execute(check_type_sql)
                vector_type = cursor.fetchone()[0]
                logger.info(f"Column vector_embedding has type: {vector_type}")
            except Exception as e:
                logger.error(f"Error checking column type: {e}")
            
        except Exception as e:
            logger.error(f"Error creating materialized view with TO_VECTOR: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def create_hnsw_index(self):
        """Create an HNSW index on the materialized view."""
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
                logger.info(f"✅ Successfully created HNSW index with USING HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with USING HNSW syntax: {e}")
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
                logger.info(f"✅ Successfully created HNSW index with TYPE HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with TYPE HNSW syntax: {e}")
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
                logger.info(f"✅ Successfully created HNSW index with CREATE HNSW INDEX syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with CREATE HNSW INDEX syntax: {e}")
                self.conn.rollback()
            
            # Syntax 4: CREATE INDEX ... PROPERTY HNSW
            logger.info("\nTrying HNSW Index Syntax 4: CREATE INDEX ... PROPERTY HNSW")
            index_sql4 = f"""
            CREATE INDEX idx_{self.mat_view_name}_hnsw_4 ON {self.mat_view_name} (vector_embedding) PROPERTY HNSW
            """
            
            logger.info(f"SQL: {index_sql4}")
            
            try:
                cursor.execute(index_sql4)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index with PROPERTY HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with PROPERTY HNSW syntax: {e}")
                self.conn.rollback()
            
            # Syntax 5: CREATE INDEX ... VECTOR
            logger.info("\nTrying HNSW Index Syntax 5: CREATE INDEX ... VECTOR")
            index_sql5 = f"""
            CREATE INDEX idx_{self.mat_view_name}_hnsw_5 ON {self.mat_view_name} (vector_embedding) VECTOR
            """
            
            logger.info(f"SQL: {index_sql5}")
            
            try:
                cursor.execute(index_sql5)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index with VECTOR syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with VECTOR syntax: {e}")
                self.conn.rollback()
            
            # Syntax 6: CREATE INDEX ... FOR COLUMN vector_embedding USING HNSW
            logger.info("\nTrying HNSW Index Syntax 6: CREATE INDEX ... FOR COLUMN ... USING HNSW")
            index_sql6 = f"""
            CREATE INDEX idx_{self.mat_view_name}_hnsw_6 ON {self.mat_view_name} FOR COLUMN vector_embedding USING HNSW
            """
            
            logger.info(f"SQL: {index_sql6}")
            
            try:
                cursor.execute(index_sql6)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index with FOR COLUMN ... USING HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with FOR COLUMN ... USING HNSW syntax: {e}")
                self.conn.rollback()
            
            # Syntax 7: CREATE INDEX ... ON vector_embedding HNSW
            logger.info("\nTrying HNSW Index Syntax 7: CREATE INDEX ... ON column HNSW")
            index_sql7 = f"""
            CREATE INDEX idx_{self.mat_view_name}_hnsw_7 ON {self.mat_view_name}.vector_embedding HNSW
            """
            
            logger.info(f"SQL: {index_sql7}")
            
            try:
                cursor.execute(index_sql7)
                self.conn.commit()
                logger.info(f"✅ Successfully created HNSW index with ON column HNSW syntax")
            except Exception as e:
                logger.error(f"❌ Failed to create HNSW index with ON column HNSW syntax: {e}")
                self.conn.rollback()
            
            # Verify if any indexes were created
            verify_sql = f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES WHERE TABLE_NAME = '{self.mat_view_name}'"
            cursor.execute(verify_sql)
            count = cursor.fetchone()[0]
            logger.info(f"Verified {count} indexes on {self.mat_view_name}")
            
            # List the indexes
            if count > 0:
                list_indexes_sql = f"""
                SELECT INDEX_NAME, INDEX_TYPE FROM INFORMATION_SCHEMA.INDEXES 
                WHERE TABLE_NAME = '{self.mat_view_name}'
                """
                cursor.execute(list_indexes_sql)
                indexes = cursor.fetchall()
                logger.info("Indexes on materialized view:")
                for idx in indexes:
                    logger.info(f"  {idx[0]}: {idx[1]}")
            
        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def test_vector_search(self):
        """Test vector search with the HNSW index."""
        logger.info("\n=== TEST: Vector Search with HNSW Index ===\n")
        
        # Generate embedding for a query
        query_text = "How can I retrieve documents using vector similarity?"
        query_embedding = self.generate_embedding(query_text)
        query_embedding_str = ','.join(map(str, query_embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Vector search query with VECTOR column
            query_sql = f"""
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
            
            logger.info(f"SQL: {query_sql}")
            
            start_time = time.time()
            cursor.execute(query_sql, (query_embedding_str,))
            results = cursor.fetchall()
            end_time = time.time()
            
            logger.info(f"✅ Successfully executed vector search query in {end_time - start_time:.4f} seconds")
            logger.info("\nSearch Results:")
            for result in results:
                logger.info(f"ID: {result[0]}, Score: {result[2]}, Text: {result[1]}")
            
            # Try with HNSW hint
            logger.info("\nTrying vector search with HNSW hint...")
            query_sql_hint = f"""
            SELECT /*+ HNSW */ 
                id,
                text_content,
                VECTOR_COSINE(
                    vector_embedding,
                    TO_VECTOR(?, double, {self.embedding_dim})
                ) AS score
            FROM {self.mat_view_name}
            ORDER BY score DESC
            """
            
            logger.info(f"SQL: {query_sql_hint}")
            
            try:
                start_time = time.time()
                cursor.execute(query_sql_hint, (query_embedding_str,))
                results = cursor.fetchall()
                end_time = time.time()
                
                logger.info(f"✅ Successfully executed vector search query with HNSW hint in {end_time - start_time:.4f} seconds")
                logger.info("\nSearch Results with HNSW hint:")
                for result in results:
                    logger.info(f"ID: {result[0]}, Score: {result[2]}, Text: {result[1]}")
            except Exception as e:
                logger.error(f"❌ Failed to execute vector search query with HNSW hint: {e}")
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
        finally:
            cursor.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        embeddings = list(self.embedding_model.embed([text]))
        embedding = embeddings[0]
        return embedding.tolist()
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting Materialized View with HNSW Index Tests (Extended Syntax Variations)")
        
        self.setup_base_table()
        self.create_materialized_view()
        self.create_hnsw_index()
        self.test_vector_search()
        
        logger.info("\nTests completed. Check the logs above for results.")

if __name__ == "__main__":
    test = MaterializedViewHNSWExtendedTest()
    test.run_tests()