#!/usr/bin/env python3
"""
Test Computed VECTOR Column with HNSW Index in IRIS 2025.1

This script tests creating a table with a computed column that uses TO_VECTOR
to convert a VARCHAR column to a VECTOR type, and then creating an HNSW index
on the computed column.

The approach:
1. Create a table with a VARCHAR column for embeddings and a computed column using TO_VECTOR
2. Insert data with embeddings as strings
3. Create an HNSW index on the computed VECTOR column
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
    print("Error: Could not import get_iris_connector. Make sure you're running this script from the project root.")
    sys.exit(1)

try:
    from fastembed import TextEmbedding
except ImportError:
    print("Error: fastembed not installed. Install it with: pip install fastembed")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_computed_vector_hnsw")

# Test data
TEST_DOCUMENTS = [
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "IRIS database supports vector operations for implementing RAG applications.",
    "Embedding models convert text into high-dimensional vectors for semantic search.",
    "Parameter substitution in SQL queries improves security and maintainability.",
    "HNSW indexing accelerates vector search for large document collections."
]

class ComputedVectorHNSWTest:
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
            
        self.table_name = "ComputedVectorHNSWTest"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        
    def setup_database(self):
        """Set up the database table with a VARCHAR column and computed VECTOR column."""
        logger.info("\n=== SETUP: Creating Table with Computed VECTOR Column ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Drop the table if it exists
            logger.info("Dropping table if it exists...")
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Try different syntax variations for creating a table with a computed VECTOR column
            
            # Syntax 1: Computed column with TO_VECTOR without quotes
            logger.info("\nTrying to create table with computed VECTOR column...")
            create_table_sql1 = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                vector_embedding AS TO_VECTOR(embedding, double, {self.embedding_dim})
            )
            """
            
            logger.info(f"SQL: {create_table_sql1}")
            
            try:
                cursor.execute(create_table_sql1)
                self.conn.commit()
                logger.info("✅ Successfully created table with computed VECTOR column")
                return
            except Exception as e:
                logger.error(f"❌ Failed to create table with computed VECTOR column: {e}")
                self.conn.rollback()
            
            # Syntax 2: Computed column with TO_VECTOR with quotes
            logger.info("\nTrying alternative syntax with quotes...")
            create_table_sql2 = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                vector_embedding AS TO_VECTOR(embedding, 'double', {self.embedding_dim})
            )
            """
            
            logger.info(f"SQL: {create_table_sql2}")
            
            try:
                cursor.execute(create_table_sql2)
                self.conn.commit()
                logger.info("✅ Successfully created table with computed VECTOR column (quotes)")
                return
            except Exception as e:
                logger.error(f"❌ Failed to create table with computed VECTOR column (quotes): {e}")
                self.conn.rollback()
            
            # Syntax 3: VECTOR data type
            logger.info("\nTrying VECTOR data type...")
            create_table_sql3 = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                vector_embedding VECTOR({self.embedding_dim}, double)
            )
            """
            
            logger.info(f"SQL: {create_table_sql3}")
            
            try:
                cursor.execute(create_table_sql3)
                self.conn.commit()
                logger.info("✅ Successfully created table with VECTOR data type")
                return
            except Exception as e:
                logger.error(f"❌ Failed to create table with VECTOR data type: {e}")
                self.conn.rollback()
            
            # Fallback: Create table with just VARCHAR column
            logger.info("\nFalling back to VARCHAR column only...")
            create_fallback_table_sql = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000)
            )
            """
            
            logger.info(f"SQL: {create_fallback_table_sql}")
            
            cursor.execute(create_fallback_table_sql)
            self.conn.commit()
            logger.info("✅ Successfully created fallback table with VARCHAR column")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def insert_documents(self):
        """Insert test documents with embeddings."""
        logger.info("\n=== TEST: Inserting Documents with Embeddings ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Check table structure
            check_columns_sql = f"""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{self.table_name}'
            """
            
            cursor.execute(check_columns_sql)
            columns = [row[0] for row in cursor.fetchall()]
            logger.info(f"Table columns: {columns}")
            
            has_vector_column = 'VECTOR_EMBEDDING' in [col.upper() for col in columns]
            has_embedding_column = 'EMBEDDING' in [col.upper() for col in columns]
            
            # Generate embeddings for test documents
            embeddings = []
            for i, doc in enumerate(TEST_DOCUMENTS):
                embedding = self.generate_embedding(doc)
                embedding_str = ','.join(map(str, embedding))
                embeddings.append((f"doc{i+1}", doc, embedding_str))
            
            if has_embedding_column:
                # Insert documents with embeddings as strings
                if has_vector_column:
                    # Table has both embedding and vector_embedding columns
                    logger.info("Table has both embedding and vector_embedding columns")
                    
                    # Check if vector_embedding is a computed column
                    check_computed_sql = f"""
                    SELECT TOP 1 * FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{self.table_name}' AND COLUMN_NAME = 'vector_embedding' AND IS_COMPUTED = 'YES'
                    """
                    
                    cursor.execute(check_computed_sql)
                    is_computed = cursor.fetchone() is not None
                    
                    if is_computed:
                        logger.info("vector_embedding is a computed column, inserting only embedding")
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (id, text_content, embedding)
                        VALUES (?, ?, ?)
                        """
                    else:
                        logger.info("vector_embedding is not a computed column, inserting both")
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (id, text_content, embedding, vector_embedding)
                        VALUES (?, ?, ?, TO_VECTOR(?, double, {self.embedding_dim}))
                        """
                else:
                    # Table has only embedding column
                    logger.info("Table has only embedding column")
                    insert_sql = f"""
                    INSERT INTO {self.table_name} (id, text_content, embedding)
                    VALUES (?, ?, ?)
                    """
                
                logger.info(f"SQL: {insert_sql}")
                logger.info(f"Inserting {len(embeddings)} documents...")
                
                for doc_data in embeddings:
                    if has_vector_column and not is_computed:
                        # Include embedding for TO_VECTOR
                        cursor.execute(insert_sql, (doc_data[0], doc_data[1], doc_data[2], doc_data[2]))
                    else:
                        cursor.execute(insert_sql, doc_data)
                
                self.conn.commit()
                logger.info(f"✅ Successfully inserted {len(embeddings)} documents")
            else:
                logger.error("Table doesn't have an embedding column, cannot insert documents")
            
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
    
    def create_hnsw_index(self):
        """Create an HNSW index on the VECTOR column."""
        logger.info("\n=== TEST: Creating HNSW Index ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Check if the table has a VECTOR column
            check_columns_sql = f"""
            SELECT TOP 1 * FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{self.table_name}' AND COLUMN_NAME = 'vector_embedding'
            """
            
            cursor.execute(check_columns_sql)
            has_vector_column = cursor.fetchone() is not None
            
            if has_vector_column:
                # Try different HNSW index syntax variations
                
                # Syntax 1: CREATE INDEX ... USING HNSW
                logger.info("\nTrying HNSW Index Syntax 1: CREATE INDEX ... USING HNSW")
                index_sql1 = f"""
                CREATE INDEX idx_{self.table_name}_hnsw_1 ON {self.table_name} (vector_embedding) USING HNSW
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
                CREATE INDEX idx_{self.table_name}_hnsw_2 ON {self.table_name} (vector_embedding) TYPE HNSW
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
                CREATE HNSW INDEX idx_{self.table_name}_hnsw_3 ON {self.table_name} (vector_embedding)
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
                CREATE INDEX idx_{self.table_name}_hnsw_4 ON {self.table_name} (vector_embedding) PROPERTY HNSW
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
                CREATE INDEX idx_{self.table_name}_hnsw_5 ON {self.table_name} (vector_embedding) VECTOR
                """
                
                logger.info(f"SQL: {index_sql5}")
                
                try:
                    cursor.execute(index_sql5)
                    self.conn.commit()
                    logger.info(f"✅ Successfully created HNSW index with VECTOR syntax")
                except Exception as e:
                    logger.error(f"❌ Failed to create HNSW index with VECTOR syntax: {e}")
                    self.conn.rollback()
            else:
                logger.info("Table doesn't have a VECTOR column, skipping HNSW index creation")
            
            # Verify if any indexes were created
            verify_sql = f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES WHERE TABLE_NAME = '{self.table_name}'"
            cursor.execute(verify_sql)
            count = cursor.fetchone()[0]
            logger.info(f"Verified {count} indexes on {self.table_name}")
            
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
            # Check if the table has a VECTOR column
            check_columns_sql = f"""
            SELECT TOP 1 * FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{self.table_name}' AND COLUMN_NAME = 'vector_embedding'
            """
            
            cursor.execute(check_columns_sql)
            has_vector_column = cursor.fetchone() is not None
            
            if has_vector_column:
                # Vector search query with VECTOR column
                query_sql = f"""
                SELECT 
                    id,
                    text_content,
                    VECTOR_COSINE(
                        vector_embedding,
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
            else:
                # Vector search query with VARCHAR column and TO_VECTOR
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
        logger.info("Starting Computed VECTOR Column with HNSW Index Tests")
        
        self.setup_database()
        self.insert_documents()
        self.create_hnsw_index()
        self.test_vector_search()
        
        logger.info("\nTests completed. Check the logs above for results.")

if __name__ == "__main__":
    test = ComputedVectorHNSWTest()
    test.run_tests()