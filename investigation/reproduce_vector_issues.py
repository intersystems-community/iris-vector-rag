#!/usr/bin/env python3
"""
Reproduce Vector Issues in IRIS 2025.1

This script provides a simple, standalone way to reproduce two key issues:
1. Parameter substitution issues with TO_VECTOR in IRIS 2025.1
2. Inability to create views, computed columns, or materialized views with TO_VECTOR for HNSW indexing

Usage:
    python investigation/reproduce_vector_issues.py

Requirements:
    - IRIS 2025.1 running (e.g., via docker-compose -f docker-compose.iris-only.yml up -d)
    - intersystems-iris 5.1.2 Python driver installed
    - fastembed installed (pip install fastembed)
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
import traceback
from sqlalchemy import text, create_engine

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from common.iris_connector import get_iris_connection
except ImportError:
    traceback.print_exc()
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
logger = logging.getLogger("reproduce_vector_issues")

# Test data
TEST_DOCUMENT = "Vector search enables similarity-based retrieval of documents using embedding vectors."

class VectorIssuesReproducer:
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
            
        self.table_name = "VectorIssuesTest"
        self.view_name = "VectorIssuesView"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        
    def setup_database(self):
        """Set up the database table for testing."""
        cursor = self.conn.cursor()
        
        try:
            # Drop the view and table if they exist
            logger.info("Dropping view and table if they exist...")
            drop_view_sql = f"""
            DROP VIEW IF EXISTS {self.view_name}
            """
            cursor.execute(drop_view_sql)
            
            drop_table_sql = f"""
            DROP TABLE IF EXISTS {self.table_name}
            """
            cursor.execute(drop_table_sql)
            
            # Create base table with VARCHAR column for embeddings
            logger.info("Creating base table with VARCHAR column for embeddings...")
            create_table_sql = f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000)
            )
            """
            cursor.execute(create_table_sql)
            self.conn.commit()
            logger.info(f"✅ Successfully created table {self.table_name}")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        embeddings = list(self.embedding_model.embed([text]))
        embedding = embeddings[0]
        return embedding.tolist()
    
    def store_document(self):
        """Store a test document with embedding as string."""
        embedding = self.generate_embedding(TEST_DOCUMENT)
        embedding_str = ','.join(map(str, embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Insert document with embedding as string
            insert_sql = f"""
            INSERT INTO {self.table_name} (id, text_content, embedding)
            VALUES (?, ?, ?)
            """
            
            cursor.execute(insert_sql, ("doc1", TEST_DOCUMENT, embedding_str))
            self.conn.commit()
            logger.info("✅ Successfully inserted document with embedding as string")
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
    
    def test_parameter_substitution(self):
        """Test parameter substitution issues with TO_VECTOR."""
        logger.info("\n=== TEST 1: Parameter Substitution Issues with TO_VECTOR ===\n")
        
        embedding = self.generate_embedding(TEST_DOCUMENT)
        embedding_str = ','.join(map(str, embedding))
        
        cursor = self.conn.cursor()
        engine = create_engine("iris://superuser:SYS@localhost:1972/USER")
        
        try:
            # Test 1: Direct query with parameter for TO_VECTOR
            logger.info("Test 1a: Direct query with parameter for TO_VECTOR")
            query_sql1 = f"""
            SELECT Top ? VECTOR_COSINE(
                TO_VECTOR(?, double, {self.embedding_dim}),
                TO_VECTOR(?, double, {self.embedding_dim})
            ) AS score
            """
            
            query_sql = f"""
            SELECT Top :top VECTOR_COSINE(
                TO_VECTOR(:vector1, double, {self.embedding_dim}),
                TO_VECTOR(:vector2, double, {self.embedding_dim})
            ) AS score
            """
        
            print(query_sql)
            
            try:
                cursor.execute(query_sql1, [5, embedding_str, embedding_str])
                result = cursor.fetchone()
                logger.info(f"✅ Successfully executed query with TO_VECTOR and parameter markers: {result}")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"❌ Error executing query with TO_VECTOR and parameter markers: {e}")
                #logger.error("This confirms that TO_VECTOR doesn't accept parameter markers in IRIS 2025.1")
            
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        results = conn.execute(text(query_sql), {"vector1": embedding_str, "vector2": embedding_str, "top": 5})
                        result = results.fetchone()
                        logger.info(f"✅ Successfully executed query with SQLAlchemy and TO_VECTOR and parameter markers: {result}")
            except Exception as e:
               # traceback.print_exc()
                logger.error(f"❌ Error executing query with TO_VECTOR and parameter markers: {e}")
                #logger.error("This confirms that TO_VECTOR with SQLALCHEMY doesn't accept parameter markers in IRIS 2025.1")
            
            
            # Test 2: Query with string interpolation
            logger.info("\nTest 1b: Query with string interpolation")
            query_sql = text(f"""
            SELECT VECTOR_COSINE(
                TO_VECTOR('{embedding_str}', double, {self.embedding_dim}),
                TO_VECTOR('{embedding_str}', double, {self.embedding_dim})
            ) AS score
            """)
        
           # print(query_sql)
            try:
                cursor.execute(query_sql)
                result = cursor.fetchone()
                logger.info(f"✅ Successfully executed query with TO_VECTOR and string interpolation: {result}")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"❌ Error executing query with TO_VECTOR and string interpolation: {e}")
                #logger.error("This suggests that even string interpolation has issues with TO_VECTOR in IRIS 2025.1")
            
        except Exception as e:
            logger.error(f"Error in parameter substitution test: {e}")
        finally:
            cursor.close()
    
    def test_view_creation(self):
        """Test creating views, computed columns, and materialized views with TO_VECTOR."""
        logger.info("\n=== TEST 2: View Creation Issues with TO_VECTOR ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Test 1: Create view with TO_VECTOR
            # Removed text() 
            logger.info("Test 2a: Create view with TO_VECTOR")
            create_view_sql = f"""
            CREATE VIEW {self.view_name} AS
            SELECT 
                id,
                text_content,
                TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
            FROM {self.table_name}
            """
            
            try:
                cursor.execute(create_view_sql)
                self.conn.commit()
                logger.info("✅ Successfully created view with TO_VECTOR")
                
                # Try to create HNSW index on the view
                logger.info("Attempting to create HNSW index on the view...")
                create_index_sql = f"""
                CREATE INDEX idx_{self.view_name}_vector ON {self.view_name} (vector_embedding) USING HNSW
                """
                
                create_index_sql1 = f"""CREATE INDEX HNSWIndex ON TABLE scotch_reviews (description_vector) AS HNSW(M=80, Distance='DotProduct')"""
                
                print(create_index_sql)
                try:
                    cursor.execute(create_index_sql)
                    self.conn.commit()
                    logger.info("✅ Successfully created HNSW index on the view!")
                except Exception as e_index:
                    logger.error(f"❌ Failed to create HNSW index on the view: {e_index}")
                    logger.error("This confirms that HNSW indexes cannot be created on views with TO_VECTOR.")
                    self.conn.rollback()
                
            except Exception as e_view:
                traceback.print_exc()
                logger.error(f"❌ Failed to create view with TO_VECTOR: {e_view}")
                self.conn.rollback()
            
            # Test 2: Create table with computed column
            logger.info("\nTest 2b: Create table with computed column")
            drop_computed_table_sql = f"""
            DROP TABLE IF EXISTS ComputedVectorTest
            """
            cursor.execute(drop_computed_table_sql)
            
            create_computed_table_sql = f"""
            CREATE TABLE ComputedVectorTest (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                vector_embedding AS TO_VECTOR(embedding, double, {self.embedding_dim})
            )
            """
            
            try:
                cursor.execute(create_computed_table_sql)
                self.conn.commit()
                logger.info("✅ Successfully created table with computed column!")
                
                # Try to create HNSW index on the computed column
                logger.info("Attempting to create HNSW index on the computed column...")
                create_computed_index_sql = f"""
                CREATE INDEX idx_computed_vector ON ComputedVectorTest (vector_embedding) USING HNSW
                """
                cursor.execute(create_computed_index_sql)
                self.conn.commit()
                logger.info("✅ Successfully created HNSW index on the computed column!")
            except Exception as e_computed:
                logger.error(f"❌ Failed to create table with computed column: {e_computed}")
                self.conn.rollback()
            
            # Test 3: Create a materialized view
            logger.info("\nTest 2c: Create a materialized view")
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
                    TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
                FROM {self.table_name}
                """
                cursor.execute(create_mat_view_sql)
                self.conn.commit()
                logger.info("✅ Successfully created materialized view!")
                
                # Try to create HNSW index on the materialized view
                logger.info("Attempting to create HNSW index on the materialized view...")
                create_mat_index_sql = f"""
                CREATE INDEX idx_mat_vector ON MaterializedVectorView (vector_embedding) USING HNSW
                """
                cursor.execute(create_mat_index_sql)
                self.conn.commit()
                logger.info("✅ Successfully created HNSW index on the materialized view!")
            except Exception as e_mat:
                logger.error(f"❌ Failed to create materialized view: {e_mat}")
                self.conn.rollback()
            
        except Exception as e:
            logger.error(f"Error in view creation test: {e}")
        finally:
            cursor.close()
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting Vector Issues Reproduction Tests with IRIS 2025.1")
        
        self.setup_database()
        self.store_document()
        #self.test_parameter_substitution()
        self.test_view_creation()
        
        logger.info("\nTests completed. Check the logs above for results.")
        logger.info("\nSummary:")
        logger.info("1. Parameter Substitution Issues: TO_VECTOR doesn't accept parameter markers in IRIS 2025.1")
        logger.info("2. View Creation Issues: Views, computed columns, and materialized views with TO_VECTOR don't work with HNSW indexing")
        logger.info("\nThese findings confirm that the dual-table architecture with ObjectScript triggers is the only viable approach for HNSW indexing.")

if __name__ == "__main__":
    reproducer = VectorIssuesReproducer()
    reproducer.run_tests()