#!/usr/bin/env python3
"""
Test Working Vector Parameter Substitution in IRIS 2025.1

This script demonstrates the correct syntax for using parameter markers with TO_VECTOR
in IRIS 2025.1, based on feedback from a knowledgeable developer.

Key syntax changes:
1. Use 'double' without quotes: TO_VECTOR(?, double, 384) instead of TO_VECTOR(?, 'double', 384)
2. Use ? with a list of parameters for direct cursor execution
3. Use :var1, :var2 with a dict for SQLAlchemy execution
4. Use text() function from SQLAlchemy in some cases
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
logger = logging.getLogger("test_working_vector_params")

# Test data
TEST_DOCUMENT = "Vector search enables similarity-based retrieval of documents using embedding vectors."

class WorkingVectorParamsTest:
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
            
        self.table_name = "WorkingVectorParamsTest"
        self.embedding_dim = 384  # dimension for all-MiniLM-L6-v2
        
    def setup_database(self):
        """Set up the database table for testing."""
        cursor = self.conn.cursor()
        
        try:
            # Drop the table if it exists
            logger.info("Dropping table if it exists...")
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
    
    def test_working_parameter_substitution(self):
        """Test working parameter substitution with TO_VECTOR."""
        logger.info("\n=== TEST: Working Parameter Substitution with TO_VECTOR ===\n")
        
        embedding = self.generate_embedding(TEST_DOCUMENT)
        embedding_str = ','.join(map(str, embedding))
        
        cursor = self.conn.cursor()
        
        try:
            # Create SQLAlchemy engine
            engine = create_engine("iris://superuser:SYS@localhost:1972/USER")
            
            # Test 1: Direct cursor with ? parameters
            logger.info("Test 1: Direct cursor with ? parameters")
            query_sql1 = f"""
            SELECT TOP ? VECTOR_COSINE(
                TO_VECTOR(?, double, {self.embedding_dim}),
                TO_VECTOR(?, double, {self.embedding_dim})
            ) AS score
            """
            
            logger.info("\n--- SQL FOR DIRECT CURSOR ---")
            logger.info(f"""
SELECT TOP ? VECTOR_COSINE(
    TO_VECTOR(?, double, {self.embedding_dim}),
    TO_VECTOR(?, double, {self.embedding_dim})
) AS score
            """)
            logger.info("Parameters: [5, <embedding_str>, <embedding_str>]")
            
            try:
                cursor.execute(query_sql1, [5, embedding_str, embedding_str])
                result = cursor.fetchone()
                logger.info(f"✅ Successfully executed query with direct cursor and ? parameters: {result}")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"❌ Error executing query with direct cursor and ? parameters: {e}")
            
            # Test 2: SQLAlchemy with named parameters
            logger.info("\nTest 2: SQLAlchemy with named parameters")
            query_sql2 = f"""
            SELECT TOP :top VECTOR_COSINE(
                TO_VECTOR(:vector1, double, {self.embedding_dim}),
                TO_VECTOR(:vector2, double, {self.embedding_dim})
            ) AS score
            """
            
            logger.info("\n--- SQL FOR SQLALCHEMY ---")
            logger.info(f"""
SELECT TOP :top VECTOR_COSINE(
    TO_VECTOR(:vector1, double, {self.embedding_dim}),
    TO_VECTOR(:vector2, double, {self.embedding_dim})
) AS score
            """)
            logger.info("Parameters: {'vector1': <embedding_str>, 'vector2': <embedding_str>, 'top': 5}")
            
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        results = conn.execute(text(query_sql2), {"vector1": embedding_str, "vector2": embedding_str, "top": 5})
                        result = results.fetchone()
                        logger.info(f"✅ Successfully executed query with SQLAlchemy and named parameters: {result}")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"❌ Error executing query with SQLAlchemy and named parameters: {e}")
            
            # Test 3: String interpolation with text()
            logger.info("\nTest 3: String interpolation with text()")
            query_sql3 = text(f"""
            SELECT VECTOR_COSINE(
                TO_VECTOR('{embedding_str}', double, {self.embedding_dim}),
                TO_VECTOR('{embedding_str}', double, {self.embedding_dim})
            ) AS score
            """)
            
            logger.info("\n--- SQL FOR STRING INTERPOLATION ---")
            logger.info(f"""
SELECT VECTOR_COSINE(
    TO_VECTOR('<embedding_str>', double, {self.embedding_dim}),
    TO_VECTOR('<embedding_str>', double, {self.embedding_dim})
) AS score
            """)
            
            try:
                cursor.execute(query_sql3)
                result = cursor.fetchone()
                logger.info(f"✅ Successfully executed query with string interpolation and text(): {result}")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"❌ Error executing query with string interpolation and text(): {e}")
            
        except Exception as e:
            logger.error(f"Error in parameter substitution test: {e}")
        finally:
            cursor.close()
    
    def test_view_creation(self):
        """Test creating views, computed columns, and materialized views with TO_VECTOR."""
        logger.info("\n=== TEST: View Creation with TO_VECTOR ===\n")
        
        cursor = self.conn.cursor()
        
        try:
            # Test 1: Create view with TO_VECTOR (without quotes around 'double')
            logger.info("Test 1: Create view with TO_VECTOR (without quotes)")
            create_view_sql = f"""
            CREATE VIEW WorkingVectorView AS
            SELECT 
                id,
                text_content,
                TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
            FROM {self.table_name}
            """
            
            logger.info("\n--- SQL FOR VIEW CREATION ---")
            logger.info(f"""
CREATE VIEW WorkingVectorView AS
SELECT 
    id,
    text_content,
    TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
FROM {self.table_name}
            """)
            
            try:
                cursor.execute(create_view_sql)
                self.conn.commit()
                logger.info("✅ Successfully created view with TO_VECTOR (without quotes)")
                
                # Try to create HNSW index on the view
                logger.info("Attempting to create HNSW index on the view...")
                create_index_sql = f"""
                CREATE INDEX idx_working_vector_view ON WorkingVectorView (vector_embedding) USING HNSW
                """
                
                logger.info("\n--- SQL FOR HNSW INDEX ON VIEW ---")
                logger.info(f"""
CREATE INDEX idx_working_vector_view ON WorkingVectorView (vector_embedding) USING HNSW
                """)
                
                try:
                    cursor.execute(create_index_sql)
                    self.conn.commit()
                    logger.info("✅ Successfully created HNSW index on the view!")
                except Exception as e_index:
                    logger.error(f"❌ Failed to create HNSW index on the view: {e_index}")
                    self.conn.rollback()
                
            except Exception as e_view:
                logger.error(f"❌ Failed to create view with TO_VECTOR (without quotes): {e_view}")
                self.conn.rollback()
            
            # Test 2: Create table with computed column (without quotes around 'double')
            logger.info("\nTest 2: Create table with computed column (without quotes)")
            drop_computed_table_sql = f"""
            DROP TABLE IF EXISTS ComputedVectorTestWorking
            """
            cursor.execute(drop_computed_table_sql)
            
            create_computed_table_sql = f"""
            CREATE TABLE ComputedVectorTestWorking (
                id VARCHAR(100) PRIMARY KEY,
                text_content TEXT,
                embedding VARCHAR(60000),
                vector_embedding AS TO_VECTOR(embedding, double, {self.embedding_dim})
            )
            """
            
            logger.info("\n--- SQL FOR COMPUTED COLUMN ---")
            logger.info(f"""
CREATE TABLE ComputedVectorTestWorking (
    id VARCHAR(100) PRIMARY KEY,
    text_content TEXT,
    embedding VARCHAR(60000),
    vector_embedding AS TO_VECTOR(embedding, double, {self.embedding_dim})
)
            """)
            
            try:
                cursor.execute(create_computed_table_sql)
                self.conn.commit()
                logger.info("✅ Successfully created table with computed column (without quotes)!")
                
                # Try to create HNSW index on the computed column
                logger.info("Attempting to create HNSW index on the computed column...")
                create_computed_index_sql = f"""
                CREATE INDEX idx_computed_vector_working ON ComputedVectorTestWorking (vector_embedding) USING HNSW
                """
                
                logger.info("\n--- SQL FOR HNSW INDEX ON COMPUTED COLUMN ---")
                logger.info(f"""
CREATE INDEX idx_computed_vector_working ON ComputedVectorTestWorking (vector_embedding) USING HNSW
                """)
                
                try:
                    cursor.execute(create_computed_index_sql)
                    self.conn.commit()
                    logger.info("✅ Successfully created HNSW index on the computed column!")
                except Exception as e_computed_index:
                    logger.error(f"❌ Failed to create HNSW index on the computed column: {e_computed_index}")
                    self.conn.rollback()
            except Exception as e_computed:
                logger.error(f"❌ Failed to create table with computed column (without quotes): {e_computed}")
                self.conn.rollback()
            
            # Test 3: Create a materialized view (without quotes around 'double')
            logger.info("\nTest 3: Create a materialized view (without quotes)")
            drop_mat_view_sql = f"""
            DROP TABLE IF EXISTS MaterializedVectorViewWorking
            """
            cursor.execute(drop_mat_view_sql)
            
            create_mat_view_sql = f"""
            CREATE TABLE MaterializedVectorViewWorking AS
            SELECT 
                id,
                text_content,
                TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
            FROM {self.table_name}
            """
            
            logger.info("\n--- SQL FOR MATERIALIZED VIEW ---")
            logger.info(f"""
CREATE TABLE MaterializedVectorViewWorking AS
SELECT 
    id,
    text_content,
    TO_VECTOR(embedding, double, {self.embedding_dim}) AS vector_embedding
FROM {self.table_name}
            """)
            
            try:
                cursor.execute(create_mat_view_sql)
                self.conn.commit()
                logger.info("✅ Successfully created materialized view (without quotes)!")
                
                # Try to create HNSW index on the materialized view
                logger.info("Attempting to create HNSW index on the materialized view...")
                create_mat_index_sql = f"""
                CREATE INDEX idx_mat_vector_working ON MaterializedVectorViewWorking (vector_embedding) USING HNSW
                """
                
                logger.info("\n--- SQL FOR HNSW INDEX ON MATERIALIZED VIEW ---")
                logger.info(f"""
CREATE INDEX idx_mat_vector_working ON MaterializedVectorViewWorking (vector_embedding) USING HNSW
                """)
                
                try:
                    cursor.execute(create_mat_index_sql)
                    self.conn.commit()
                    logger.info("✅ Successfully created HNSW index on the materialized view!")
                except Exception as e_mat_index:
                    logger.error(f"❌ Failed to create HNSW index on the materialized view: {e_mat_index}")
                    self.conn.rollback()
            except Exception as e_mat:
                logger.error(f"❌ Failed to create materialized view (without quotes): {e_mat}")
                self.conn.rollback()
            
        except Exception as e:
            logger.error(f"Error in view creation test: {e}")
        finally:
            cursor.close()
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting Working Vector Parameters Tests with IRIS 2025.1")
        
        self.setup_database()
        self.store_document()
        self.test_working_parameter_substitution()
        self.test_view_creation()
        
        logger.info("\nTests completed. Check the logs above for results.")
        logger.info("\nSummary of Key Findings:")
        logger.info("1. TO_VECTOR works with parameter markers when 'double' is used without quotes")
        logger.info("2. Direct cursor execution requires ? parameters with a list")
        logger.info("3. SQLAlchemy execution requires :var parameters with a dict")
        logger.info("4. String interpolation works with text() function from SQLAlchemy")
        logger.info("5. Views, computed columns, and materialized views work with TO_VECTOR when 'double' is used without quotes")

if __name__ == "__main__":
    test = WorkingVectorParamsTest()
    test.run_tests()