#!/usr/bin/env python3
"""
Debug script to check IRIS database schema for vector columns.
"""

import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_iris_schema():
    """Check the IRIS database schema for DocumentEntities table."""
    
    try:
        # Initialize connection
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Check if table exists and get its structure
        try:
            cursor.execute("DESCRIBE RAG.DocumentEntities")
            columns = cursor.fetchall()
            
            logger.info("RAG.DocumentEntities table structure:")
            for column in columns:
                logger.info(f"  {column}")
                
        except Exception as e:
            logger.error(f"Failed to describe table: {e}")
            
        # Check for vector-specific information
        try:
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentEntities'
                ORDER BY ORDINAL_POSITION
            """)
            
            schema_info = cursor.fetchall()
            logger.info("\nDetailed column information:")
            for col_name, data_type, max_length, nullable in schema_info:
                logger.info(f"  {col_name}: {data_type} (max_length: {max_length}, nullable: {nullable})")
                
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            
        # Try to check if there are any existing embeddings
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities WHERE embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
            logger.info(f"\nExisting records with embeddings: {embedding_count}")
            
            if embedding_count > 0:
                cursor.execute("SELECT TOP 1 entity_id, embedding FROM RAG.DocumentEntities WHERE embedding IS NOT NULL")
                result = cursor.fetchone()
                if result:
                    entity_id, embedding = result
                    logger.info(f"Sample embedding: {entity_id} -> {str(embedding)[:100]}...")
                    
        except Exception as e:
            logger.error(f"Failed to check existing embeddings: {e}")
            
        # Check if we can create a simple test table with vector column
        try:
            cursor.execute("DROP TABLE IF EXISTS RAG.VectorTest")
            cursor.execute("""
                CREATE TABLE RAG.VectorTest (
                    id VARCHAR(50),
                    test_vector VECTOR(DOUBLE, 3)
                )
            """)
            
            # Try to insert a simple vector
            cursor.execute("INSERT INTO RAG.VectorTest (id, test_vector) VALUES (?, TO_VECTOR(?))", 
                          ["test1", "0.1,0.2,0.3"])
            
            # Check if it worked
            cursor.execute("SELECT id, test_vector FROM RAG.VectorTest")
            result = cursor.fetchone()
            if result:
                logger.info(f"✅ Vector test table works: {result}")
            
            cursor.execute("DROP TABLE RAG.VectorTest")
            connection.commit()
            
        except Exception as e:
            logger.error(f"❌ Vector test table failed: {e}")
            
        cursor.close()
        
    except Exception as e:
        logger.error(f"Schema check failed: {e}")

if __name__ == "__main__":
    check_iris_schema()