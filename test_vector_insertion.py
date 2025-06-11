#!/usr/bin/env python3
"""
Test script to verify vector insertion works with other tables.
This will help us understand what's different about our ColBERT token embedding insertion.
"""

import os
import sys
import logging
from typing import List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_insertion():
    """Test vector insertion into DocumentChunks table to verify it works."""
    
    try:
        # Get database connection using the same approach as populate script
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        iris_connector = connection_manager.get_connection()
        cursor = iris_connector.cursor()
        
        # Test 1: Insert into DocumentChunks table (which has chunk_embedding VECTOR(FLOAT, 384))
        logger.info("=== TEST 1: DocumentChunks table insertion ===")
        
        # Create a test vector (384 dimensions for DocumentChunks)
        test_vector = [0.1] * 384  # Simple test vector
        test_vector_str = str(test_vector)  # Convert to string like working code
        
        logger.info(f"Test vector format: {test_vector_str[:100]}...")
        logger.info(f"Vector length: {len(test_vector)}")
        
        # Try inserting into DocumentChunks
        insert_chunks_sql = """
            INSERT INTO RAG.DocumentChunks (chunk_id, doc_id, chunk_text, chunk_embedding, chunk_index)
            VALUES (?, ?, ?, TO_VECTOR(?), ?)
        """
        
        test_data = (
            'test_chunk_vector_insertion',
            'test_doc_vector_insertion', 
            'This is a test chunk for vector insertion',
            test_vector_str,
            0
        )
        
        cursor.execute(insert_chunks_sql, test_data)
        iris_connector.commit()
        logger.info("✓ SUCCESS: DocumentChunks vector insertion worked!")
        
        # Verify the insertion
        cursor.execute("SELECT chunk_id, chunk_text FROM RAG.DocumentChunks WHERE chunk_id = ?", ('test_chunk_vector_insertion',))
        result = cursor.fetchone()
        if result:
            logger.info(f"✓ VERIFIED: Inserted chunk found: {result}")
        else:
            logger.error("✗ FAILED: Inserted chunk not found")
            
    except Exception as e:
        logger.error(f"✗ FAILED: DocumentChunks insertion failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    try:
        # Test 2: Try the exact same approach with DocumentTokenEmbeddings
        logger.info("\n=== TEST 2: DocumentTokenEmbeddings table insertion ===")
        
        # Create a test vector (128 dimensions for ColBERT)
        test_token_vector = [0.1] * 128  # Simple test vector
        test_token_vector_str = str(test_token_vector)  # Convert to string like working code
        
        logger.info(f"Test token vector format: {test_token_vector_str[:100]}...")
        logger.info(f"Token vector length: {len(test_token_vector)}")
        
        # Try inserting into DocumentTokenEmbeddings using EXACT same pattern as DocumentChunks
        insert_token_sql = """
            INSERT INTO RAG.DocumentTokenEmbeddings (doc_id, token_index, token_text, token_embedding)
            VALUES (?, ?, ?, TO_VECTOR(?))
        """
        
        test_token_data = (
            'test_doc_token_insertion',
            0,
            'test_token',
            test_token_vector_str
        )
        
        cursor.execute(insert_token_sql, test_token_data)
        iris_connector.commit()
        logger.info("✓ SUCCESS: DocumentTokenEmbeddings vector insertion worked!")
        
        # Verify the insertion
        cursor.execute("SELECT doc_id, token_text FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?", ('test_doc_token_insertion',))
        result = cursor.fetchone()
        if result:
            logger.info(f"✓ VERIFIED: Inserted token found: {result}")
        else:
            logger.error("✗ FAILED: Inserted token not found")
            
    except Exception as e:
        logger.error(f"✗ FAILED: DocumentTokenEmbeddings insertion failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    try:
        # Test 3: Check what's different about the table schemas
        logger.info("\n=== TEST 3: Schema comparison ===")
        
        # Check DocumentChunks schema
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks' 
            AND COLUMN_NAME = 'chunk_embedding'
        """)
        chunks_schema = cursor.fetchone()
        logger.info(f"DocumentChunks.chunk_embedding schema: {chunks_schema}")
        
        # Check DocumentTokenEmbeddings schema
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings' 
            AND COLUMN_NAME = 'token_embedding'
        """)
        token_schema = cursor.fetchone()
        logger.info(f"DocumentTokenEmbeddings.token_embedding schema: {token_schema}")
        
    except Exception as e:
        logger.error(f"✗ FAILED: Schema check failed: {e}")
    
    finally:
        # Clean up test data
        try:
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id = ?", ('test_chunk_vector_insertion',))
            cursor.execute("DELETE FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?", ('test_doc_token_insertion',))
            iris_connector.commit()
            logger.info("✓ Cleaned up test data")
        except:
            pass
        
        cursor.close()

if __name__ == "__main__":
    test_vector_insertion()