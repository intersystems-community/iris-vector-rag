#!/usr/bin/env python3
"""
Debug script to understand IRIS vector format requirements.
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

def test_iris_vector_formats():
    """Test different vector formats to see what IRIS accepts."""
    
    try:
        # Initialize connection
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Test different vector formats
        test_formats = [
            ("Comma-separated", "0.1,0.2,0.3"),
            ("Bracketed", "[0.1,0.2,0.3]"),
            ("JSON array", "[0.1, 0.2, 0.3]"),
            ("Space-separated", "0.1 0.2 0.3"),
            ("Pipe-separated", "0.1|0.2|0.3"),
        ]
        
        test_entity_id = "debug_vector_test_001"
        
        for format_name, vector_str in test_formats:
            try:
                logger.info(f"Testing {format_name}: {vector_str}")
                
                # Try with TO_VECTOR function
                sql = """
                INSERT INTO RAG.DocumentEntities
                (entity_id, document_id, entity_text, entity_type, position, embedding)
                VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
                """
                
                cursor.execute(sql, [
                    f"{test_entity_id}_{format_name.replace(' ', '_').replace('-', '_')}",
                    "debug_doc_001",
                    "Test entity",
                    "TEST",
                    0,
                    vector_str
                ])
                
                logger.info(f"✅ {format_name} format SUCCEEDED with TO_VECTOR()")
                
            except Exception as e:
                logger.error(f"❌ {format_name} format FAILED with TO_VECTOR(): {e}")
                
                # Try without TO_VECTOR function (direct insertion)
                try:
                    sql_direct = """
                    INSERT INTO RAG.DocumentEntities
                    (entity_id, document_id, entity_text, entity_type, position, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(sql_direct, [
                        f"{test_entity_id}_{format_name.replace(' ', '_').replace('-', '_')}_direct",
                        "debug_doc_001",
                        "Test entity direct",
                        "TEST",
                        0,
                        vector_str
                    ])
                    
                    logger.info(f"✅ {format_name} format SUCCEEDED with direct insertion")
                    
                except Exception as e2:
                    logger.error(f"❌ {format_name} format FAILED with direct insertion: {e2}")
        
        # Test what's actually in the database
        cursor.execute("SELECT entity_id, embedding FROM RAG.DocumentEntities WHERE document_id = 'debug_doc_001'")
        results = cursor.fetchall()
        
        logger.info(f"Found {len(results)} test records in database:")
        for entity_id, embedding in results:
            logger.info(f"  - {entity_id}: {embedding}")
        
        # Clean up
        cursor.execute("DELETE FROM RAG.DocumentEntities WHERE document_id = 'debug_doc_001'")
        connection.commit()
        cursor.close()
        
    except Exception as e:
        logger.error(f"Debug test failed: {e}")

if __name__ == "__main__":
    test_iris_vector_formats()