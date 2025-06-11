#!/usr/bin/env python3
"""
Quick test to determine the exact vector dimension expected by DocumentEntities.
"""

import logging
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from common.vector_format_fix import format_vector_for_iris, create_iris_vector_string

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_actual_embedding_dimensions():
    """Test with actual sentence transformer embeddings to find the right dimension."""
    
    try:
        # Initialize connection
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Test common embedding dimensions
        dimensions_to_test = [384, 768, 1536]  # Common sentence transformer dimensions
        
        for dim in dimensions_to_test:
            try:
                logger.info(f"Testing dimension {dim}...")
                
                # Create a simple test vector
                test_vector = [0.1] * dim  # Simple repeated values
                
                # Format using our utility
                formatted_vector = format_vector_for_iris(test_vector)
                vector_str = create_iris_vector_string(formatted_vector)
                
                logger.info(f"Vector string length: {len(vector_str)} chars")
                
                # Try to insert
                sql = """
                INSERT INTO RAG.DocumentEntities
                (entity_id, document_id, entity_text, entity_type, position, embedding)
                VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
                """
                
                test_id = f"quick_test_{dim}"
                cursor.execute(sql, [
                    test_id,
                    "quick_test_doc",
                    f"Test entity for dimension {dim}",
                    "TEST",
                    0,
                    vector_str
                ])
                
                logger.info(f"✅ Dimension {dim} SUCCEEDED!")
                
            except Exception as e:
                logger.error(f"❌ Dimension {dim} FAILED: {e}")
        
        # Clean up
        cursor.execute("DELETE FROM RAG.DocumentEntities WHERE document_id = 'quick_test_doc'")
        connection.commit()
        cursor.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_actual_embedding_dimensions()