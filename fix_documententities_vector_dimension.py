#!/usr/bin/env python3
"""
Fix DocumentEntities table to use 384-dimensional vectors instead of 1536.
This matches the all-MiniLM-L6-v2 sentence transformer model.
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

def fix_documententities_table():
    """Recreate DocumentEntities table with correct vector dimension."""
    
    try:
        # Initialize connection
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        logger.info("Checking current DocumentEntities table...")
        
        # Check if table exists and has data
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
            row_count = cursor.fetchone()[0]
            logger.info(f"Current table has {row_count} rows")
            
            if row_count > 0:
                logger.warning(f"Table contains {row_count} rows. These will be lost when recreating the table.")
                response = input("Continue? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Operation cancelled.")
                    return False
        except Exception as e:
            logger.info(f"Table doesn't exist or is inaccessible: {e}")
        
        # Drop the existing table
        logger.info("Dropping existing DocumentEntities table...")
        try:
            cursor.execute("DROP TABLE RAG.DocumentEntities")
            logger.info("✅ Dropped existing table")
        except Exception as e:
            logger.warning(f"Could not drop table (may not exist): {e}")
        
        # Create new table with 384-dimensional vectors
        logger.info("Creating new DocumentEntities table with 384-dimensional vectors...")
        
        create_sql = """
        CREATE TABLE RAG.DocumentEntities (
            entity_id VARCHAR(255) NOT NULL,
            document_id VARCHAR(255) NOT NULL,
            entity_text VARCHAR(1000) NOT NULL,
            entity_type VARCHAR(100),
            position INTEGER,
            embedding VECTOR(DOUBLE, 384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_id)
        )
        """
        
        cursor.execute(create_sql)
        logger.info("✅ Created new table with 384-dimensional vectors")
        
        # Create indexes for better performance
        logger.info("Creating indexes...")
        
        indexes = [
            "CREATE INDEX idx_documententities_document_id ON RAG.DocumentEntities (document_id)",
            "CREATE INDEX idx_documententities_entity_type ON RAG.DocumentEntities (entity_type)",
            "CREATE INDEX idx_documententities_created_at ON RAG.DocumentEntities (created_at)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"✅ Created index: {index_sql.split('CREATE INDEX ')[1].split(' ON')[0]}")
            except Exception as e:
                logger.warning(f"Could not create index: {e}")
        
        # Test the new table with a 384-dimensional vector
        logger.info("Testing new table with 384-dimensional vector...")
        
        test_vector = [0.1] * 384
        vector_str = ','.join(f'{x:.8f}' for x in test_vector)
        
        test_sql = """
        INSERT INTO RAG.DocumentEntities
        (entity_id, document_id, entity_text, entity_type, position, embedding)
        VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
        """
        
        cursor.execute(test_sql, [
            "test_384_vector",
            "test_doc",
            "Test entity with 384-dimensional vector",
            "TEST",
            0,
            vector_str
        ])
        
        # Verify it was stored
        cursor.execute("SELECT entity_id, embedding FROM RAG.DocumentEntities WHERE entity_id = 'test_384_vector'")
        result = cursor.fetchone()
        
        if result:
            entity_id, stored_embedding = result
            logger.info(f"✅ Test vector stored successfully: {entity_id}")
            logger.info(f"   Stored embedding preview: {str(stored_embedding)[:100]}...")
        else:
            logger.error("❌ Test vector was not stored correctly")
            return False
        
        # Clean up test data
        cursor.execute("DELETE FROM RAG.DocumentEntities WHERE entity_id = 'test_384_vector'")
        
        connection.commit()
        cursor.close()
        
        logger.info("🎉 DocumentEntities table successfully recreated with 384-dimensional vectors!")
        logger.info("GraphRAG entity embeddings should now work correctly.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fix DocumentEntities table: {e}")
        return False

if __name__ == "__main__":
    success = fix_documententities_table()
    sys.exit(0 if success else 1)