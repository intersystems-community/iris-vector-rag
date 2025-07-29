#!/usr/bin/env python3
"""
Apply ColBERT dimension fix to database schema.

This script fixes the dimension mismatch between the database schema (128 dimensions)
and the actual ColBERT model output (384 dimensions).
"""

import sys
import os
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from common.iris_connection_manager import IRISConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_dimension_fix():
    """Apply the ColBERT dimension fix to the database."""
    
    # SQL statements to execute
    sql_statements = [
        """
        CREATE TABLE RAG.DocumentTokenEmbeddings_New (
            doc_id VARCHAR(255),
            token_index INTEGER,
            token_text VARCHAR(500),
            token_embedding VECTOR(FLOAT, 384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (doc_id, token_index),
            FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
        )
        """,
        "DROP TABLE RAG.DocumentTokenEmbeddings",
        "RENAME TABLE RAG.DocumentTokenEmbeddings_New TO RAG.DocumentTokenEmbeddings",
        "CREATE INDEX idx_doc_token_embeddings_doc_id ON RAG.DocumentTokenEmbeddings(doc_id)",
        "CREATE INDEX idx_doc_token_embeddings_token_index ON RAG.DocumentTokenEmbeddings(token_index)",
        "CREATE INDEX idx_doc_token_embeddings_vector ON RAG.DocumentTokenEmbeddings(token_embedding)"
    ]
    
    try:
        # Connect to IRIS
        logger.info("Connecting to IRIS database...")
        iris_connector = IRISConnectionManager()
        connection = iris_connector.get_connection()
        cursor = connection.cursor()
        
        # Execute each SQL statement
        for i, statement in enumerate(sql_statements):
            logger.info(f"Executing statement {i+1}/{len(sql_statements)}...")
            logger.debug(f"Statement: {statement.strip()[:100]}...")
            
            try:
                cursor.execute(statement.strip())
                logger.info("✅ Success")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                if 'already exists' not in str(e).lower() and 'does not exist' not in str(e).lower():
                    raise
                else:
                    logger.info("Continuing (expected error)...")
        
        # Commit changes
        connection.commit()
        logger.info("✅ Database migration completed successfully")
        
        # Verify the new schema
        logger.info("Verifying new schema...")
        cursor.execute("DESCRIBE RAG.DocumentTokenEmbeddings")
        for row in cursor.fetchall():
            if 'token_embedding' in str(row).lower():
                logger.info(f"New schema: {row}")
        
        cursor.close()
        connection.close()
        logger.info("✅ Database connection closed")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting ColBERT dimension fix migration...")
    success = apply_dimension_fix()
    
    if success:
        logger.info("✅ Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Migration failed")
        sys.exit(1)