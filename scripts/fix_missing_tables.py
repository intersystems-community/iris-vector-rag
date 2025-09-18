#!/usr/bin/env python3
"""
Fix missing tables for GraphRAG, ColBERT, and CRAG/NodeRAG pipelines.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from common.iris_connection_manager import get_iris_connection
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_missing_tables():
    """Create missing tables for pipelines."""

    # Initialize managers
    config_manager = ConfigurationManager()
    connection = get_iris_connection()
    schema_manager = SchemaManager(connection, config_manager)

    # Get connection
    connection = get_iris_connection()
    cursor = connection.cursor()

    try:
        # 1. Create DocumentEntities table for GraphRAG
        logger.info("Creating DocumentEntities table for GraphRAG...")

        # Ensure the table is created with correct schema
        if schema_manager.ensure_table_schema("DocumentEntities"):
            logger.info("✅ DocumentEntities table created successfully")
        else:
            logger.error("❌ Failed to create DocumentEntities table")

        # 2. Check/create other required tables
        logger.info("\nChecking other required tables...")

        # Check if we need ChunkedDocuments table for CRAG
        cursor.execute(
            """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'ChunkedDocuments'
        """
        )
        if cursor.fetchone()[0] == 0:
            logger.info("Creating ChunkedDocuments table for CRAG...")
            cursor.execute(
                """
                CREATE TABLE RAG.ChunkedDocuments (
                    chunk_id VARCHAR(255) PRIMARY KEY,
                    document_id VARCHAR(255),
                    chunk_text TEXT,
                    chunk_embedding VECTOR(FLOAT, 384),
                    chunk_index INTEGER,
                    chunk_type VARCHAR(100),
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES RAG.SourceDocuments(doc_id)
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX idx_chunked_docs_doc_id ON RAG.ChunkedDocuments (document_id)"
            )
            cursor.execute(
                "CREATE INDEX idx_chunked_docs_type ON RAG.ChunkedDocuments (chunk_type)"
            )
            logger.info("✅ ChunkedDocuments table created")
        else:
            logger.info("✅ ChunkedDocuments table already exists")

        # 3. Check DocumentTokenEmbeddings for ColBERT
        cursor.execute(
            """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings'
        """
        )
        if cursor.fetchone()[0] == 0:
            logger.info("Creating DocumentTokenEmbeddings table for ColBERT...")
            cursor.execute(
                """
                CREATE TABLE RAG.DocumentTokenEmbeddings (
                    doc_id VARCHAR(255),
                    token_index INTEGER,
                    token_text VARCHAR(500),
                    token_embedding VECTOR(FLOAT, 384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (doc_id, token_index),
                    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings (doc_id)"
            )
            logger.info("✅ DocumentTokenEmbeddings table created")
        else:
            logger.info("✅ DocumentTokenEmbeddings table already exists")

        # 4. Check table structures
        logger.info("\nVerifying table structures...")

        # Check DocumentEntities columns
        cursor.execute(
            """
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentEntities'
            ORDER BY ORDINAL_POSITION
        """
        )
        columns = [row[0] for row in cursor.fetchall()]
        logger.info(f"DocumentEntities columns: {columns}")

        # Check ChunkedDocuments columns
        cursor.execute(
            """
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'ChunkedDocuments'
            ORDER BY ORDINAL_POSITION
        """
        )
        columns = [row[0] for row in cursor.fetchall()]
        logger.info(f"ChunkedDocuments columns: {columns}")

        connection.commit()
        logger.info("\n✅ All table fixes completed successfully!")

    except Exception as e:
        logger.error(f"Error fixing tables: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    create_missing_tables()
