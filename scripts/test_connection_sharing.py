#!/usr/bin/env python3
"""
Test script to verify connection sharing is working properly.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from common.connection_singleton import (
    get_shared_iris_connection,
    reset_shared_connection,
)
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection_sharing():
    """Test that connection sharing works properly."""

    # Reset connection to start fresh
    reset_shared_connection()

    # Get shared connection
    shared_conn = get_shared_iris_connection()
    logger.info(f"Shared connection ID: {id(shared_conn)}")

    # Create config and schema manager
    config_manager = ConfigurationManager()

    # Create connection manager wrapper
    class ConnectionManager:
        def __init__(self, connection):
            self._connection = connection

        def get_connection(self):
            return self._connection

    connection_manager = ConnectionManager(shared_conn)
    schema_manager = SchemaManager(connection_manager, config_manager)

    # Create vector store with shared connection
    vector_store = IRISVectorStore(
        config_manager=config_manager,
        schema_manager=schema_manager,
        connection_manager=connection_manager,
    )

    # Check if vector store is using the same connection
    vs_conn = vector_store._connection
    logger.info(f"Vector store connection ID: {id(vs_conn)}")
    logger.info(f"Connections are same: {id(shared_conn) == id(vs_conn)}")

    # Test inserting and immediately querying
    cursor1 = shared_conn.cursor()
    cursor2 = vs_conn.cursor()

    try:
        # Clear any existing data
        cursor1.execute("DELETE FROM RAG.SourceDocuments")
        logger.info("Cleared existing data")

        # Insert a test document using shared connection
        cursor1.execute(
            """
            INSERT INTO RAG.SourceDocuments (id, title, content, metadata)
            VALUES (?, ?, ?, ?)
        """,
            ["test-1", "Test Document", "This is test content", "{}"],
        )
        logger.info("Inserted test document using shared connection")

        # Query using vector store connection
        cursor2.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count = cursor2.fetchone()[0]
        logger.info(f"Document count using vector store connection: {count}")

        if count == 1:
            logger.info(
                "✅ Connection sharing is working - document visible immediately"
            )
        else:
            logger.error("❌ Connection sharing is NOT working - document not visible")

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        cursor1.close()
        cursor2.close()


if __name__ == "__main__":
    test_connection_sharing()
