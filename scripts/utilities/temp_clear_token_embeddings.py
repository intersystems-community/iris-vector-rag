import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_token_embeddings_table():
    logger.info("Attempting to clear RAG.DocumentTokenEmbeddings table...")
    try:
        iris_conn = get_iris_connection()
        cursor = iris_conn.cursor()
        
        # Check if table is empty first
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        count_before = cursor.fetchone()[0]
        logger.info(f"Found {count_before} rows in RAG.DocumentTokenEmbeddings before clearing.")

        if count_before == 0:
            logger.info("RAG.DocumentTokenEmbeddings is already empty. No action taken.")
        else:
            cursor.execute("DELETE FROM RAG.DocumentTokenEmbeddings")
            iris_conn.commit()
            logger.info("Successfully executed DELETE FROM RAG.DocumentTokenEmbeddings.")

            # Verify
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            count_after = cursor.fetchone()[0]
            logger.info(f"Found {count_after} rows in RAG.DocumentTokenEmbeddings after clearing.")
            if count_after == 0:
                logger.info("Table successfully cleared.")
            else:
                logger.error(f"Table clearing FAILED. Expected 0 rows, found {count_after}.")

        cursor.close()
        iris_conn.close()
    except Exception as e:
        logger.error(f"Error clearing RAG.DocumentTokenEmbeddings table: {e}", exc_info=True)

if __name__ == "__main__":
    clear_token_embeddings_table()