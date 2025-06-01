import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TABLES_TO_CLEAR = [
    "RAG.DocumentTokenEmbeddings", # Depends on SourceDocuments (indirectly via how it was populated) or DocumentChunks
    "RAG.DocumentEntities",        # Depends on Entities and SourceDocuments
    "RAG.EntityRelationships",     # Depends on Entities
    "RAG.KnowledgeGraphEdges",     # Depends on KnowledgeGraphNodes
    "RAG.DocumentChunks",          # Depends on SourceDocuments
    "RAG.Entities",                # Depends on SourceDocuments (potentially, for source_doc_id)
    "RAG.KnowledgeGraphNodes",     # Standalone or depends on how it's populated
    "RAG.SourceDocuments"          # Base table
]

# A more robust order considering FKs:
ORDERED_TABLES_TO_CLEAR = [
    "RAG.DocumentTokenEmbeddings", # FK to SourceDocuments (or was intended for chunks)
    "RAG.DocumentEntities",        # FK to Entities, SourceDocuments
    "RAG.EntityRelationships",     # FK to Entities
    "RAG.KnowledgeGraphEdges",     # FK to KnowledgeGraphNodes
    "RAG.DocumentChunks",          # FK to SourceDocuments
    "RAG.Entities",                # No direct FKs from others in this list, but logically dependent
    "RAG.KnowledgeGraphNodes",     # No direct FKs from others in this list
    "RAG.SourceDocuments"          # Cleared last among these
]


def clear_all_rag_tables():
    logger.info("Attempting to clear all RAG data tables...")
    try:
        iris_conn = get_iris_connection()
        cursor = iris_conn.cursor()
        
        for table_name in ORDERED_TABLES_TO_CLEAR:
            try:
                logger.info(f"Attempting to clear table: {table_name}")
                # Check if table is empty first
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count_before = cursor.fetchone()[0]
                logger.info(f"Found {count_before} rows in {table_name} before clearing.")

                if count_before == 0:
                    logger.info(f"{table_name} is already empty. No action taken.")
                else:
                    cursor.execute(f"DELETE FROM {table_name}")
                    logger.info(f"Successfully executed DELETE FROM {table_name}.")
                    
                    # Verify
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count_after = cursor.fetchone()[0]
                    logger.info(f"Found {count_after} rows in {table_name} after clearing.")
                    if count_after == 0:
                        logger.info(f"Table {table_name} successfully cleared.")
                    else:
                        logger.error(f"Table {table_name} clearing FAILED. Expected 0 rows, found {count_after}.")
            except Exception as e_table:
                logger.error(f"Error clearing table {table_name}: {e_table}")
                # Optionally, decide if you want to continue or stop on error
                # For now, let's try to clear other tables

        iris_conn.commit()
        logger.info("All pending DELETE operations committed.")
        
        cursor.close()
        iris_conn.close()
        logger.info("Database connection closed.")
        logger.info("All RAG data tables attempted to be cleared.")

    except Exception as e:
        logger.error(f"Error during RAG data clearing process: {e}", exc_info=True)
        try:
            if 'iris_conn' in locals() and iris_conn:
                iris_conn.close()
        except Exception as ex_close:
            logger.error(f"Failed to close connection during error handling: {ex_close}")

if __name__ == "__main__":
    clear_all_rag_tables()