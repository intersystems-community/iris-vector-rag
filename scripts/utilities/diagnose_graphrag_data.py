import logging
import os
import sys

# Add project root to sys.path to allow imports from common, etc.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
except ImportError as e:
    print(f"Error importing common.iris_connector: {e}. Ensure common.iris_connector.py exists and project root is in PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_diagnostics(conn):
    """
    Runs diagnostic queries against the IRIS database.
    """
    cursor = None
    try:
        cursor = conn.cursor()

        # Query 1: Count total entities
        logger.info("Query 1: Counting total entities in RAG.Entities...")
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        total_entities_result = cursor.fetchone()
        total_entities = total_entities_result[0] if total_entities_result else 0
        logger.info(f"Total entities: {total_entities}")

        # Query 2: Count entities with non-NULL source_doc_id
        logger.info("Query 2: Counting entities with non-NULL and non-empty source_doc_id in RAG.Entities...")
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE source_doc_id IS NOT NULL AND source_doc_id <> ''")
        entities_with_source_doc_id_result = cursor.fetchone()
        entities_with_source_doc_id = entities_with_source_doc_id_result[0] if entities_with_source_doc_id_result else 0
        logger.info(f"Entities with source_doc_id: {entities_with_source_doc_id}")

        if total_entities > 0:
            percentage_linked = (entities_with_source_doc_id / total_entities) * 100
            logger.info(f"Percentage of entities linked to a source_doc_id: {percentage_linked:.2f}%")
        else:
            logger.info("Percentage of entities linked: N/A (no entities found)")

        # Query 3: Check if source_doc_id values match actual doc_id values in RAG.SourceDocuments
        logger.info("Query 3: Checking for orphaned source_doc_id references (entities with source_doc_id not in RAG.SourceDocuments)...")
        query_orphaned = """
        SELECT COUNT(e.id)
        FROM RAG.Entities e
        LEFT JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
        WHERE e.source_doc_id IS NOT NULL AND e.source_doc_id <> '' AND sd.doc_id IS NULL
        """
        cursor.execute(query_orphaned)
        orphaned_references_count_result = cursor.fetchone()
        orphaned_references_count = orphaned_references_count_result[0] if orphaned_references_count_result else 0
        logger.info(f"Number of entities with source_doc_id not found in RAG.SourceDocuments: {orphaned_references_count}")

        # Query 4: Sample entities related to "BRCA1" or "protein"
        logger.info("Query 4: Sampling entities related to 'BRCA1' or 'protein' (case-insensitive)...")
        # Assuming entity names are in a column like 'name'. Adjust if schema is different.
        # Using TOP 10 for IRIS SQL
        query_sample_entities = """
        SELECT TOP 10 entity_id, entity_name, source_doc_id
        FROM RAG.Entities
        WHERE (LOWER(entity_name) LIKE '%brca1%' OR LOWER(entity_name) LIKE '%protein%')
        ORDER BY entity_id
        """
        try:
            cursor.execute(query_sample_entities)
            sample_entities = cursor.fetchall()
            if sample_entities:
                logger.info("Sample entities (entity_id, entity_name, source_doc_id):")
                for entity in sample_entities:
                    logger.info(f"  - Entity ID: {entity[0]}, Name: {entity[1]}, Source Doc ID: {entity[2]}")
            else:
                logger.info("No entities found matching 'BRCA1' or 'protein' in RAG.Entities.entity_name.")
        except Exception as e_sample:
            logger.warning(f"Could not sample entities (Error: {e_sample}). This might be due to RAG.Entities not having an 'entity_name' column or other schema mismatch. Please check your RAG.Entities schema and adjust the query if needed.")

        # Query 5: Check if there are any documents in RAG.SourceDocuments that contain "BRCA1" or related terms
        logger.info("Query 5: Checking for documents in RAG.SourceDocuments containing 'BRCA1' or 'protein'...")
        logger.info("         (Note: Searching stream field 'text_content' without LOWER() due to SQL limitations. Query uses multiple LIKEs for common cases.)")
        # Assuming document content is in 'text_content' and title in 'title'. Adjust if schema is different.
        # Using TOP 5 for IRIS SQL
        query_sample_docs = """
        SELECT TOP 5 doc_id, title
        FROM RAG.SourceDocuments
        WHERE (text_content LIKE '%brca1%' OR text_content LIKE '%BRCA1%' OR text_content LIKE '%Protein%' OR text_content LIKE '%protein%')
        ORDER BY doc_id
        """
        try:
            cursor.execute(query_sample_docs)
            sample_docs = cursor.fetchall()
            if sample_docs:
                logger.info("Sample documents containing 'BRCA1' or 'protein' (doc_id, title):")
                for doc in sample_docs:
                    logger.info(f"  - Doc ID: {doc[0]}, Title: {doc[1]}")
                
                # Count total matching documents
                count_query_docs = """
                    SELECT COUNT(*)
                    FROM RAG.SourceDocuments
                    WHERE (text_content LIKE '%brca1%' OR text_content LIKE '%BRCA1%' OR text_content LIKE '%Protein%' OR text_content LIKE '%protein%')
                """
                cursor.execute(count_query_docs)
                count_matching_docs_result = cursor.fetchone()
                count_matching_docs = count_matching_docs_result[0] if count_matching_docs_result else 0
                logger.info(f"Total documents found in RAG.SourceDocuments containing these terms in 'text_content': {count_matching_docs}")
            else:
                logger.info("No documents found in RAG.SourceDocuments containing 'BRCA1' or 'protein' in 'text_content' with the specified LIKE patterns.")
        except Exception as e_docs:
            logger.warning(f"Could not sample documents (Error: {e_docs}). This might be due to RAG.SourceDocuments not having 'text_content' or 'title' columns, issues with LIKE on stream fields, or other schema mismatch. Please check your RAG.SourceDocuments schema and adjust the query if needed.")

    except Exception as e:
        logger.error(f"An error occurred during diagnostics: {e}", exc_info=True)
    finally:
        if cursor:
            cursor.close()

def main():
    logger.info("Starting GraphRAG data diagnostics script...")
    conn = None
    try:
        logger.info("Attempting to connect to IRIS database using common.iris_connector...")
        # Ensure environment variables for JDBC connection are set:
        # IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME, IRIS_PASSWORD
        # And intersystems-jdbc-3.8.4.jar is in the project root.
        conn = get_iris_connection() 
        logger.info("Successfully connected to IRIS database.")

        run_diagnostics(conn)

    except IRISConnectionError as e:
        logger.error(f"Failed to connect to IRIS: {e}")
        logger.error("Please ensure your IRIS instance is running and connection parameters (env vars) are correctly set.")
        logger.error(f"JDBC JAR expected at: {os.path.abspath(os.path.join(project_root, 'intersystems-jdbc-3.8.4.jar'))}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        if conn:
            try:
                conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing database connection: {e_close}")
        logger.info("Diagnostics script finished.")

if __name__ == "__main__":
    main()