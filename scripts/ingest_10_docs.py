import os
import sys
import logging
import time
import json # For authors and keywords if stored as JSON strings

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import get_config
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
from data.pmc_processor import process_pmc_files # For extract_pmc_metadata or process_pmc_files

# Configure logging
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
CONFIG = get_config()

def get_config_values():
    """Helper function to expose config to tests if needed, primarily for test setup."""
    if not CONFIG:
        raise RuntimeError("Configuration could not be loaded.")
    
    db_conf = CONFIG.get("database", {})
    model_conf = CONFIG.get("embedding_model", {})
    paths_conf = CONFIG.get("paths", {})
    log_conf = CONFIG.get("logging", {})

    return {
        "database": {
            "host": db_conf.get("db_host", "localhost"),
            "port": db_conf.get("db_port", 1972),
            "namespace": db_conf.get("db_namespace", "USER"), # Default to USER as per config.yaml
            "user": db_conf.get("db_user", "SuperUser"),
            "password": db_conf.get("db_password", "SYS"),
        },
        "embedding_model_name": model_conf.get("name", "all-MiniLM-L6-v2"),
        "sample_docs_path": paths_conf.get("data_dir", "data/") + "sample_10_docs/", # Construct full path
        "log_level": log_conf.get("log_level", "INFO"),
        "log_format": log_conf.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    }

def setup_logging():
    """Sets up logging based on configuration."""
    if not CONFIG:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.warning("Failed to load configuration for logging. Using basic INFO setup.")
        return

    log_config = CONFIG.get("logging", {})
    log_level_str = log_config.get("log_level", "INFO").upper()
    log_format = log_config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    numeric_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=numeric_level, format=log_format)
    logger.info(f"Logging configured to level: {log_level_str}")


def ingest_10_sample_docs():
    """
    Reads 10 sample PMC XML files, generates embeddings, and stores them in IRIS.
    Ensures idempotency by deleting existing records for these 10 docs before insertion.
    """
    setup_logging()
    
    if not CONFIG:
        logger.error("Configuration not loaded. Aborting ingestion.")
        return

    cfg_values = get_config_values()
    db_config = cfg_values["database"]
    embedding_model_name = cfg_values["embedding_model_name"]
    sample_docs_dir = cfg_values["sample_docs_path"]
    
    # Define schema and base table name separately for clarity and correct quoting
    DB_NAMESPACE = db_config["namespace"]
    SCHEMA_NAME = "RAG"
    BASE_TABLE_NAME = "SourceDocuments"
    TABLE_NAME = f'"{DB_NAMESPACE}"."{SCHEMA_NAME}"."{BASE_TABLE_NAME}"'
    EXPECTED_DOC_COUNT = 10

    logger.info(f"Starting ingestion of {EXPECTED_DOC_COUNT} sample documents from: {sample_docs_dir}")
    logger.info(f"Using embedding model: {embedding_model_name}")
    logger.info(f"Target table: {db_config['namespace']}.{TABLE_NAME}")

    iris_conn = None
    try:
        # Get embedding function
        # The get_embedding_func from common.utils uses the model name from config by default if not passed
        # but here we pass it explicitly from our loaded config.
        embed_func = get_embedding_func(model_name=embedding_model_name)
        if embed_func is None:
            logger.error("Failed to initialize embedding function.")
            return

        # Process XML files from the sample directory
        # We use process_pmc_files with a limit of 10.
        # The pmc_processor yields dicts with 'doc_id', 'title', 'abstract', 'authors', 'keywords'
        documents_to_ingest = []
        doc_ids_to_process = []

        logger.info(f"Processing XML files from {sample_docs_dir}...")
        processed_docs_generator = process_pmc_files(directory=sample_docs_dir, limit=EXPECTED_DOC_COUNT)
        
        for doc_data in processed_docs_generator:
            doc_id = doc_data.get("doc_id")
            if not doc_id:
                logger.warning(f"Skipping document due to missing doc_id. Data: {doc_data.get('metadata', {}).get('file_path')}")
                continue
            
            doc_ids_to_process.append(doc_id)
            
            title = doc_data.get("title", "")
            # The 'text_content' for RAG.SourceDocuments should be the main content used for embedding.
            # Typically, this is the abstract.
            abstract = doc_data.get("abstract", "")
            text_for_embedding = abstract if abstract else title # Fallback to title if abstract is empty
            
            if not text_for_embedding:
                logger.warning(f"Document {doc_id} has no abstract or title for embedding. Skipping embedding generation.")
                embedding_value_for_db = None
            else:
                try:
                    # embedding_vector is a list of floats, e.g., [0.1, 0.2, ...]
                    embedding_vector = embed_func([text_for_embedding])[0] # This is a list of floats
                    # For native VECTOR type, IRIS expects a string like "[d1, d2, ...]"
                    embedding_value_for_db = f"[{','.join(map(str, embedding_vector))}]"
                except Exception as e:
                    logger.error(f"Error generating embedding for doc {doc_id}: {e}")
                    embedding_value_for_db = None
            
            documents_to_ingest.append({
                "doc_id": doc_id,
                "title": title,
                "text_content": abstract,
                "authors": json.dumps(doc_data.get("authors", [])),
                "keywords": json.dumps(doc_data.get("keywords", [])),
                "embedding": embedding_value_for_db # This is now a list of floats or None
            })

        if len(documents_to_ingest) != EXPECTED_DOC_COUNT:
            logger.warning(f"Expected to process {EXPECTED_DOC_COUNT} documents, but processed {len(documents_to_ingest)}.")
            # Decide if to proceed or abort. For this script, we'll proceed with what we have.

        if not documents_to_ingest:
            logger.info("No documents processed. Exiting.")
            return

        # Connect to IRIS
        logger.info(f"Connecting to IRIS: {db_config['host']}:{db_config['port']}, Namespace: {db_config['namespace']}")
        # Pass the db_config dictionary directly to the 'config' parameter
        iris_conn = get_iris_connection(config=db_config)
        if iris_conn is None:
            logger.error("Failed to connect to IRIS database.")
            return
        
        cursor = iris_conn.cursor()

        # Idempotency: Delete existing records for these 10 doc_ids
        if doc_ids_to_process:
            logger.info(f"Ensuring idempotency: Deleting existing records for {len(doc_ids_to_process)} doc_ids...")
            placeholders = ', '.join(['?'] * len(doc_ids_to_process))
            # TABLE_NAME already includes schema, properly quoted
            sql_delete = f"DELETE FROM {TABLE_NAME} WHERE doc_id IN ({placeholders})"
            try:
                cursor.execute(sql_delete, doc_ids_to_process)
                logger.info(f"Deleted {cursor.rowcount} existing rows for the sample doc_ids.")
            except Exception as e:
                # This might fail if the table doesn't exist yet, which is fine on a very first run.
                logger.warning(f"Could not execute pre-delete for idempotency (table might not exist or other issue): {e}")
                iris_conn.rollback() # Rollback if delete fails to ensure clean state for inserts
        
        # Insert new records
        # TABLE_NAME already includes schema, properly quoted
        logger.info(f"Inserting {len(documents_to_ingest)} documents into {TABLE_NAME}...")
        sql_insert = f"""
        INSERT INTO {TABLE_NAME}
        (doc_id, title, text_content, authors, keywords, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        insert_params = [
            (
                doc["doc_id"], 
                doc["title"], 
                doc["text_content"],
                doc["authors"],
                doc["keywords"],
                doc["embedding"]
            ) for doc in documents_to_ingest
        ]

        try:
            cursor.executemany(sql_insert, insert_params)
            iris_conn.commit()
            logger.info(f"Successfully inserted/updated {len(documents_to_ingest)} documents.")
        except Exception as e:
            logger.error(f"Error during batch insert: {e}")
            iris_conn.rollback()
            raise # Re-raise to indicate failure to the caller/test

    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)
        if iris_conn:
            iris_conn.rollback()
        raise # Re-raise the exception to signal failure to the caller
    finally:
        if iris_conn:
            iris_conn.close()
            logger.info("IRIS connection closed.")
        logger.info("Ingestion process finished.")

if __name__ == "__main__":
    ingest_10_sample_docs()