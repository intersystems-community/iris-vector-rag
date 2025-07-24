import os
import yaml
import logging
import glob
import argparse
from xml.etree import ElementTree as ET
import time
from sentence_transformers import SentenceTransformer
import sys
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from common.iris_connector import get_iris_connection
except ImportError:
    # This might happen if common.iris_connector is not found during initial generation
    # The __main__ block below tries to create __init__.py files to help with this.
    print("Error: common.iris_connector not found. Ensure it's in the PYTHONPATH.")
    # Define a dummy get_iris_connection if not available, so script can be written
    def get_iris_connection(config_file=None, use_mock=False):
        print(f"Dummy get_iris_connection called with config_file={config_file}, use_mock={use_mock}")
        return None

# Global logger instance
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

def setup_logging(log_level_str: str, log_format_str: Optional[str] = None):
    """Configures basic logging."""
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    if log_format_str is None:
        log_format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format_str, stream=sys.stdout)
    logger.info(f"Logging configured at level: {log_level_str}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Basic validation
        if not all(k in config for k in ['database', 'embedding_model', 'paths']):
            raise ValueError("Config missing required sections: database, embedding_model, paths")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration at {config_path}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Configuration error in {config_path}: {e}")
        raise

def get_document_filepaths(docs_dir: str, specific_doc_ids: Optional[List[str]] = None, limit: Optional[int] = None) -> List[str]:
    """
    Retrieves a list of XML file paths to process from the given directory.
    Filters by specific_doc_ids if provided, or applies a limit.
    """
    if not os.path.isdir(docs_dir):
        logger.error(f"Documents directory not found: {docs_dir}")
        return []

    all_xml_files = sorted(glob.glob(os.path.join(docs_dir, "*.xml")))

    if specific_doc_ids:
        selected_files = []
        for doc_id in specific_doc_ids:
            found = False
            for f_path in all_xml_files:
                if os.path.splitext(os.path.basename(f_path))[0] == doc_id:
                    selected_files.append(f_path)
                    found = True
                    break
            if not found:
                logger.warning(f"Specified doc_id '{doc_id}' not found in {docs_dir}")
        filepaths = selected_files
    else:
        filepaths = all_xml_files

    if limit is not None and len(filepaths) > limit:
        filepaths = filepaths[:limit]
    
    logger.info(f"Found {len(filepaths)} XML files to process in {docs_dir} (filters: ids={specific_doc_ids}, limit={limit}).")
    return filepaths

def parse_pmc_xml(file_path: str) -> Optional[Dict[str, str]]:
    """
    Parses a PMC XML file to extract doc_id, textual content, and source_filename.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML file {file_path}: {e}")
        return None

    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    source_filename = os.path.basename(file_path)

    title_elements = root.findall(".//article-title")
    title = " ".join([elem.text.strip() for elem in title_elements if elem.text]) if title_elements else ""
    
    abstract_elements = root.findall(".//abstract//p") or root.findall(".//abstract")
    abstract = " ".join([ET.tostring(elem, method='text', encoding='unicode').strip() for elem in abstract_elements])
    abstract = ' '.join(abstract.replace("\n", " ").strip().split())

    body_elements = root.findall(".//body//p")
    body = " ".join([ET.tostring(elem, method='text', encoding='unicode').strip() for elem in body_elements])
    body = ' '.join(body.replace("\n", " ").strip().split())
    
    full_content = f"{title} {abstract} {body}".strip()
    full_content = ' '.join(full_content.split())

    if not full_content:
        all_text_content = "".join(root.itertext())
        full_content = ' '.join(all_text_content.split())
        if not full_content:
            logger.warning(f"Doc ID {doc_id}: Extracted content is empty from {file_path}.")
            # Return with empty content to allow tracking, but it won't be embedded meaningfully
            return {"doc_id": doc_id, "content": "", "source_filename": source_filename}

    return {"doc_id": doc_id, "content": full_content, "source_filename": source_filename}

def generate_embeddings_for_docs(documents_data: List[Dict[str, str]], embedding_model_instance) -> List[Dict[str, Any]]:
    """
    Generates embeddings for a list of document data.
    Modifies documents_data in-place by adding 'embedding_str'.
    """
    docs_with_embeddings = []
    for doc_data in documents_data:
        content = doc_data.get("content", "")
        doc_id = doc_data.get("doc_id", "N/A")
        embedding_str = None
        if content and content.strip():
            try:
                embedding_vector = embedding_model_instance.encode(content)
                embedding_str = ",".join(map(str, embedding_vector))
            except Exception as e:
                logger.error(f"Error generating embedding for doc_id {doc_id}: {e}")
        else:
            logger.warning(f"No content to embed for doc_id {doc_id}. Embedding will be NULL.")
        
        # Create a new dict to avoid modifying the input list's dicts directly if they are reused
        processed_doc = doc_data.copy()
        processed_doc["embedding_str"] = embedding_str
        # processed_doc["text_length"] = len(content) # Removed
        # processed_doc["last_updated"] = datetime.datetime.now() # Removed
        docs_with_embeddings.append(processed_doc)
    return docs_with_embeddings

def ingest_data_to_iris(db_conn, documents_to_ingest: List[Dict[str, Any]], clear_doc_ids_first: List[str]) -> int:
    """
    Ingests a batch of processed documents into IRIS.
    Clears existing data for doc_ids in clear_doc_ids_first before inserting.
    Returns the number of successfully inserted documents.
    """
    inserted_count = 0
    if not documents_to_ingest:
        return 0

    cursor = db_conn.cursor()

    # 1. Clear existing data for the doc_ids in this specific batch
    if clear_doc_ids_first:
        placeholders = ','.join(['?'] * len(clear_doc_ids_first))
        delete_sql = f"DELETE FROM RAG.SourceDocuments WHERE doc_id IN ({placeholders})"
        try:
            logger.debug(f"Clearing {len(clear_doc_ids_first)} doc_ids before ingest: {clear_doc_ids_first[:5]}...")
            cursor.execute(delete_sql, tuple(clear_doc_ids_first))
            # db_conn.commit() # Commit delete separately or as part of the main transaction
            logger.info(f"Cleared {cursor.rowcount} existing records for {len(clear_doc_ids_first)} doc_ids.")
        except Exception as e:
            logger.error(f"Error deleting existing records for doc_ids {clear_doc_ids_first[:5]}...: {e}")
            # db_conn.rollback() # Rollback if delete fails
            raise # Re-raise to stop this batch if cleanup fails

    # 2. Prepare parameters for insertion
    insert_params = []
    for doc_data in documents_to_ingest:
        params = (
            doc_data["doc_id"],
            doc_data["content"], # This is the actual text content from parsing
            doc_data["embedding_str"]
            # doc_data["source_filename"], # Removed source_filename
            # doc_data["last_updated"], # Removed
            # doc_data["text_length"] # Removed
        )
        insert_params.append(params)

    # 3. Insert new data
    insert_sql = """
    INSERT INTO RAG.SourceDocuments (doc_id, text_content, embedding)
    VALUES (?, ?, ?)
    """
    try:
        cursor.executemany(insert_sql, insert_params)
        inserted_count = cursor.rowcount # executemany might not return reliable rowcount on all drivers for inserts
        if inserted_count is None or inserted_count == -1 : # some drivers return -1 or None
             inserted_count = len(insert_params) # Assume all were attempted if no specific error
        logger.debug(f"Attempted to insert {len(insert_params)} documents. Driver reported rowcount: {cursor.rowcount}")
    except Exception as e:
        logger.error(f"Error during batch insert: {e}")
        # db_conn.rollback() # Rollback handled by the caller (main function) for the whole transaction
        raise # Re-raise to be caught by main
    
    # db_conn.commit() # Commit handled by the caller (main function)
    return inserted_count


def main():
    parser = argparse.ArgumentParser(description="Ingest PMC XML documents into IRIS RAG.SourceDocuments table.")
    parser.add_argument("--docs_path", required=True, help="Directory containing PMC XML files.")
    parser.add_argument("--doc_ids", type=str, help="Comma-separated list of specific doc_ids (filenames without extension) to process.")
    parser.add_argument("--limit", type=int, help="Maximum number of documents to process from docs_path if doc_ids not given.")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of documents per database transaction batch.")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH, help=f"Path to config.yaml (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Logging level.")
    parser.add_argument("--clear_before_ingest", action='store_true', help="Clear existing entries for the processed documents before ingestion. This is the default behavior for selected docs.")

    args = parser.parse_args()

    # Setup logging based on config or command-line arg
    # Config logging settings can be loaded after initial config load if preferred
    # For now, command-line arg for log_level takes precedence for initial setup.
    config_log_format = None
    try:
        temp_config_for_log = load_config(args.config_path)
        config_log_format = temp_config_for_log.get('logging', {}).get('log_format')
    except Exception: # If config load fails, use basic format
        pass
    setup_logging(args.log_level, config_log_format)
    
    start_time = time.time()
    total_files_processed = 0
    total_docs_successfully_ingested = 0
    total_errors = 0

    try:
        config = load_config(args.config_path)
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}. Exiting.")
        sys.exit(1)

    specific_doc_ids_list = [doc_id.strip() for doc_id in args.doc_ids.split(',')] if args.doc_ids else None
    
    filepaths_to_process = get_document_filepaths(args.docs_path, specific_doc_ids_list, args.limit)
    total_files_to_process = len(filepaths_to_process)
    if not filepaths_to_process:
        logger.info("No files selected for processing. Exiting.")
        sys.exit(0)

    # Load embedding model
    model_name = config['embedding_model']['name']
    try:
        logger.info(f"Loading embedding model: {model_name}...")
        embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load SentenceTransformer model '{model_name}': {e}. Exiting.")
        sys.exit(1)

    db_conn = None
    try:
        # Corrected: Pass the loaded config dictionary to the 'config' parameter
        # The get_iris_connection function expects the actual config dict for its 'config' param,
        # not the path to the config file for 'config_file'.
        # The config object is already loaded earlier in main().
        db_config_params = config.get('database') if config else None
        db_conn = get_iris_connection(config=db_config_params) # Use the 'config' parameter
        if db_conn is None:
            logger.critical("Failed to establish database connection. Exiting.")
            sys.exit(1)
        
        logger.info(f"Processing {total_files_to_process} documents in batches of {args.batch_size}...")

        for i in range(0, total_files_to_process, args.batch_size):
            batch_filepaths = filepaths_to_process[i:i + args.batch_size]
            logger.info(f"Processing batch {i // args.batch_size + 1}/{(total_files_to_process + args.batch_size -1) // args.batch_size} ({len(batch_filepaths)} files)")
            
            parsed_docs_data = []
            current_batch_doc_ids_to_clear = []

            for fp in batch_filepaths:
                total_files_processed += 1
                parsed_data = parse_pmc_xml(fp)
                if parsed_data:
                    parsed_docs_data.append(parsed_data)
                    current_batch_doc_ids_to_clear.append(parsed_data["doc_id"])
                else:
                    total_errors += 1
                    logger.warning(f"Skipping file {fp} due to parsing error.")
            
            if not parsed_docs_data:
                logger.info("No documents successfully parsed in this batch. Moving to next batch.")
                continue

            # Generate embeddings for the successfully parsed documents in the batch
            docs_for_ingestion_with_embeddings = generate_embeddings_for_docs(parsed_docs_data, embedding_model)

            try:
                # The clear_before_ingest flag from args determines if we clear.
                # The list of doc_ids to clear is specific to this batch.
                # Default behavior is to clear if --clear_before_ingest is set or implied.
                # For this script, the instruction is to clear for the processed docs.
                # So, current_batch_doc_ids_to_clear will always be passed if non-empty.
                
                ingested_in_batch = ingest_data_to_iris(db_conn, docs_for_ingestion_with_embeddings, current_batch_doc_ids_to_clear)
                db_conn.commit() # Commit after each successful batch transaction
                total_docs_successfully_ingested += ingested_in_batch
                logger.info(f"Batch committed. Successfully ingested {ingested_in_batch} documents in this batch.")
            except Exception as batch_db_error:
                total_errors += len(docs_for_ingestion_with_embeddings) # Assume all in batch failed if DB error
                logger.error(f"Error ingesting batch to IRIS: {batch_db_error}. Rolling back this batch.")
                if db_conn:
                    try:
                        db_conn.rollback()
                    except Exception as rb_err:
                        logger.error(f"Error during rollback: {rb_err}")
        
    except Exception as e:
        logger.critical(f"An critical error occurred during the ingestion process: {e}", exc_info=True)
        total_errors += (total_files_to_process - total_files_processed) # Count unprocessed files as errors
        if db_conn:
            try:
                db_conn.rollback()
            except Exception as rb_err:
                logger.error(f"Error during final rollback: {rb_err}")
    finally:
        if db_conn:
            db_conn.close()
            logger.info("Database connection closed.")

    duration = time.time() - start_time
    logger.info("--- Ingestion Summary ---")
    logger.info(f"Total files found to process: {total_files_to_process}")
    logger.info(f"Total files actually processed (parsed or attempted): {total_files_processed}")
    logger.info(f"Total documents successfully ingested: {total_docs_successfully_ingested}")
    logger.info(f"Total errors (parsing/embedding/DB): {total_errors}")
    logger.info(f"Total duration: {duration:.2f} seconds")
    if duration > 0 and total_docs_successfully_ingested > 0 :
        logger.info(f"Ingestion rate: {total_docs_successfully_ingested / duration:.2f} docs/sec")
    logger.info("Ingestion script finished.")

if __name__ == "__main__":
    # Ensure __init__.py exists for sibling imports if run directly
    # This helps if common/ or scripts/ are not in PYTHONPATH during dev
    for p_part in ['common', 'scripts']:
        dir_path = os.path.join(os.path.dirname(__file__), '..', p_part)
        os.makedirs(dir_path, exist_ok=True)
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {p_part} module\n")
    
    # For standalone execution, ensure common.iris_connector is truly available
    # The dummy definition at the top is a fallback for generation time.
    try:
        from common.iris_connector import get_iris_connection
    except ImportError:
        print("FATAL: common.iris_connector could not be imported. Ensure PYTHONPATH is set correctly or common/ is a package.")
        print("       The script might have created a dummy __init__.py. You might need to restart or fix paths.")
        sys.exit(1)
        
    main()