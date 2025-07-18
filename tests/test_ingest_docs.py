import pytest
import os
import yaml
import logging
import re
import subprocess
import sys
from typing import List, Optional # Added List, Optional
from xml.etree import ElementTree as ET

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
# from scripts.ingest_docs import main as ingest_main # For direct call if preferred

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, '..', 'config', 'config.yaml')
# Use a dedicated test data path if different from main sample_10_docs, or ensure cleanup
SAMPLE_DOCS_PATH_NAME = 'sample_10_docs' # This is the sub-directory name in 'data/'
DATA_DIR_FROM_CONFIG = 'data/' # As per default config.yaml paths.data_dir
ABSOLUTE_SAMPLE_DOCS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', DATA_DIR_FROM_CONFIG, SAMPLE_DOCS_PATH_NAME))

INGEST_SCRIPT_PATH = os.path.join(BASE_DIR, '..', 'scripts', 'ingest_docs.py')
EXPECTED_DOC_COUNT_FULL_SAMPLE = 10 # Total docs in sample_10_docs

# --- Helper Functions ---

def run_ingestion_script(script_args: List[str]) -> subprocess.CompletedProcess:
    """Runs the ingestion script as a subprocess."""
    base_command = [sys.executable, INGEST_SCRIPT_PATH]
    command = base_command + script_args
    logger.info(f"Running ingestion script with command: {' '.join(command)}")
    # Capture output for debugging
    result = subprocess.run(command, capture_output=True, text=True, check=False) # check=False to inspect output on failure
    if result.returncode != 0:
        logger.error(f"Ingestion script failed with exit code {result.returncode}")
        logger.error(f"Stdout:\n{result.stdout}")
        logger.error(f"Stderr:\n{result.stderr}")
    else:
        logger.debug(f"Ingestion script stdout:\n{result.stdout}")
        if result.stderr:
             logger.debug(f"Ingestion script stderr:\n{result.stderr}") # Stderr might contain warnings
    return result

def get_sample_doc_ids_from_dir(doc_dir_path, limit=None) -> List[str]:
    """Gets doc_ids from XML filenames in a directory."""
    doc_ids = []
    if not os.path.exists(doc_dir_path):
        return []
    for filename in sorted(os.listdir(doc_dir_path)): # sorted for consistency
        if filename.endswith(".xml"):
            doc_ids.append(os.path.splitext(filename)[0])
    return doc_ids[:limit] if limit is not None else doc_ids

def count_docs_in_db(cursor, doc_ids_list: Optional[List[str]] = None) -> int:
    if doc_ids_list:
        placeholders = ','.join(['?'] * len(doc_ids_list))
        query = f"SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id IN ({placeholders})"
        cursor.execute(query, tuple(doc_ids_list))
    else:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    return cursor.fetchone()[0]

def get_docs_from_db(cursor, doc_ids_list: List[str]) -> List[tuple]:
    if not doc_ids_list:
        return []
    placeholders = ','.join(['?'] * len(doc_ids_list))
    query = f"SELECT doc_id, text_content, embedding FROM RAG.SourceDocuments WHERE doc_id IN ({placeholders})" # Removed text_length
    cursor.execute(query, tuple(doc_ids_list))
    return cursor.fetchall()

# --- Fixtures ---

@pytest.fixture(scope="module")
def db_connection(config_data): # Added config_data dependency
    """Module-scoped database connection."""
    try:
        # Pass the 'database' section of the loaded config
        conn = get_iris_connection(config=config_data.get('database'))
        if conn is None:
            pytest.fail("Failed to connect to the database for testing.")
        yield conn
        conn.close()
    except Exception as e:
        pytest.fail(f"Database connection fixture failed: {e}")

@pytest.fixture(scope="module")
def config_data():
    """Loads configuration data once per module."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="function", autouse=True)
def cleanup_db_before_each_test(db_connection):
    """Cleans up specific sample documents from DB before each test function."""
    # This ensures each test starts with a clean slate for the sample docs
    # It's important if tests might insert overlapping doc_ids
    sample_doc_ids_to_clear = get_sample_doc_ids_from_dir(ABSOLUTE_SAMPLE_DOCS_PATH, EXPECTED_DOC_COUNT_FULL_SAMPLE)
    if sample_doc_ids_to_clear:
        cursor = db_connection.cursor()
        placeholders = ','.join(['?'] * len(sample_doc_ids_to_clear))
        delete_query = f"DELETE FROM RAG.SourceDocuments WHERE doc_id IN ({placeholders})"
        try:
            cursor.execute(delete_query, tuple(sample_doc_ids_to_clear))
            db_connection.commit()
            logger.debug(f"Pre-test cleanup: Deleted {cursor.rowcount} records for sample doc IDs.")
        except Exception as e:
            logger.error(f"Error during pre-test cleanup: {e}")
            db_connection.rollback()
            pytest.fail(f"DB cleanup failed: {e}")
    yield # Test runs here

# --- Test Cases ---

def test_script_runs_and_ingests_all_sample_docs(db_connection, config_data):
    """Test basic run: ingest all 10 sample docs."""
    args = ["--docs_path", ABSOLUTE_SAMPLE_DOCS_PATH, "--config_path", CONFIG_PATH, "--log_level", "DEBUG"]
    result = run_ingestion_script(args)
    assert result.returncode == 0, "Ingestion script failed to run."

    cursor = db_connection.cursor()
    sample_doc_ids = get_sample_doc_ids_from_dir(ABSOLUTE_SAMPLE_DOCS_PATH, EXPECTED_DOC_COUNT_FULL_SAMPLE)
    
    db_docs = get_docs_from_db(cursor, sample_doc_ids)
    assert len(db_docs) == EXPECTED_DOC_COUNT_FULL_SAMPLE, \
        f"Expected {EXPECTED_DOC_COUNT_FULL_SAMPLE} docs in DB, found {len(db_docs)}."

    embedding_dim = config_data['embedding_model']['dimension']
    for row in db_docs:
        doc_id, text_content_stream, embedding_str = row # text_content_stream might be an InputStream

        # Handle potential InputStream for CLOB data from JDBC
        text_content_actual = ""
        if hasattr(text_content_stream, 'read'): # Check if it's a stream-like object
            try:
                # Read the stream; amount to read might need adjustment or iterative reading
                # For now, assume it can be read in one go or a large chunk
                stream_bytes = text_content_stream.read() # Read all available bytes
                if isinstance(stream_bytes, bytes):
                    text_content_actual = stream_bytes.decode('utf-8')
                elif isinstance(stream_bytes, str): # Some drivers might return str directly
                    text_content_actual = stream_bytes
                else: # Fallback if it's neither bytes nor str but readable
                    text_content_actual = str(text_content_stream)
            except Exception as e:
                logger.error(f"Error reading text_content stream for doc {doc_id}: {e}")
                text_content_actual = "" # Default to empty if read fails
        elif isinstance(text_content_stream, str):
            text_content_actual = text_content_stream
        
        assert text_content_actual is not None and len(text_content_actual) > 0, f"Doc {doc_id} has no content or content is not a string (type: {type(text_content_stream)})."
        # text_length check was removed as the column was removed from ingest script
        assert embedding_str is not None, f"Doc {doc_id} has NULL embedding."
        assert not embedding_str.startswith("[") and not embedding_str.endswith("]"), f"Doc {doc_id} embedding has brackets."
        parts = embedding_str.split(',')
        assert len(parts) == embedding_dim, f"Doc {doc_id} embedding dimension mismatch. Expected {embedding_dim}, got {len(parts)}."
        try:
            [float(p) for p in parts]
        except ValueError:
            pytest.fail(f"Doc {doc_id} embedding parts not all floats: {parts[:5]}")
    logger.info("test_script_runs_and_ingests_all_sample_docs PASSED")


def test_ingestion_with_limit_argument(db_connection):
    """Test the --limit argument."""
    limit = 3
    args = ["--docs_path", ABSOLUTE_SAMPLE_DOCS_PATH, "--limit", str(limit), "--config_path", CONFIG_PATH, "--log_level", "DEBUG"]
    result = run_ingestion_script(args)
    assert result.returncode == 0, "Ingestion script with --limit failed."

    cursor = db_connection.cursor()
    # We check all sample doc IDs because the script should only insert 'limit' number of them
    all_sample_ids = get_sample_doc_ids_from_dir(ABSOLUTE_SAMPLE_DOCS_PATH)
    
    # Count how many of the *potential* sample docs were actually inserted
    inserted_count = 0
    for doc_id in all_sample_ids:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id = ?", (doc_id,))
        if cursor.fetchone()[0] > 0:
            inserted_count +=1
            
    assert inserted_count == limit, f"Expected {limit} docs with --limit, found {inserted_count}."
    logger.info("test_ingestion_with_limit_argument PASSED")


def test_ingestion_with_doc_ids_argument(db_connection):
    """Test the --doc_ids argument."""
    all_sample_ids = get_sample_doc_ids_from_dir(ABSOLUTE_SAMPLE_DOCS_PATH)
    ids_to_ingest = all_sample_ids[:2] + all_sample_ids[5:6] # Pick 3 specific, non-contiguous IDs
    assert len(ids_to_ingest) == 3
    
    doc_ids_str = ",".join(ids_to_ingest)
    args = ["--docs_path", ABSOLUTE_SAMPLE_DOCS_PATH, "--doc_ids", doc_ids_str, "--config_path", CONFIG_PATH, "--log_level", "DEBUG"]
    result = run_ingestion_script(args)
    assert result.returncode == 0, "Ingestion script with --doc_ids failed."

    cursor = db_connection.cursor()
    db_docs = get_docs_from_db(cursor, ids_to_ingest) # Fetch only the ones we asked for
    assert len(db_docs) == len(ids_to_ingest), \
        f"Expected {len(ids_to_ingest)} docs with --doc_ids, found {len(db_docs)}."

    # Verify that ONLY these doc_ids were inserted from the sample set
    other_sample_ids = [id_ for id_ in all_sample_ids if id_ not in ids_to_ingest]
    if other_sample_ids:
        count_others = count_docs_in_db(cursor, other_sample_ids)
        assert count_others == 0, f"Found {count_others} unexpected documents from sample set in DB."
    logger.info("test_ingestion_with_doc_ids_argument PASSED")

def test_idempotency_clear_before_ingest(db_connection, config_data):
    """Test that running twice with clear_before_ingest results in correct state (no duplicates)."""
    args = ["--docs_path", ABSOLUTE_SAMPLE_DOCS_PATH, "--config_path", CONFIG_PATH, "--log_level", "DEBUG", "--clear_before_ingest"]
    
    # Run 1
    result1 = run_ingestion_script(args)
    assert result1.returncode == 0, "First run of ingestion script failed."
    
    cursor = db_connection.cursor()
    sample_doc_ids = get_sample_doc_ids_from_dir(ABSOLUTE_SAMPLE_DOCS_PATH, EXPECTED_DOC_COUNT_FULL_SAMPLE)
    count_after_run1 = count_docs_in_db(cursor, sample_doc_ids)
    assert count_after_run1 == EXPECTED_DOC_COUNT_FULL_SAMPLE, f"Expected {EXPECTED_DOC_COUNT_FULL_SAMPLE} docs after first run, got {count_after_run1}"

    # Run 2
    logger.info("Running ingestion script for the second time to test idempotency...")
    result2 = run_ingestion_script(args)
    assert result2.returncode == 0, "Second run of ingestion script failed."

    count_after_run2 = count_docs_in_db(cursor, sample_doc_ids)
    assert count_after_run2 == EXPECTED_DOC_COUNT_FULL_SAMPLE, \
        f"Expected {EXPECTED_DOC_COUNT_FULL_SAMPLE} docs after second run (idempotency check), got {count_after_run2}."

    # Additionally check that each doc_id appears exactly once
    for doc_id in sample_doc_ids:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 1, f"Doc {doc_id} does not have exactly one entry after two runs."
    logger.info("test_idempotency_clear_before_ingest PASSED")


def test_nonexistent_docs_path(caplog):
    """Test script behavior with a non-existent docs_path."""
    non_existent_path = os.path.join(ABSOLUTE_SAMPLE_DOCS_PATH, "non_existent_subdir")
    args = ["--docs_path", non_existent_path, "--config_path", CONFIG_PATH, "--log_level", "INFO"]
    
    # We expect the script to log an error and exit gracefully (or process 0 files)
    # It shouldn't raise an unhandled exception.
    # The script's main function has sys.exit(0) if no files, or sys.exit(1) for critical errors.
    # Here, get_document_filepaths returns [], so it should exit 0.
    result = run_ingestion_script(args)
    assert result.returncode == 0, f"Script should exit gracefully with non-existent path, got {result.returncode}"
    
    # Check logs for appropriate messages
    assert f"Documents directory not found: {non_existent_path}" in result.stderr or f"Documents directory not found: {non_existent_path}" in result.stdout
    assert "No files selected for processing. Exiting." in result.stdout
    logger.info("test_nonexistent_docs_path PASSED")

def test_empty_docs_directory(tmp_path, db_connection):
    """Test script behavior with an empty document directory."""
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    
    args = ["--docs_path", str(empty_dir), "--config_path", CONFIG_PATH, "--log_level", "DEBUG"]
    result = run_ingestion_script(args)
    assert result.returncode == 0, "Script should exit gracefully with empty dir."
    assert "No files selected for processing. Exiting." in result.stdout
    
    # Ensure no documents were added to the DB
    cursor = db_connection.cursor()
    assert count_docs_in_db(cursor) == 0, "Database should be empty after processing an empty directory."
    logger.info("test_empty_docs_directory PASSED")

def test_xml_parsing_error_handling(tmp_path, db_connection, caplog):
    """Test how the script handles a malformed XML file."""
    malformed_dir = tmp_path / "malformed_xml_dir"
    malformed_dir.mkdir()
    
    # Create one good XML and one bad XML
    good_xml_path = malformed_dir / "good_PMC1.xml"
    with open(good_xml_path, "w") as f:
        f.write("<article><front><article-meta><article-id pub-id-type='pmc'>PMC1</article-id><title-group><article-title>Good Title</article-title></title-group></article-meta></front><body><p>Good content.</p></body></article>")

    bad_xml_path = malformed_dir / "bad_PMC2.xml"
    with open(bad_xml_path, "w") as f:
        f.write("<article><title>This is not well-formed XML<body></article>") # Missing closing tags

    args = ["--docs_path", str(malformed_dir), "--config_path", CONFIG_PATH, "--log_level", "DEBUG"]
    result = run_ingestion_script(args)
    assert result.returncode == 0, "Script should complete even with parsing errors." # It logs errors and continues

    # Check logs for parsing error message
    # The script logs to stdout/stderr when run as subprocess
    assert f"Failed to parse XML file {str(bad_xml_path)}" in result.stderr or f"Failed to parse XML file {str(bad_xml_path)}" in result.stdout
    assert "Skipping file" in result.stderr or "Skipping file" in result.stdout
    
    # Check DB: only the good document should be ingested
    cursor = db_connection.cursor()
    good_doc = get_docs_from_db(cursor, ["PMC1"])
    bad_doc_count = count_docs_in_db(cursor, ["PMC2"])
    
    assert len(good_doc) == 1, "The good XML document was not ingested."
    assert bad_doc_count == 0, "The malformed XML document should not have been ingested."
    logger.info("test_xml_parsing_error_handling PASSED")


if __name__ == "__main__":
    # This allows running pytest via `python tests/test_ingest_docs.py`
    # Create __init__.py files if they don't exist for local test runs
    for p_part in ['common', 'scripts', 'tests']:
        dir_path = os.path.join(BASE_DIR, '..', p_part)
        os.makedirs(dir_path, exist_ok=True)
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {p_part} module\n")
    
    pytest.main([__file__, "-v", "-s"]) # -s to show stdout/stderr from script