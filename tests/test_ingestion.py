import pytest
import os
import glob
import xml.etree.ElementTree as ET
import re
import logging # Import logging
from sentence_transformers import SentenceTransformer # For checking embedding dimension if needed

logger = logging.getLogger(__name__) # Define logger

# Attempt to import the yet-to-be-created ingestion script/function
try:
    from scripts.ingest_10_docs import ingest_10_sample_docs, get_config_values
except ImportError:
    # This is expected initially. We'll define a placeholder so tests can be defined.
    def ingest_10_sample_docs():
        raise NotImplementedError("ingest_10_docs.py is not yet implemented.")
    def get_config_values():
        # Return a mock config for tests to be defined, actual config will be loaded by the script
        return {
            "database": {
                "namespace": "RAGTEST", # Example, will be from actual config
                "host": "localhost",
                "port": 1972,
                "user": "superuser",
                "password": "SYS"
            },
            "embedding_model": "all-MiniLM-L6-v2",
            "sample_docs_path": "data/sample_10_docs/"
        }


# Assuming common.iris_connector is available and works
from common.iris_connector import get_iris_connection

SAMPLE_DOCS_PATH = "data/sample_10_docs/"
EXPECTED_DOC_COUNT = 10
TABLE_NAME = "RAG.SourceDocuments"

def get_doc_ids_from_folder(folder_path):
    """Helper to get expected doc_ids from filenames."""
    doc_ids = []
    if os.path.exists(folder_path):
        for filepath in glob.glob(os.path.join(folder_path, "*.xml")):
            filename = os.path.basename(filepath)
            doc_id = filename.replace(".xml", "")
            doc_ids.append(doc_id)
    return sorted(doc_ids)

EXPECTED_DOC_IDS = get_doc_ids_from_folder(SAMPLE_DOCS_PATH)

@pytest.fixture(scope="module")
def iris_conn():
    """Fixture to provide an IRIS connection."""
    # In a real scenario, config would be loaded more robustly,
    # but for tests, we might rely on the script's config or a test-specific one.
    # For now, let's assume get_iris_connection can use its defaults or env vars.
    # The actual script will use config/config.yaml
    conn_details = get_config_values()["database"]
    # Pass conn_details dictionary to the 'config' parameter
    conn = get_iris_connection(config=conn_details)
    if conn is None:
        pytest.fail("Failed to connect to IRIS database.")
    yield conn
    conn.close()

@pytest.fixture(scope="function", autouse=True)
def clean_db_before_each_test(iris_conn):
    """Cleans the specific 10 docs from the table before each relevant test run."""
    cursor = iris_conn.cursor()
    try:
        # Delete only the 10 sample docs to ensure idempotency test is valid
        # and other data in the table (if any) is not affected.
        if EXPECTED_DOC_IDS:
            placeholders = ', '.join(['?'] * len(EXPECTED_DOC_IDS))
            sql_delete = f"DELETE FROM {TABLE_NAME} WHERE doc_id IN ({placeholders})"
            cursor.execute(sql_delete, EXPECTED_DOC_IDS)
            iris_conn.commit()
    except Exception as e:
        # Table might not exist on first run, which is fine for some tests.
        print(f"Note: Could not pre-clean {TABLE_NAME} for sample docs: {e}")
    finally:
        cursor.close()

def test_db_connection(iris_conn):
    """Test that a connection to the database can be established."""
    assert iris_conn is not None
    # For jaydebeapi, a successful connection means the object is not None.
    # It doesn't have an is_connected() method.
    # We can perform a simple query to be absolutely sure.
    try:
        cursor = iris_conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone() is not None
        cursor.close()
    except Exception as e:
        pytest.fail(f"Database connection check query failed: {e}")

def test_read_sample_documents_existence():
    """Test that the 10 sample documents can be found."""
    assert os.path.exists(SAMPLE_DOCS_PATH), f"Sample documents path {SAMPLE_DOCS_PATH} does not exist."
    xml_files = glob.glob(os.path.join(SAMPLE_DOCS_PATH, "*.xml"))
    assert len(xml_files) == EXPECTED_DOC_COUNT, \
        f"Expected {EXPECTED_DOC_COUNT} XML files in {SAMPLE_DOCS_PATH}, found {len(xml_files)}."
    assert len(EXPECTED_DOC_IDS) == EXPECTED_DOC_COUNT, \
        f"Expected {EXPECTED_DOC_COUNT} doc_ids, found {len(EXPECTED_DOC_IDS)} from filenames."


# This is the main test that will initially fail because ingest_10_sample_docs is not implemented.
def test_ingest_and_verify_documents(iris_conn):
    """
    Tests the ingestion process:
    1. Runs the ingestion script.
    2. Verifies connection to the database.
    3. Verifies reading the 10 sample documents.
    4. Verifies proper insertion into RAG.SourceDocuments.
    5. Verifies embeddings are stored correctly as CLOB (clean, comma-separated string).
    6. Verifies no duplicate insertions (idempotency for the 10 docs).
    """
    try:
        ingest_10_sample_docs() # This will call the (future) script's main function
    except NotImplementedError:
        pytest.fail("Ingestion script 'scripts/ingest_10_docs.py' is not implemented yet.")
    except Exception as e:
        pytest.fail(f"Ingestion script failed: {e}")

    cursor = iris_conn.cursor()

    # Verify insertion
    placeholders = ', '.join(['?'] * len(EXPECTED_DOC_IDS))
    cursor.execute(f"SELECT doc_id, text_content, embedding FROM {TABLE_NAME} WHERE doc_id IN ({placeholders}) ORDER BY doc_id", EXPECTED_DOC_IDS)
    rows = cursor.fetchall()
    
    assert len(rows) == EXPECTED_DOC_COUNT, \
        f"Expected {EXPECTED_DOC_COUNT} documents in {TABLE_NAME} for the sample set, found {len(rows)}."

    # Verify content and embedding format for each document
    db_doc_ids = []
    for row in rows:
        doc_id, text_content_stream, embedding_str = row
        db_doc_ids.append(doc_id)
        assert doc_id in EXPECTED_DOC_IDS, f"Unexpected doc_id {doc_id} found in database."

        text_content = ""
        if hasattr(text_content_stream, 'read'): # Check if it's a stream-like object
            # Read the stream content. JDBC might return CLOBs as streams.
            # Assuming UTF-8 encoding for the stream.
            try:
                # JayDeBeApi might return LOBs as Java objects that can be directly converted to string,
                # or sometimes as actual stream objects.
                # If it's a stream with a read method:
                if hasattr(text_content_stream, 'read') and callable(text_content_stream.read):
                    # Attempt to read. This might return bytes or a Java array of bytes.
                    raw_data = text_content_stream.read()
                    if isinstance(raw_data, bytes):
                        text_content = raw_data.decode('utf-8')
                    elif raw_data is not None: # If it's some other Java type (like JInt if read() returned length)
                                               # or if the stream itself is the content (less likely for .read())
                                               # This part is tricky without knowing exact JayDeBeApi LOB behavior.
                                               # Let's assume if .read() doesn't give bytes, the stream itself might be convertible.
                                               # Or, the driver might have already converted small CLOBs.
                        # If .read() returned a JInt (like a length), then text_content_stream is the stream.
                        # This case is complex. For now, let's assume if it's not bytes, try converting the stream obj.
                        # A more robust solution would inspect the specific type from JayDeBeApi for CLOBs.
                        # For now, if .read() doesn't yield bytes, we'll rely on the direct str conversion below.
                        # This means the previous .read() might not have been what we wanted if it returned JInt.
                        # Let's simplify: if it has .read, try to read and decode. If that fails, or no .read, try str().
                        pass # Will fall through to str() conversion if bytes not obtained

                # If not read as bytes, or if it's not a stream with .read(), try direct conversion
                if not text_content: # if text_content is still empty
                    text_content = str(text_content_stream)

            except Exception as e:
                logger.warning(f"Could not read/convert text_content_stream for {doc_id}. Type: {type(text_content_stream)}, Error: {e}")
                text_content = "" # Fallback to empty string if read fails
        elif isinstance(text_content_stream, str): # If it's already a Python string
            text_content = text_content_stream
        else: # Fallback for unknown types
            try:
                text_content = str(text_content_stream)
            except Exception as e:
                logger.warning(f"Could not convert text_content_stream to string for {doc_id}. Type: {type(text_content_stream)}, Error: {e}")
                text_content = ""

        assert text_content is not None and len(text_content) > 0, f"Text content for {doc_id} is empty or could not be read. Final type: {type(text_content)}, Value: '{str(text_content)[:100]}...'"
        
        assert embedding_str is not None, f"Embedding for {doc_id} is NULL."
        assert isinstance(embedding_str, str), f"Embedding for {doc_id} is not a string."
        # Regex to check for comma-separated floats.
        # Allows for numbers like: 123, 123.45, .45, -123, -123.45, -.45
        # Does not allow brackets.
        # Pattern for one number: -?(\d+(\.\d*)?|\.\d+)
        # Full regex: ^<one_number>(,<one_number>)*$
        float_pattern = r"-?(\d+(\.\d*)?|\.\d+)"
        full_regex = f"^{float_pattern}(,{float_pattern})*$"
        assert re.match(full_regex, embedding_str), \
            f"Embedding for {doc_id} is not a clean, comma-separated string. Value: '{embedding_str[:100]}...'"
        
        # Optional: Check embedding dimension if model is easily available
        # model_name = get_config_values()["embedding_model"]
        # model = SentenceTransformer(model_name)
        # expected_dim = model.get_sentence_embedding_dimension()
        # assert len(embedding_str.split(',')) == expected_dim, \
        #    f"Embedding for {doc_id} has incorrect dimension. Expected {expected_dim}, got {len(embedding_str.split(','))}"

    assert sorted(db_doc_ids) == sorted(EXPECTED_DOC_IDS), "Mismatch between expected and actual doc_ids in DB."

    # Close the cursor before the second ingestion run to potentially release LOB locks
    cursor.close()

    # Verify idempotency: run ingestion again and check
    try:
        ingest_10_sample_docs() # Run ingestion a second time
    except Exception as e:
        pytest.fail(f"Ingestion script failed on second run for idempotency check: {e}")
    
    # Re-open cursor for subsequent checks
    cursor = iris_conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE doc_id IN ({placeholders})", EXPECTED_DOC_IDS)
    count_after_second_run = cursor.fetchone()[0]
    assert count_after_second_run == EXPECTED_DOC_COUNT, \
        f"Idempotency check failed: Expected {EXPECTED_DOC_COUNT} documents after second run, found {count_after_second_run}."

    # Further check if content is identical (optional, can be heavy)
    # For instance, check one document's embedding string remains identical
    cursor.execute(f"SELECT embedding FROM {TABLE_NAME} WHERE doc_id = ?", (EXPECTED_DOC_IDS[0],))
    embedding_after_second_run = cursor.fetchone()[0]
    
    # Find the original embedding string for comparison
    original_embedding_str = ""
    for row_tuple in rows: # rows from the first fetch
        if row_tuple[0] == EXPECTED_DOC_IDS[0]:
            original_embedding_str = row_tuple[2]
            break
    
    assert embedding_after_second_run == original_embedding_str, \
        f"Idempotency check failed: Embedding for {EXPECTED_DOC_IDS[0]} changed after second run."

    cursor.close()

if __name__ == "__main__":
    # This allows running pytest via "python tests/test_ingestion.py"
    # Add specific arguments if needed, e.g. for verbose output
    pytest.main([__file__, "-v"])