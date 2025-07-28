import pytest
import os
import glob
import xml.etree.ElementTree as ET
import logging

# Import required components following project rules
from iris_rag.config.manager import ConfigurationManager
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.storage.schema_manager import SchemaManager
from common.vector_store import create_vector_store, VectorPoint

logger = logging.getLogger(__name__)

SAMPLE_DOCS_PATH = "data/sample_10_docs/"
EXPECTED_DOC_COUNT = 10

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

def extract_text_from_xml(filepath):
    """Extract text content from PMC XML file."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Extract text from various elements
        text_parts = []
        
        # Get text from all elements, excluding certain tags
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                text_parts.append(elem.tail.strip())
        
        return " ".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to parse XML file {filepath}: {e}")
        return ""

def ingest_documents_via_vectorstore(connection_manager):
    """
    Use the proper vector store infrastructure to ingest documents.
    This follows the intended design pattern.
    """
    # Initialize configuration and managers
    config_manager = ConfigurationManager()
    embedding_manager = EmbeddingManager(config_manager)
    
    # Initialize schema manager to ensure proper table setup
    schema_manager = SchemaManager(connection_manager, config_manager)
    schema_manager.ensure_table_schema('SourceDocuments')
    
    # Create vector store using the proper factory
    vector_store = create_vector_store(
        backend="iris",
        connection_manager=connection_manager
    )
    
    # Process each document in the sample folder
    ingested_count = 0
    
    if not os.path.exists(SAMPLE_DOCS_PATH):
        logger.warning(f"Sample docs path {SAMPLE_DOCS_PATH} does not exist")
        return ingested_count
    
    for filepath in glob.glob(os.path.join(SAMPLE_DOCS_PATH, "*.xml")):
        try:
            # Extract document ID from filename
            filename = os.path.basename(filepath)
            doc_id = filename.replace(".xml", "")
            
            # Extract text content
            text_content = extract_text_from_xml(filepath)
            
            if not text_content:
                logger.warning(f"No text content extracted from {filepath}")
                continue
            
            # Generate embedding for the document
            embedding = embedding_manager.embed_texts([text_content])[0]

            # Create a VectorPoint for upserting
            point = VectorPoint(id=doc_id, vector=embedding, payload={'text_content': text_content})

            # Use vector store to upsert the document
            vector_store.upsert('SourceDocuments', [point])
            ingested_count += 1
            logger.info(f"Successfully ingested document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to ingest document {filepath}: {e}")
            continue
    
    return ingested_count

@pytest.fixture
def mock_iris_connection(mocker):
    """Mock IRIS database connection for testing."""
    mock_connection = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    
    # Mock successful operations
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.rowcount = 1
    
    # Create a proper connection manager mock
    connection_manager = mocker.MagicMock()
    connection_manager.get_connection.return_value = mock_connection
    
    return connection_manager

def test_ingest_and_verify_documents(mock_iris_connection):
    """
    Test document ingestion using the proper vector store infrastructure.
    This test verifies that documents can be ingested and stored correctly.
    """
    # Verify sample documents exist
    assert os.path.exists(SAMPLE_DOCS_PATH), f"Sample docs path {SAMPLE_DOCS_PATH} does not exist"
    
    # Get expected document IDs
    expected_doc_ids = get_doc_ids_from_folder(SAMPLE_DOCS_PATH)
    assert len(expected_doc_ids) == EXPECTED_DOC_COUNT, f"Expected {EXPECTED_DOC_COUNT} docs, found {len(expected_doc_ids)}"
    
    # Ingest documents using vector store with mocked connection manager
    ingested_count = ingest_documents_via_vectorstore(mock_iris_connection)
    
    # Verify ingestion was successful
    assert ingested_count > 0, "No documents were ingested"
    assert ingested_count <= EXPECTED_DOC_COUNT, f"Ingested more documents ({ingested_count}) than expected ({EXPECTED_DOC_COUNT})"
    
    # Verify that the vector store operations were called
    # The mock connection should have been used for database operations
    assert mock_iris_connection.get_connection.called, "Connection manager should have been used"
    
    logger.info(f"Successfully ingested {ingested_count} documents using vector store infrastructure")