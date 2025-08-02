import pytest
import os
import glob
import xml.etree.ElementTree as ET
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.storage.enterprise_storage import IRISStorage
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.core.models import Document

# --- Configuration & Fixtures ---

@pytest.fixture(scope="module")
def config_manager():
    """Fixture for ConfigurationManager. Assumes environment variables are set for test DB."""
    # Ensure critical env vars are set for the test to run
    # Example:
    # os.environ["IRIS_HOST"] = "localhost"
    # os.environ["IRIS_PORT"] = "1972"
    # os.environ["IRIS_NAMESPACE"] = "TESTNS_E2E" # Use a dedicated test namespace if possible
    # os.environ["IRIS_USERNAME"] = "testuser"
    # os.environ["IRIS_PASSWORD"] = "testpass"
    # os.environ["IRIS_DRIVER_PATH"] = "/path/to/your/driver/intersystems-iris-native-2023.1.0.235.0-macx64/lib/libirisodbcdriver.dylib"
    # os.environ["LOG_LEVEL"] = "DEBUG"
    # os.environ["LOG_PATH"] = "logs/iris_rag_e2e_test.log"
    # os.environ["EMBEDDING_MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2" # A common small model
    # os.environ["DEFAULT_TABLE_NAME"] = "rag_documents_e2e_test"
    # os.environ["DEFAULT_TOP_K"] = "3"

    # For tests, ensure required env vars are present or raise an error
    required_env_vars = ["IRIS_HOST", "IRIS_PORT", "IRIS_NAMESPACE", "IRIS_USERNAME", "IRIS_PASSWORD", "EMBEDDING_MODEL_NAME", "DEFAULT_TABLE_NAME"]
    for var in required_env_vars:
        if var not in os.environ:
            # In a real CI, you might load a .env.test file or have these pre-set
            pytest.skip(f"Environment variable {var} not set. Skipping E2E pipeline tests.")
    return ConfigurationManager()

@pytest.fixture(scope="module")
def connection_manager(config_manager):
    """Fixture for ConnectionManager."""
    return ConnectionManager(config_manager)

@pytest.fixture(scope="module")
def embedding_manager(config_manager):
    """Fixture for EmbeddingManager."""
    return EmbeddingManager(config_manager)

@pytest.fixture(scope="module")
def vector_storage(connection_manager, embedding_manager, config_manager):
    """Fixture for IRISStorage."""
    storage = IRISStorage(connection_manager, embedding_manager, table_name=config_manager.get_default_table_name())
    # Setup: ensure table is clean or created
    try:
        storage.create_table(if_not_exists=True) # Assuming this also clears if it exists, or add a clear method
        # Clear table before tests in this module
        with connection_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DELETE FROM {storage.table_name}")
            conn.commit()

    except Exception as e:
        pytest.fail(f"Failed to set up vector storage table: {e}")
    return storage

def mock_llm_func(query: str, context: str) -> str:
    """A simple mock LLM function for testing."""
    if not context:
        return f"Mock LLM: No context provided for query: {query}"
    return f"Mock LLM response to '{query}' based on context: {context[:100]}..."

@pytest.fixture(scope="module")
def basic_rag_pipeline(connection_manager, embedding_manager, vector_storage, config_manager):
    """Fixture for BasicRAGPipeline."""
    return BasicRAGPipeline(
        connection_manager=connection_manager,
        embedding_manager=embedding_manager,
        vector_storage=vector_storage,
        llm_func=mock_llm_func,
        config_manager=config_manager
    )

@pytest.fixture(scope="module")
def sample_documents():
    """Loads sample documents from data/sample_10_docs/."""
    docs = []
    # Path relative to workspace root
    sample_docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_10_docs", "*.xml")
    for filepath in glob.glob(sample_docs_path):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            # Extract text content - this is a simplified parser
            # A more robust parser would handle different XML structures
            content_parts = [elem.text for elem in root.iter() if elem.text]
            content = "\n".join(filter(None, content_parts)).strip()
            if not content: # If no text found, use a placeholder
                content = "Placeholder content for " + os.path.basename(filepath)

            doc_id = os.path.basename(filepath).replace(".xml", "")
            docs.append(Document(id=doc_id, content=content, metadata={"source": filepath}))
        except ET.ParseError:
            print(f"Warning: Could not parse XML {filepath}")
        except Exception as e:
            print(f"Warning: Could not load document {filepath}: {e}")

    if not docs:
         pytest.skip("No sample documents found or loaded. Skipping E2E pipeline tests.")
    # Ensure we have at least a few docs for the test to be meaningful
    if len(docs) < 5: # Expecting 10, but be lenient
        print(f"Warning: Loaded only {len(docs)} documents. Test might be less effective.")

    return docs[:10] # Limit to 10 for test speed

# --- Test Cases ---

def test_full_rag_pipeline_e2e(basic_rag_pipeline, sample_documents, vector_storage):
    """Tests the full RAG pipeline: ingest, store, query, respond."""
    assert len(sample_documents) > 0, "No sample documents loaded for the test."

    # 1. Ingest Documents
    try:
        print(f"Ingesting {len(sample_documents)} documents...")
        basic_rag_pipeline.ingest_documents(sample_documents)
        print("Ingestion complete.")
    except Exception as e:
        pytest.fail(f"Document ingestion failed: {e}")

    # Verify documents are in storage (basic check)
    # A more thorough check would query the DB directly or use a vector_storage.count() method
    # For now, we rely on the query step to implicitly test storage.
    # Example direct check (if vector_storage had a count method or similar):
    # assert vector_storage.count_documents() >= len(sample_documents)

    # 2. Query the Pipeline
    test_query = "What is the role of apoptosis?" # A generic query likely to hit some PMC content
    response_data = None
    try:
        print(f"Querying pipeline with: '{test_query}'")
        response_data = basic_rag_pipeline.query(test_query)
        print(f"Pipeline response: {response_data}")
    except Exception as e:
        pytest.fail(f"Pipeline query failed: {e}")

    # 3. Validate Response
    assert response_data is not None, "Pipeline did not return a response."
    assert "query" in response_data, "Response missing 'query' field."
    assert response_data["query"] == test_query
    assert "answer" in response_data, "Response missing 'answer' field."
    assert response_data["answer"].startswith("Mock LLM response"), "LLM was not called or mock failed."
    assert "retrieved_documents" in response_data, "Response missing 'retrieved_documents' field."

    retrieved_docs = response_data["retrieved_documents"]
    assert isinstance(retrieved_docs, list), "Retrieved documents should be a list."

    # Depending on DEFAULT_TOP_K, we expect some documents if context was found
    # If the mock LLM indicates no context, this part might be different
    if "No context provided" not in response_data["answer"]:
        assert len(retrieved_docs) > 0, "No documents retrieved for a query that should have context."
        for doc in retrieved_docs:
            assert isinstance(doc, Document), "Retrieved item is not a Document instance."
            assert doc.id is not None
            assert doc.content is not None
            assert "similarity_score" in doc.metadata, "Retrieved document missing similarity_score."
    else:
        print("Warning: Mock LLM reported no context, so no documents might have been passed to it.")
        # In this case, retrieved_docs might be empty or not used by the mock LLM.
        # The core test is that the pipeline ran and the mock LLM responded.

    # 4. (Optional) Cleanup - if not handled by fixtures or a dedicated test namespace
    # For this example, we assume the table is either in a test namespace or cleared by the fixture.
    # If explicit cleanup is needed:
    # try:
    #     with vector_storage.connection_manager.get_connection() as conn:
    #         with conn.cursor() as cursor:
    #             cursor.execute(f"DELETE FROM {vector_storage.table_name}")
    #         conn.commit()
    # except Exception as e:
    #     print(f"Warning: Failed to clean up test data: {e}")


# To run these tests:
# 1. Ensure IRIS database is running and accessible.
# 2. Set ALL required environment variables (IRIS_HOST, ..., EMBEDDING_MODEL_NAME, DEFAULT_TABLE_NAME).
#    Consider using a .env file and a library like python-dotenv for local testing.
# 3. Ensure `sentence-transformers/all-MiniLM-L6-v2` is downloadable or cached.
# 4. PYTHONPATH=. pytest tests/test_e2e_iris_rag_full_pipeline.py