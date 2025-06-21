import pytest
import os
import glob
import xml.etree.ElementTree as ET
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.storage.iris import IRISStorage
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.core.models import Document

# --- Configuration & Fixtures (largely reused from full_pipeline test) ---

@pytest.fixture(scope="module")
def config_manager():
    """Fixture for ConfigurationManager. Assumes environment variables are set for test DB."""
    required_env_vars = ["IRIS_HOST", "IRIS_PORT", "IRIS_NAMESPACE", "IRIS_USERNAME", "IRIS_PASSWORD", "EMBEDDING_MODEL_NAME", "DEFAULT_TABLE_NAME"]
    for var in required_env_vars:
        if var not in os.environ:
            pytest.skip(f"Environment variable {var} not set. Skipping E2E integration tests.")
    # Use a distinct log path for integration tests if desired
    os.environ["LOG_PATH"] = os.environ.get("LOG_PATH", "logs/iris_rag_integration_test.log")
    os.environ["DEFAULT_TABLE_NAME"] = os.environ.get("DEFAULT_TABLE_NAME", "rag_documents_integration_test")
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
    table_name = config_manager.get_default_table_name()
    storage = IRISStorage(connection_manager, embedding_manager, table_name=table_name)
    try:
        # Ensure table is clean or created for this module
        with connection_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Drop if exists to ensure clean state for integration tests
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                storage.create_table(if_not_exists=False) # Create it fresh
            conn.commit()
    except Exception as e:
        pytest.fail(f"Failed to set up vector storage table '{table_name}': {e}")
    return storage

def mock_llm_func_integration(query: str, context: str) -> str:
    """A simple mock LLM function for integration testing, returns context for inspection."""
    if not context:
        return f"NO_CONTEXT_PROVIDED_FOR_QUERY:{query}"
    return f"CONTEXT_FOR_QUERY:{query}|CONTEXT_START:{context}CONTEXT_END"

@pytest.fixture(scope="module")
def basic_rag_pipeline_integration(connection_manager, embedding_manager, vector_storage, config_manager):
    """Fixture for BasicRAGPipeline for integration tests."""
    pipeline = BasicRAGPipeline(
        connection_manager=connection_manager,
        embedding_manager=embedding_manager,
        vector_storage=vector_storage,
        llm_func=mock_llm_func_integration, # Use the integration-specific mock
        config_manager=config_manager
    )
    return pipeline

@pytest.fixture(scope="module")
def sample_documents_with_known_content():
    """
    Loads sample documents and injects/identifies known content for targeted queries.
    This is a simplified approach. A real test might have dedicated small text files.
    """
    docs = []
    sample_docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_10_docs", "*.xml")
    
    # Define some keywords we expect or will inject into specific documents for testing
    # For this example, let's assume PMC524367.xml will be about "mitochondrial function"
    # and PMC526216.xml about "neural pathways".
    known_keywords = {
        "PMC524367": "Detailed analysis of mitochondrial function and energy metabolism.",
        "PMC526216": "Exploring novel neural pathways in cognitive development."
    }

    for filepath in glob.glob(sample_docs_path):
        doc_id = os.path.basename(filepath).replace(".xml", "")
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            content_parts = [elem.text for elem in root.iter() if elem.text]
            base_content = "\n".join(filter(None, content_parts)).strip()
            
            # Inject known keyword phrase if this doc_id is in our map
            final_content = base_content
            if doc_id in known_keywords:
                final_content = known_keywords[doc_id] + "\n" + base_content
            
            if not final_content:
                final_content = "Placeholder content for " + doc_id
            
            docs.append(Document(id=doc_id, content=final_content, metadata={"source": filepath}))
        except Exception as e:
            print(f"Warning: Could not load/process document {filepath}: {e}")
            # Add a placeholder if loading fails to keep doc count consistent
            docs.append(Document(id=doc_id, content=f"Error loading {doc_id}. {known_keywords.get(doc_id, '')}", metadata={"source": filepath, "error": True}))


    if not docs or len(docs) < 2: # Need at least our two known keyword docs
         pytest.skip("Not enough sample documents with known content found. Skipping E2E integration tests.")
    return docs[:10] # Use up to 10

# --- Test Cases ---

def test_pipeline_integration_with_known_content(basic_rag_pipeline_integration, sample_documents_with_known_content, vector_storage):
    """
    Tests the RAG pipeline's ability to retrieve relevant documents based on specific content.
    """
    assert len(sample_documents_with_known_content) > 0, "No sample documents loaded."

    # 1. Ingest Documents
    try:
        basic_rag_pipeline_integration.ingest_documents(sample_documents_with_known_content)
    except Exception as e:
        pytest.fail(f"Document ingestion failed during integration test: {e}")

    # 2. Query for "mitochondrial function" - expecting PMC524367
    query1 = "Tell me about mitochondrial function"
    response1_data = basic_rag_pipeline_integration.query(query1)

    assert response1_data is not None
    assert "answer" in response1_data
    answer1 = response1_data["answer"]
    retrieved_docs1 = response1_data["retrieved_documents"]

    assert not answer1.startswith("NO_CONTEXT_PROVIDED"), f"Query '{query1}' did not get context."
    assert "CONTEXT_FOR_QUERY:" in answer1
    # Check if the context passed to LLM contains our keyword phrase for PMC524367
    assert "mitochondrial function and energy metabolism" in answer1, \
        f"Expected context for '{query1}' not found in LLM input: {answer1}"
    
    assert len(retrieved_docs1) > 0, f"No documents retrieved for query '{query1}'"
    found_doc1 = any(doc.id == "PMC524367" for doc in retrieved_docs1)
    assert found_doc1, f"Document PMC524367 not found in retrieved set for query '{query1}'"
    # Check similarity score of the target document (optional, but good)
    for doc in retrieved_docs1:
        if doc.id == "PMC524367":
            assert "similarity_score" in doc.metadata
            assert doc.metadata["similarity_score"] > 0.5, \
                f"PMC524367 retrieved for '{query1}' but with low similarity: {doc.metadata['similarity_score']}"


    # 3. Query for "neural pathways" - expecting PMC526216
    query2 = "Information on neural pathways"
    response2_data = basic_rag_pipeline_integration.query(query2)

    assert response2_data is not None
    assert "answer" in response2_data
    answer2 = response2_data["answer"]
    retrieved_docs2 = response2_data["retrieved_documents"]

    assert not answer2.startswith("NO_CONTEXT_PROVIDED"), f"Query '{query2}' did not get context."
    assert "CONTEXT_FOR_QUERY:" in answer2
    assert "novel neural pathways in cognitive development" in answer2, \
        f"Expected context for '{query2}' not found in LLM input: {answer2}"

    assert len(retrieved_docs2) > 0, f"No documents retrieved for query '{query2}'"
    found_doc2 = any(doc.id == "PMC526216" for doc in retrieved_docs2)
    assert found_doc2, f"Document PMC526216 not found in retrieved set for query '{query2}'"
    for doc in retrieved_docs2:
        if doc.id == "PMC526216":
            assert "similarity_score" in doc.metadata
            assert doc.metadata["similarity_score"] > 0.5, \
                 f"PMC526216 retrieved for '{query2}' but with low similarity: {doc.metadata['similarity_score']}"

    # 4. Query for something completely unrelated - expecting no specific known context or low scores
    query3 = "syzygy astronomy calculations"
    response3_data = basic_rag_pipeline_integration.query(query3)
    answer3 = response3_data["answer"]
    retrieved_docs3 = response3_data["retrieved_documents"]

    # It might still find *some* documents, but our specific keywords shouldn't be prominent in context
    if not answer3.startswith("NO_CONTEXT_PROVIDED"):
        assert "mitochondrial function" not in answer3.split("CONTEXT_START:")[1] # Check only actual context
        assert "neural pathways" not in answer3.split("CONTEXT_START:")[1]
    
    # Ensure our specific test docs are not retrieved with high confidence for an unrelated query
    for doc in retrieved_docs3:
        if doc.id in ["PMC524367", "PMC526216"]:
            assert doc.metadata.get("similarity_score", 0) < 0.4, \
                f"Known document {doc.id} unexpectedly retrieved with high score for unrelated query '{query3}'"

# To run these tests:
# 1. Ensure IRIS database is running and accessible.
# 2. Set ALL required environment variables (IRIS_HOST, ..., EMBEDDING_MODEL_NAME, DEFAULT_TABLE_NAME).
# 3. Ensure `sentence-transformers/all-MiniLM-L6-v2` (or configured model) is downloadable.
# 4. PYTHONPATH=. pytest tests/test_e2e_iris_rag_integration.py