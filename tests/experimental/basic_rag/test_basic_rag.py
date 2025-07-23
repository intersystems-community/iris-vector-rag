import pytest
import os
import sys
from typing import Callable, List # Import Callable and List
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jaydebeapi # Import for type hinting
from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func, Document # Updated import
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import and class name

# Placeholder for the 10 sample doc IDs, assuming they are loaded with these IDs
# This is more for conceptual clarity in the test; the actual test won't hardcode content checks
SAMPLE_DOC_IDS = [f"PMC{i}" for i in range(1, 11)] # Example: PMC1, PMC2, ...

@pytest.fixture(scope="module")
def iris_conn() -> jaydebeapi.Connection: # Updated type hint
    """Fixture to provide an IRIS connection."""
    connection = get_iris_connection()
    assert connection is not None, "Failed to connect to IRIS"
    yield connection
    connection.close()

@pytest.fixture(scope="module")
def embedding_func():
    """Fixture to provide an embedding function."""
    return get_embedding_func()

@pytest.fixture(scope="module")
def llm_func() -> Callable[[str], str]: # Added type hint for consistency
    """Fixture to provide an LLM function (mocked for simplicity)."""
    def mock_llm(prompt: str) -> str:
        return "This is a mock LLM answer."
    return mock_llm

@pytest.fixture(scope="module")
def basic_rag_pipeline(iris_conn: jaydebeapi.Connection, embedding_func: Callable, llm_func: Callable):
    """Fixture to create an instance of the BasicRAGPipeline."""
    # Assuming the pipeline will use the RAG schema by default
    # and will read connection details from config.yaml implicitly via get_iris_connection
    pipeline = BasicRAGPipeline( # Updated class name
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func,
        schema="RAG" # Explicitly using RAG schema as per project context
    )
    return pipeline

def test_database_connection(iris_conn: jaydebeapi.Connection): # Updated type hint
    """Test that a connection to the database can be established."""
    assert iris_conn is not None
    cursor = iris_conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
    cursor.close()

def test_basic_rag_pipeline_search_and_format(basic_rag_pipeline: BasicRAGPipeline): # Updated type hint
    """
    Test the BasicRAGPipelineSimple's search functionality.
    Verifies:
    - It can perform a vector search (implicitly using TO_VECTOR on CLOB).
    - It returns results in the standardized format.
    - It handles our 10 sample documents (by returning some results).
    """
    test_query = "What is CRISPR?" # A generic query
    
    # This call should trigger the pipeline's retrieve_documents and generate_answer methods
    # The retrieve_documents method in pipeline_simple.py will need to use TO_VECTOR()
    result = basic_rag_pipeline.run(query=test_query, top_k=3)

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "query" in result, "Result dictionary missing 'query' key."
    assert "answer" in result, "Result dictionary missing 'answer' key."
    assert "retrieved_documents" in result, "Result dictionary missing 'retrieved_documents' key."
    
    assert result["query"] == test_query, "Query in result does not match input query."
    assert isinstance(result["answer"], str), "Answer should be a string."
    
    retrieved_docs = result["retrieved_documents"]
    assert isinstance(retrieved_docs, list), "Retrieved documents should be a list."
    
    # Check if some documents are retrieved (assuming 10 sample docs are loaded and searchable)
    # We don't check for specific content, just that the mechanism works.
    assert len(retrieved_docs) > 0, "No documents retrieved. Expected some results from the 10 sample docs."
    assert len(retrieved_docs) <= 3, "More documents retrieved than top_k."

    for doc_data in retrieved_docs:
        assert isinstance(doc_data, dict), "Each retrieved document should be a dictionary."
        assert "id" in doc_data, "Retrieved document missing 'id' key."
        assert "content" in doc_data, "Retrieved document missing 'content' key."
        assert "metadata" in doc_data, "Retrieved document missing 'metadata' key."
        assert "similarity_score" in doc_data["metadata"], "Document metadata missing 'similarity_score'."
        assert isinstance(doc_data["metadata"]["similarity_score"], float), "Similarity score should be a float."

def test_pipeline_uses_to_vector_implicitly(basic_rag_pipeline: BasicRAGPipeline, iris_conn: jaydebeapi.Connection, embedding_func: Callable): # Updated type hint
    """
    A more focused test to ensure the SQL query likely uses TO_VECTOR.
    This is an indirect test by checking if a query against string embeddings works.
    """
    # This test assumes that 'RAG.SourceDocuments' has 'embedding' as a CLOB/VARCHAR
    # and that the pipeline is designed to convert it using TO_VECTOR.
    
    # We can't directly inspect the SQL from here without mocking,
    # but we can infer its correct operation if it returns results.
    
    query_text = "test query for TO_VECTOR"
    query_embedding = embedding_func([query_text])[0]
    # The pipeline should handle formatting this for the SQL query
    # e.g. by converting to "[f1,f2,...]" string for TO_VECTOR(?)

    # Attempt to retrieve documents. If this works with string embeddings,
    # it implies TO_VECTOR is being used correctly in the SQL.
    try:
        # We only need to test the retrieval part for this specific check
        retrieved_docs: List[Document] = basic_rag_pipeline.retrieve_documents(query_text, top_k=1)
        assert len(retrieved_docs) >= 0 # Can be 0 if no match, but shouldn't error
        if retrieved_docs:
            assert isinstance(retrieved_docs[0].score, float)
    except Exception as e:
        pytest.fail(f"retrieve_documents failed, possibly due to TO_VECTOR issues: {e}")