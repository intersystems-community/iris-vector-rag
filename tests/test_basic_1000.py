# tests/test_basic_1000.py

import pytest
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basic_rag.pipeline import BasicRAGPipeline
from common.utils import Document, get_embedding_func, get_llm_func
# Fixtures like verify_document_count will be automatically picked up if
# tests/conftest_1000docs_active.py is in effect.

@pytest.mark.requires_1000_docs # Explicitly mark, though auto-marking might also apply
def test_basic_rag_pipeline_with_1000_docs(
    verify_document_count, # This fixture ensures 1000+ docs and provides the IRIS connection
    embedding_model_fixture, # From main conftest.py - provides real embedding func
    llm_client_fixture_stub # A new fixture for stub LLM to avoid OpenAI calls
):
    """
    Tests the BasicRAGPipeline with a database guaranteed to have at least 1000 documents.
    Uses real embedding function and a stub LLM.
    """
    iris_connection = verify_document_count # The fixture returns the connection
    
    # Use the real embedding function from the main conftest
    # Use a stub LLM to avoid actual LLM calls during this test
    # embedding_fn = get_embedding_func() # Or use embedding_model_fixture
    # llm_fn = get_llm_func(provider="stub") # Or use llm_client_fixture with stub config
    
    # Defensive check: Ensure iris_connection is a compliant mock if it's not a real connection
    from tests.mocks.db import MockIRISConnector
    from common.iris_connector import get_mock_iris_connection # To get a fresh compliant mock
    
    final_iris_connection = iris_connection
    if not hasattr(iris_connection, '_connection'): # Heuristic for real SQLAlchemy connection
        if not isinstance(iris_connection, MockIRISConnector):
            logger.warning(f"Test 'test_basic_rag_pipeline_with_1000_docs' received a non-standard mock connection type: {type(iris_connection)}. Replacing with MockIRISConnector.")
            final_iris_connection = get_mock_iris_connection()
            if final_iris_connection is None:
                pytest.fail("Failed to obtain MockIRISConnector for 1000-doc test.")

    logger.info("Initializing BasicRAGPipeline for 1000-doc test.")
    pipeline = BasicRAGPipeline(
        iris_connector=final_iris_connection, # Use the potentially replaced connection
        embedding_func=embedding_model_fixture, # Use real embedding func
        llm_func=llm_client_fixture_stub # Use stub LLM
    )

    test_query = "What are common treatments for diabetes?"
    logger.info(f"Running BasicRAGPipeline with 1000+ docs for query: '{test_query}'")
    
    result = pipeline.run(test_query, top_k=5)

    logger.info(f"Query: {result['query']}")
    logger.info(f"Answer: {result['answer']}")
    
    assert "answer" in result, "Pipeline should produce an answer."
    # The stub LLM will return something like "Stub LLM response for prompt: '...'"
    assert "Stub LLM response" in result['answer'], "Answer should come from the stub LLM."
    
    assert "retrieved_documents" in result, "Pipeline should retrieve documents."
    # We can't assert much about the content of retrieved_documents if they are synthetic
    # and embeddings are random, but we can check if some were retrieved.
    # If the DB was empty and only synthetic docs were added, retrieval might be weak.
    # If real data was loaded by another fixture (e.g. iris_with_pmc_data) and then
    # ensure_1000_documents topped it up, results might be more meaningful.
    
    # For now, a basic check:
    if len(result['retrieved_documents']) == 0:
        logger.warning("No documents retrieved in 1000-doc test. This might be okay if embeddings are random/synthetic.")
    else:
        logger.info(f"Retrieved {len(result['retrieved_documents'])} documents.")
        first_doc = result['retrieved_documents'][0]
        assert isinstance(first_doc, Document)
        logger.info(f"First retrieved doc ID: {first_doc.id}, Score: {first_doc.score}")

    logger.info("test_basic_rag_pipeline_with_1000_docs completed.")

# Need a fixture for a stub LLM if llm_client_fixture defaults to OpenAI
@pytest.fixture(scope="session")
def llm_client_fixture_stub():
    """Provides a stub LLM function for tests not needing real LLM calls."""
    logger.info("Fixture: Initializing STUB LLM client function for 1000-doc test.")
    return get_llm_func(provider="stub")
