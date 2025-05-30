# tests/test_basic_rag.py

import pytest
from unittest.mock import MagicMock, patch
# import sqlalchemy # No longer needed
import os
import sys
from typing import Any # For mock type hints

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basic_rag.pipeline import BasicRAGPipeline # This will now use the updated IRISConnection type hint
from common.utils import Document

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnectionTypes, Cursor as IRISCursorTypes
except ImportError:
    IRISConnectionTypes = Any
    IRISCursorTypes = Any


# --- Mock Fixtures ---

@pytest.fixture
def mock_iris_connector():
    """Simplified mock for the InterSystems IRIS DB-API connection object."""
    mock_conn = MagicMock(spec=IRISConnectionTypes) 
    
    mock_cursor_method = MagicMock()
    mock_conn.cursor = mock_cursor_method

    mock_cursor_instance = MagicMock(spec=IRISCursorTypes) 
    mock_cursor_method.return_value = mock_cursor_instance
    
    # Explicitly create fetchall as a MagicMock and set its return_value
    mock_cursor_instance.fetchall = MagicMock(return_value=[
        ("mock_doc_1", "Mocked document content 1.", 0.95),
        ("mock_doc_2", "Mocked document content 2.", 0.88)
    ])
    mock_cursor_instance.execute = MagicMock()
    # Add a close method to the mock cursor instance
    mock_cursor_instance.close = MagicMock()
    # Add a close method to the mock connection instance
    mock_conn.close = MagicMock()
    return mock_conn

@pytest.fixture
def mock_embedding_func():
    """Mocks the embedding function, returning a fixed embedding."""
    # Returns a list containing one embedding (list of floats) for a list of input texts
    return MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]) 

# No fixture for mock_llm_func, we'll use the stub from common.utils

@pytest.fixture
def basic_rag_pipeline_under_test(mock_iris_connector, mock_embedding_func):
    """Initializes BasicRAGPipeline with some mock and some real (stub) dependencies."""
    from common.utils import get_llm_func # Import locally to use the stub
    
    stub_llm_func = get_llm_func(provider="stub") # Use the stub LLM

    return BasicRAGPipeline(
        iris_connector=mock_iris_connector,
        embedding_func=mock_embedding_func,
        llm_func=stub_llm_func # Use the stubbed LLM
    )

# --- Unit Tests ---

def test_retrieve_documents_calls_embedding_and_iris(basic_rag_pipeline_under_test, mock_embedding_func, mock_iris_connector):
    """
    Tests that retrieve_documents calls the embedding function and executes a query on IRIS.
    """
    query_text = "Test query for retrieval"
    top_k = 3
    
    # Get the mock cursor from the mock connector
    mock_cursor = mock_iris_connector.cursor.return_value

    retrieved_docs = basic_rag_pipeline_under_test.retrieve_documents(query_text, top_k=top_k) # Changed fixture name

    # Assert embedding_func was called with the query text
    mock_embedding_func.assert_called_once_with([query_text])

    # Assert IRIS cursor was obtained and execute was called
    mock_iris_connector.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()

    # Check the SQL query structure and parameters
    args, kwargs = mock_cursor.execute.call_args
    executed_sql = args[0]
    executed_params = args[1]
    
    assert "SELECT TOP ?" in executed_sql # Check for placeholder
    assert "VECTOR_COSINE" in executed_sql
    assert "TO_VECTOR(embedding, double, " in executed_sql # Check for stored embedding
    assert "TO_VECTOR(?, double, " in executed_sql      # Check for query embedding
    assert "FROM RAG.SourceDocuments_V2" in executed_sql
    assert "ORDER BY similarity_score DESC" in executed_sql
    
    # Check that top_k is the first parameter
    assert executed_params[0] == top_k
    # Check that the second parameter is the stringified query embedding
    # (mock_embedding_func.return_value[0] is the embedding list)
    expected_embedding_str = ','.join(map(str, mock_embedding_func.return_value[0]))
    assert executed_params[1] == expected_embedding_str
    # Ensure the key components are in the SQL, allowing for flexibility in exact spacing/formatting
    assert "VECTOR_COSINE" in executed_sql
    assert "TO_VECTOR(embedding, double, 768)" in executed_sql  # Check for stored embedding
    assert "TO_VECTOR(?, double, 768)" in executed_sql       # Check for query embedding
    assert "FROM RAG.SourceDocuments_V2" in executed_sql # Check for schema-qualified table
    assert "ORDER BY similarity_score DESC" in executed_sql # Alias is similarity_score
    
    # Assert fetchall was called
    mock_cursor.fetchall.assert_called_once()

    # Assert the structure of returned documents
    assert len(retrieved_docs) == 2 # Based on default mock_cursor.fetchall.return_value
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    assert retrieved_docs[0].id == "mock_doc_1" # Updated to match new mock_iris_connector default
    assert retrieved_docs[0].score == 0.95

def test_retrieve_documents_handles_iris_error(basic_rag_pipeline_under_test, mock_iris_connector):
    """
    Tests that retrieve_documents handles exceptions from IRIS gracefully.
    """
    mock_cursor = mock_iris_connector.cursor.return_value
    mock_cursor.execute.side_effect = Exception("IRIS DB Error")

    retrieved_docs = basic_rag_pipeline_under_test.retrieve_documents("query", top_k=3)
    
    assert retrieved_docs == [] # Should return empty list on error

def test_generate_answer_constructs_prompt_and_calls_stub_llm(basic_rag_pipeline_under_test):
    """
    Tests that generate_answer correctly constructs the prompt and calls the stub LLM.
    """
    query_text = "Test query for answer generation"
    retrieved_docs = [
        Document(id="doc1", content="Document 1 content.", score=0.9),
        Document(id="doc2", content="Document 2 provides more details.", score=0.85)
    ]
    
    # The stub LLM is used, so we check its characteristic output.
    # The prompt construction itself is an internal detail of generate_answer.
    # We trust the pipeline to call its llm_func (the stub).
    answer = basic_rag_pipeline_under_test.generate_answer(query_text, retrieved_docs)

    assert "Stub LLM response for prompt:" in answer
    # The prompt is complex, so checking for a substring of the query within the prompt part of the stub response is fragile.
    # A more robust check for this stub is that it contains its characteristic prefix.
    # If we need to check prompt content, we'd mock the llm_func itself.
    # For now, confirming it's the stub's response is sufficient for this test's scope.

def test_generate_answer_no_documents(basic_rag_pipeline_under_test):
    """
    Tests generate_answer behavior when no documents are retrieved.
    The stub LLM should not be called.
    """
    query_text = "Query with no retrieved docs"
    retrieved_docs = []
    
    # Patch the llm_func on the instance to see if it's called
    # This is a light way to check non-invocation without a separate mock fixture
    basic_rag_pipeline_under_test.llm_func = MagicMock(wraps=basic_rag_pipeline_under_test.llm_func)
    
    answer = basic_rag_pipeline_under_test.generate_answer(query_text, retrieved_docs)
    
    basic_rag_pipeline_under_test.llm_func.assert_not_called() 
    assert answer == "I could not find enough information to answer your question."


def test_run_orchestrates_retrieval_and_generation(basic_rag_pipeline_under_test):
    """
    Tests the main 'run' method to ensure it calls retrieval and generation.
    Uses MagicMock for sub-methods to isolate 'run' logic.
    """
    query_text = "Full pipeline test query"
    top_k = 3

    # Mock the instance's methods for this specific test of orchestration
    basic_rag_pipeline_under_test.retrieve_documents = MagicMock(return_value=[Document(id="d1", content="c1", score=0.9)])
    # The generate_answer method will use the stub LLM by default from the fixture.
    # If we want to control its output for *this specific test*, we can mock it on the instance.
    basic_rag_pipeline_under_test.generate_answer = MagicMock(return_value="Orchestration Test Final Answer")

    result = basic_rag_pipeline_under_test.run(query_text, top_k=top_k)

    basic_rag_pipeline_under_test.retrieve_documents.assert_called_once_with(query_text, top_k)
    basic_rag_pipeline_under_test.generate_answer.assert_called_once_with(query_text, basic_rag_pipeline_under_test.retrieve_documents.return_value)
    
    assert result["query"] == query_text
    assert result["answer"] == "Orchestration Test Final Answer"
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0].id == "d1"

# --- Placeholder for Parametrized E2E Tests ---
# These tests will use real services and a shared evaluation dataset.
# They will be more fully fleshed out when conftest.py and eval data are ready.

# @pytest.mark.e2e # Custom marker for end-to-end tests
# @pytest.mark.skip(reason="E2E test setup not yet complete")
# def test_basic_rag_pipeline_e2e_metrics(
#     real_iris_connector, # Fixture from conftest.py (to be created)
#     real_embedding_func, # Fixture from conftest.py (to be created)
#     real_llm_func,       # Fixture from conftest.py (to be created)
#     sample_eval_query    # Fixture from conftest.py providing one query from eval set
# ):
#     """
#     End-to-end test for BasicRAGPipeline using real services and metrics.
#     """
#     pipeline = BasicRAGPipeline(
#         iris_connector=real_iris_connector,
#         embedding_func=real_embedding_func,
#         llm_func=real_llm_func
#     )

#     query = sample_eval_query["query"]
#     # ground_truth_contexts = sample_eval_query["ground_truth_contexts"]
#     # ground_truth_answer = sample_eval_query["ground_truth_answer"]

#     result = pipeline.run(query)
    
#     # retrieved_contexts = [doc.content for doc in result['retrieved_documents']]
#     # generated_answer = result['answer']

#     # TODO: Add assertions for RAGAS metrics (recall, faithfulness)
#     # e.g., recall_score = calculate_ragas_recall(query, retrieved_contexts, ground_truth_contexts)
#     # assert recall_score >= 0.8 
#     # faithfulness_score = calculate_ragas_faithfulness(query, generated_answer, retrieved_contexts)
#     # assert faithfulness_score >= 0.7

#     assert "answer" in result
#     assert len(result["retrieved_documents"]) > 0 # Basic check
