# tests/test_hyde.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
# import sqlalchemy # No longer needed
from typing import Any # For mock type hints

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hyde.pipeline import HyDEPipeline # This will use the updated IRISConnection type hint
from common.utils import Document, get_llm_func # For stub LLM

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
    mock_cursor_method = MagicMock() # Mock for the .cursor() method call
    mock_conn.cursor = mock_cursor_method
    
    mock_cursor_instance = MagicMock(spec=IRISCursorTypes) # Mock for the cursor object
    mock_cursor_method.return_value = mock_cursor_instance
    
    # Explicitly create fetchall as a MagicMock and set its return_value
    mock_cursor_instance.fetchall = MagicMock(return_value=[
        ("retrieved_doc1", "Actual content from DB for doc 1", 0.92),
        ("retrieved_doc2", "Actual content from DB for doc 2", 0.88)
    ])
    mock_cursor_instance.execute = MagicMock() # Ensure execute is a callable mock
    mock_cursor_instance.close = MagicMock()
    mock_conn.close = MagicMock()
    return mock_conn

@pytest.fixture
def mock_embedding_func():
    """Mocks the embedding function."""
    return MagicMock(return_value=[[0.5, 0.4, 0.3, 0.2, 0.1]]) # Different embedding for hypo doc

@pytest.fixture
def mock_llm_for_hyde():
    """
    Mocks the LLM function for HyDE.
    It needs to return a hypothetical document for the first call,
    and a final answer for the second call.
    """
    mock = MagicMock()
    # Configure side_effect to return different values on subsequent calls
    mock.side_effect = [
        "This is a generated hypothetical document about the query.", # First call (hypothetical doc)
        "This is the final answer based on retrieved context."      # Second call (final answer)
    ]
    return mock

@pytest.fixture
def hyde_pipeline(mock_iris_connector, mock_embedding_func, mock_llm_for_hyde):
    """Initializes HyDEPipeline with mock dependencies."""
    return HyDEPipeline(
        iris_connector=mock_iris_connector,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_for_hyde 
    )

# --- Unit Tests ---

def test_generate_hypothetical_document(hyde_pipeline, mock_llm_for_hyde):
    """Tests the _generate_hypothetical_document method."""
    query_text = "What is HyDE?"
    # The mock_llm_for_hyde is already configured with side_effect for its first call
    hypo_doc = hyde_pipeline._generate_hypothetical_document(query_text)
    
    mock_llm_for_hyde.assert_any_call(
        f"Write a short, concise passage that directly answers the following question. "
        f"Focus on providing a factual-sounding answer, even if you need to make up plausible details. "
        f"Do not state that you are an AI or that the answer is hypothetical.\n\n"
        f"Question: {query_text}\n\n"
        f"Passage:"
    )
    assert hypo_doc == "This is a generated hypothetical document about the query."

def test_retrieve_documents_flow(hyde_pipeline, mock_embedding_func, mock_iris_connector, mock_llm_for_hyde):
    """Tests the retrieve_documents method flow."""
    query_text = "Test query for HyDE retrieval"
    top_k = 2
    
    # Mock _generate_hypothetical_document to control its output directly for this test
    # and to avoid consuming the first side_effect of mock_llm_for_hyde here.
    hyde_pipeline._generate_hypothetical_document = MagicMock(return_value="Specific hypo doc for this test")

    mock_cursor = mock_iris_connector.cursor.return_value
    
    retrieved_docs = hyde_pipeline.retrieve_documents(query_text, top_k=top_k)

    hyde_pipeline._generate_hypothetical_document.assert_called_once_with(query_text)
    mock_embedding_func.assert_called_once_with(["Specific hypo doc for this test"])
    
    mock_iris_connector.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert f"SELECT TOP {top_k}" in executed_sql
    assert "VECTOR_COSINE(embedding, TO_VECTOR(" in executed_sql
    assert "'DOUBLE', 768" in executed_sql
    assert "FROM RAG.SourceDocuments" in executed_sql
    
    mock_cursor.fetchall.assert_called_once()
    assert len(retrieved_docs) == 2
    assert retrieved_docs[0].id == "retrieved_doc1"

def test_generate_final_answer(hyde_pipeline, mock_llm_for_hyde):
    """Tests the generate_answer method for the final answer."""
    query_text = "Final answer query"
    retrieved_docs = [Document(id="d1", content="Content1"), Document(id="d2", content="Content2")]
    
    # Reset mock_llm_for_hyde for this specific test if it was called by _generate_hypothetical_document
    # or ensure its side_effect is set for the *second* type of call.
    # For this test, we assume mock_llm_for_hyde is fresh or its side_effect list is managed.
    # The fixture mock_llm_for_hyde is set up with a list of side_effects.
    # The first call (hypo doc) is made by retrieve_documents.
    # This call to generate_answer should trigger the *second* side_effect.
    
    # To be safe, let's ensure the mock_llm_for_hyde's call count is reset if needed,
    # or rely on the fixture providing a fresh mock for each test function.
    # Pytest fixtures are typically instantiated per test function unless session/module scoped.
    # This mock_llm_for_hyde is function-scoped.

    # If _generate_hypothetical_document was called by another part of the test setup using the same mock instance,
    # we need to account for that. Here, we are testing generate_answer in isolation.
    # Let's assume the mock_llm_for_hyde is "ready" for its second type of call.
    # To make it explicit, we can advance its side_effect if it were a shared instance,
    # but for a function-scoped fixture, it's simpler.

    # The mock_llm_for_hyde is function scoped, so it's fresh.
    # The first call to it would be for hypothetical doc, second for final answer.
    # We are directly calling generate_answer, so it will be the *first* call to this instance of the mock.
    # This means the side_effect needs to be configured for *this* call.
    # The current mock_llm_for_hyde fixture is designed for the full run.

    # Let's use a dedicated mock for this unit test of generate_answer
    dedicated_llm_mock = MagicMock(return_value="Specific final answer for this test")
    hyde_pipeline.llm_func = dedicated_llm_mock # Temporarily override

    answer = hyde_pipeline.generate_answer(query_text, retrieved_docs)

    expected_context = "Content1\n\nContent2"
    expected_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{expected_context}

Question: {query_text}

Answer:"""
    dedicated_llm_mock.assert_called_once_with(expected_prompt)
    assert answer == "Specific final answer for this test"


def test_hyde_pipeline_run_orchestration(hyde_pipeline, mock_llm_for_hyde, mock_embedding_func, mock_iris_connector):
    """Tests the full run method orchestration."""
    query_text = "Run HyDE query"
    
    # mock_llm_for_hyde is already set up with side_effect for two calls.
    # 1st call: in _generate_hypothetical_document (via retrieve_documents)
    # 2nd call: in generate_answer

    result = hyde_pipeline.run(query_text, top_k=2)

    assert mock_llm_for_hyde.call_count == 2
    
    # Check first call (hypothetical document generation)
    hypo_doc_prompt_args = mock_llm_for_hyde.call_args_list[0][0]
    assert f"Question: {query_text}" in hypo_doc_prompt_args[0]
    
    # Check embedding of hypothetical document
    mock_embedding_func.assert_called_with(["This is a generated hypothetical document about the query."])

    # Check database query
    mock_iris_connector.cursor.return_value.execute.assert_called()
    
    # Check second call (final answer generation)
    final_answer_prompt_args = mock_llm_for_hyde.call_args_list[1][0]
    assert f"Question: {query_text}" in final_answer_prompt_args[0]
    assert "Actual content from DB for doc 1" in final_answer_prompt_args[0] # Context from mock DB

    assert result["query"] == query_text
    assert result["answer"] == "This is the final answer based on retrieved context."
    assert len(result["retrieved_documents"]) == 2
    assert result["retrieved_documents"][0].id == "retrieved_doc1"
