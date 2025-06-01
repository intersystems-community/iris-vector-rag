# tests/test_crag.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
# import sqlalchemy # No longer needed
from typing import Any # For mock type hints

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experimental.crag.pipeline import CRAGPipeline, RetrievalStatus, RetrievalEvaluator # Updated import
from src.common.utils import Document # Updated import

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
    
    # Default fetchall return for initial retrieval
    # Explicitly create fetchall as a MagicMock and set its return_value
    mock_cursor_instance.fetchall = MagicMock(return_value=[
        ("initial_doc1", "Initial content 1 (score 0.9)", 0.9),
        ("initial_doc2", "Initial content 2 (score 0.6)", 0.6),
    ])
    mock_cursor_instance.execute = MagicMock()
    mock_cursor_instance.close = MagicMock()
    mock_conn.close = MagicMock()
    return mock_conn

@pytest.fixture
def mock_embedding_func():
    """Mocks the embedding function."""
    return MagicMock(return_value=[[0.1]*384]) # Returns a single embedding

@pytest.fixture
def mock_llm_func():
    """Mocks the LLM function."""
    return MagicMock(return_value="Mocked CRAG LLM answer.")

@pytest.fixture
def mock_web_search_func():
    """Mocks the web search function."""
    return MagicMock(return_value=["Web result A", "Web result B"])

@pytest.fixture
def mock_retrieval_evaluator():
    """Mocks the RetrievalEvaluator."""
    mock_evaluator = MagicMock(spec=RetrievalEvaluator)
    # Default evaluation result
    mock_evaluator.evaluate.return_value = "confident"
    return mock_evaluator


@pytest.fixture
def crag_pipeline(mock_iris_connector, mock_embedding_func, mock_llm_func, mock_web_search_func, mock_retrieval_evaluator):
    """Initializes CRAGPipeline with mock dependencies."""
    pipeline = CRAGPipeline(
        iris_connector=mock_iris_connector,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func,
        web_search_func=mock_web_search_func
    )
    # Replace the pipeline's internal evaluator with our mock
    pipeline.retrieval_evaluator = mock_retrieval_evaluator
    return pipeline

# --- Unit Tests ---

def test_retrieval_evaluator_logic():
    """Tests the placeholder RetrievalEvaluator logic."""
    evaluator = RetrievalEvaluator() # Use real evaluator for this test

    # Test case 1: No documents
    assert evaluator.evaluate("query", []) == "disoriented"

    # Test case 2: Documents with high scores (confident)
    docs_high_score = [Document(id="d1", content="c1", score=0.9), Document(id="d2", content="c2", score=0.85)]
    assert evaluator.evaluate("query", docs_high_score) == "confident"

    # Test case 3: Documents with medium scores (ambiguous)
    docs_med_score = [Document(id="d3", content="c3", score=0.6), Document(id="d4", content="c4", score=0.7)]
    assert evaluator.evaluate("query", docs_med_score) == "ambiguous"

    # Test case 4: Documents with low scores (disoriented)
    docs_low_score = [Document(id="d5", content="c5", score=0.4), Document(id="d6", content="c6", score=0.3)]
    assert evaluator.evaluate("query", docs_low_score) == "disoriented"
    
    # Test case 5: Documents with mixed scores (average matters)
    docs_mixed_score = [Document(id="d7", content="c7", score=0.9), Document(id="d8", content="c8", score=0.3)] # Avg = 0.6
    assert evaluator.evaluate("query", docs_mixed_score) == "ambiguous"

    # Test case 6: Documents with no scores
    docs_no_score = [Document(id="d9", content="c9"), Document(id="d10", content="c10")]
    assert evaluator.evaluate("query", docs_no_score) == "disoriented" # Sum of None scores is 0

def test_initial_retrieve(crag_pipeline, mock_iris_connector, mock_embedding_func):
    """Tests the _initial_retrieve method (delegates to BasicRAG-like logic)."""
    query_text = "Initial retrieve query"
    top_k = 3
    
    mock_cursor = mock_iris_connector.cursor.return_value
    
    retrieved_docs = crag_pipeline._initial_retrieve(query_text, top_k=top_k)

    mock_embedding_func.assert_called_once_with([query_text])
    mock_iris_connector.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert f"SELECT TOP {top_k}" in executed_sql
    assert "VECTOR_COSINE(embedding, TO_VECTOR(" in executed_sql
    assert "'DOUBLE', 768" in executed_sql # Assuming 768 from pipeline.py
    assert "FROM RAG.SourceDocuments_V2" in executed_sql
    
    mock_cursor.fetchall.assert_called_once()
    assert len(retrieved_docs) == 2 # Based on mock_iris_connector default
    assert retrieved_docs[0].id == "initial_doc1"

def test_augment_with_web_search(crag_pipeline, mock_web_search_func):
    """Tests the _augment_with_web_search method."""
    query_text = "Web search query"
    initial_docs = [Document(id="d1", content="Initial content")]
    web_top_k = 2

    augmented_docs = crag_pipeline._augment_with_web_search(query_text, initial_docs, web_top_k)

    mock_web_search_func.assert_called_once_with(query_text, num_results=web_top_k)
    assert len(augmented_docs) == len(initial_docs) + web_top_k
    assert augmented_docs[0].id == "d1"
    assert augmented_docs[1].id == "web_0"
    assert augmented_docs[1].content == "Web result A"
    assert augmented_docs[2].id == "web_1"
    assert augmented_docs[2].content == "Web result B"

def test_augment_with_web_search_no_func(crag_pipeline, mock_web_search_func):
    """Tests _augment_with_web_search when web_search_func is None."""
    crag_pipeline.web_search_func = None # Remove web search capability
    query_text = "No web search query"
    initial_docs = [Document(id="d1", content="Initial content")]
    web_top_k = 2

    augmented_docs = crag_pipeline._augment_with_web_search(query_text, initial_docs, web_top_k)

    mock_web_search_func.assert_not_called()
    assert augmented_docs == initial_docs # Should return original docs

def test_decompose_recompose_filter(crag_pipeline):
    """Tests the placeholder _decompose_recompose_filter logic."""
    query_text = "diabetes treatment"
    documents = [
        Document(id="d1", content="This document discusses diabetes treatments."), # Relevant
        Document(id="d2", content="Information about cancer research."), # Not relevant
        Document(id="d3", content="Another document on diabetes management.", score=0.8), # Relevant
        Document(id="d4", content="General health tips."), # Not relevant
    ]
    
    # The placeholder filter checks for keywords.
    # "diabetes" and "treatment" are keywords.
    
    relevant_chunks = crag_pipeline._decompose_recompose_filter(query_text, documents)

    # Expecting content of d1 and d3 as chunks
    assert len(relevant_chunks) == 2
    assert documents[0].content in relevant_chunks
    assert documents[2].content in relevant_chunks
    assert documents[1].content not in relevant_chunks
    assert documents[3].content not in relevant_chunks

def test_retrieve_and_correct_confident(crag_pipeline, mock_retrieval_evaluator, mock_web_search_func):
    """Tests retrieve_and_correct when status is confident."""
    query_text = "Confident query"
    initial_docs = [Document(id="d1", content="c1", score=0.9)]
    
    # Configure evaluator to return confident
    mock_retrieval_evaluator.evaluate.return_value = "confident"
    
    # Mock _initial_retrieve and _decompose_recompose_filter to control their output
    crag_pipeline._initial_retrieve = MagicMock(return_value=initial_docs)
    crag_pipeline._decompose_recompose_filter = MagicMock(return_value=["Refined chunk 1"])

    refined_context = crag_pipeline.retrieve_and_correct(query_text)

    crag_pipeline._initial_retrieve.assert_called_once_with(query_text, 5) # Default top_k
    mock_retrieval_evaluator.evaluate.assert_called_once_with(query_text, initial_docs)
    mock_web_search_func.assert_not_called() # Web search should NOT be called
    crag_pipeline._decompose_recompose_filter.assert_called_once_with(query_text, initial_docs) # Called with initial docs
    assert refined_context == ["Refined chunk 1"]

def test_retrieve_and_correct_ambiguous(crag_pipeline, mock_retrieval_evaluator, mock_web_search_func):
    """Tests retrieve_and_correct when status is ambiguous."""
    query_text = "Ambiguous query"
    initial_docs = [Document(id="d1", content="c1", score=0.6)]
    augmented_docs = initial_docs + [Document(id="web_0", content="web c1")] # Expected augmented docs
    
    # Configure evaluator to return ambiguous
    mock_retrieval_evaluator.evaluate.return_value = "ambiguous"
    
    # Mock sub-methods
    crag_pipeline._initial_retrieve = MagicMock(return_value=initial_docs)
    crag_pipeline._augment_with_web_search = MagicMock(return_value=augmented_docs)
    crag_pipeline._decompose_recompose_filter = MagicMock(return_value=["Refined chunk 2", "Refined chunk 3"])

    refined_context = crag_pipeline.retrieve_and_correct(query_text)

    crag_pipeline._initial_retrieve.assert_called_once_with(query_text, 5)
    mock_retrieval_evaluator.evaluate.assert_called_once_with(query_text, initial_docs)
    crag_pipeline._augment_with_web_search.assert_called_once_with(query_text, initial_docs, 3) # Web search SHOULD be called
    crag_pipeline._decompose_recompose_filter.assert_called_once_with(query_text, augmented_docs) # Called with augmented docs
    assert refined_context == ["Refined chunk 2", "Refined chunk 3"]

def test_retrieve_and_correct_disoriented(crag_pipeline, mock_retrieval_evaluator, mock_web_search_func):
    """Tests retrieve_and_correct when status is disoriented."""
    query_text = "Disoriented query"
    initial_docs = [] # No initial docs
    augmented_docs = [Document(id="web_0", content="web c1")] # Expected augmented docs (only web)
    
    # Configure evaluator to return disoriented
    mock_retrieval_evaluator.evaluate.return_value = "disoriented"
    
    # Mock sub-methods
    crag_pipeline._initial_retrieve = MagicMock(return_value=initial_docs)
    crag_pipeline._augment_with_web_search = MagicMock(return_value=augmented_docs)
    crag_pipeline._decompose_recompose_filter = MagicMock(return_value=["Refined chunk 4"])

    refined_context = crag_pipeline.retrieve_and_correct(query_text)

    crag_pipeline._initial_retrieve.assert_called_once_with(query_text, 5)
    mock_retrieval_evaluator.evaluate.assert_called_once_with(query_text, initial_docs)
    crag_pipeline._augment_with_web_search.assert_called_once_with(query_text, initial_docs, 3) # Web search SHOULD be called
    crag_pipeline._decompose_recompose_filter.assert_called_once_with(query_text, augmented_docs) # Called with augmented docs
    assert refined_context == ["Refined chunk 4"]


def test_generate_answer(crag_pipeline, mock_llm_func):
    """Tests the generate_answer method."""
    query_text = "CRAG final answer query"
    refined_context_list = ["Chunk 1 content.", "Chunk 2 content."]
    
    answer = crag_pipeline.generate_answer(query_text, refined_context_list)

    expected_context = "Chunk 1 content.\n\nChunk 2 content."
    expected_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{expected_context}

Question: {query_text}

Answer:"""
    mock_llm_func.assert_called_once_with(expected_prompt)
    assert answer == "Mocked CRAG LLM answer."

def test_run_orchestration(crag_pipeline, mock_retrieval_evaluator, mock_llm_func):
    """Tests the full run method orchestration."""
    query_text = "Run CRAG query"
    
    # Mock sub-methods to test run orchestration
    crag_pipeline.retrieve_and_correct = MagicMock(return_value=["Final refined chunk"])
    crag_pipeline.generate_answer = MagicMock(return_value="Final CRAG Answer")

    result = crag_pipeline.run(query_text, top_k=10, web_top_k=5) # Use different top_k to test passing args

    crag_pipeline.retrieve_and_correct.assert_called_once_with(query_text, 10, 5)
    crag_pipeline.generate_answer.assert_called_once_with(query_text, ["Final refined chunk"])

    assert result["query"] == query_text
    assert result["answer"] == "Final CRAG Answer"
    assert result["retrieved_context_chunks"] == ["Final refined chunk"]
