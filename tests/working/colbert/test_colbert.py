# tests/test_colbert.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
# import sqlalchemy # No longer needed
import numpy as np
from typing import Any # For mock type hints

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.pipelines.colbert import ColBERTRAGPipeline as ColBERTRAGPipeline
from common.utils import Document # Updated import

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnectionTypes, Cursor as IRISCursorTypes
except ImportError:
    IRISConnectionTypes = Any
    IRISCursorTypes = Any

# --- Mock Fixtures ---

@pytest.fixture
def mock_iris_connector_for_colbert():
    """
    Mock for IRIS connection specifically for ColBERT tests.
    Needs to return token embeddings for fetching.
    """
    mock_conn = MagicMock(spec=IRISConnectionTypes)
    mock_cursor_method = MagicMock()
    mock_conn.cursor = mock_cursor_method
    
    mock_cursor_instance = MagicMock(spec=IRISCursorTypes)
    mock_cursor_method.return_value = mock_cursor_instance
    
    # Mock fetchall to return token embeddings for the fetch_tokens query
    # Format: (doc_id, token_sequence_index, token_embedding_str)
    # Need to simulate data for a few documents
    # Explicitly create fetchall as a MagicMock and set its side_effect
    mock_cursor_instance.fetchall = MagicMock(side_effect=[
        # First fetchall call (fetch_tokens)
        [
            ("doc_colbert_1", 0, str([0.1]*10)),
            ("doc_colbert_1", 1, str([0.2]*10)),
            ("doc_colbert_1", 2, str([0.3]*10)),
            ("doc_colbert_2", 0, str([0.8]*10)),
            ("doc_colbert_2", 1, str([0.7]*10)),
        ],
        # Second fetchall call (fetch_content for top-k)
        [
            ("doc_colbert_1", "Content for doc colbert 1"),
            ("doc_colbert_2", "Content for doc colbert 2"),
        ]
    ])
    
    mock_cursor_instance.execute = MagicMock()
    mock_cursor_instance.close = MagicMock()
    mock_conn.close = MagicMock()
    return mock_conn

@pytest.fixture
def mock_colbert_query_encoder():
    """Mocks the ColBERT query encoder."""
    # Returns a list of mock token embeddings for a query
    return MagicMock(return_value=[[0.1]*10, [0.9]*10]) # Example: 2 query tokens, 10 dim each

@pytest.fixture
def mock_colbert_doc_encoder():
    """Mocks the ColBERT document encoder (used in loader, but needed for pipeline init)."""
    return MagicMock(return_value=[[0.5]*10, [0.6]*10]) # Example: 2 doc tokens, 10 dim each

@pytest.fixture
def mock_llm_func():
    """Mocks the LLM function."""
    return MagicMock(return_value="Mocked ColBERT LLM answer.")


@pytest.fixture
def colbert_rag_pipeline(mock_iris_connector_for_colbert, mock_colbert_query_encoder, mock_colbert_doc_encoder, mock_llm_func):
    """Initializes ColBERTRAGPipeline with mock dependencies."""
    return ColBERTRAGPipeline(
        iris_connector=mock_iris_connector_for_colbert,
        colbert_query_encoder_func=mock_colbert_query_encoder,
        colbert_doc_encoder_func=mock_colbert_doc_encoder,
        llm_func=mock_llm_func
    )

# --- Unit Tests ---

def test_calculate_cosine_similarity():
    """Tests the cosine similarity calculation."""
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    vec3 = [1.0, 1.0]
    vec4 = [-1.0, 0.0]
    
    assert ColBERTRAGPipeline(None, None, None, None)._calculate_cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    assert ColBERTRAGPipeline(None, None, None, None)._calculate_cosine_similarity(vec1, vec1) == pytest.approx(1.0)
    assert ColBERTRAGPipeline(None, None, None, None)._calculate_cosine_similarity(vec1, vec3) == pytest.approx(1.0 / np.sqrt(2))
    assert ColBERTRAGPipeline(None, None, None, None)._calculate_cosine_similarity(vec1, vec4) == pytest.approx(-1.0)
    assert ColBERTRAGPipeline(None, None, None, None)._calculate_cosine_similarity([], []) == 0.0 # Test empty vectors

def test_calculate_maxsim():
    """Tests the MaxSim calculation."""
    pipeline = ColBERTRAGPipeline(None, None, None, None) # No real dependencies needed for this method
    
    query_embeds = [[1.0, 0.0], [0.0, 1.0]] # Query tokens Q1, Q2
    doc_embeds = [[1.0, 0.1], [0.1, 1.0], [0.5, 0.5]] # Doc tokens D1, D2, D3
    
    # Sim(Q1, D1) = cosine([1,0], [1,0.1]) = 1 / sqrt(1+0.01) = 1 / 1.005 = 0.995
    # Sim(Q1, D2) = cosine([1,0], [0.1,1]) = 0.1 / sqrt(1+0.01) = 0.0995
    # Sim(Q1, D3) = cosine([1,0], [0.5,0.5]) = 0.5 / sqrt(0.25+0.25) = 0.5 / sqrt(0.5) = 0.5 / 0.707 = 0.707
    # MaxSim(Q1, Doc) = max(0.995, 0.0995, 0.707) = 0.995
    
    # Sim(Q2, D1) = cosine([0,1], [1,0.1]) = 0.1 / 1.005 = 0.0995
    # Sim(Q2, D2) = cosine([0,1], [0.1,1]) = 1 / 1.005 = 0.995
    # Sim(Q2, D3) = cosine([0,1], [0.5,0.5]) = 0.5 / 0.707 = 0.707
    # MaxSim(Q2, Doc) = max(0.0995, 0.995, 0.707) = 0.995
    
    # Total MaxSim = MaxSim(Q1, Doc) + MaxSim(Q2, Doc) = 0.995 + 0.995 = 1.99
    
    score = pipeline._calculate_maxsim(query_embeds, doc_embeds)
    assert score == pytest.approx(1.99, abs=1e-2) # Allow small floating point error

    assert pipeline._calculate_maxsim([], doc_embeds) == 0.0
    assert pipeline._calculate_maxsim(query_embeds, []) == 0.0
    assert pipeline._calculate_maxsim([], []) == 0.0


def test_retrieve_documents_flow(colbert_rag_pipeline, mock_iris_connector_for_colbert, mock_colbert_query_encoder):
    """Tests the retrieve_documents method flow (client-side MaxSim)."""
    query_text = "Test query for ColBERT retrieval"
    top_k = 2
    
    mock_cursor = mock_iris_connector_for_colbert.cursor() # Call the method to get the cursor instance
    
    # Mock _calculate_maxsim to control scoring logic.
    # The conftest mock_iris_connector_for_colbert is set up for 5 docs.
    # Provide 5 scores for the 5 mock documents.
    mock_maxsim_scores = [0.95, 0.85, 0.75, 0.65, 0.55]
    colbert_rag_pipeline._calculate_maxsim = MagicMock(side_effect=mock_maxsim_scores)

    retrieved_docs = colbert_rag_pipeline.retrieve_documents(query_text, top_k=top_k)

    mock_colbert_query_encoder.assert_called_once_with(query_text)
    
    # Check DB calls
    # The connector's cursor() method is a MagicMock.
    # It's called once by the test, and once by the pipeline.
    mock_iris_connector_for_colbert.cursor.assert_any_call() # Looser check, or assert call_count == 2
    assert mock_iris_connector_for_colbert.cursor.call_count >= 1 # Ensure it was called

    # mock_cursor is now a MagicMock instance from the fixture.
    # Its methods (execute, fetchall, fetchone) are also MagicMocks.
    
    # Expected execute calls:
    # 1 (all_doc_ids) + 5 * (1 for tokens + 1 for content) = 11
    assert mock_cursor.execute.call_count == 11
    
    # Expected fetchall calls:
    # 1 (all_doc_ids) + 5 * (1 for tokens) = 6
    assert mock_cursor.fetchall.call_count == 6

    # Expected fetchone calls:
    # 5 * (1 for content) = 5
    assert mock_cursor.fetchone.call_count == 5

    # Check _calculate_maxsim calls (should be called for each of the 5 mock documents)
    assert colbert_rag_pipeline._calculate_maxsim.call_count == 5
    
    # Check the content of retrieved documents
    assert len(retrieved_docs) == top_k # top_k is 2
    
    # Based on the mock_maxsim_scores, doc_colbert_1 and doc_colbert_2 should be the top 2
    # The mock_iris_connector_for_colbert provides doc_ids as "doc_colbert_1", "doc_colbert_2", etc.
    # and content as "Content for mock ColBERT doc 1.", etc.
    
    assert retrieved_docs[0].id == "doc_colbert_1"
    assert retrieved_docs[0].score == 0.95
    assert "Content for mock ColBERT doc 1" in retrieved_docs[0].content
    
    assert retrieved_docs[1].id == "doc_colbert_2"
    assert retrieved_docs[1].score == 0.85
    assert "Content for mock ColBERT doc 2" in retrieved_docs[1].content


def test_generate_answer(colbert_rag_pipeline, mock_llm_func):
    """Tests the generate_answer method."""
    query_text = "ColBERT final answer query"
    retrieved_docs = [Document(id="d1", content="ContentA"), Document(id="d2", content="ContentB")]
    
    answer = colbert_rag_pipeline.generate_answer(query_text, retrieved_docs)

    expected_context = "ContentA\n\nContentB"
    expected_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{expected_context}

Question: {query_text}

Answer:"""
    mock_llm_func.assert_called_once_with(expected_prompt)
    assert answer == "Mocked ColBERT LLM answer."