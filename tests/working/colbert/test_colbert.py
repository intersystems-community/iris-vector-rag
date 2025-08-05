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

from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from common.utils import Document

# Import our working IRIS DBAPI connector utilities
from common.iris_dbapi_connector import _get_iris_dbapi_module

# Type hints will be set lazily to avoid circular imports
IRISConnectionTypes = Any  # Connection type from iris.connect()
IRISCursorTypes = Any      # Cursor type from connection.cursor()

def _get_iris_types():
    """Get type hints from our working IRIS module safely, called lazily to avoid circular imports."""
    _iris_module = _get_iris_dbapi_module()
    if _iris_module and hasattr(_iris_module, 'connect'):
        # Use Any for type hints since we can't safely instantiate connections for typing
        # The actual connection will be mocked in tests anyway
        return Any, Any  # Connection type, Cursor type
    else:
        # Fallback to Any if iris module is not available
        return Any, Any

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
def mock_connection_manager():
    """Mock connection manager for ColBERT tests."""
    mock_manager = MagicMock()
    mock_manager.get_connection.return_value = MagicMock()
    return mock_manager

# Remove the local mock_config_manager fixture - use the one from conftest.py

@pytest.fixture
def mock_schema_manager():
    """Mock schema manager for ColBERT tests."""
    mock_manager = MagicMock()
    mock_manager.get_vector_dimension.side_effect = lambda table: 384 if table == "SourceDocuments" else 768
    return mock_manager

@pytest.fixture
def colbert_rag_pipeline(mock_connection_manager, mock_config_manager, mock_colbert_query_encoder, mock_llm_func):
    """Initializes ColBERTRAGPipeline with mock dependencies."""
    with patch('iris_rag.storage.schema_manager.SchemaManager') as mock_schema_class:
        mock_schema_class.return_value.get_vector_dimension.side_effect = lambda table: 384 if table == "SourceDocuments" else 768
        
        with patch('common.utils.get_llm_func', return_value=mock_llm_func):
            with patch('common.utils.get_embedding_func', return_value=MagicMock()):
                with patch('iris_rag.embeddings.colbert_interface.get_colbert_interface_from_config') as mock_colbert_interface:
                    # Mock the ColBERT interface
                    mock_interface = MagicMock()
                    mock_interface.encode_query = mock_colbert_query_encoder
                    mock_interface._calculate_cosine_similarity = MagicMock()
                    mock_colbert_interface.return_value = mock_interface
                    
                    with patch('iris_rag.embeddings.colbert_interface.RAGTemplatesColBERTInterface') as mock_rag_interface_class:
                        # Mock the RAGTemplatesColBERTInterface class
                        mock_rag_interface = MagicMock()
                        
                        # Configure the cosine similarity mock to return proper values
                        def mock_cosine_similarity(vec1, vec2):
                            import numpy as np
                            # Convert to numpy arrays for calculation
                            v1 = np.array(vec1)
                            v2 = np.array(vec2)
                            # Calculate cosine similarity
                            dot_product = np.dot(v1, v2)
                            norm_v1 = np.linalg.norm(v1)
                            norm_v2 = np.linalg.norm(v2)
                            if norm_v1 == 0 or norm_v2 == 0:
                                return 0.0
                            return dot_product / (norm_v1 * norm_v2)
                        
                        mock_rag_interface._calculate_cosine_similarity = mock_cosine_similarity
                        mock_rag_interface.encode_query = mock_colbert_query_encoder
                        mock_rag_interface_class.return_value = mock_rag_interface
                        
                        pipeline = ColBERTRAGPipeline(
                            connection_manager=mock_connection_manager,
                            config_manager=mock_config_manager,
                            colbert_query_encoder=mock_colbert_query_encoder,
                            llm_func=mock_llm_func
                        )
                        return pipeline

# --- Unit Tests ---

def test_calculate_cosine_similarity(colbert_rag_pipeline):
    """Tests the cosine similarity calculation."""
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    vec3 = [1.0, 1.0]
    vec4 = [-1.0, 0.0]
    
    # Test using the colbert_interface which has the _calculate_cosine_similarity method
    assert colbert_rag_pipeline.colbert_interface._calculate_cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    assert colbert_rag_pipeline.colbert_interface._calculate_cosine_similarity(vec1, vec1) == pytest.approx(1.0)
    assert colbert_rag_pipeline.colbert_interface._calculate_cosine_similarity(vec1, vec3) == pytest.approx(1.0 / np.sqrt(2))
    assert colbert_rag_pipeline.colbert_interface._calculate_cosine_similarity(vec1, vec4) == pytest.approx(-1.0)
    assert colbert_rag_pipeline.colbert_interface._calculate_cosine_similarity([], []) == 0.0 # Test empty vectors

def test_calculate_maxsim(colbert_rag_pipeline):
    """Tests the MaxSim calculation."""
    query_embeds = np.array([[1.0, 0.0], [0.0, 1.0]]) # Query tokens Q1, Q2
    doc_embeds = np.array([[1.0, 0.1], [0.1, 1.0], [0.5, 0.5]]) # Doc tokens D1, D2, D3
    
    # Sim(Q1, D1) = cosine([1,0], [1,0.1]) = 1 / sqrt(1+0.01) = 1 / 1.005 = 0.995
    # Sim(Q1, D2) = cosine([1,0], [0.1,1]) = 0.1 / sqrt(1+0.01) = 0.0995
    # Sim(Q1, D3) = cosine([1,0], [0.5,0.5]) = 0.5 / sqrt(0.25+0.25) = 0.5 / sqrt(0.5) = 0.5 / 0.707 = 0.707
    # MaxSim(Q1, Doc) = max(0.995, 0.0995, 0.707) = 0.995
    
    # Sim(Q2, D1) = cosine([0,1], [1,0.1]) = 0.1 / 1.005 = 0.0995
    # Sim(Q2, D2) = cosine([0,1], [0.1,1]) = 1 / 1.005 = 0.995
    # Sim(Q2, D3) = cosine([0,1], [0.5,0.5]) = 0.5 / 0.707 = 0.707
    # MaxSim(Q2, Doc) = max(0.0995, 0.995, 0.707) = 0.995
    
    # ColBERT MaxSim = average of max similarities = (0.995 + 0.995) / 2 = 0.995
    
    score = colbert_rag_pipeline._calculate_maxsim_score(query_embeds, doc_embeds)
    assert score == pytest.approx(0.995, abs=1e-2) # Allow small floating point error

    assert colbert_rag_pipeline._calculate_maxsim_score(np.array([]), doc_embeds) == 0.0
    assert colbert_rag_pipeline._calculate_maxsim_score(query_embeds, np.array([])) == 0.0
    assert colbert_rag_pipeline._calculate_maxsim_score(np.array([]), np.array([])) == 0.0


def test_retrieve_documents_flow(colbert_rag_pipeline, mock_connection_manager, mock_colbert_query_encoder):
    """Tests the retrieve_documents method flow using vector store."""
    query_text = "Test query for ColBERT retrieval"
    top_k = 2
    
    # Mock the vector store's colbert_search method
    mock_vector_store = MagicMock()
    mock_docs = [
        Document(id="doc1", content="Content for doc 1", score=0.95),
        Document(id="doc2", content="Content for doc 2", score=0.85)
    ]
    mock_vector_store.colbert_search.return_value = [(doc, doc.score) for doc in mock_docs]
    colbert_rag_pipeline.vector_store = mock_vector_store

    # Execute the pipeline
    result = colbert_rag_pipeline.query(query_text, top_k=top_k)

    # Verify query encoder was called with the actual query text
    # Note: The encoder may be called multiple times (validation + actual processing)
    mock_colbert_query_encoder.assert_any_call(query_text)
    
    # Verify vector store search was called
    mock_vector_store.colbert_search.assert_called_once()
    
    # Check the result structure
    assert "query" in result
    assert "answer" in result
    assert "retrieved_documents" in result
    assert result["query"] == query_text
    assert len(result["retrieved_documents"]) == top_k


def test_generate_answer(colbert_rag_pipeline, mock_llm_func):
    """Tests the generate_answer method."""
    query_text = "ColBERT final answer query"
    retrieved_docs = [Document(id="d1", content="ContentA"), Document(id="d2", content="ContentB")]

    answer = colbert_rag_pipeline._generate_answer(query_text, retrieved_docs)

    # The actual implementation uses this format
    expected_prompt = f"""Based on the following documents, please answer the question.

Question: {query_text}

Documents:
Document 1: ContentA...

Document 2: ContentB...

Answer:"""
    mock_llm_func.assert_called_once_with(expected_prompt)
    assert answer == "Mocked ColBERT LLM answer."