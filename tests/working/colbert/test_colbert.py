# tests/test_colbert.py

import pytest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
from typing import Any, List, Callable

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from iris_rag.core.models import Document
from common.iris_connection_manager import get_iris_connection
from iris_rag.config.manager import ConfigurationManager

# --- Concrete Subclass for Testing ---

class TestColBERTRAGPipeline(ColBERTRAGPipeline):
    """
    A concrete subclass of ColBERTRAGPipeline for testing purposes.
    This class allows for mocking dependencies without a full initialization.
    """
    def __init__(self,
                 iris_connector,
                 config_manager: ConfigurationManager,
                 colbert_query_encoder: Callable,
                 llm_func: Callable,
                 embedding_func: Callable,
                 vector_store: Any):
        # Mock superclass initialization to avoid heavy setup
        self.iris_connector = iris_connector
        self.config_manager = config_manager
        self.vector_store = vector_store
        self.doc_embedding_func = embedding_func
        self.colbert_query_encoder = colbert_query_encoder
        self.llm_func = llm_func
        
        # Mock retriever, as it's a key component
        self.retriever = MagicMock()

    def load_documents(self, documents: List[Document]):
        """Mock implementation for abstract method."""
        pass

    def query(self, query_text: str, **kwargs) -> dict:
        """Mock implementation for abstract method."""
        return self.execute(query_text, **kwargs)

# --- Mock Fixtures ---

@pytest.fixture
def mock_connection_manager():
    """Mocks the ConnectionManager."""
    return MagicMock()

@pytest.fixture
def mock_config_manager():
    """Mocks the ConfigurationManager."""
    mock_cm = MagicMock(spec=ConfigurationManager)
    # Mock the get method to return default values for pipeline config
    mock_cm.get.return_value = {
        'llm': {'provider': 'mock'},
        'embeddings': {'provider': 'mock'},
        'colbert': {'query_encoder': 'mock'}
    }
    return mock_cm

@pytest.fixture
def mock_colbert_query_encoder():
    """Mocks the ColBERT query encoder."""
    return MagicMock(return_value=[[0.1]*10, [0.9]*10])

@pytest.fixture
def mock_embedding_func():
    """Mocks the document embedding function."""
    return MagicMock(return_value=[0.5]*10)

@pytest.fixture
def mock_llm_func():
    """Mocks the LLM function."""
    return MagicMock(return_value="Mocked ColBERT LLM answer.")

@pytest.fixture
def mock_vector_store():
    """Mocks the VectorStore."""
    return MagicMock()

@pytest.fixture
def colbert_rag_pipeline(mock_connection_manager, mock_config_manager, mock_colbert_query_encoder, mock_llm_func, mock_embedding_func, mock_vector_store):
    """Initializes TestColBERTRAGPipeline with mock dependencies."""
    return TestColBERTRAGPipeline(
        iris_connector=mock_connection_manager,
        config_manager=mock_config_manager,
        colbert_query_encoder=mock_colbert_query_encoder,
        llm_func=mock_llm_func,
        embedding_func=mock_embedding_func,
        vector_store=mock_vector_store
    )

# --- Unit Tests ---

def test_pipeline_execution_flow(colbert_rag_pipeline, mock_colbert_query_encoder):
    """Tests the main execute method flow of the pipeline."""
    query_text = "Test query for ColBERT retrieval"
    top_k = 2
    
    # Mock the retriever's behavior
    mock_retrieved_docs = [
        Document(id="doc_colbert_1", page_content="Content for doc colbert 1", score=0.95),
        Document(id="doc_colbert_2", page_content="Content for doc colbert 2", score=0.85),
    ]
    colbert_rag_pipeline.retriever._retrieve_documents_with_colbert.return_value = mock_retrieved_docs
    
    # Mock the validation method to always pass
    colbert_rag_pipeline.validate_setup = MagicMock(return_value=True)

    # Execute the pipeline
    result = colbert_rag_pipeline.execute(query_text, top_k=top_k)

    # Assertions
    mock_colbert_query_encoder.assert_called_once_with(query_text)
    
    # Check that the retriever was called correctly
    colbert_rag_pipeline.retriever._retrieve_documents_with_colbert.assert_called_once()
    call_args, call_kwargs = colbert_rag_pipeline.retriever._retrieve_documents_with_colbert.call_args
    assert call_kwargs['query_text'] == query_text
    assert call_kwargs['top_k'] == top_k
    
    # Check the final result
    assert result['query'] == query_text
    assert result['answer'] == "Mocked ColBERT LLM answer."
    assert len(result['retrieved_documents']) == top_k
    assert result['retrieved_documents'][0].id == "doc_colbert_1"
    assert result['retrieved_documents'][1].id == "doc_colbert_2"

def test_generate_answer(colbert_rag_pipeline, mock_llm_func):
    """Tests the _generate_answer method."""
    query_text = "ColBERT final answer query"
    retrieved_docs = [
        Document(id="d1", page_content="ContentA", score=0.9),
        Document(id="d2", page_content="ContentB", score=0.8)
    ]
    
    # We need to call the private method directly for this unit test
    answer = colbert_rag_pipeline._generate_answer(query_text, retrieved_docs)

    # Verify the prompt format
    expected_context = "Document 1: ContentA...\n\nDocument 2: ContentB..."
    expected_prompt = f"""Based on the following documents, please answer the question.

Question: {query_text}

Documents:
{expected_context}

Answer:"""
    mock_llm_func.assert_called_once_with(expected_prompt)
    assert answer == "Mocked ColBERT LLM answer."