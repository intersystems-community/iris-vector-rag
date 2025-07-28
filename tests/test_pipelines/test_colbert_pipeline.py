"""
Test suite for the refactored ColBERTRAGPipeline.

These tests verify that the ColBERTRAGPipeline correctly orchestrates the
retrieval process by delegating to the ColBERTRetriever, and that the
end-to-end flow works as expected with mocked components.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from iris_rag.core.models import Document
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from common.iris_connection_manager import IRISConnectionManager as ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.vector_store_iris import IRISVectorStore

# A concrete implementation of the abstract pipeline for testing purposes
class ConcreteTestColBERTRAGPipeline(ColBERTRAGPipeline):
    """A concrete ColBERT pipeline for testing with mocked encoders."""
    def __init__(self, connection_manager, config_manager, vector_store):
        # Mock external dependencies for initialization
        mock_colbert_query_encoder = MagicMock(return_value=np.random.rand(5, 128).astype(np.float32))
        mock_doc_embedding_func = MagicMock(return_value=[0.1] * 384)
        mock_llm_func = MagicMock(return_value="This is a mock answer.")

        # Patch schema manager during init to avoid DB calls
        with patch('iris_rag.storage.schema_manager.SchemaManager') as mock_schema_manager:
            # doc_dim, token_dim
            mock_schema_manager.return_value.get_vector_dimension.side_effect = [384, 128]
            super().__init__(
                iris_connector=connection_manager,
                config_manager=config_manager,
                colbert_query_encoder=mock_colbert_query_encoder,
                llm_func=mock_llm_func,
                vector_store=vector_store
            )
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Mock implementation of load_documents for testing."""
        pass
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> list:
        """Mock implementation of query for testing."""
        return []

@pytest.fixture
def mock_config_manager():
    """Fixture for a mocked ConfigurationManager."""
    mock_mgr = MagicMock(spec=ConfigurationManager)
    config_map = {
        "pipelines:colbert:num_candidates": 30,
    }
    mock_mgr.get.side_effect = lambda key, default=None: config_map.get(key, default)
    return mock_mgr

@pytest.fixture
def mock_connection_manager():
    """Fixture for a mocked ConnectionManager."""
    mock_conn_mgr = MagicMock(spec=ConnectionManager)
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn_mgr.get_connection.return_value = mock_conn
    return mock_conn_mgr

@pytest.fixture
def mock_vector_store():
    """Fixture for a mocked IRISVectorStore."""
    return MagicMock(spec=IRISVectorStore)

@pytest.fixture
def pipeline(mock_connection_manager, mock_config_manager, mock_vector_store):
    """Fixture for a concrete ColBERTRAGPipeline instance."""
    return ConcreteTestColBERTRAGPipeline(
        connection_manager=mock_connection_manager,
        config_manager=mock_config_manager,
        vector_store=mock_vector_store
    )

# --- Test Cases ---

def test_pipeline_initialization(pipeline):
    """Verify that the pipeline and its retriever are initialized correctly."""
    assert pipeline is not None
    assert hasattr(pipeline, 'retriever')
    assert pipeline.doc_embedding_dim == 384
    assert pipeline.token_embedding_dim == 128

@patch('iris_rag.pipelines.colbert.retriever.ColBERTRetriever._retrieve_documents_with_colbert')
def test_run_successful_retrieval(mock_retrieve_docs, pipeline):
    """Test the full `run` method with a successful retrieval and answer generation."""
    query = "What is the capital of France?"
    mock_retrieved = [Document(page_content="Paris is the capital of France.", id="1")]
    mock_retrieve_docs.return_value = mock_retrieved

    result = pipeline.run(query)

    mock_retrieve_docs.assert_called_once()
    pipeline.llm_func.assert_called_once()
    
    assert result['query'] == query
    assert result['answer'] == "This is a mock answer."
    assert result['retrieved_documents'] == mock_retrieved
    assert 'execution_time' in result

@patch('iris_rag.pipelines.colbert.retriever.ColBERTRetriever._retrieve_documents_with_colbert')
def test_run_no_documents_found(mock_retrieve_docs, pipeline):
    """Test the `run` method when no documents are retrieved."""
    query = "A query that finds nothing."
    mock_retrieve_docs.return_value = []

    result = pipeline.run(query)

    mock_retrieve_docs.assert_called_once()
    # Should not generate answer if no docs, but return a message
    pipeline.llm_func.assert_not_called()

    assert result['query'] == query
    assert result['answer'] == "No relevant documents found to answer the query."
    assert result['retrieved_documents'] == []

@patch('iris_rag.pipelines.colbert.pipeline.ColBERTRAGPipeline.validate_setup', return_value=False)
@patch('iris_rag.pipelines.colbert.retriever.ColBERTRetriever._fallback_to_basic_retrieval')
def test_run_colbert_failure_fallback(mock_fallback, mock_validate, pipeline):
    """Test that the pipeline falls back to basic retrieval if ColBERT setup is invalid."""
    query = "Test fallback mechanism."
    fallback_docs = [Document(page_content="Fallback document.", id="100")]
    mock_fallback.return_value = fallback_docs

    # Make the main retrieval path fail by having the encoder return nothing
    pipeline.colbert_query_encoder.return_value = []

    result = pipeline.run(query)

    mock_fallback.assert_called_once_with(query, 5)
    assert result['technique'] == "ColBERT (fallback)"
    assert result['retrieved_documents'] == fallback_docs

def test_retriever_e2e_flow(pipeline):
    """
    Test the retriever's end-to-end logic by mocking DB interactions
    but letting the retriever's own logic execute.
    """
    query = "e2e test query"
    query_token_embeddings = np.random.rand(5, 128).astype(np.float32)
    pipeline.colbert_query_encoder.return_value = query_token_embeddings

    # Mock the retriever's methods that interact with the database
    with patch.object(pipeline.retriever, 'vector_store') as mock_vs, \
         patch.object(pipeline.retriever, '_load_token_embeddings_for_candidates') as mock_load_tokens, \
         patch.object(pipeline.retriever, '_fetch_documents_by_ids') as mock_fetch_docs:

        # Stage 1: Candidate retrieval
        mock_vs.similarity_search_by_embedding.return_value = [
            (Document(page_content="Test content 101", id="101"), 0.9),
            (Document(page_content="Test content 102", id="102"), 0.8)
        ]

        # Stage 2: Token loading
        mock_load_tokens.return_value = {
            101: np.random.rand(10, 128).astype(np.float32),
            102: np.random.rand(15, 128).astype(np.float32)
        }

        # Stage 3: Fetching full documents after reranking
        def fetch_side_effect(doc_ids, **kwargs):
            docs = []
            # Ensure we check against integer IDs as in retriever logic
            if 101 in doc_ids:
                docs.append(Document(page_content="Content 101", id="101"))
            if 102 in doc_ids:
                docs.append(Document(page_content="Content 102", id="102"))
            
            # Reorder docs to match the input `doc_ids` order
            id_map = {int(d.id): d for d in docs}
            return [id_map[doc_id] for doc_id in doc_ids if doc_id in id_map]

        mock_fetch_docs.side_effect = fetch_side_effect

        # Execute the method under test
        retrieved_docs = pipeline.retriever._retrieve_documents_with_colbert(
            query_text=query,
            query_token_embeddings=query_token_embeddings,
            top_k=2
        )

        # Assertions
        mock_vs.similarity_search_by_embedding.assert_called_once()
        mock_load_tokens.assert_called_once_with([101, 102])
        mock_fetch_docs.assert_called_once()

        assert len(retrieved_docs) == 2
        assert retrieved_docs[0].id in ["101", "102"]
        assert 'maxsim_score' in retrieved_docs[0].metadata
        assert retrieved_docs[0].metadata['retrieval_method'] == 'colbert_v2_hybrid'