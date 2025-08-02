"""
Test suite for ColBERT V2 restoration in ColBERTRAGPipeline.

These tests are designed to follow the TDD (Test-Driven Development) approach,
meaning they will initially fail and guide the implementation of the
ColBERT V2 hybrid retrieval logic as specified in:
- specs/COLBERT_OPTIMIZATION_SPECIFICATION_REVISED.md
- specs/COLBERT_V2_RESTORATION_TDD_TESTS.md
"""
import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock, call

from iris_rag.core.models import Document
from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

# Placeholder for actual ColBERTRAGPipeline if it needs to be imported
# from iris_rag.pipelines.colbert import ColBERTRAGPipeline

@pytest.fixture
def mock_config_manager():
    """Fixture for a mocked ConfigurationManager."""
    mock_mgr = MagicMock(spec=ConfigurationManager)
    
    # Mock the get method to return appropriate values based on key
    def mock_get(key, default=None):
        config_map = {
            "pipelines:colbert:num_candidates": 30,
            "pipelines:colbert:max_query_length": 32,
            "pipelines:colbert:doc_maxlen": 180,
            "pipelines:colbert:token_embedding_dimension": 384,
            "embeddings:dimension": 128
        }
        return config_map.get(key, default)
    
    mock_mgr.get.side_effect = mock_get
    
    # Mock get_pipeline_config method that the current implementation might use
    def mock_get_pipeline_config(pipeline_name):
        if pipeline_name == 'colbert':
            return {
                "num_candidates": 30,
                "max_query_length": 32,
                "doc_maxlen": 180,
                "token_embedding_dimension": 384
            }
        return {}
    
    mock_mgr.get_pipeline_config = mock_get_pipeline_config
    return mock_mgr

@pytest.fixture
def mock_connection_manager():
    """Fixture for a mocked ConnectionManager."""
    mock_conn_mgr = MagicMock(spec=ConnectionManager)
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn_mgr.get_connection.return_value = mock_conn
    return mock_conn_mgr, mock_cursor # Return cursor for direct mocking

@pytest.fixture
def sample_query_token_embeddings():
    """Fixture for sample query token embeddings."""
    # Example: 3 tokens, 128 dimensions each
    return np.random.rand(3, 128).astype(np.float32)

@pytest.fixture
def colbert_rag_pipeline_instance(mock_config_manager, mock_connection_manager):
    """Fixture for a ColBERTRAGPipeline instance with mocked dependencies."""
    conn_mgr, _ = mock_connection_manager
    
    # Mock embedding function that returns proper dimensions
    mock_embedding_func = MagicMock()
    # Document-level embedding should return 384D list of floats (not numpy array)
    mock_embedding_func.return_value = [0.1] * 384  # 384D document embedding
    
    # Mock ColBERT query encoder for token embeddings (768D)
    mock_colbert_encoder = MagicMock()
    mock_colbert_encoder.return_value = np.random.rand(5, 768).astype(np.float32)  # 5 tokens, 768D each

    # Mock vector store
    mock_vector_store = MagicMock()

    pipeline = ColBERTRAGPipeline(
        connection_manager=conn_mgr,
        config_manager=mock_config_manager,
        embedding_func=mock_embedding_func,
        colbert_query_encoder=mock_colbert_encoder,
        llm_func=MagicMock(), # Not used in retrieval tests
        vector_store=mock_vector_store
    )
    return pipeline

# --- Test Cases ---

def test_candidate_document_retrieval(colbert_rag_pipeline_instance, mock_connection_manager, sample_query_token_embeddings):
    """
    Verify that the first stage (document-level HNSW search on RAG.SourceDocuments)
    correctly retrieves a small set of relevant candidate doc_ids.
    """
    pipeline = colbert_rag_pipeline_instance
    _, mock_cursor = mock_connection_manager
    query_text = "sample query for candidate retrieval"
    
    # Mock the database cursor results for the current SQL-based implementation
    # First mock the count query
    mock_cursor.fetchone.return_value = (100, 100)  # total_docs, docs_with_embeddings
    
    # Then mock the vector search results
    mock_cursor.fetchall.return_value = [
        (101, "Title 1", "Content 1", 0.9),
        (102, "Title 2", "Content 2", 0.85),
        (103, "Title 3", "Content 3", 0.8)
    ]

    # Call the method under test - this uses direct SQL approach
    candidate_docs_ids = pipeline._retrieve_candidate_documents_hnsw(query_text, k=3)

    # Assertions
    assert len(candidate_docs_ids) == 3
    assert set(candidate_docs_ids) == {101, 102, 103}
    
    # Verify SQL was executed (the current implementation uses direct SQL)
    assert mock_cursor.execute.call_count >= 1  # At least one SQL call
    assert mock_cursor.fetchone.called  # Count query was called
    assert mock_cursor.fetchall.called  # Vector search was called


def test_selective_token_embedding_loading(colbert_rag_pipeline_instance, mock_connection_manager):
    """
    Given a list of candidate doc_ids, verify that token embeddings are loaded
    *only* for these specific documents from RAG.DocumentTokenEmbeddings.
    """
    pipeline = colbert_rag_pipeline_instance
    _, mock_cursor = mock_connection_manager
    candidate_doc_ids = [101, 102, 105]

    # Mock the token embedding fetch result
    # (doc_id, token_index, token_embedding)
    # Note: The test expects the token_embedding to be a VECTOR string, not bytes
    mock_cursor.fetchall.return_value = [
        (101, 0, "[0.1,0.2,0.3,0.4]"),  # Simple vector string format
        (101, 1, "[0.5,0.6,0.7,0.8]"),
        (102, 0, "[0.9,1.0,1.1,1.2]"),
        # No embeddings for 105 to test partial results
    ]

    doc_embeddings_map = pipeline._load_token_embeddings_for_candidates(candidate_doc_ids)

    # Assertions - account for schema setup call + actual query
    assert mock_cursor.execute.call_count >= 1  # At least one call for the token query
    
    # Find the token embedding query call (not the schema setup)
    token_query_call = None
    for call in mock_cursor.execute.call_args_list:
        if "DocumentTokenEmbeddings" in str(call):
            token_query_call = call
            break
    
    assert token_query_call is not None, "Token embedding query should have been called"
    args, _ = token_query_call
    sql_query = args[0]
    assert "RAG.DocumentTokenEmbeddings" in sql_query
    assert "WHERE doc_id IN" in sql_query  # Check for IN clause
    assert 101 in doc_embeddings_map
    assert 102 in doc_embeddings_map
    assert 105 not in doc_embeddings_map  # No embeddings for 105
    assert len(doc_embeddings_map[101]) == 2  # 2 tokens for doc 101
    assert len(doc_embeddings_map[102]) == 1  # 1 token for doc 102


def test_maxsim_reranking_on_candidates(colbert_rag_pipeline_instance, sample_query_token_embeddings):
    """
    Verify that the _calculate_maxsim_score is applied correctly to the
    selectively loaded token embeddings of candidate documents.
    """
    pipeline = colbert_rag_pipeline_instance
    query_tokens_embeddings = sample_query_token_embeddings # (Q_len, dim)

    # Mock candidate document token embeddings: {doc_id: np.array(num_tokens, dim)}
    doc_embeddings_map = {
        201: np.random.rand(10, 128).astype(np.float32), # Doc 201, 10 tokens
        202: np.random.rand(5, 128).astype(np.float32),  # Doc 202, 5 tokens
        203: np.random.rand(15, 128).astype(np.float32)  # Doc 203, 15 tokens
    }
    
    # Mock documents (simplified) that would be passed to _calculate_maxsim_score
    # This structure depends on how _calculate_maxsim_score will be integrated
    candidate_documents_with_embeddings = [
        Document(page_content="doc 201 content", metadata={"doc_id": 201}, id="201"),
        Document(page_content="doc 202 content", metadata={"doc_id": 202}, id="202"),
        Document(page_content="doc 203 content", metadata={"doc_id": 203}, id="203"),
    ]

    # This method is expected to be part of ColBERTRAGPipeline
    # For TDD, we assume it exists and will be implemented.
    # It might take query_tokens_embeddings and the map, or individual docs.
    # Let's assume it takes query embeddings and a single document's token embeddings.
    
    # To make it fail for TDD, we can assert that a call to a non-existent or
    # differently-behaving _calculate_maxsim_score happens, or that the method itself
    # is not yet implemented in the way we expect for V2.

    with patch.object(pipeline, '_calculate_maxsim_score', side_effect=NotImplementedError("MaxSim calculation for V2 not implemented")) as mock_maxsim:
        try:
            # This part of the test simulates how the pipeline would use _calculate_maxsim_score
            # The actual call might be inside a larger reranking method.
            # For now, we'll just check if it would be called.
            # pipeline._rerank_with_maxsim(query_tokens_embeddings, candidate_documents_with_embeddings)
            # This test focuses on _calculate_maxsim_score itself.
            # Let's assume it's called for each document.
            for doc_id, embeddings in doc_embeddings_map.items():
                pipeline._calculate_maxsim_score(query_tokens_embeddings, embeddings) # This will raise NotImplementedError
        except NotImplementedError:
            pass # Expected for TDD
        
        # assert mock_maxsim.call_count == len(doc_embeddings_map)
        # For a more direct test of _calculate_maxsim_score, if it were public/static:
        # score1 = pipeline._calculate_maxsim_score(query_tokens_embeddings, doc_embeddings_map[201])
        # score2 = pipeline._calculate_maxsim_score(query_tokens_embeddings, doc_embeddings_map[202])
        # assert isinstance(score1, float)
        # Add assertions for expected score ranges or relative scores if logic was known
    
    # Test the _calculate_maxsim_score method directly
    # Remove the mock and test the actual implementation
    for doc_id, embeddings in doc_embeddings_map.items():
        score = pipeline._calculate_maxsim_score(query_tokens_embeddings, embeddings)
        assert isinstance(score, float)
        assert score >= 0.0  # MaxSim scores should be non-negative
    
    # Test that different documents get different scores (with high probability)
    scores = []
    for doc_id, embeddings in doc_embeddings_map.items():
        score = pipeline._calculate_maxsim_score(query_tokens_embeddings, embeddings)
        scores.append(score)
    
    # With random embeddings, scores should likely be different
    assert len(set(scores)) > 1 or len(scores) == 1  # Allow for edge case of single document


@patch('iris_rag.pipelines.colbert.ColBERTRAGPipeline._load_token_embeddings_for_candidates')
@patch('iris_rag.pipelines.colbert.ColBERTRAGPipeline._calculate_maxsim_score') # Or a reranking method
def test_end_to_end_colbert_v2_retrieval_logic(
    mock_calc_maxsim, mock_load_tokens,
    colbert_rag_pipeline_instance, sample_query_token_embeddings
):
    """
    Test the refactored _retrieve_documents_with_colbert method end-to-end
    (with mocks for DB interactions and internal methods).
    """
    pipeline = colbert_rag_pipeline_instance
    query_text = "sample end-to-end query"
    
    # Mock pipeline's colbert_query_encoder for query tokenization
    pipeline.colbert_query_encoder.return_value = sample_query_token_embeddings

    # Stage 1: Mock vector store for candidate document retrieval
    from langchain_core.documents import Document
    mock_candidate_docs = [
        (Document(page_content="Content 101", metadata={"title": "Title 101"}, id="101"), 0.95),
        (Document(page_content="Content 102", metadata={"title": "Title 102"}, id="102"), 0.85)
    ]
    pipeline.vector_store.similarity_search_by_embedding.return_value = mock_candidate_docs

    # Stage 2: Selective token embedding loading
    mock_doc_embeddings_map = {
        101: np.random.rand(10, 768).astype(np.float32),  # 768D for ColBERT token embeddings
        102: np.random.rand(5, 768).astype(np.float32)   # 768D for ColBERT token embeddings
    }
    # This mock needs to return a structure that _retrieve_documents_with_colbert expects
    # e.g., a list of Document objects with their token embeddings pre-loaded, or the map
    mock_load_tokens.return_value = mock_doc_embeddings_map # Or Document objects

    # Stage 3: MaxSim calculation and reranking
    # Mock _calculate_maxsim_score to return predictable scores
    # If _calculate_maxsim_score is called per document:
    mock_calc_maxsim.side_effect = [0.95, 0.85] # Score for doc 101, then for doc 102

    # Mock the part that fetches full document content for the final Document objects
    # This might be part of _load_token_embeddings_for_candidates or a separate step
    # For simplicity, assume _retrieve_documents_with_colbert handles this.
    # We need to mock the DB call that gets content for doc_ids 101, 102
    mock_conn_mgr, mock_cursor = pipeline.connection_manager, pipeline.connection_manager.get_connection().cursor()
    
    # This mock is for fetching Document objects by their IDs after reranking
    # to populate content, title etc.
    def mock_fetch_docs_by_ids_side_effect(doc_ids_list, table_name="RAG.Documents"):
        docs = []
        if 101 in doc_ids_list:
            docs.append(Document(page_content="Content for 101", metadata={"title": "Title 101"}, id="101"))
        if 102 in doc_ids_list:
            docs.append(Document(page_content="Content for 102", metadata={"title": "Title 102"}, id="102"))
        # Ensure order matches input if necessary, or handle sorting in test
        return docs
    
    # This assumes a helper method like _fetch_documents_by_ids exists and is called
    # If not, the mock_cursor for the main DB connection needs to be set up.
    # For TDD, let's assume _retrieve_documents_with_colbert will call it.
    with patch.object(pipeline, '_fetch_documents_by_ids', side_effect=mock_fetch_docs_by_ids_side_effect) as mock_fetch_full_docs:
        # Call the method under test with proper parameters
        retrieved_documents = pipeline._retrieve_documents_with_colbert(query_text, sample_query_token_embeddings, top_k=5)

        # Verify vector store was called for candidate retrieval
        pipeline.vector_store.similarity_search_by_embedding.assert_called_once()
        call_args = pipeline.vector_store.similarity_search_by_embedding.call_args
        assert call_args[1]['top_k'] == 30  # Use the configured value
        
        mock_load_tokens.assert_called_once_with([101, 102])
        
        # Assertions for _calculate_maxsim_score calls
        # This depends on how _calculate_maxsim_score is integrated.
        # If it's called inside a loop:
        expected_maxsim_calls = [
            call(sample_query_token_embeddings, mock_doc_embeddings_map[101]),
            call(sample_query_token_embeddings, mock_doc_embeddings_map[102])
        ]
        # mock_calc_maxsim.assert_has_calls(expected_maxsim_calls, any_order=True) # any_order might be true if parallel

        assert len(retrieved_documents) == 2
        
        # Check ranking (doc 101 should be first due to higher mocked score)
        assert retrieved_documents[0].id == "101"
        assert retrieved_documents[1].id == "102"
        
        # Check metadata
        assert retrieved_documents[0].metadata['maxsim_score'] == 0.95
        assert retrieved_documents[0].metadata['retrieval_method'] == 'colbert_v2_hybrid'
        assert retrieved_documents[0].page_content == "Content for 101"

        assert retrieved_documents[1].metadata['maxsim_score'] == 0.85
        assert retrieved_documents[1].metadata['retrieval_method'] == 'colbert_v2_hybrid'
        assert retrieved_documents[1].page_content == "Content for 102"

        mock_fetch_full_docs.assert_called_once() # Check it was called to get full docs


@pytest.mark.slow
def test_performance_colbert_v2_retrieval_logic(colbert_rag_pipeline_instance):
    """
    Measure the execution time of the refactored _retrieve_documents_with_colbert method.
    Initially, this test will use mocks and focus on logic flow.
    Performance assertion can be commented out or set to a high threshold.
    """
    pipeline = colbert_rag_pipeline_instance
    query_text = "sample performance test query"

    # For initial TDD, we can mock heavily and not assert performance yet,
    # or assert that it simply completes.
    # Later, this test would involve more realistic (but still controlled) data.

    # Mock all underlying methods as in the end-to-end test for now
    with patch.object(pipeline, '_retrieve_candidate_documents_hnsw', return_value=[1,2]) as mock_cand, \
         patch.object(pipeline, '_load_token_embeddings_for_candidates', return_value={1: np.array([]), 2: np.array([])}) as mock_load, \
         patch.object(pipeline, '_calculate_maxsim_score', side_effect=[0.9, 0.8]) as mock_maxsim, \
         patch.object(pipeline, '_fetch_documents_by_ids', return_value=[
            Document(page_content="c1", metadata={}, id="1"),
             Document(page_content="c2", metadata={}, id="2")
            ]) as mock_fetch_full:

        start_time = time.time()
        # retrieved_documents = pipeline._retrieve_documents_with_colbert(query_text)
        # For TDD, make it fail if the method isn't there or doesn't run.
        # Call with proper parameters
        query_embeddings = np.random.rand(3, 128).astype(np.float32)
        retrieved_documents = pipeline._retrieve_documents_with_colbert(query_text, query_embeddings, top_k=5)
        
        end_time = time.time()
        execution_time = end_time - start_time

        # print(f"ColBERT V2 retrieval execution time: {execution_time:.4f}s")

        # Initial assertion (can be very lenient or commented out)
        # assert execution_time < 60.0, "Execution time exceeds initial high threshold."
        # Target: < 10s, ideally < 5s
        # assert execution_time < 10.0, "Execution time target not met."

    # Performance test completed successfully
    assert execution_time < 60.0, f"Execution time {execution_time:.4f}s exceeds threshold"
    print(f"ColBERT V2 retrieval execution time: {execution_time:.4f}s")


def test_edge_case_no_candidate_documents(colbert_rag_pipeline_instance):
    """
    Verify behavior when the initial document-level search returns no candidates.
    Assert that the method returns an empty list of documents gracefully.
    """
    pipeline = colbert_rag_pipeline_instance
    query_text = "query that finds no candidates"
    
    # Mock vector store to return no candidates
    pipeline.vector_store.similarity_search_by_embedding.return_value = []
    
    with patch.object(pipeline, '_load_token_embeddings_for_candidates') as mock_load_tokens: # Should not be called

        # Call with proper parameters - should return empty list gracefully
        query_embeddings = np.random.rand(3, 768).astype(np.float32)  # 768D for ColBERT query embeddings
        retrieved_documents = pipeline._retrieve_documents_with_colbert(query_text, query_embeddings, top_k=5)

        # Should return empty list when no candidates found
        assert retrieved_documents == []
        
        # Verify vector store was called for candidate retrieval
        pipeline.vector_store.similarity_search_by_embedding.assert_called_once()
        
        mock_load_tokens.assert_not_called()  # Should not be called if no candidates

        # mock_retrieve_candidates.assert_called_once_with(query_text, k=pipeline.colbert_config["candidate_doc_k"])
        # mock_load_tokens.assert_not_called()
        # assert retrieved_documents == []
    
    # Test passed - the method correctly handled no candidates
    pass


def test_edge_case_no_token_embeddings_for_candidates(colbert_rag_pipeline_instance):
    """
    Verify behavior when candidate documents are found, but they have no
    token embeddings in RAG.DocumentTokenEmbeddings.
    Assert that the method handles this gracefully.
    """
    pipeline = colbert_rag_pipeline_instance
    query_text = "query with candidates but no token embeddings"

    # Mock vector store to return candidates
    from langchain_core.documents import Document
    mock_candidate_docs = [
        (Document(page_content="Content 301", metadata={"title": "Title 301"}, id="301"), 0.95),
        (Document(page_content="Content 302", metadata={"title": "Title 302"}, id="302"), 0.85)
    ]
    pipeline.vector_store.similarity_search_by_embedding.return_value = mock_candidate_docs
    
    with patch.object(pipeline, '_load_token_embeddings_for_candidates', return_value={}) as mock_load_tokens, \
         patch.object(pipeline, '_calculate_maxsim_score') as mock_maxsim, \
         patch.object(pipeline, '_fetch_documents_by_ids', return_value=[]) as mock_fetch_full: # Should not fetch if no valid items after maxsim

        # Call with proper parameters
        query_embeddings = np.random.rand(3, 768).astype(np.float32)  # 768D for ColBERT query embeddings
        retrieved_documents = pipeline._retrieve_documents_with_colbert(query_text, query_embeddings, top_k=5)
        
        # Should return empty list when no token embeddings found
        assert retrieved_documents == []
        
        # Verify vector store was called for candidate retrieval
        pipeline.vector_store.similarity_search_by_embedding.assert_called_once()
        
        mock_load_tokens.assert_called_once_with([301, 302])
        mock_maxsim.assert_not_called()  # Should not be called if no embeddings
        mock_fetch_full.assert_not_called()  # Should not fetch if no valid items after maxsim

        # mock_retrieve_candidates.assert_called_once_with(query_text, k=pipeline.colbert_config["candidate_doc_k"])
        # mock_load_tokens.assert_called_once_with([301, 302])
        # mock_maxsim.assert_not_called() # Or called but results in zero scores / no valid docs
        
        # Depending on design:
        # assert retrieved_documents == []
        # OR documents are returned but with zero scores / specific metadata
        # For now, assume empty list if no token embeddings mean no scoring possible.
        # assert mock_fetch_full.assert_not_called() # If no valid docs after scoring phase

    # Test passed - the method correctly handled no token embeddings
    pass