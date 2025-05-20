"""
Test all RAG techniques with 1000+ documents.

This test file tests all six RAG techniques with at least 1000 documents,
as required by the project's .clinerules file:
"Tests must use real PMC documents, not synthetic data. At least
1000 documents should be used."

The techniques tested include:
1. BasicRAG
2. HyDE
3. ColBERT
4. NodeRAG
5. GraphRAG
6. CRAG

Each test performs a full pipeline execution and verifies that results
are returned correctly. The tests use fixtures to ensure the database
contains at least 1000 documents.
"""

import pytest
import logging
import time
from typing import Dict, Any, List, Callable
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_all_with_1000_docs")

# Import standard mock functions directly (not using fixtures)
from tests.mocks.models import mock_embedding_func, mock_llm_func

# Import RAG pipelines
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline 
from colbert.pipeline import ColbertRAGPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline import GraphRAGPipeline
from crag.pipeline import CRAGPipeline

# Test constants
TEST_QUERY = "What are the major cardiovascular complications of diabetes?"
EXPECTED_DOC_COUNT = 5  # Minimum number of documents to retrieve


@pytest.mark.requires_1000_docs
def test_basic_rag_with_1000_docs(verify_document_count, iris_with_pmc_data):
    # Create mock functions directly
    embed_mock = MagicMock(side_effect=mock_embedding_func)
    llm_mock = MagicMock(side_effect=mock_llm_func)
    """Test BasicRAG with 1000+ documents."""
    doc_count = verify_document_count
    logger.info(f"Running BasicRAG test with {doc_count} documents")
    
    # Time the retrieval process
    start_time = time.time()
    
    # Run pipeline
    pipeline = BasicRAGPipeline(
        iris_connector=iris_with_pmc_data,  # Use the iris_with_pmc_data fixture
        embedding_func=embed_mock,
        llm_func=llm_mock
    )
    result = pipeline.run(query_text=TEST_QUERY, top_k=EXPECTED_DOC_COUNT)
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    # Log performance
    logger.info(f"BasicRAG retrieval took {retrieval_time:.2f} seconds")
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain a generated answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    
    # Verify correct number of documents retrieved
    assert len(result["retrieved_documents"]) >= EXPECTED_DOC_COUNT, \
        f"Expected at least {EXPECTED_DOC_COUNT} documents, got {len(result['retrieved_documents'])}"
    
    logger.info("BasicRAG with 1000+ documents test passed ✅")


@pytest.mark.requires_1000_docs
def test_hyde_with_1000_docs(verify_document_count, iris_with_pmc_data):
    # Create mock functions directly
    embed_mock = MagicMock(side_effect=mock_embedding_func)
    llm_mock = MagicMock(side_effect=mock_llm_func)
    """Test HyDE with 1000+ documents."""
    doc_count = verify_document_count
    logger.info(f"Running HyDE test with {doc_count} documents")
    
    # Time the retrieval process
    start_time = time.time()
    
    # Run pipeline
    pipeline = HyDEPipeline(
        iris_connector=iris_with_pmc_data,  # Use the iris_with_pmc_data fixture
        embedding_func=embed_mock,
        llm_func=llm_mock
    )
    result = pipeline.run(query_text=TEST_QUERY, top_k=EXPECTED_DOC_COUNT)
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    # Log performance
    logger.info(f"HyDE retrieval took {retrieval_time:.2f} seconds")
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain a generated answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    assert "hypothetical_document" in result, "Result should contain the hypothetical document"
    
    # Verify correct number of documents retrieved
    assert len(result["retrieved_documents"]) >= EXPECTED_DOC_COUNT, \
        f"Expected at least {EXPECTED_DOC_COUNT} documents, got {len(result['retrieved_documents'])}"
    
    logger.info("HyDE with 1000+ documents test passed ✅")


@pytest.mark.requires_1000_docs
def test_colbert_with_1000_docs(verify_document_count, iris_with_pmc_data):
    # Create mock functions directly
    embed_mock = MagicMock(side_effect=mock_embedding_func)
    llm_mock = MagicMock(side_effect=mock_llm_func)
    """Test ColBERT with 1000+ documents."""
    doc_count = verify_document_count
    logger.info(f"Running ColBERT test with {doc_count} documents")
    
    # Time the retrieval process
    start_time = time.time()
    
    # Run pipeline
    pipeline = ColbertRAGPipeline(
        iris_connector=iris_with_pmc_data,  # Use the iris_with_pmc_data fixture
        colbert_query_encoder_func=embed_mock,  # Using mock_embedding_func as placeholder
        colbert_doc_encoder_func=embed_mock,    # Using mock_embedding_func as placeholder
        llm_func=llm_mock
    )
    result = pipeline.run(query_text=TEST_QUERY, top_k=EXPECTED_DOC_COUNT)
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    # Log performance
    logger.info(f"ColBERT retrieval took {retrieval_time:.2f} seconds")
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain a generated answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    # ColBERT pipeline doesn't return token_vectors in the final result
    
    # Verify correct number of documents retrieved
    assert len(result["retrieved_documents"]) >= EXPECTED_DOC_COUNT, \
        f"Expected at least {EXPECTED_DOC_COUNT} documents, got {len(result['retrieved_documents'])}"
    
    logger.info("ColBERT with 1000+ documents test passed ✅")


@pytest.mark.requires_1000_docs
def test_noderag_with_1000_docs(verify_document_count, iris_with_pmc_data):
    # Create mock functions directly
    embed_mock = MagicMock(side_effect=mock_embedding_func)
    llm_mock = MagicMock(side_effect=mock_llm_func)
    """Test NodeRAG with 1000+ documents."""
    doc_count = verify_document_count
    logger.info(f"Running NodeRAG test with {doc_count} documents")
    
    # Time the retrieval process
    start_time = time.time()
    
    # Run pipeline
    pipeline = NodeRAGPipeline(
        iris_connector=iris_with_pmc_data,  # Use the iris_with_pmc_data fixture
        embedding_func=embed_mock,
        llm_func=llm_mock,
        graph_lib=None  # Using None as placeholder for graph library
    )
    result = pipeline.run(query_text=TEST_QUERY, top_k_seeds=EXPECTED_DOC_COUNT)
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    # Log performance
    logger.info(f"NodeRAG retrieval took {retrieval_time:.2f} seconds")
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain a generated answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    # NodeRAG doesn't return separate "nodes" in the API
    
    # Verify correct number of documents retrieved
    assert len(result["retrieved_documents"]) >= 1, "Expected at least 1 document"
    
    logger.info("NodeRAG with 1000+ documents test passed ✅")


@pytest.mark.requires_1000_docs
def test_graphrag_with_1000_docs(verify_document_count, iris_with_pmc_data):
    # Create mock functions directly
    embed_mock = MagicMock(side_effect=mock_embedding_func)
    llm_mock = MagicMock(side_effect=mock_llm_func)
    """Test GraphRAG with 1000+ documents."""
    doc_count = verify_document_count
    logger.info(f"Running GraphRAG test with {doc_count} documents")
    
    # Time the retrieval process
    start_time = time.time()
    
    # Run pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_pmc_data,  # Use the iris_with_pmc_data fixture
        embedding_func=embed_mock,
        llm_func=llm_mock
    )
    result = pipeline.run(query_text=TEST_QUERY, top_n_start_nodes=EXPECTED_DOC_COUNT)
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    # Log performance
    logger.info(f"GraphRAG retrieval took {retrieval_time:.2f} seconds")
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain a generated answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    # GraphRAG doesn't return separate graph_nodes/edges in the API
    
    # Verify correct number of documents retrieved 
    assert len(result["retrieved_documents"]) >= 1, "Expected at least 1 document"
    
    logger.info("GraphRAG with 1000+ documents test passed ✅")


@pytest.mark.requires_1000_docs
def test_crag_with_1000_docs(verify_document_count, iris_with_pmc_data):
    # Create mock functions directly
    embed_mock = MagicMock(side_effect=mock_embedding_func)
    llm_mock = MagicMock(side_effect=mock_llm_func)
    """Test CRAG with 1000+ documents."""
    doc_count = verify_document_count
    logger.info(f"Running CRAG test with {doc_count} documents")
    
    # Time the retrieval process
    start_time = time.time()
    
    # Run pipeline
    pipeline = CRAGPipeline(
        iris_connector=iris_with_pmc_data,  # Use the iris_with_pmc_data fixture
        embedding_func=embed_mock,
        llm_func=llm_mock,
        web_search_func=lambda query: [f"Mock web result for {query}"]  # Using lambda as placeholder for web search
    )
    result = pipeline.run(query_text=TEST_QUERY, top_k=EXPECTED_DOC_COUNT)
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    # Log performance
    logger.info(f"CRAG retrieval took {retrieval_time:.2f} seconds")
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain a generated answer"
    # CRAG returns retrieved_context_chunks instead of retrieved_documents
    assert "retrieved_context_chunks" in result, "Result should contain retrieved context chunks"
    
    # Verify retrieved context chunks
    # CRAG uses retrieved_context_chunks instead of retrieved_documents
    logger.info(f"CRAG found {len(result['retrieved_context_chunks'])} context chunks")
    
    logger.info("CRAG with 1000+ documents test passed ✅")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("This test file is intended to be run with pytest")
    print("Run with: python -m pytest tests/test_all_with_1000_docs.py -v")
