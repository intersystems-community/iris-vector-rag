"""
Test to validate HyDERAG embedding generation fix.

This test specifically validates that HyDERAG now properly generates embeddings
during document ingestion, fixing the production-blocking issue where documents
were stored without embeddings causing vector search to return 0 results.
"""

import pytest
import logging
from typing import List
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document
from common.iris_connection_manager import get_iris_connection

logger = logging.getLogger(__name__)


@pytest.fixture
def config_manager():
    """Create a configuration manager for testing."""
    return ConfigurationManager()


@pytest.fixture
def test_documents():
    """Create test documents for embedding validation."""
    return [
        Document(
            id="test_embed_doc_1",
            page_content="This is a test document about machine learning and artificial intelligence.",
            metadata={"title": "ML Test Doc", "source": "test"}
        ),
        Document(
            id="test_embed_doc_2", 
            page_content="This document discusses natural language processing and text analysis techniques.",
            metadata={"title": "NLP Test Doc", "source": "test"}
        ),
        Document(
            id="test_embed_doc_3",
            page_content="Computer vision and image recognition are important areas of AI research.",
            metadata={"title": "CV Test Doc", "source": "test"}
        )
    ]


@pytest.fixture
def mock_llm_func():
    """Mock LLM function for testing."""
    def llm_func(prompt: str) -> str:
        return "This is a mock response for testing purposes."
    return llm_func


def test_hyde_embedding_generation(config_manager, test_documents, mock_llm_func):
    """
    Test that HyDERAG properly generates embeddings during document ingestion.
    
    This test validates the fix for the critical issue where HyDERAG was storing
    documents without embeddings, causing vector search to return 0 results.
    """
    # Initialize HyDERAG pipeline
    pipeline = HyDERAGPipeline(
        config_manager=config_manager,
        llm_func=mock_llm_func
    )
    
    # Clear any existing test documents
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    # Clean up test documents
    for doc in test_documents:
        try:
            cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", [doc.id])
        except Exception:
            pass  # Ignore if document doesn't exist
    
    # Test document ingestion with embedding generation
    logger.info("Testing HyDERAG document ingestion with embedding generation")
    
    result = pipeline.ingest_documents(test_documents)
    
    # Validate ingestion was successful
    assert result["status"] == "success", f"Ingestion failed: {result.get('error', 'Unknown error')}"
    assert "processing_time" in result
    assert result["pipeline_type"] == "hyde_rag"
    
    # Verify documents were stored with embeddings by checking vector search works
    logger.info("Testing vector search to validate embeddings were generated")
    
    # Test query that should find relevant documents
    query_result = pipeline.query("machine learning artificial intelligence", top_k=2)
    
    # Validate query results
    assert "query" in query_result
    assert "answer" in query_result
    assert "retrieved_documents" in query_result
    assert query_result["num_documents_retrieved"] > 0, "Vector search returned 0 results - embeddings not generated!"
    
    # Verify we got actual documents back
    retrieved_docs = query_result["retrieved_documents"]
    assert len(retrieved_docs) > 0, "No documents retrieved - embedding generation failed!"
    
    # Verify the retrieved documents are actual Document objects with content
    for doc in retrieved_docs:
        assert hasattr(doc, 'page_content'), "Retrieved document missing page_content"
        assert hasattr(doc, 'metadata'), "Retrieved document missing metadata"
        assert len(doc.page_content) > 0, "Retrieved document has empty content"
    
    logger.info(f"SUCCESS: HyDERAG retrieved {len(retrieved_docs)} documents with embeddings")
    
    # Test another query to ensure consistent behavior
    query_result_2 = pipeline.query("natural language processing", top_k=1)
    assert query_result_2["num_documents_retrieved"] > 0, "Second query failed - inconsistent embedding behavior"
    
    # Clean up test documents
    for doc in test_documents:
        try:
            cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", [doc.id])
        except Exception:
            pass
    
    logger.info("HyDERAG embedding generation test completed successfully!")


def test_hyde_embedding_generation_with_chunking(config_manager, mock_llm_func):
    """
    Test HyDERAG embedding generation with chunking enabled.
    
    This ensures the fix works correctly when documents are chunked.
    """
    # Create a larger document that will trigger chunking
    large_doc = Document(
        id="test_large_doc",
        page_content="This is a very long document about machine learning. " * 100,  # Make it large enough to chunk
        metadata={"title": "Large ML Doc", "source": "test"}
    )
    
    pipeline = HyDERAGPipeline(
        config_manager=config_manager,
        llm_func=mock_llm_func
    )
    
    # Clean up
    connection = get_iris_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", [large_doc.id])
    except Exception:
        pass
    
    # Test ingestion with chunking
    result = pipeline.ingest_documents([large_doc], auto_chunk=True)
    
    assert result["status"] == "success", f"Chunked ingestion failed: {result.get('error', 'Unknown error')}"
    
    # Test that vector search works with chunked documents
    query_result = pipeline.query("machine learning", top_k=1)
    assert query_result["num_documents_retrieved"] > 0, "Vector search failed with chunked documents - embeddings not generated!"
    
    # Clean up
    try:
        cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", [large_doc.id])
        # Also clean up any chunks
        cursor.execute("DELETE FROM RAG.SourceDocuments WHERE metadata LIKE ?", [f'%{large_doc.id}%'])
    except Exception:
        pass
    
    logger.info("HyDERAG chunking with embeddings test completed successfully!")


if __name__ == "__main__":
    # Run the test directly for debugging
    config_mgr = ConfigurationManager()
    test_docs = [
        Document(
            id="direct_test_doc",
            page_content="Direct test document for debugging.",
            metadata={"title": "Debug Doc", "source": "direct_test"}
        )
    ]
    
    def mock_llm(prompt: str) -> str:
        return "Mock response"
    
    test_hyde_embedding_generation(config_mgr, test_docs, mock_llm)
    print("Direct test completed successfully!")