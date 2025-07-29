import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import pytest
from unittest.mock import Mock, patch, MagicMock
from iris_rag.pipelines.hyde import HyDERAGPipeline
from common.utils import get_llm_func

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@patch('iris_rag.storage.vector_store_iris.IRISVectorStore')
@patch('iris_rag.core.connection.ConnectionManager')
@patch('iris_rag.config.manager.ConfigurationManager')
def test_hyde_document_retrieval(mock_config_manager, mock_connection_manager, mock_vector_store):
    logger.info("Starting HyDE document retrieval test...")
    
    # Mock the configuration manager
    mock_config_instance = Mock()
    mock_config_manager.return_value = mock_config_instance
    
    # Mock the connection manager
    mock_connection_instance = Mock()
    mock_connection_manager.return_value = mock_connection_instance
    
    # Mock the vector store
    mock_vector_store_instance = Mock()
    mock_vector_store.return_value = mock_vector_store_instance
    
    # Create mock documents for retrieval
    mock_doc1 = Mock()
    mock_doc1.id = "doc1"
    mock_doc1.score = 0.85
    mock_doc1.content = "Climate change significantly affects polar bear populations by reducing sea ice habitat..."
    
    mock_doc2 = Mock()
    mock_doc2.id = "doc2"
    mock_doc2.score = 0.78
    mock_doc2.content = "Arctic warming leads to habitat loss for polar bears, forcing them to travel longer distances..."
    
    # Mock the vector store's similarity search method
    mock_vector_store_instance.similarity_search.return_value = [mock_doc1, mock_doc2]
    
    # Get LLM function
    llm_fn = get_llm_func(provider="stub")
    
    # Create the HyDE pipeline with mocked dependencies
    pipeline = HyDERAGPipeline(
        connection_manager=mock_connection_instance,
        config_manager=mock_config_instance,
        llm_func=llm_fn
    )
    
    # Override the vector store with our mock
    pipeline.vector_store = mock_vector_store_instance
    
    # Mock the embedding manager
    pipeline.embedding_manager = Mock()
    pipeline.embedding_manager.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Mock the _retrieve_documents method to return our mock documents
    pipeline._retrieve_documents = Mock(return_value=[
        {"doc_id": "doc1", "title": "Climate Change Effects", "content": "Climate change significantly affects polar bear populations by reducing sea ice habitat...", "similarity_score": 0.85},
        {"doc_id": "doc2", "title": "Arctic Warming", "content": "Arctic warming leads to habitat loss for polar bears, forcing them to travel longer distances...", "similarity_score": 0.78}
    ])
    
    test_query = "What are the effects of climate change on polar bears?"
    logger.info(f"Test query: '{test_query}'")
    
    # Test hypothetical document generation
    hypothetical_doc_text = pipeline._generate_hypothetical_document(test_query)
    logger.info(f"Generated hypothetical document text: '{hypothetical_doc_text}'")
    
    # Verify hypothetical document was generated
    assert hypothetical_doc_text is not None
    assert len(hypothetical_doc_text) > 0
    
    # Test the full query method which includes document retrieval
    result = pipeline.query(test_query, top_k=3)
    
    logger.info(f"Query result keys: {result.keys()}")
    
    # Verify the result structure
    assert "query" in result
    assert "retrieved_documents" in result
    assert result["query"] == test_query
    
    retrieved_docs = result["retrieved_documents"]
    logger.info(f"Number of documents retrieved: {len(retrieved_docs)}")
    
    # Verify documents were retrieved
    assert len(retrieved_docs) > 0, "HyDE should retrieve at least one document."
    
    logger.info("Retrieved documents:")
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"  Doc {i+1}: ID={doc.get('doc_id', 'unknown')}, Score={doc.get('similarity_score', 0):.4f}, Content='{doc.get('content', '')[:100]}...'")
    
    # Verify the embedding manager was called for text embedding
    pipeline.embedding_manager.embed_text.assert_called()
    
    logger.info("HyDE document retrieval test PASSED.")

if __name__ == "__main__":
    test_hyde_document_retrieval()