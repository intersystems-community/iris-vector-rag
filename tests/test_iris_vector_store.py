"""
Test script for IRISVectorStore implementation

This script demonstrates the core functionality of the IRISVectorStore
including payload/metadata support for searchable chunk metadata.
"""

import sys
import os
import logging
from typing import List
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.vector_store import IRISVectorStore, VectorPoint, VectorSearchResult, create_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_iris_vector_store():
    """Test the IRISVectorStore implementation with payload functionality."""
    
    logger.info("Starting IRISVectorStore test...")
    
    # Create vector store instance
    vector_store = create_vector_store("iris")
    
    # Test collection creation
    collection_name = "test_chunks"
    vector_size = 768
    
    logger.info(f"Creating collection '{collection_name}' with {vector_size}D vectors...")
    success = vector_store.create_collection(collection_name, vector_size, "cosine")
    if not success:
        logger.error("Failed to create collection")
        return False
    
    # Check if collection exists
    exists = vector_store.collection_exists(collection_name)
    logger.info(f"Collection exists: {exists}")
    
    # Create test vectors with payload metadata
    test_points = []
    for i in range(5):
        # Generate random 768-dimensional vector
        vector = np.random.randn(768).tolist()
        
        # Create payload with chunk metadata
        payload = {
            "file_path": f"documents/paper_{i}.pdf",
            "chunk_index": i,
            "chunk_text": f"This is chunk {i} containing important information about topic {i}.",
            "document_type": "research_paper",
            "section": "introduction" if i < 3 else "conclusion",
            "page_number": i + 1
        }
        
        point = VectorPoint(
            id=f"chunk_{i}",
            vector=vector,
            payload=payload
        )
        test_points.append(point)
    
    # Test upsert functionality
    logger.info(f"Upserting {len(test_points)} vectors with payload metadata...")
    success = vector_store.upsert(collection_name, test_points)
    if not success:
        logger.error("Failed to upsert vectors")
        return False
    
    # Test search functionality
    logger.info("Testing vector search...")
    query_vector = np.random.randn(768).tolist()
    
    # Basic search
    results = vector_store.search(collection_name, query_vector, limit=3)
    logger.info(f"Found {len(results)} results from basic search")
    
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: ID={result.id}, Score={result.score:.4f}")
        logger.info(f"  File: {result.payload.get('file_path')}")
        logger.info(f"  Section: {result.payload.get('section')}")
        logger.info(f"  Text: {result.payload.get('chunk_text')[:50]}...")
    
    # Test payload filtering
    logger.info("Testing payload filtering...")
    
    # Filter by file path
    file_filter = {"file_path": "documents/paper_2.pdf"}
    filtered_results = vector_store.search(collection_name, query_vector, 
                                         limit=5, payload_filter=file_filter)
    logger.info(f"Found {len(filtered_results)} results with file filter")
    
    # Filter by section
    section_filter = {"section": "introduction"}
    section_results = vector_store.search(collection_name, query_vector,
                                        limit=5, payload_filter=section_filter)
    logger.info(f"Found {len(section_results)} results with section filter")
    
    # Test collection info
    info = vector_store.get_collection_info(collection_name)
    logger.info(f"Collection info: {info}")
    
    # Test deletion by ID
    logger.info("Testing deletion by ID...")
    delete_ids = ["chunk_0", "chunk_1"]
    success = vector_store.delete_by_id(collection_name, delete_ids)
    if success:
        logger.info(f"Successfully deleted {len(delete_ids)} vectors")
    
    # Test deletion by filter (file-based deletion)
    logger.info("Testing deletion by filter...")
    delete_filter = {"file_path": "documents/paper_3.pdf"}
    success = vector_store.delete_by_filter(collection_name, delete_filter)
    if success:
        logger.info("Successfully deleted vectors by file filter")
    
    # Verify remaining vectors
    remaining_results = vector_store.search(collection_name, query_vector, limit=10)
    logger.info(f"Remaining vectors after deletion: {len(remaining_results)}")
    
    # Clean up - delete collection
    logger.info("Cleaning up test collection...")
    success = vector_store.delete_collection(collection_name)
    if success:
        logger.info("Successfully deleted test collection")
    
    logger.info("IRISVectorStore test completed successfully!")
    return True

def test_drop_in_replacement():
    """Test that IRISVectorStore can be used as a drop-in replacement."""
    
    logger.info("Testing drop-in replacement functionality...")
    
    # This demonstrates how the vector store can be used in existing code
    # that expects a QdrantVectorStore interface
    
    def simulate_rag_pipeline(vector_store):
        """Simulate a RAG pipeline using the vector store."""
        
        collection_name = "rag_test"
        
        # Create collection
        vector_store.create_collection(collection_name, 384, "cosine")
        
        # Add some documents
        documents = [
            {"id": "doc1", "text": "Machine learning is a subset of artificial intelligence."},
            {"id": "doc2", "text": "Deep learning uses neural networks with multiple layers."},
            {"id": "doc3", "text": "Natural language processing enables computers to understand text."}
        ]
        
        points = []
        for doc in documents:
            # Simulate embedding generation (random for demo)
            vector = np.random.randn(384).tolist()
            payload = {
                "document_id": doc["id"],
                "text": doc["text"],
                "source": "knowledge_base"
            }
            points.append(VectorPoint(doc["id"], vector, payload))
        
        # Store vectors
        vector_store.upsert(collection_name, points)
        
        # Simulate query
        query_vector = np.random.randn(384).tolist()
        results = vector_store.search(collection_name, query_vector, limit=2)
        
        logger.info(f"RAG pipeline found {len(results)} relevant documents")
        for result in results:
            logger.info(f"  {result.id}: {result.payload['text'][:50]}...")
        
        # Cleanup
        vector_store.delete_collection(collection_name)
        
        return len(results) > 0
    
    # Test with IRIS implementation
    iris_store = create_vector_store("iris")
    success = simulate_rag_pipeline(iris_store)
    
    logger.info(f"Drop-in replacement test: {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    try:
        # Run basic functionality test
        test1_success = test_iris_vector_store()
        
        # Run drop-in replacement test
        test2_success = test_drop_in_replacement()
        
        if test1_success and test2_success:
            logger.info("All tests PASSED! IRISVectorStore is ready for use.")
        else:
            logger.error("Some tests FAILED. Please check the implementation.")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()