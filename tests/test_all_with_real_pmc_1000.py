"""
Comprehensive test suite that runs ALL RAG techniques with 1000+ REAL PMC documents.

This test file ensures that all RAG techniques are tested with real PMC data
using at least 1000 documents as required by the project standards.
No tests are skipped.
"""

import pytest
import logging
import time
from typing import Dict, Any, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define test markers
pytestmark = [
    pytest.mark.large_scale,  # Mark all tests as large scale
    pytest.mark.real_data,    # Using real PMC data
    pytest.mark.unskippable,  # These tests must not be skipped
]

# Minimum number of documents for testing
MIN_DOCUMENTS = 1000

@pytest.fixture(scope="module")
def verify_real_pmc_count(request, iris_with_pmc_data):
    """Verify that we have the minimum 1000 real PMC documents."""
    conn = iris_with_pmc_data
    
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        
        pmc_dir = os.path.join(os.getcwd(), "data", "pmc_oas_downloaded")
        xml_count = 0
        for dirpath, dirnames, filenames in os.walk(pmc_dir):
            for filename in filenames:
                if filename.endswith('.xml'):
                    xml_count += 1
        
        logger.info(f"Testing with {count} documents in database")
        logger.info(f"Found {xml_count} PMC XML files in {pmc_dir}")
        
        assert count >= MIN_DOCUMENTS, f"Expected at least {MIN_DOCUMENTS} documents, but found {count}"
    
    # Log test information
    logger.info(f"Running test: {request.node.name}")
    logger.info(f"Using real PMC documents with count >= {MIN_DOCUMENTS}")
    
    return conn

def test_basic_rag_with_real_pmc(verify_real_pmc_count):
    """Test BasicRAG with 1000+ real PMC documents."""
    from basic_rag.pipeline import BasicRAGPipeline
    
    # Create pipeline with test connection
    pipeline = BasicRAGPipeline(
        iris_connector=verify_real_pmc_count,
        # Use simple functions for testing
        embedding_func=lambda text: [0.1] * 10,
        llm_func=lambda prompt: f"Answer from BasicRAG with real PMC data"
    )
    
    # Run pipeline with a medical query
    query = "What is the role of insulin in diabetes?"
    start_time = time.time()
    result = pipeline.run(query, top_k=5)
    duration = time.time() - start_time
    
    # Assert result format
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    assert len(result["retrieved_documents"]) > 0, "Should retrieve at least one document"
    
    # Verify document count in database
    with verify_real_pmc_count.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        assert count >= MIN_DOCUMENTS, f"Expected at least {MIN_DOCUMENTS} documents, but found {count}"
    
    # Log results
    logger.info(f"BasicRAG query executed in {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
    logger.info(f"Answer: {result['answer']}")
    
    return result

def test_colbert_with_real_pmc(verify_real_pmc_count):
    """Test ColBERT with 1000+ real PMC documents."""
    from colbert.pipeline import ColBERTPipeline
    
    # Simple token-level embedding function
    def token_encoder(text):
        tokens = text.split()[:5]  # First 5 tokens
        return [[0.1, 0.2, 0.3] for _ in tokens]  # Same embedding for each token
    
    # Create pipeline
    pipeline = ColBERTPipeline(
        iris_connector=verify_real_pmc_count,
        colbert_query_encoder=token_encoder,
        llm_func=lambda prompt: f"Answer from ColBERT with real PMC data"
    )
    
    # Run query
    query = "What are the latest treatments for cancer?"
    start_time = time.time()
    result = pipeline.run(query, top_k=5)
    duration = time.time() - start_time
    
    # Assert result format
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    assert len(result["retrieved_documents"]) > 0, "Should retrieve at least one document"
    
    # Log results
    logger.info(f"ColBERT query executed in {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
    logger.info(f"Answer: {result['answer']}")
    
    return result

def test_noderag_with_real_pmc(verify_real_pmc_count):
    """Test NodeRAG with 1000+ real PMC documents."""
    from noderag.pipeline import NodeRAGPipeline
    
    # Create pipeline
    pipeline = NodeRAGPipeline(
        iris_connector=verify_real_pmc_count,
        embedding_func=lambda text: [0.1] * 10,
        llm_func=lambda prompt: f"Answer from NodeRAG with real PMC data"
    )
    
    # Run query
    query = "How does insulin relate to diabetes treatment?"
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    # Assert result format
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    assert len(result["retrieved_documents"]) > 0, "Should retrieve at least one document/node"
    
    # Log results
    logger.info(f"NodeRAG query executed in {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} nodes")
    logger.info(f"Answer: {result['answer']}")
    
    return result

def test_graphrag_with_real_pmc(verify_real_pmc_count):
    """Test GraphRAG with 1000+ real PMC documents."""
    from graphrag.pipeline import GraphRAGPipeline
    
    # Create pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=verify_real_pmc_count,
        embedding_func=lambda text: [0.1] * 10,
        llm_func=lambda prompt: f"Answer from GraphRAG with real PMC data"
    )
    
    # Run query
    query = "What is the relationship between cancer and diabetes?"
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    # Assert result format
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    assert len(result["retrieved_documents"]) > 0, "Should retrieve at least one document/node"
    
    # Log results
    logger.info(f"GraphRAG query executed in {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} nodes/documents")
    logger.info(f"Answer: {result['answer']}")
    
    return result

def test_context_reduction_with_real_pmc(verify_real_pmc_count):
    """Test context reduction with 1000+ real PMC documents."""
    from common.context_reduction import reduce_context
    from basic_rag.pipeline import BasicRAGPipeline
    
    # First retrieve documents using BasicRAG
    basic_pipeline = BasicRAGPipeline(
        iris_connector=verify_real_pmc_count,
        embedding_func=lambda text: [0.1] * 10,
        llm_func=lambda prompt: f"Answer with context reduction on real PMC data"
    )
    
    # Retrieve many documents
    query = "What are the relationships between various diseases?"
    documents = basic_pipeline._retrieve_documents(query, top_k=20)
    
    assert len(documents) > 0, "Should retrieve documents for context reduction"
    
    # Convert to format needed for context reduction
    docs_for_reduction = [{"content": doc.content} for doc in documents]
    
    # Run context reduction
    start_time = time.time()
    reduced_context = reduce_context(
        query=query,
        documents=docs_for_reduction,
        max_tokens=1000,
        strategy="semantic_clustering"
    )
    duration = time.time() - start_time
    
    # Assert context was reduced
    assert reduced_context is not None, "Reduced context should not be None"
    original_content = " ".join(doc["content"] for doc in docs_for_reduction)
    assert len(reduced_context) < len(original_content), "Context should be reduced in size"
    
    # Log results
    logger.info(f"Context reduction executed in {duration:.2f} seconds")
    logger.info(f"Original context size: {len(original_content)} characters")
    logger.info(f"Reduced context size: {len(reduced_context)} characters")
    logger.info(f"Reduction ratio: {len(reduced_context) / len(original_content):.2%}")
    
    return reduced_context

def test_all_techniques_together_with_real_pmc(verify_real_pmc_count):
    """Integration test running all RAG techniques with 1000+ real PMC documents."""
    # Run all techniques in sequence
    basic_result = test_basic_rag_with_real_pmc(verify_real_pmc_count)
    colbert_result = test_colbert_with_real_pmc(verify_real_pmc_count)
    noderag_result = test_noderag_with_real_pmc(verify_real_pmc_count)
    graphrag_result = test_graphrag_with_real_pmc(verify_real_pmc_count)
    context_result = test_context_reduction_with_real_pmc(verify_real_pmc_count)
    
    # Verify all results are present
    assert basic_result is not None, "BasicRAG should return a result"
    assert colbert_result is not None, "ColBERT should return a result"
    assert noderag_result is not None, "NodeRAG should return a result"
    assert graphrag_result is not None, "GraphRAG should return a result"
    assert context_result is not None, "Context reduction should return a result"
    
    # Log summary
    logger.info("All RAG techniques successfully tested with 1000+ real PMC documents")
    logger.info(f"BasicRAG retrieved {len(basic_result['retrieved_documents'])} documents")
    logger.info(f"ColBERT retrieved {len(colbert_result['retrieved_documents'])} documents")
    logger.info(f"NodeRAG retrieved {len(noderag_result['retrieved_documents'])} nodes")
    logger.info(f"GraphRAG retrieved {len(graphrag_result['retrieved_documents'])} nodes/documents")
    
    # Return a summary of all results
    return {
        "basic_rag": basic_result["answer"],
        "colbert": colbert_result["answer"],
        "noderag": noderag_result["answer"],
        "graphrag": graphrag_result["answer"],
        "all_techniques_passed": True
    }

if __name__ == "__main__":
    # Use pytest.main to run the tests directly
    result = pytest.main(["-xvs", __file__])
    print(f"Test result: {result}")
