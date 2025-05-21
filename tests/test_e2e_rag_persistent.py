"""
Comprehensive end-to-end tests for all RAG pipelines with real data.

This test file follows TDD principles:
1. Red: Define failing tests that specify expected behavior
2. Green: Implement code to make tests pass
3. Refactor: Clean up while keeping tests passing

Tests verify that all RAG techniques work with real PMC documents and
include meaningful assertions to validate results.

This version is modified to use the persistent IRIS container instead of testcontainers.
"""

import pytest
import os
import sys
import logging
from typing import List, Dict, Any, Callable, Optional
import time
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import RAG pipelines
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from crag.pipeline import CRAGPipeline
from colbert.pipeline import ColbertRAGPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline import GraphRAGPipeline

# Import common utilities
from common.utils import Document, timing_decorator
from common.iris_connector import get_iris_connection

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def sample_medical_queries() -> List[Dict[str, Any]]:
    """
    Provides a list of medical queries for testing RAG pipelines.
    
    Each query includes:
    - query: The question text
    - expected_keywords: Keywords that should appear in retrieved documents
    - min_doc_count: Minimum number of documents that should be retrieved
    """
    return [
        {
            "query": "What are the symptoms of diabetes?",
            "expected_keywords": ["diabetes", "symptom", "glucose", "insulin"],
            "min_doc_count": 2
        },
        {
            "query": "How does COVID-19 affect the respiratory system?",
            "expected_keywords": ["covid", "respiratory", "lung", "breathing"],
            "min_doc_count": 2
        },
        {
            "query": "What treatments are available for Alzheimer's disease?",
            "expected_keywords": ["alzheimer", "treatment", "cognitive", "memory"],
            "min_doc_count": 2
        }
    ]

@pytest.fixture(scope="module")
def real_iris_connection():
    """
    Provides a real IRIS connection to the persistent Docker container.
    """
    conn = get_iris_connection(use_mock=False)
    if not conn:
        pytest.skip("Could not connect to IRIS database")
    
    # Verify document count
    with conn.cursor() as cursor:
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            count = cursor.fetchone()[0]
        except Exception:
            try:
                cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
                count = cursor.fetchone()[0]
            except Exception as e:
                pytest.skip(f"Error querying document count: {e}")
                return None
    
    if count < 1000:
        pytest.skip(f"Insufficient documents: found {count}, need at least 1000")
    
    logger.info(f"Using real IRIS connection with {count} documents")
    return conn

@pytest.fixture(scope="module")
def real_embedding_func():
    """
    Provides a real embedding function for testing.
    """
    from common.utils import get_embedding_func
    embed_func = get_embedding_func()
    logger.info(f"Using real embedding function")
    return embed_func

@pytest.fixture(scope="module")
def real_llm_func():
    """
    Provides a real LLM function for testing.
    Falls back to stub if real LLM is not available.
    """
    from common.utils import get_llm_func
    try:
        # Try to get a real LLM function (e.g., OpenAI)
        llm_func = get_llm_func(provider="openai")
        logger.info(f"Using real OpenAI LLM function")
    except (ImportError, ValueError, EnvironmentError):
        # Fall back to stub LLM if real LLM is not available
        logger.warning(f"Real LLM not available, falling back to stub LLM")
        llm_func = get_llm_func(provider="stub")
    
    return llm_func

@pytest.fixture(scope="module")
def colbert_query_encoder():
    """
    Provides a ColBERT query encoder function for testing.
    """
    from common.utils import get_embedding_func
    
    # For testing purposes, we'll use a simple wrapper around the embedding function
    # In a real implementation, this would be a proper ColBERT query encoder
    embedding_func = get_embedding_func()
    
    def simple_colbert_query_encoder(text):
        # Split text into tokens and get embeddings for each token
        tokens = text.split()
        if not tokens:
            tokens = [text]
        
        # Limit tokens to avoid excessive processing
        tokens = tokens[:20]
        
        # Get embeddings for each token
        try:
            token_embeddings = embedding_func(tokens)
            logger.info(f"Generated {len(token_embeddings)} token embeddings for ColBERT query")
            return token_embeddings
        except Exception as e:
            logger.error(f"Error generating ColBERT token embeddings: {e}")
            # Return dummy embeddings (3 tokens with 10-dim vectors)
            return [[0.1] * 10 for _ in range(3)]
    
    logger.info(f"Using simple ColBERT query encoder")
    return simple_colbert_query_encoder

@pytest.fixture(scope="module")
def web_search_func():
    """
    Provides a web search function for CRAG testing.
    
    This is a mock function that returns predefined results based on the query.
    In a real implementation, this would call an actual web search API.
    """
    def mock_web_search(query, num_results=3):
        # Create mock web search results based on the query
        results = []
        keywords = query.lower().split()
        
        # Include relevant keywords in the results based on the query
        if "diabetes" in query.lower():
            for i in range(num_results):
                result = f"Web search result {i+1} for query about diabetes symptoms. "
                result += f"Common symptoms of diabetes include increased thirst, frequent urination, "
                result += f"unexplained weight loss, extreme hunger, blurred vision, fatigue, and slow-healing sores. "
                result += f"These symptoms are related to high blood glucose levels and insulin resistance."
                results.append(result)
        elif "covid" in query.lower():
            for i in range(num_results):
                result = f"Web search result {i+1} for query about COVID-19 respiratory effects. "
                result += f"COVID-19 affects the respiratory system by causing inflammation in the lungs, "
                result += f"leading to breathing difficulties, cough, and in severe cases, pneumonia and respiratory failure."
                results.append(result)
        elif "alzheimer" in query.lower():
            for i in range(num_results):
                result = f"Web search result {i+1} for query about Alzheimer's treatment. "
                result += f"Current treatments for Alzheimer's disease include medications like cholinesterase inhibitors "
                result += f"and memantine that can temporarily improve cognitive symptoms. Therapeutic approaches also focus "
                result += f"on maintaining memory function and managing behavioral changes."
                results.append(result)
        else:
            # Generic results for other queries
            for i in range(num_results):
                result = f"Web search result {i+1} for query about {' and '.join(keywords[:2])}. "
                result += f"This result contains information related to {keywords[0] if keywords else 'medicine'}."
                results.append(result)
        
        logger.info(f"Mock web search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    logger.info(f"Using mock web search function")
    return mock_web_search

# --- Test Helper Functions ---

def verify_rag_result(result: Dict[str, Any], query: str, expected_keywords: List[str], min_doc_count: int, pipeline_type: str = "standard"):
    """
    Verifies that a RAG result meets the expected criteria.
    
    Args:
        result: The result from a RAG pipeline
        query: The original query
        expected_keywords: Keywords that should appear in retrieved documents
        min_doc_count: Minimum number of documents that should be retrieved
        pipeline_type: Type of pipeline ("standard", "crag")
    """
    # Check that the result contains the expected keys
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain an answer"
    
    # Different pipelines use different keys for retrieved documents
    if pipeline_type == "crag":
        assert "retrieved_context_chunks" in result, "CRAG result should contain retrieved_context_chunks"
        retrieved_items = result["retrieved_context_chunks"]
    else:
        assert "retrieved_documents" in result, "Result should contain retrieved documents"
        retrieved_items = result["retrieved_documents"]
    
    # Check that the query matches the original query
    assert result["query"] == query, f"Result query '{result['query']}' does not match original query '{query}'"
    
    # Check that the answer is not empty
    assert result["answer"], "Answer should not be empty"
    
    # Check that the answer is a string
    assert isinstance(result["answer"], str), f"Answer should be a string, got {type(result['answer'])}"
    
    # Check that the answer is not too short
    assert len(result["answer"]) > 20, f"Answer is too short: '{result['answer']}'"
    
    # Check that the retrieved documents meet the minimum count
    assert len(retrieved_items) >= min_doc_count, f"Expected at least {min_doc_count} retrieved items, got {len(retrieved_items)}"
    
    # Check that the retrieved documents contain the expected keywords
    # For each document, check if it contains at least one of the expected keywords
    items_with_keywords = 0
    for item in retrieved_items:
        if pipeline_type == "crag":
            # For CRAG, items are strings
            item_content = item.lower()
        else:
            # For other pipelines, items are Document objects or dicts
            if hasattr(item, 'content'):
                item_content = item.content.lower()
            elif isinstance(item, dict) and 'content' in item:
                item_content = item['content'].lower()
            else:
                item_content = str(item).lower()
        
        if any(keyword.lower() in item_content for keyword in expected_keywords):
            items_with_keywords += 1
    
    # At least half of the documents should contain at least one of the expected keywords
    min_items_with_keywords = max(1, min_doc_count // 2)
    assert items_with_keywords >= min_items_with_keywords, \
        f"Expected at least {min_items_with_keywords} items to contain keywords {expected_keywords}, got {items_with_keywords}"
    
    logger.info(f"RAG result verification passed for query: {query[:50]}...")

# --- Test Cases ---

def test_basic_rag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    """
    Verifies that Basic RAG works with real PMC documents.
    """
    logger.info("Running test_basic_rag_with_real_data")
    
    # Initialize the BasicRAGPipeline
    pipeline = BasicRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func
    )
    
    # Test with a sample query
    query_data = sample_medical_queries[0]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Run the pipeline
    result = pipeline.run(query, top_k=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_basic_rag_with_real_data passed")

def test_hyde_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    """
    Verifies that HyDE works with real PMC documents.
    """
    logger.info("Running test_hyde_with_real_data")
    
    # Initialize the HyDEPipeline
    pipeline = HyDEPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func
    )
    
    # Test with a sample query
    query_data = sample_medical_queries[1]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Run the pipeline
    result = pipeline.run(query, top_k=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_hyde_with_real_data passed")

def test_crag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, web_search_func, sample_medical_queries):
    """
    Verifies that CRAG works with real PMC documents.
    """
    logger.info("Running test_crag_with_real_data")
    
    # Initialize the CRAGPipeline
    pipeline = CRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func,
        web_search_func=web_search_func
    )
    
    # Test with a sample query
    query_data = sample_medical_queries[2]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Run the pipeline
    result = pipeline.run(query, top_k=5)
    
    # Verify the result - note the "crag" pipeline type
    verify_rag_result(result, query, expected_keywords, min_doc_count, pipeline_type="crag")
    
    logger.info("test_crag_with_real_data passed")

def test_colbert_with_real_data(real_iris_connection, real_llm_func, colbert_query_encoder, sample_medical_queries):
    """
    Verifies that ColBERT works with real PMC documents.
    """
    logger.info("Running test_colbert_with_real_data")
    
    # Initialize the ColbertRAGPipeline - note the different parameters
    pipeline = ColbertRAGPipeline(
        iris_connector=real_iris_connection,
        colbert_query_encoder_func=colbert_query_encoder,
        colbert_doc_encoder_func=colbert_query_encoder,  # Use the same function for both
        llm_func=real_llm_func
    )
    
    # Test with a sample query
    query_data = sample_medical_queries[0]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Run the pipeline
    result = pipeline.run(query, top_k=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_colbert_with_real_data passed")

def test_noderag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    """
    Verifies that NodeRAG works with real PMC documents.
    """
    logger.info("Running test_noderag_with_real_data")
    
    # Initialize the NodeRAGPipeline
    # NodeRAG might need a graph_lib, but we'll use None for now
    pipeline = NodeRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func,
        graph_lib=None  # Use None as a placeholder
    )
    
    # Test with a sample query
    query_data = sample_medical_queries[1]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Run the pipeline - note the top_k_seeds parameter instead of top_k
    result = pipeline.run(query, top_k_seeds=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_noderag_with_real_data passed")

def test_graphrag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    """
    Verifies that GraphRAG works with real PMC documents.
    """
    logger.info("Running test_graphrag_with_real_data")
    
    # Initialize the GraphRAGPipeline
    pipeline = GraphRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func
    )
    
    # Test with a sample query
    query_data = sample_medical_queries[2]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Run the pipeline - note the top_n_start_nodes parameter instead of top_k
    result = pipeline.run(query, top_n_start_nodes=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_graphrag_with_real_data passed")

def test_all_pipelines_with_same_query(
    real_iris_connection, 
    real_embedding_func, 
    real_llm_func, 
    colbert_query_encoder,
    web_search_func,
    sample_medical_queries
):
    """
    Compares results from all pipelines with the same query.
    """
    logger.info("Running test_all_pipelines_with_same_query")
    
    # Use the same query for all pipelines
    query_data = sample_medical_queries[0]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    # Initialize all pipelines with their correct parameters
    pipelines = {
        "BasicRAG": BasicRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        ),
        "HyDE": HyDEPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        ),
        "CRAG": CRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func,
            web_search_func=web_search_func
        ),
        "ColBERT": ColbertRAGPipeline(
            iris_connector=real_iris_connection,
            colbert_query_encoder_func=colbert_query_encoder,
            colbert_doc_encoder_func=colbert_query_encoder,  # Use the same function for both
            llm_func=real_llm_func
        ),
        "NodeRAG": NodeRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func,
            graph_lib=None  # Use None as a placeholder
        ),
        "GraphRAG": GraphRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        )
    }
    
    # Run all pipelines and collect results
    results = {}
    for name, pipeline in pipelines.items():
        logger.info(f"Running {name} pipeline with query: {query}")
        try:
            start_time = time.time()
            
            # Call run with the appropriate parameters based on pipeline type
            if name == "NodeRAG":
                result = pipeline.run(query, top_k_seeds=5)
            elif name == "GraphRAG":
                result = pipeline.run(query, top_n_start_nodes=5)
            else:
                result = pipeline.run(query, top_k=5)
                
            elapsed_time = time.time() - start_time
            
            # Verify the result with the appropriate pipeline type
            if name == "CRAG":
                verify_rag_result(result, query, expected_keywords, min_doc_count, pipeline_type="crag")
            else:
                verify_rag_result(result, query, expected_keywords, min_doc_count)
            
            # Store the result and elapsed time
            results[name] = {
                "result": result,
                "elapsed_time": elapsed_time
            }
            
            logger.info(f"{name} pipeline completed in {elapsed_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error running {name} pipeline: {e}")
            results[name] = {
                "error": str(e)
            }
    
    # Compare the results
    # Check that all pipelines returned results
    assert len(results) == len(pipelines), f"Expected results from all {len(pipelines)} pipelines, got {len(results)}"
    
    # Check that all pipelines returned valid results
    for name, result_data in results.items():
        assert "error" not in result_data, f"{name} pipeline failed with error: {result_data.get('error')}"
        assert "result" in result_data, f"{name} pipeline did not return a result"
        
        result = result_data["result"]
        assert "answer" in result, f"{name} pipeline result does not contain an answer"
        
        # Check for retrieved documents or context chunks based on pipeline type
        if name == "CRAG":
            assert "retrieved_context_chunks" in result, f"{name} pipeline result does not contain retrieved context chunks"
        else:
            assert "retrieved_documents" in result, f"{name} pipeline result does not contain retrieved documents"
    
    # Log the answers from all pipelines
    logger.info(f"Query: {query}")
    for name, result_data in results.items():
        answer = result_data["result"]["answer"]
        logger.info(f"{name} answer: {answer[:100]}...")
    
    logger.info("test_all_pipelines_with_same_query passed")

if __name__ == "__main__":
    # This allows running the tests directly with python -m tests.test_e2e_rag_persistent
    pytest.main(["-v", __file__])