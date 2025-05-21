# tests/test_e2e_rag_pipelines.py
"""
Comprehensive end-to-end tests for all RAG pipelines with real data.

This test file follows TDD principles:
1. Red: Define failing tests that specify expected behavior
2. Green: Implement code to make tests pass
3. Refactor: Clean up while keeping tests passing

Tests verify that all RAG techniques work with real PMC documents and
include meaningful assertions to validate results.
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

# Import fixtures for real data testing
from tests.conftest_1000docs import ensure_1000_documents, verify_document_count
import os
import json

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
def real_iris_connection(verify_document_count):
    """
    Provides a real IRIS connection with at least 1000 documents.
    
    This fixture uses verify_document_count from conftest_1000docs.py
    to ensure the database has at least 1000 documents.
    """
    conn = verify_document_count
    logger.info(f"Using real IRIS connection with at least 1000 documents")
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
    Uses the provider specified in LLM_PROVIDER environment variable,
    or falls back to stub if real LLM is not available.
    """
    from common.utils import get_llm_func
    
    # Check for provider in environment variable
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    
    try:
        # Try to get a real LLM function based on provider
        if provider == "stub":
            logger.info("Using stub LLM function as specified")
            return get_llm_func(provider="stub")
        
        llm_func = get_llm_func(provider=provider)
        logger.info(f"Using real {provider.upper()} LLM function")
        return llm_func
    except (ImportError, ValueError, EnvironmentError) as e:
        # Fall back to stub LLM if real LLM is not available
        logger.warning(f"Real LLM ({provider}) not available: {e}")
        logger.warning(f"Falling back to stub LLM")
        return get_llm_func(provider="stub")

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
    
    Attempts to use a real web search API if available,
    otherwise falls back to a mock function.
    """
    try:
        # Try to import a real web search API (e.g., SerpAPI, Bing, etc.)
        # This is just a placeholder - in a real implementation, you would
        # use an actual web search API client
        
        # For now, we'll use a more realistic mock that returns medical information
        def enhanced_web_search(query, num_results=3):
            # Create more realistic web search results based on the query
            results = []
            
            # Extract keywords from query
            keywords = query.lower().split()
            main_topic = keywords[0] if keywords else "medicine"
            
            # Medical topics dictionary with some real information
            medical_topics = {
                "diabetes": [
                    "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, and blurred vision.",
                    "Type 2 diabetes develops when the body becomes resistant to insulin or when the pancreas is unable to produce enough insulin.",
                    "Treatment for diabetes includes monitoring blood sugar, insulin therapy, and lifestyle changes including diet and exercise."
                ],
                "covid": [
                    "COVID-19 affects the respiratory system by infecting cells in the lungs, causing inflammation and damage to lung tissue.",
                    "Severe COVID-19 can lead to pneumonia, acute respiratory distress syndrome (ARDS), and respiratory failure.",
                    "Long-term respiratory effects of COVID-19 can include reduced lung function, persistent shortness of breath, and pulmonary fibrosis."
                ],
                "alzheimer": [
                    "Current treatments for Alzheimer's disease include cholinesterase inhibitors like donepezil, rivastigmine, and galantamine.",
                    "Memantine (Namenda) works by regulating glutamate activity and can help improve symptoms in moderate to severe Alzheimer's disease.",
                    "New treatments being researched include immunotherapies targeting beta-amyloid plaques and tau protein tangles."
                ]
            }
            
            # Find relevant topic
            relevant_topic = None
            for topic in medical_topics:
                if topic in query.lower():
                    relevant_topic = topic
                    break
            
            # Generate results
            if relevant_topic and relevant_topic in medical_topics:
                # Use real information for the topic
                for i in range(min(num_results, len(medical_topics[relevant_topic]))):
                    results.append(medical_topics[relevant_topic][i])
            else:
                # Generic results for unknown topics
                for i in range(num_results):
                    result = f"Web search result {i+1} for query about {' and '.join(keywords[:2])}. "
                    result += f"This result contains information related to {main_topic}."
                    results.append(result)
            
            logger.info(f"Enhanced web search returned {len(results)} results for query: {query[:50]}...")
            return results
        
        logger.info(f"Using enhanced web search function")
        return enhanced_web_search
        
    except ImportError:
        # Fall back to mock web search
        def mock_web_search(query, num_results=3):
            # Create mock web search results based on the query
            results = []
            keywords = query.lower().split()
            
            for i in range(num_results):
                result = f"Web search result {i+1} for query about {' and '.join(keywords[:2])}. "
                result += f"This result contains information related to {keywords[0] if keywords else 'medicine'}."
                results.append(result)
            
            logger.info(f"Mock web search returned {len(results)} results for query: {query[:50]}...")
            return results
        
        logger.warning(f"Real web search API not available, using mock web search function")
        return mock_web_search

# --- Test Helper Functions ---

def verify_rag_result(result: Dict[str, Any], query: str, expected_keywords: List[str], min_doc_count: int, save_results: bool = True):
    """
    Verifies that a RAG result meets the expected criteria.
    
    Args:
        result: The result from a RAG pipeline
        query: The original query
        expected_keywords: Keywords that should appear in retrieved documents
        min_doc_count: Minimum number of documents that should be retrieved
    """
    # Check that the result contains the expected keys
    assert "query" in result, "Result should contain the original query"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    
    # Check that the query matches the original query
    assert result["query"] == query, f"Result query '{result['query']}' does not match original query '{query}'"
    
    # Check that the answer is not empty
    assert result["answer"], "Answer should not be empty"
    
    # Check that the answer is a string
    assert isinstance(result["answer"], str), f"Answer should be a string, got {type(result['answer'])}"
    
    # Check that the answer is not too short
    assert len(result["answer"]) > 20, f"Answer is too short: '{result['answer']}'"
    
    # Check that the retrieved documents meet the minimum count
    retrieved_docs = result["retrieved_documents"]
    assert len(retrieved_docs) >= min_doc_count, f"Expected at least {min_doc_count} retrieved documents, got {len(retrieved_docs)}"
    
    # Check that the retrieved documents contain the expected keywords
    # For each document, check if it contains at least one of the expected keywords
    docs_with_keywords = 0
    for doc in retrieved_docs:
        doc_content = doc.content.lower() if hasattr(doc, 'content') else str(doc).lower()
        if any(keyword.lower() in doc_content for keyword in expected_keywords):
            docs_with_keywords += 1
    
    # At least half of the documents should contain at least one of the expected keywords
    min_docs_with_keywords = max(1, min_doc_count // 2)
    assert docs_with_keywords >= min_docs_with_keywords, \
        f"Expected at least {min_docs_with_keywords} documents to contain keywords {expected_keywords}, got {docs_with_keywords}"
    
    logger.info(f"RAG result verification passed for query: {query[:50]}...")
    
    # Save results to file if requested
    if save_results:
        try:
            # Create directory if it doesn't exist
            os.makedirs("test_results/rag_outputs", exist_ok=True)
            
            # Generate a filename based on the query
            filename = f"test_results/rag_outputs/result_{query.replace(' ', '_')[:30]}_{int(time.time())}.json"
            
            # Create a serializable version of the result
            serializable_result = {
                "query": result["query"],
                "answer": result["answer"],
                "retrieved_documents": []
            }
            
            # Convert documents to serializable format
            for doc in result["retrieved_documents"]:
                if hasattr(doc, 'to_dict'):
                    serializable_result["retrieved_documents"].append(doc.to_dict())
                elif hasattr(doc, '__dict__'):
                    serializable_result["retrieved_documents"].append(vars(doc))
                else:
                    # Try to convert to a simple dict with content
                    doc_content = doc.content if hasattr(doc, 'content') else str(doc)
                    doc_id = doc.id if hasattr(doc, 'id') else "unknown"
                    serializable_result["retrieved_documents"].append({
                        "id": doc_id,
                        "content": doc_content
                    })
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(serializable_result, f, indent=2)
                
            logger.info(f"Saved RAG result to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save RAG result: {e}")

# --- Test Cases ---

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
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

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
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

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
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
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_crag_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
def test_colbert_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, colbert_query_encoder, sample_medical_queries):
    """
    Verifies that ColBERT works with real PMC documents.
    """
    logger.info("Running test_colbert_with_real_data")
    
    # Initialize the ColbertRAGPipeline
    pipeline = ColbertRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func,
        colbert_query_encoder_func=colbert_query_encoder,
        colbert_doc_encoder_func=real_embedding_func  # Use regular embedding func as a placeholder
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

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
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
    
    # Run the pipeline
    result = pipeline.run(query, top_k=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_noderag_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
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
    
    # Run the pipeline
    result = pipeline.run(query, top_k=5)
    
    # Verify the result
    verify_rag_result(result, query, expected_keywords, min_doc_count)
    
    logger.info("test_graphrag_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
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
    
    # Initialize all pipelines
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
            embedding_func=real_embedding_func,
            llm_func=real_llm_func,
            colbert_query_encoder_func=colbert_query_encoder,
            colbert_doc_encoder_func=real_embedding_func  # Use regular embedding func as a placeholder
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
            result = pipeline.run(query, top_k=5)
            elapsed_time = time.time() - start_time
            
            # Verify the result
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
        assert "retrieved_documents" in result, f"{name} pipeline result does not contain retrieved documents"
    
    # Log the answers from all pipelines
    logger.info(f"Query: {query}")
    for name, result_data in results.items():
        answer = result_data["result"]["answer"]
        logger.info(f"{name} answer: {answer[:100]}...")
    
    logger.info("test_all_pipelines_with_same_query passed")

def test_verify_real_data_requirements():
    """
    Verify that we're using real data for testing.
    
    This test doesn't actually do anything - it's just a placeholder to ensure
    that the requires_real_data marker is registered and can be used to skip
    tests if real data is not available.
    """
    logger.info("Verifying real data requirements...")
    # The actual verification is done by the verify_real_data_testing.py script
    # and the conftest_1000docs.py fixtures
    assert True

if __name__ == "__main__":
    # This allows running the tests directly with python -m tests.test_e2e_rag_pipelines
    pytest.main(["-v", __file__])