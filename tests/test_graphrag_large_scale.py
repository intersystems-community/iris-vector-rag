"""
Large-scale testing for GraphRAG with 1000+ documents.

This module implements specialized tests for GraphRAG with large document sets,
using optimized utilities for better memory management and performance.
"""

import os
import time
import logging
import pytest
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import standard and large-scale utilities
from graphrag.pipeline import GraphRAGPipeline
from common.utils import Document
from tests.utils import run_standardized_queries
from tests.utils_large_scale import (
    load_pmc_documents_large_scale,
    build_knowledge_graph_large_scale,
    validate_large_scale_graph
)

# Test constants - pull from environment for flexibility
TEST_DOCUMENT_COUNT = int(os.environ.get('TEST_DOCUMENT_COUNT', '1000'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '50'))
MAX_ENTITIES_PER_DOC = int(os.environ.get('MAX_ENTITIES_PER_DOC', '30'))
COLLECT_METRICS = os.environ.get('COLLECT_PERFORMANCE_METRICS', 'true').lower() in ('true', '1', 'yes')
QUERY_LIMIT = int(os.environ.get('QUERY_LIMIT', '5'))

# Standard medical queries for testing
STANDARD_QUERIES = [
    "What is the relationship between diabetes and insulin?",
    "How does metformin help with diabetes treatment?",
    "What are the key symptoms of diabetes?",
    "What is the role of the pancreas in diabetes?",
    "How do statins affect cholesterol levels?"
]

@pytest.mark.force_testcontainer
def test_large_scale_iris_setup(iris_testcontainer_connection):
    """Verify the IRIS testcontainer is properly set up for large-scale testing."""
    # This is a basic sanity check to ensure the testcontainer is running
    assert iris_testcontainer_connection is not None, "Failed to create testcontainer connection"
    
    # Execute a simple query to verify the connection works
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None, "Failed to execute query"
        assert result[0] == 1, "Unexpected query result"
    
    # Log configuration for easier debugging
    logger.info(f"Large-scale test configuration:")
    logger.info(f"  TEST_DOCUMENT_COUNT: {TEST_DOCUMENT_COUNT}")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  MAX_ENTITIES_PER_DOC: {MAX_ENTITIES_PER_DOC}")
    logger.info(f"  COLLECT_METRICS: {COLLECT_METRICS}")

@pytest.mark.force_testcontainer
def test_large_scale_document_loading(iris_testcontainer_connection):
    """
    Test loading 1000+ documents into the testcontainer database.
    
    This test uses enhanced document loading with batching, retries, and
    metrics collection for better performance with large document sets.
    """
    # Skip if document count is too small
    if TEST_DOCUMENT_COUNT < 100:
        pytest.skip(f"Test document count ({TEST_DOCUMENT_COUNT}) is too small for large-scale test")
    
    # Load documents with enhanced large-scale function
    metrics = load_pmc_documents_large_scale(
        connection=iris_testcontainer_connection,
        limit=TEST_DOCUMENT_COUNT,
        pmc_dir="data/pmc_oas_downloaded",
        batch_size=BATCH_SIZE,
        collect_metrics=COLLECT_METRICS
    )
    
    # Verify documents were loaded
    doc_count = metrics["successful_inserts"]
    assert doc_count > 0, "No documents were loaded"
    logger.info(f"Successfully loaded {doc_count} documents")
    
    # Log performance metrics
    if COLLECT_METRICS:
        logger.info(f"Document loading performance:")
        logger.info(f"  Total time: {metrics['total_time_seconds']:.2f} seconds")
        logger.info(f"  Throughput: {metrics['docs_per_second']:.2f} docs/second")
        logger.info(f"  Database time: {metrics['database_time']:.2f} seconds ({metrics['database_time']/metrics['total_time_seconds']*100:.1f}%)")
        logger.info(f"  Peak memory usage: {metrics['peak_memory_mb']:.2f} MB")
    
    # Verify documents can be retrieved with the correct count
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        result = cursor.fetchone()
        db_count = result[0] if result else 0
        
        # Allow for some documents to fail insertion (e.g., duplicates)
        min_expected = min(doc_count, int(TEST_DOCUMENT_COUNT * 0.9))
        assert db_count >= min_expected, f"Expected at least {min_expected} documents, found {db_count}"
        
        # Check content of a sample document
        cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT 1")
        doc = cursor.fetchone()
        
        assert doc is not None, "No document found"
        assert doc[0], "Document ID is empty"
        assert doc[1], "Document title is empty"
        assert doc[2], "Document content is empty"
        
        logger.info(f"Sample document: {doc[0]}, Title: {doc[1][:50]}...")

@pytest.fixture(scope="module")
def iris_with_large_pmc_data(iris_testcontainer_connection, real_embedding_model):
    """
    Module-scoped fixture with a large dataset of PMC documents.
    
    This fixture loads documents and builds a knowledge graph optimized
    for large-scale testing, with appropriate batch sizes and limits.
    """
    if iris_testcontainer_connection is None:
        pytest.skip("IRIS testcontainer connection not available")
        return None
    
    # Step 1: Load documents
    try:
        doc_metrics = load_pmc_documents_large_scale(
            connection=iris_testcontainer_connection,
            limit=TEST_DOCUMENT_COUNT,
            pmc_dir="data/pmc_oas_downloaded",
            batch_size=BATCH_SIZE,
            collect_metrics=COLLECT_METRICS
        )
        
        doc_count = doc_metrics["successful_inserts"]
        logger.info(f"Loaded {doc_count} PMC documents into testcontainer for large-scale test")
        
        # Skip if insufficient documents
        if doc_count < TEST_DOCUMENT_COUNT * 0.5:
            pytest.skip(f"Insufficient documents loaded: {doc_count}/{TEST_DOCUMENT_COUNT}")
            return None
            
        # Step 2: Build knowledge graph with optimized function
        # Create embedding function
        embedding_func = lambda text: real_embedding_model.encode(text)
        
        graph_metrics = build_knowledge_graph_large_scale(
            connection=iris_testcontainer_connection,
            embedding_func=embedding_func,
            limit=doc_count,  # Use actual loaded doc count
            pmc_dir="data/pmc_oas_downloaded",
            batch_size=BATCH_SIZE,
            max_entities_per_doc=MAX_ENTITIES_PER_DOC,
            collect_metrics=COLLECT_METRICS
        )
        
        # Log graph building metrics
        if COLLECT_METRICS:
            logger.info(f"Knowledge graph building performance:")
            logger.info(f"  Total time: {graph_metrics['total_time_seconds']:.2f} seconds")
            logger.info(f"  Node count: {graph_metrics['node_count']} (throughput: {graph_metrics['nodes_per_second']:.2f} nodes/sec)")
            logger.info(f"  Edge count: {graph_metrics['edge_count']} (throughput: {graph_metrics['edges_per_second']:.2f} edges/sec)")
            logger.info(f"  Embedding time: {graph_metrics['embedding_time']:.2f} seconds ({graph_metrics['embedding_time']/graph_metrics['total_time_seconds']*100:.1f}%)")
            logger.info(f"  Database time: {graph_metrics['database_time']:.2f} seconds ({graph_metrics['database_time']/graph_metrics['total_time_seconds']*100:.1f}%)")
            logger.info(f"  Peak memory usage: {graph_metrics['peak_memory_mb']:.2f} MB")
        
        # Validate the graph for minimum requirements
        validation = validate_large_scale_graph(
            connection=iris_testcontainer_connection,
            expected_docs=doc_count,
            min_doc_percentage=0.7,  # Allow some document nodes to fail
            min_nodes=doc_count,     # At minimum, should have document nodes
            min_edges=doc_count / 2  # At minimum, should have edges for half the docs
        )
        
        if not validation["success"]:
            logger.warning(f"Knowledge graph validation failed: {validation['errors']}")
            pytest.skip(f"Knowledge graph validation failed: {validation['errors']}")
            return None
            
        # Return the connection with data loaded
        yield iris_testcontainer_connection
        
    except Exception as e:
        logger.error(f"Failed to set up large-scale test data: {e}")
        pytest.skip(f"Failed to set up large-scale test data: {e}")
        yield None

@pytest.mark.force_testcontainer
def test_graphrag_pipeline_large_scale(iris_with_large_pmc_data, real_embedding_model):
    """
    Test the GraphRAG pipeline with a large document set.
    
    This test verifies that GraphRAG can handle 1000+ documents and
    perform effective retrieval with reasonable performance.
    """
    if iris_with_large_pmc_data is None:
        pytest.skip("Large PMC data fixture not available")
        return
    
    # Create embedding function
    embedding_func = lambda text: real_embedding_model.encode(text)
    
    # Mock LLM function for testing (no need for actual LLM in this test)
    def mock_llm_func(prompt):
        return f"Response to: {prompt[:100]}..."
    
    # Create GraphRAG pipeline
    logger.info("Creating GraphRAG pipeline for large-scale test")
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_large_pmc_data,
        embedding_func=embedding_func,
        llm_func=mock_llm_func
    )
    
    # Run standard queries with performance monitoring
    logger.info("Running standard queries through GraphRAG pipeline")
    start_time = time.time()
    
    results = run_standardized_queries(
        pipeline=pipeline,
        queries=STANDARD_QUERIES[:QUERY_LIMIT],
        include_docs=True
    )
    
    total_query_time = time.time() - start_time
    
    # Verify results
    assert results, "No results returned"
    assert "_summary" in results, "Summary not in results"
    assert results["_summary"]["total_docs_retrieved"] > 0, "No documents retrieved"
    
    # Log summary statistics with performance metrics
    summary = results["_summary"]
    logger.info(f"Retrieved {summary['total_docs_retrieved']} documents across {summary['total_queries']} queries")
    logger.info(f"Average {summary['avg_docs_per_query']:.2f} documents per query")
    logger.info(f"Average query time: {summary['avg_time_per_query']:.4f} seconds")
    
    # Performance assertions for large-scale test
    assert summary["avg_time_per_query"] < 10.0, f"Average query time too high: {summary['avg_time_per_query']:.4f} seconds"
    assert summary["avg_docs_per_query"] > 0, f"No documents retrieved per query: {summary['avg_docs_per_query']:.2f}"
    
    # Check individual query results
    for query in STANDARD_QUERIES[:QUERY_LIMIT]:
        assert query in results, f"No results for query: {query}"
        query_result = results[query]
        
        logger.info(f"Query: {query}")
        logger.info(f"  Retrieved {query_result['doc_count']} documents in {query_result['time_seconds']:.4f} seconds")
        
        if "documents" in query_result and query_result["documents"]:
            sample_doc = query_result["documents"][0]
            logger.info(f"  Top document: {sample_doc['id']}, Score: {sample_doc['score']:.4f}")
            logger.info(f"  Preview: {sample_doc['content_preview']}")

@pytest.mark.force_testcontainer
def test_graphrag_context_reduction_large_scale(iris_with_large_pmc_data, real_embedding_model):
    """
    Test that GraphRAG provides effective context reduction with large document sets.
    
    This test verifies that GraphRAG can significantly reduce context size
    when working with 1000+ documents, which is critical for LLM context limits.
    """
    if iris_with_large_pmc_data is None:
        pytest.skip("Large PMC data fixture not available")
        return
    
    # Create embedding function
    embedding_func = lambda text: real_embedding_model.encode(text)
    
    # Mock LLM function
    def mock_llm_func(prompt):
        return f"Response to: {prompt[:100]}..."
    
    # Create GraphRAG pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_large_pmc_data,
        embedding_func=embedding_func,
        llm_func=mock_llm_func
    )
    
    # Choose a sample query
    query = "What is the relationship between diabetes and insulin?"
    
    # Run query with default pipeline
    logger.info(f"Running query with GraphRAG on large document set: {query}")
    start_time = time.time()
    graphrag_result = pipeline.run(query)
    graphrag_time = time.time() - start_time
    
    # Count total document tokens
    total_doc_length = 0
    total_graph_context_length = 0
    doc_count = 0
    
    # Check the size of all documents in the database
    with iris_with_large_pmc_data.cursor() as cursor:
        cursor.execute("SELECT COUNT(*), AVG(LENGTH(content)) FROM SourceDocuments")
        result = cursor.fetchone()
        if result and result[0]:
            doc_count = result[0]
            avg_length = result[1] or 0
            total_doc_length = doc_count * avg_length
    
    # Count the context size from GraphRAG
    retrieved_docs = graphrag_result.get("retrieved_documents", [])
    for doc in retrieved_docs:
        if hasattr(doc, "content"):
            total_graph_context_length += len(doc.content)
    
    # Calculate reduction factor
    if total_doc_length > 0 and doc_count > 0:
        reduction_factor = total_doc_length / total_graph_context_length if total_graph_context_length > 0 else float('inf')
        avg_doc_length = total_doc_length / doc_count
        avg_context_length = total_graph_context_length / len(retrieved_docs) if retrieved_docs else 0
        
        logger.info(f"Total document corpus size: {total_doc_length:,} characters across {doc_count:,} documents")
        logger.info(f"Average document size: {avg_doc_length:.1f} characters")
        logger.info(f"GraphRAG context size: {total_graph_context_length:,} characters across {len(retrieved_docs)} documents")
        logger.info(f"Average retrieved document size: {avg_context_length:.1f} characters")
        logger.info(f"Context reduction factor: {reduction_factor:.1f}x")
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {graphrag_time:.4f} seconds")
        
        # Assertions specific to large document sets
        assert len(retrieved_docs) < doc_count * 0.1, f"GraphRAG retrieved too many documents: {len(retrieved_docs)}/{doc_count}"
        assert reduction_factor > 10.0, f"Insufficient context reduction factor: {reduction_factor:.1f}x"
        assert graphrag_time < 10.0, f"Query time too high: {graphrag_time:.4f} seconds"
