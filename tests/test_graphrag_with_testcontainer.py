"""
Test GraphRAG with real PMC data using testcontainers.

This module uses testcontainers-iris to spin up an ephemeral IRIS database,
process real PMC XML files, build a knowledge graph, and test GraphRAG with
real data in an isolated environment. This follows the Test-Driven Development
approach with proper database isolation.
"""

import os
import logging
import pytest
import time
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import modules for testing
from graphrag.pipeline import GraphRAGPipeline
from common.utils import Document
from tests.utils import build_knowledge_graph, run_standardized_queries, compare_rag_techniques

# Test constants
TEST_PMC_LIMIT = int(os.environ.get('TEST_PMC_LIMIT', '20'))
QUERY_LIMIT = int(os.environ.get('QUERY_LIMIT', '5'))
DOCUMENT_LIMIT = int(os.environ.get('DOCUMENT_LIMIT', '3'))

# Sample medical queries
STANDARD_QUERIES = [
    "What is the relationship between diabetes and insulin?",
    "How does metformin help with diabetes treatment?",
    "What are the key symptoms of diabetes?",
    "What is the role of the pancreas in diabetes?",
    "How do statins affect cholesterol levels?"
]

@pytest.mark.force_testcontainer
def test_iris_testcontainer_setup(iris_testcontainer_connection):
    """
    Test that the IRIS testcontainer setup works correctly.
    
    This is a basic sanity check to ensure the testcontainer is running
    and can be connected to before running more complex tests.
    """
    assert iris_testcontainer_connection is not None, "Failed to create testcontainer connection"
    
    # Execute a simple query to verify the connection works
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None, "Failed to execute query"
        assert result[0] == 1, "Unexpected query result"

@pytest.mark.force_testcontainer
def test_load_pmc_documents(iris_testcontainer_connection):
    """
    Test loading PMC documents into the testcontainer database.
    """
    from tests.utils import load_pmc_documents
    
    # Load documents
    doc_count = load_pmc_documents(
        connection=iris_testcontainer_connection,
        limit=TEST_PMC_LIMIT,
        pmc_dir="data/pmc_oas_downloaded"
    )
    
    # Verify documents were loaded
    assert doc_count > 0, "No documents were loaded"
    logger.info(f"Successfully loaded {doc_count} documents")
    
    # Verify documents can be retrieved
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        result = cursor.fetchone()
        count = result[0] if result else 0
        
        assert count >= doc_count, f"Expected at least {doc_count} documents, found {count}"
        
        # Check content of a document
        cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT 1")
        doc = cursor.fetchone()
        
        assert doc is not None, "No document found"
        assert doc[0], "Document ID is empty"
        assert doc[1], "Document title is empty"
        assert doc[2], "Document content is empty"
        
        logger.info(f"Sample document: {doc[0]}, Title: {doc[1][:50]}...")

@pytest.mark.force_testcontainer
def test_build_knowledge_graph(iris_testcontainer_connection, real_embedding_model):
    """
    Test building a knowledge graph from real PMC data.
    """
    # Create embedding function
    embedding_func = lambda text: real_embedding_model.encode(text)
    
    # Build knowledge graph
    node_count, edge_count = build_knowledge_graph(
        connection=iris_testcontainer_connection,
        embedding_func=embedding_func,
        limit=TEST_PMC_LIMIT,
        pmc_dir="data/pmc_oas_downloaded"
    )
    
    # Verify graph was built
    assert node_count > 0, "No nodes were created"
    assert edge_count > 0, "No edges were created"
    logger.info(f"Built knowledge graph with {node_count} nodes and {edge_count} edges")
    
    # Verify graph can be retrieved
    with iris_testcontainer_connection.cursor() as cursor:
        # Check nodes
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
        result = cursor.fetchone()
        db_node_count = result[0] if result else 0
        
        assert db_node_count == node_count, f"Expected {node_count} nodes, found {db_node_count}"
        
        # Check node types
        cursor.execute("""
            SELECT node_type, COUNT(*) 
            FROM KnowledgeGraphNodes 
            GROUP BY node_type
        """)
        node_types = cursor.fetchall()
        
        logger.info("Node type distribution:")
        for node_type, count in node_types:
            logger.info(f"  {node_type}: {count}")
        
        # Check edges
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
        result = cursor.fetchone()
        db_edge_count = result[0] if result else 0
        
        assert db_edge_count == edge_count, f"Expected {edge_count} edges, found {db_edge_count}"

@pytest.mark.force_testcontainer
def test_graphrag_pipeline(iris_with_pmc_data, real_embedding_model):
    """
    Test the GraphRAG pipeline with real PMC data in a testcontainer.
    """
    # Create embedding function
    embedding_func = lambda text: real_embedding_model.encode(text)
    
    # Mock LLM function for testing (no need for actual LLM in this test)
    def mock_llm_func(prompt):
        return f"Response to: {prompt[:100]}..."
    
    # Create GraphRAG pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_pmc_data,
        embedding_func=embedding_func,
        llm_func=mock_llm_func
    )
    
    # Run standard queries
    logger.info("Running standard queries through GraphRAG pipeline")
    results = run_standardized_queries(
        pipeline=pipeline,
        queries=STANDARD_QUERIES[:QUERY_LIMIT],
        include_docs=True
    )
    
    # Verify results
    assert results, "No results returned"
    assert "_summary" in results, "Summary not in results"
    assert results["_summary"]["total_docs_retrieved"] > 0, "No documents retrieved"
    
    # Log summary statistics
    summary = results["_summary"]
    logger.info(f"Retrieved {summary['total_docs_retrieved']} documents across {summary['total_queries']} queries")
    logger.info(f"Average {summary['avg_docs_per_query']:.2f} documents per query")
    logger.info(f"Average query time: {summary['avg_time_per_query']:.4f} seconds")
    
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
def test_graphrag_kg_verification(iris_with_pmc_data):
    """
    Verify the knowledge graph structure and content.
    """
    # Query nodes and their types
    with iris_with_pmc_data.cursor() as cursor:
        # Get document nodes
        cursor.execute("""
            SELECT node_id, node_name, content
            FROM KnowledgeGraphNodes
            WHERE node_type = 'Document'
            LIMIT 3
        """)
        doc_nodes = cursor.fetchall()
        
        # Get entity nodes
        cursor.execute("""
            SELECT node_id, node_type, node_name, content
            FROM KnowledgeGraphNodes
            WHERE node_type != 'Document'
            LIMIT 5
        """)
        entity_nodes = cursor.fetchall()
        
        # Get edges connecting entities to documents
        cursor.execute("""
            SELECT e.source_node_id, e.target_node_id, e.relationship_type,
                   ns.node_name as source_name, nt.node_name as target_name
            FROM KnowledgeGraphEdges e
            JOIN KnowledgeGraphNodes ns ON e.source_node_id = ns.node_id
            JOIN KnowledgeGraphNodes nt ON e.target_node_id = nt.node_id
            LIMIT 5
        """)
        edges = cursor.fetchall()
    
    # Verify document nodes
    assert len(doc_nodes) > 0, "No document nodes found"
    logger.info("Document nodes:")
    for node_id, node_name, content in doc_nodes:
        logger.info(f"  ID: {node_id}, Title: {node_name[:30]}...")
        assert node_id.startswith("doc_"), "Document node ID should start with 'doc_'"
        assert node_name, "Document node name should not be empty"
        assert content, "Document node content should not be empty"
    
    # Verify entity nodes
    assert len(entity_nodes) > 0, "No entity nodes found"
    logger.info("Entity nodes:")
    for node_id, node_type, node_name, content in entity_nodes:
        logger.info(f"  {node_type}: {node_name} ({node_id})")
        assert node_type in ["Disease", "Medication", "Organ", "Symptom", "Treatment"], f"Unknown entity type: {node_type}"
        assert node_name, "Entity node name should not be empty"
        assert content, "Entity node content should not be empty"
    
    # Verify edges
    assert len(edges) > 0, "No edges found"
    logger.info("Edges:")
    for source_id, target_id, rel_type, source_name, target_name in edges:
        logger.info(f"  {source_name} --[{rel_type}]--> {target_name}")
        assert source_id, "Edge source ID should not be empty"
        assert target_id, "Edge target ID should not be empty"
        assert rel_type, "Edge relationship type should not be empty"

@pytest.mark.force_testcontainer
def test_graphrag_context_reduction(iris_with_pmc_data, real_embedding_model):
    """
    Test that GraphRAG provides effective context reduction.
    
    This test verifies that GraphRAG reduces the context by selecting
    relevant documents and entities during retrieval.
    """
    # Create embedding function
    embedding_func = lambda text: real_embedding_model.encode(text)
    
    # Mock LLM function
    def mock_llm_func(prompt):
        return f"Response to: {prompt[:100]}..."
    
    # Create GraphRAG pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_pmc_data,
        embedding_func=embedding_func,
        llm_func=mock_llm_func
    )
    
    # Choose a sample query
    query = "What is the relationship between diabetes and insulin?"
    
    # Run query with default pipeline
    logger.info(f"Running query with GraphRAG: {query}")
    start_time = time.time()
    graphrag_result = pipeline.run(query)
    graphrag_time = time.time() - start_time
    
    # Count total document tokens
    total_doc_length = 0
    total_graph_context_length = 0
    
    # Check the size of all documents in the database
    with iris_with_pmc_data.cursor() as cursor:
        cursor.execute("SELECT content FROM SourceDocuments")
        all_docs = cursor.fetchall()
        for doc in all_docs:
            total_doc_length += len(doc[0])
    
    # Count the context size from GraphRAG
    retrieved_docs = graphrag_result.get("retrieved_documents", [])
    for doc in retrieved_docs:
        if hasattr(doc, "content"):
            total_graph_context_length += len(doc.content)
    
    # Calculate reduction factor
    if total_doc_length > 0 and len(all_docs) > 0:
        reduction_factor = total_doc_length / total_graph_context_length if total_graph_context_length > 0 else float('inf')
        avg_doc_length = total_doc_length / len(all_docs)
        avg_context_length = total_graph_context_length / len(retrieved_docs) if retrieved_docs else 0
        
        logger.info(f"Total document corpus size: {total_doc_length} characters across {len(all_docs)} documents")
        logger.info(f"Average document size: {avg_doc_length:.1f} characters")
        logger.info(f"GraphRAG context size: {total_graph_context_length} characters across {len(retrieved_docs)} documents")
        logger.info(f"Average retrieved document size: {avg_context_length:.1f} characters")
        logger.info(f"Context reduction factor: {reduction_factor:.1f}x")
        
        assert len(retrieved_docs) < len(all_docs), "GraphRAG should retrieve a subset of documents"
        assert reduction_factor > 1.0, f"GraphRAG should reduce context (factor: {reduction_factor:.1f})"
