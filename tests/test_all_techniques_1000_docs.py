"""
Comprehensive tests for all RAG techniques with 1000 documents.

This module tests all implemented RAG techniques with 1000 documents using
pytest fixtures to properly handle setup and teardown.
"""

import pytest
import logging
import os
import time
import uuid
import random
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom markers
pytestmark = [
    pytest.mark.force_testcontainer,  # Always use testcontainer
    pytest.mark.document_count(1000)  # Mark as 1000 document test
]

# Make sure environment variables are set for all tests in this module
os.environ["TEST_IRIS"] = "true" 
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"  # Use mock embeddings for faster tests
os.environ["COLLECT_PERFORMANCE_METRICS"] = "true"

@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    """Setup environment for 1000 document tests."""
    logger.info("=== Starting comprehensive 1000 document test session ===")
    logger.info(f"TEST_DOCUMENT_COUNT: {os.environ.get('TEST_DOCUMENT_COUNT')}")
    logger.info(f"USE_MOCK_EMBEDDINGS: {os.environ.get('USE_MOCK_EMBEDDINGS')}")
    
    yield
    
    logger.info("=== Completed comprehensive 1000 document test session ===")

@pytest.fixture(scope="module")
def mock_query_vector():
    """Generate a mock query vector for testing."""
    return [random.random() for _ in range(384)]  # Standard embedding size

@pytest.fixture(scope="module")
def sample_queries():
    """Provide a list of sample queries for testing."""
    return [
        "What is the main function of the human immune system?",
        "Explain the pathophysiology of COVID-19",
        "What are the symptoms of diabetes?",
        "How do vaccines work?",
        "What treatments are available for cancer?"
    ]

def test_db_connection(iris_testcontainer_connection):
    """Verify database connection works."""
    assert iris_testcontainer_connection is not None
    
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        logger.info("Database connection verified")

def test_load_1000_documents(iris_testcontainer_connection):
    """Load 1000 documents for the test suite."""
    assert iris_testcontainer_connection is not None
    start_time = time.time()
    
    # Clear existing documents
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("DELETE FROM SourceDocuments")
        logger.info("Cleared existing documents")
    
    # Generate and insert 1000 documents in batches
    batch_size = 50
    total_docs = 1000
    batches = total_docs // batch_size
    
    logger.info(f"Inserting {total_docs} documents in {batches} batches of {batch_size}...")
    
    for batch in range(batches):
        batch_docs = []
        for i in range(batch_size):
            doc_idx = batch * batch_size + i
            doc_id = f"doc_1000_{doc_idx}"
            title = f"Test Medical Document {doc_idx}"
            
            # Create content with medical terminology for more realistic testing
            if doc_idx % 5 == 0:
                content = f"This document discusses immune system functions. The immune system defends against infections and diseases through a network of cells, tissues, and organs."
            elif doc_idx % 5 == 1:
                content = f"COVID-19 is caused by SARS-CoV-2 virus. It primarily affects the respiratory system and can cause a range of symptoms from mild to severe."
            elif doc_idx % 5 == 2:
                content = f"Diabetes is a chronic condition affecting how the body processes blood sugar. Symptoms include increased thirst, frequent urination, and fatigue."
            elif doc_idx % 5 == 3:
                content = f"Vaccines work by stimulating the immune system to recognize and fight specific pathogens. They contain weakened or inactivated parts of a particular organism."
            else:
                content = f"Cancer treatments include surgery, chemotherapy, radiation therapy, immunotherapy, and targeted therapy depending on the type and stage."
            
            # Create mock embedding as a serialized vector
            mock_embedding = ",".join([str(random.random()) for _ in range(10)])
            
            batch_docs.append((doc_id, title, content, mock_embedding))
        
        # Insert batch with embeddings
        with iris_testcontainer_connection.cursor() as cursor:
            for doc in batch_docs:
                cursor.execute(
                    "INSERT INTO SourceDocuments (doc_id, title, content, embedding) VALUES (?, ?, ?, ?)",
                    doc
                )
        
        # Log progress
        if (batch + 1) % 5 == 0 or batch == 0:
            logger.info(f"Inserted {(batch + 1) * batch_size} documents...")
    
    # Verify final count
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        assert count == total_docs, f"Expected {total_docs} documents, found {count}"
    
    end_time = time.time()
    duration = end_time - start_time
    docs_per_second = total_docs / duration
    
    logger.info(f"Successfully loaded {count} documents in {duration:.2f} seconds")
    logger.info(f"Performance: {docs_per_second:.2f} docs/sec")
    
    return total_docs

def test_basic_rag_with_1000_docs(iris_testcontainer_connection, mock_query_vector, sample_queries):
    """Test Basic RAG technique with 1000 documents."""
    from basic_rag.pipeline import BasicRAGPipeline
    
    # Load 1000 documents first
    doc_count = test_load_1000_documents(iris_testcontainer_connection)
    assert doc_count == 1000
    
    logger.info("Testing Basic RAG with 1000 documents...")
    
    # Create pipeline with mock functions
    pipeline = BasicRAGPipeline(
        iris_connector=iris_testcontainer_connection,
        embedding_func=lambda x: [np.random.rand(384).tolist()],
        llm_func=lambda prompt: f"Answer based on the provided context with {prompt.count('Context:') + prompt.count('content')} documents"
    )
    
    # Test with a sample query
    query = sample_queries[0]
    logger.info(f"Running Basic RAG query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query, top_k=10)
    duration = time.time() - start_time
    
    assert result is not None
    assert "Answer based on" in result["answer"]
    assert "retrieved_documents" in result
    
    logger.info(f"Basic RAG query completed in {duration:.2f} seconds")
    logger.info(f"Basic RAG result: {result['answer']}")
    logger.info(f"Context documents used: {len(result['retrieved_documents'])}")
    
    # Test multiple queries to verify stability
    for i, query in enumerate(sample_queries[1:3]):
        result = pipeline.run(query, top_k=5)
        assert result is not None, f"Query {i+1} failed"
        logger.info(f"Additional query {i+1} successful")
    
    logger.info("Basic RAG with 1000 documents test passed")

def test_colbert_with_1000_docs(iris_testcontainer_connection, sample_queries):
    """Test ColBERT technique with 1000 documents."""
    from colbert.pipeline import ColBERTPipeline
    from colbert.query_encoder import ColBERTQueryEncoder
    
    # Make sure documents are loaded
    doc_count = test_load_1000_documents(iris_testcontainer_connection)
    assert doc_count == 1000
    
    logger.info("Testing ColBERT with 1000 documents...")
    
    # Create mock query encoder (token-level embeddings)
    query_encoder = lambda x: [np.random.rand(384).tolist() for _ in range(len(x.split()))]
    
    # Create pipeline with mock functions
    pipeline = ColBERTPipeline(
        iris_connector=iris_testcontainer_connection,
        colbert_query_encoder=query_encoder,
        llm_func=lambda prompt: f"ColBERT answer based on the provided context with {prompt.count('Context:') + prompt.count('content')} documents",
        client_side_maxsim=True  # Use client-side maxsim for testing
    )
    
    # Test with sample query
    query = sample_queries[0]
    logger.info(f"Running ColBERT query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query, top_k=10)
    duration = time.time() - start_time
    
    assert result is not None
    assert "answer" in result
    assert "ColBERT answer" in result["answer"]
    assert "retrieved_documents" in result
    
    logger.info(f"ColBERT query completed in {duration:.2f} seconds")
    logger.info(f"ColBERT result: {result['answer']}")
    logger.info(f"Context documents used: {len(result['retrieved_documents'])}")
    
    logger.info("ColBERT with 1000 documents test passed")

def test_noderag_with_1000_docs(iris_testcontainer_connection, mock_query_vector, sample_queries):
    """Test NodeRAG technique with 1000 documents."""
    from noderag.pipeline import NodeRAGPipeline
    
    # Make sure documents are loaded
    doc_count = test_load_1000_documents(iris_testcontainer_connection)
    assert doc_count == 1000
    
    logger.info("Testing NodeRAG with 1000 documents...")
    
    # First, let's create some nodes in the knowledge graph
    with iris_testcontainer_connection.cursor() as cursor:
        # Clear existing nodes
        cursor.execute("DELETE FROM KnowledgeGraphNodes")
        
        # Insert test nodes (100 nodes is sufficient for testing)
        node_count = 100
        for i in range(node_count):
            node_id = f"node_{i}"
            node_type = random.choice(["disease", "symptom", "treatment", "organ", "process"])
            node_name = f"Medical {node_type} {i}"
            description = f"This is a {node_type} node with ID {node_id}"
            embedding = ",".join([str(random.random()) for _ in range(10)])
            metadata = "{}"
            
            cursor.execute(
                "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, node_type, node_name, description, embedding, metadata)
            )
        
        # Verify nodes were inserted
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
        count = cursor.fetchone()[0]
        assert count == node_count, f"Expected {node_count} nodes, found {count}"
        logger.info(f"Created {count} test nodes for NodeRAG")
    
    # Create pipeline with correct parameter names
    pipeline = NodeRAGPipeline(
        iris_connector=iris_testcontainer_connection,
        embedding_func=lambda x: np.array([random.random() for _ in range(384)]),
        llm_func=lambda prompt: f"NodeRAG answer based on the provided knowledge graph with {prompt.count('Entity') + prompt.count('Document') + prompt.count('Concept')} nodes"
    )
    
    # Test with sample query
    query = sample_queries[0]
    logger.info(f"Running NodeRAG query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    assert result is not None
    assert "answer" in result
    assert "NodeRAG answer" in result["answer"]
    assert "retrieved_documents" in result
    
    logger.info(f"NodeRAG query completed in {duration:.2f} seconds")
    logger.info(f"NodeRAG result: {result['answer']}")
    logger.info(f"Context items used: {len(result['retrieved_documents'])}")
    
    logger.info("NodeRAG with 1000 documents test passed")

def test_graphrag_with_1000_docs(iris_testcontainer_connection, mock_query_vector, sample_queries):
    """Test GraphRAG technique with 1000 documents."""
    from graphrag.pipeline import GraphRAGPipeline
    
    # Make sure documents and nodes are loaded
    doc_count = test_load_1000_documents(iris_testcontainer_connection)
    assert doc_count == 1000
    
    logger.info("Testing GraphRAG with 1000 documents...")
    
    # First, let's create some nodes and edges in the knowledge graph
    with iris_testcontainer_connection.cursor() as cursor:
        # We'll reuse the nodes from the NodeRAG test
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
        node_count = cursor.fetchone()[0]
        
        if node_count < 100:
            # If nodes weren't created in the previous test, create them now
            cursor.execute("DELETE FROM KnowledgeGraphNodes")
            
            # Insert test nodes (100 nodes is sufficient for testing)
            node_count = 100
            for i in range(node_count):
                node_id = f"node_{i}"
                node_type = random.choice(["disease", "symptom", "treatment", "organ", "process"])
                node_name = f"Medical {node_type} {i}"
                description = f"This is a {node_type} node with ID {node_id}"
                embedding = ",".join([str(random.random()) for _ in range(10)])
                metadata = "{}"
                
                cursor.execute(
                    "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (node_id, node_type, node_name, description, embedding, metadata)
                )
            
            # Verify nodes were inserted
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
            node_count = cursor.fetchone()[0]
            assert node_count == 100, f"Expected 100 nodes, found {node_count}"
            logger.info(f"Created {node_count} test nodes for GraphRAG")
        
        # Clear existing edges
        cursor.execute("DELETE FROM KnowledgeGraphEdges")
        
        # Insert test edges (300 edges to create a reasonably connected graph)
        edge_count = 300
        for i in range(edge_count):
            edge_id = i
            # Select random nodes for source and target
            source_id = f"node_{random.randint(0, node_count-1)}"
            target_id = f"node_{random.randint(0, node_count-1)}"
            # Avoid self-loops
            while source_id == target_id:
                target_id = f"node_{random.randint(0, node_count-1)}"
                
            rel_type = random.choice(["causes", "treats", "located_in", "part_of", "related_to"])
            weight = random.uniform(0.1, 1.0)
            properties = "{}"
            
            cursor.execute(
                "INSERT INTO KnowledgeGraphEdges (edge_id, source_node_id, target_node_id, relationship_type, weight, properties_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (edge_id, source_id, target_id, rel_type, weight, properties)
            )
        
        # Verify edges were inserted
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
        count = cursor.fetchone()[0]
        assert count == edge_count, f"Expected {edge_count} edges, found {count}"
        logger.info(f"Created {count} test edges for GraphRAG")
    
    # Create pipeline with correct parameter names
    pipeline = GraphRAGPipeline(
        iris_connector=iris_testcontainer_connection,
        embedding_func=lambda x: np.random.rand(384).tolist(),
        llm_func=lambda prompt: f"GraphRAG answer based on information from a knowledge graph with {prompt.count('Context:') + prompt.count('Knowledge Graph:')} nodes"
    )
    
    # Test with sample query
    query = sample_queries[0]
    logger.info(f"Running GraphRAG query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    assert result is not None
    assert "answer" in result
    assert "GraphRAG answer" in result["answer"]
    assert "retrieved_documents" in result
    
    logger.info(f"GraphRAG query completed in {duration:.2f} seconds")
    logger.info(f"GraphRAG result: {result['answer']}")
    logger.info(f"Context items used: {len(result['retrieved_documents'])}")
    
    logger.info("GraphRAG with 1000 documents test passed")

def test_context_reduction_with_1000_docs(iris_testcontainer_connection, sample_queries):
    """Test Context Reduction technique with 1000 documents."""
    from common.context_reduction import reduce_context
    
    # Make sure documents are loaded
    doc_count = test_load_1000_documents(iris_testcontainer_connection)
    assert doc_count == 1000
    
    logger.info("Testing Context Reduction with 1000 documents...")
    
    # First, we need to retrieve some documents to reduce
    with iris_testcontainer_connection.cursor() as cursor:
        # Get 20 random documents to test with
        cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT 20")
        docs = cursor.fetchall()
        
        # Convert to a list of dictionaries for the context reduction function
        context_docs = [
            {"id": doc[0], "title": doc[1], "content": doc[2]} 
            for doc in docs
        ]
    
    # Test the context reduction function
    query = sample_queries[0]
    logger.info(f"Running Context Reduction for query: '{query}'")
    
    start_time = time.time()
    reduced_context = reduce_context(
        query=query,
        documents=context_docs,
        llm_func=lambda query, context: "This is a mock summary of the reduced context"
    )
    duration = time.time() - start_time
    
    assert reduced_context is not None
    assert isinstance(reduced_context, str)
    assert len(reduced_context) > 0
    
    logger.info(f"Context Reduction completed in {duration:.2f} seconds")
    logger.info(f"Original context size: {sum(len(doc['content']) for doc in context_docs)} characters")
    logger.info(f"Reduced context size: {len(reduced_context)} characters")
    
    logger.info("Context Reduction with 1000 documents test passed")
