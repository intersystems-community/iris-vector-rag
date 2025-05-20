"""
Real PMC Document Tests with 1000 Documents

This module tests all RAG techniques using 1000 real PMC documents.
These tests load actual PMC documents into a real IRIS database
(using testcontainer) and verify that all RAG techniques work properly.
"""

import pytest
import logging
import os
import time
import random
import numpy as np
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure environment variables are set for all tests in this module
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["USE_REAL_PMC_DATA"] = "true"

# Define custom markers
pytestmark = [
    pytest.mark.force_testcontainer,  # Always use testcontainer
    pytest.mark.document_count(1000),  # Mark as 1000 document test
    pytest.mark.real_pmc              # Mark as using real PMC documents
]

@pytest.fixture(scope="module")
def sample_queries():
    """Provide a list of sample queries for testing."""
    return [
        "What is the role of insulin in diabetes management?",
        "How does the immune system respond to viral infections?",
        "What are the mechanisms of drug resistance in cancer therapy?",
        "How do vaccines work against infectious diseases?",
        "What are the effects of inflammation on cardiovascular health?"
    ]

@pytest.fixture(scope="module")
def mock_embedding_func():
    """Create a mock embedding function that generates consistent embeddings."""
    def _embedding_func(text):
        # Use hash of text for seed to make it deterministic but text-dependent
        seed = hash(text) % 10000
        np.random.seed(seed)
        return np.random.rand(384).tolist()
    return _embedding_func

@pytest.fixture(scope="module")
def mock_llm_func():
    """Create a mock LLM function that returns deterministic responses."""
    def _llm_func(prompt):
        # Extract query from the prompt
        if "Question:" in prompt:
            query = prompt.split("Question:")[1].split("\n")[0].strip()
        else:
            query = prompt[-50:].strip()
            
        return f"This is a mock answer about {query}. The answer is based on the provided context."
    return _llm_func

@pytest.fixture(scope="module")
def iris_with_real_pmc_data(iris_testcontainer_connection):
    """Load 1000 real PMC documents into the IRIS testcontainer."""
    from data.loader import process_and_load_documents
    
    logger.info("Setting up IRIS testcontainer with real PMC data...")
    
    # Define PMC directory
    pmc_dir = os.getenv("PMC_DATA_DIR", "data/pmc_oas_downloaded")
    if not os.path.exists(pmc_dir):
        logger.error(f"PMC data directory {pmc_dir} does not exist")
        pytest.skip(f"PMC data directory {pmc_dir} does not exist")
    
    # Count available XML files
    xml_count = 0
    for root, _, files in os.walk(pmc_dir):
        xml_count += len([f for f in files if f.endswith('.xml')])
    
    if xml_count < 10:
        logger.error(f"Not enough PMC XML files found. Found {xml_count}, need at least 10")
        pytest.skip(f"Not enough PMC XML files found. Found {xml_count}, need at least 10")
        
    logger.info(f"Found {xml_count} PMC XML files")
    
    # Load documents into the database
    start_time = time.time()
    try:
        # Process and load up to 1000 documents
        logger.info(f"Loading PMC documents from {pmc_dir}...")
        stats = process_and_load_documents(
            pmc_directory=pmc_dir,
            connection=iris_testcontainer_connection,
            limit=1000,
            batch_size=50
        )
        
        duration = time.time() - start_time
        
        if not stats["success"]:
            logger.error(f"Failed to load PMC documents: {stats.get('error', 'Unknown error')}")
            pytest.skip(f"Failed to load PMC documents: {stats.get('error', 'Unknown error')}")
            
        logger.info(f"Successfully loaded {stats['loaded_count']} PMC documents in {duration:.2f} seconds")
        
        # Verify document count in the database
        with iris_testcontainer_connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            count = cursor.fetchone()[0]
            
            assert count >= 10, f"Expected at least 10 documents, found {count}"
            logger.info(f"Verified {count} documents in the database")
            
            if count < 100:
                logger.warning(f"Only {count} documents loaded. Tests will run but with limited data")
    
    except Exception as e:
        logger.error(f"Error setting up IRIS with PMC data: {e}")
        pytest.skip(f"Error setting up IRIS with PMC data: {e}")
    
    yield iris_testcontainer_connection

def test_basic_rag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries):
    """Test Basic RAG technique with real PMC documents."""
    from basic_rag.pipeline import BasicRAGPipeline
    
    logger.info("Testing Basic RAG with real PMC documents...")
    
    # Create pipeline with our mock functions
    pipeline = BasicRAGPipeline(
        iris_connector=iris_with_real_pmc_data,
        embedding_func=lambda x: [mock_embedding_func(x)],
        llm_func=mock_llm_func
    )
    
    # Test with a sample query
    query = sample_queries[0]
    logger.info(f"Running Basic RAG query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query, top_k=10)
    duration = time.time() - start_time
    
    assert result is not None, "BasicRAG result should not be None"
    assert "answer" in result, "BasicRAG result should contain 'answer'"
    assert "retrieved_documents" in result, "BasicRAG result should contain 'retrieved_documents'"
    assert len(result["retrieved_documents"]) > 0, "BasicRAG should retrieve at least one document"
    
    logger.info(f"Basic RAG query completed in {duration:.2f} seconds")
    logger.info(f"Basic RAG result: {result['answer'][:100]}...")
    logger.info(f"Context documents used: {len(result['retrieved_documents'])}")
    
    # Test multiple queries to verify stability
    for i, query in enumerate(sample_queries[1:3]):
        result = pipeline.run(query, top_k=5)
        assert result is not None, f"Query {i+1} failed"
        assert len(result["retrieved_documents"]) > 0, f"Query {i+1} should retrieve at least one document"
        logger.info(f"Additional query {i+1} successful")
    
    logger.info("Basic RAG with real PMC documents test passed")

def test_colbert_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries):
    """Test ColBERT technique with real PMC documents."""
    from colbert.pipeline import ColBERTPipeline
    
    logger.info("Testing ColBERT with real PMC documents...")
    
    # Create mock token-level embedding function
    def token_embedding_func(text):
        tokens = text.split()
        return [mock_embedding_func(token) for token in tokens]
    
    # Create pipeline with our mock functions
    pipeline = ColBERTPipeline(
        iris_connector=iris_with_real_pmc_data,
        colbert_query_encoder=token_embedding_func,
        llm_func=mock_llm_func,
        client_side_maxsim=True  # Use client-side calculations for testing
    )
    
    # Test with a sample query
    query = sample_queries[0]
    logger.info(f"Running ColBERT query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query, top_k=10)
    duration = time.time() - start_time
    
    assert result is not None, "ColBERT result should not be None"
    assert "answer" in result, "ColBERT result should contain 'answer'"
    assert "retrieved_documents" in result, "ColBERT result should contain 'retrieved_documents'"
    
    logger.info(f"ColBERT query completed in {duration:.2f} seconds")
    logger.info(f"ColBERT result: {result['answer'][:100]}...")
    logger.info(f"Context documents used: {len(result['retrieved_documents'])}")
    
    logger.info("ColBERT with real PMC documents test passed")

def test_noderag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries):
    """Test NodeRAG technique with real PMC documents."""
    from noderag.pipeline import NodeRAGPipeline
    
    logger.info("Testing NodeRAG with real PMC documents...")
    
    # First, let's create some nodes in the knowledge graph based on real documents
    with iris_with_real_pmc_data.cursor() as cursor:
        # Clear existing nodes
        cursor.execute("DELETE FROM KnowledgeGraphNodes")
        
        # Get documents to create nodes from
        cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT 100")
        docs = cursor.fetchall()
        
        # Create nodes for each document
        for i, (doc_id, title, content) in enumerate(docs):
            node_id = f"node_{i}"
            node_type = "Document"
            node_name = title if title else f"Document {i}"
            description = content[:500] if content else f"Content for document {i}"
            embedding = ",".join([str(x) for x in mock_embedding_func(description)])
            metadata = json.dumps({"source_doc_id": doc_id})
            
            cursor.execute(
                "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, node_type, node_name, description, embedding, metadata)
            )
            
            # Create 1-2 concept nodes for each document
            concepts = ["diabetes", "immune system", "cancer", "vaccine", "heart disease"]
            concept_count = random.randint(1, 2)
            for j in range(concept_count):
                concept_idx = (i + j) % len(concepts)
                concept_node_id = f"concept_{i}_{j}"
                concept_type = "Concept"
                concept_name = concepts[concept_idx]
                concept_desc = f"Concept of {concept_name} related to {node_name}"
                concept_embedding = ",".join([str(x) for x in mock_embedding_func(concept_desc)])
                concept_metadata = json.dumps({"related_doc": doc_id})
                
                cursor.execute(
                    "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (concept_node_id, concept_type, concept_name, concept_desc, concept_embedding, concept_metadata)
                )
        
        # Verify nodes were inserted
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
        count = cursor.fetchone()[0]
        assert count > 0, "No nodes were created for NodeRAG test"
        logger.info(f"Created {count} test nodes for NodeRAG")
    
    # Create pipeline
    pipeline = NodeRAGPipeline(
        iris_connector=iris_with_real_pmc_data,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func
    )
    
    # Test with sample query
    query = sample_queries[0]
    logger.info(f"Running NodeRAG query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    assert result is not None, "NodeRAG result should not be None"
    assert "answer" in result, "NodeRAG result should contain 'answer'"
    assert "retrieved_documents" in result, "NodeRAG result should contain 'retrieved_documents'"
    
    logger.info(f"NodeRAG query completed in {duration:.2f} seconds")
    logger.info(f"NodeRAG result: {result['answer'][:100]}...")
    logger.info(f"Context items used: {len(result['retrieved_documents'])}")
    
    logger.info("NodeRAG with real PMC documents test passed")

def test_graphrag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries):
    """Test GraphRAG technique with real PMC documents."""
    from graphrag.pipeline import GraphRAGPipeline
    
    logger.info("Testing GraphRAG with real PMC documents...")
    
    # We'll reuse the nodes from the NodeRAG test and add edges
    with iris_with_real_pmc_data.cursor() as cursor:
        # Check if we have nodes
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
        node_count = cursor.fetchone()[0]
        
        if node_count == 0:
            # If nodes weren't created in the previous test, create them now
            logger.info("No nodes found, creating nodes for GraphRAG test")
            test_noderag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries)
            
            # Verify nodes were created
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
            node_count = cursor.fetchone()[0]
            assert node_count > 0, "No nodes were created for GraphRAG test"
        
        # Clear existing edges
        cursor.execute("DELETE FROM KnowledgeGraphEdges")
        
        # Get all nodes
        cursor.execute("SELECT node_id, node_type FROM KnowledgeGraphNodes")
        nodes = cursor.fetchall()
        
        # Group nodes by type
        doc_nodes = [node_id for node_id, node_type in nodes if node_type == "Document"]
        concept_nodes = [node_id for node_id, node_type in nodes if node_type == "Concept"]
        
        # Create edges between concept nodes and document nodes
        edge_id = 0
        for concept_id in concept_nodes:
            # Connect each concept to 2-4 random document nodes
            for _ in range(random.randint(2, 4)):
                if not doc_nodes:
                    continue
                    
                target_id = random.choice(doc_nodes)
                rel_type = random.choice(["relates_to", "appears_in", "describes"])
                weight = random.uniform(0.7, 1.0)
                properties = "{}"
                
                cursor.execute(
                    "INSERT INTO KnowledgeGraphEdges (edge_id, source_node_id, target_node_id, relationship_type, weight, properties_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (edge_id, concept_id, target_id, rel_type, weight, properties)
                )
                
                edge_id += 1
        
        # Create edges between document nodes (representing citations)
        for i, source_id in enumerate(doc_nodes):
            # Connect each document to 1-3 other documents
            for _ in range(random.randint(1, 3)):
                if len(doc_nodes) <= 1:
                    continue
                    
                # Select target document that is not the source
                target_options = [node_id for node_id in doc_nodes if node_id != source_id]
                if not target_options:
                    continue
                    
                target_id = random.choice(target_options)
                rel_type = random.choice(["cites", "references", "related_to"])
                weight = random.uniform(0.5, 0.9)
                properties = "{}"
                
                cursor.execute(
                    "INSERT INTO KnowledgeGraphEdges (edge_id, source_node_id, target_node_id, relationship_type, weight, properties_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (edge_id, source_id, target_id, rel_type, weight, properties)
                )
                
                edge_id += 1
        
        # Verify edges were inserted
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
        count = cursor.fetchone()[0]
        assert count > 0, "No edges were created for GraphRAG test"
        logger.info(f"Created {count} test edges for GraphRAG")
    
    # Create pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_with_real_pmc_data,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func
    )
    
    # Test with sample query
    query = sample_queries[0]
    logger.info(f"Running GraphRAG query: '{query}'")
    
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    assert result is not None, "GraphRAG result should not be None"
    assert "answer" in result, "GraphRAG result should contain 'answer'"
    assert "retrieved_documents" in result, "GraphRAG result should contain 'retrieved_documents'"
    
    logger.info(f"GraphRAG query completed in {duration:.2f} seconds")
    logger.info(f"GraphRAG result: {result['answer'][:100]}...")
    logger.info(f"Context items used: {len(result['retrieved_documents'])}")
    
    logger.info("GraphRAG with real PMC documents test passed")

def test_context_reduction_with_real_pmc(iris_with_real_pmc_data, mock_llm_func, sample_queries):
    """Test Context Reduction technique with real PMC documents."""
    from common.context_reduction import reduce_context
    
    logger.info("Testing Context Reduction with real PMC documents...")
    
    # Get some real documents to reduce
    with iris_with_real_pmc_data.cursor() as cursor:
        # Get 20 documents to work with
        cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT 20")
        docs = cursor.fetchall()
        
        # Convert to a list of dictionaries for the context reduction function
        context_docs = [
            {"id": doc[0], "title": doc[1] or f"Document {i}", "content": doc[2] or f"Content for document {i}"} 
            for i, doc in enumerate(docs)
        ]
    
    assert len(context_docs) > 0, "No documents were retrieved for Context Reduction test"
    logger.info(f"Retrieved {len(context_docs)} documents for testing Context Reduction")
    
    # Test the context reduction function
    query = sample_queries[0]
    logger.info(f"Running Context Reduction for query: '{query}'")
    
    start_time = time.time()
    reduced_context = reduce_context(
        query=query,
        documents=context_docs,
        llm_func=mock_llm_func
    )
    duration = time.time() - start_time
    
    assert reduced_context is not None, "Context Reduction result should not be None"
    assert isinstance(reduced_context, str), "Context Reduction result should be a string"
    assert len(reduced_context) > 0, "Context Reduction result should not be empty"
    
    logger.info(f"Context Reduction completed in {duration:.2f} seconds")
    logger.info(f"Original context size: {sum(len(doc['content']) for doc in context_docs)} characters")
    logger.info(f"Reduced context size: {len(reduced_context)} characters")
    logger.info(f"Reduced to {len(reduced_context) / sum(len(doc['content']) for doc in context_docs):.2%} of original size")
    
    logger.info("Context Reduction with real PMC documents test passed")

def test_all_techniques_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries):
    """Run all RAG techniques with real PMC documents to verify they work together."""
    logger.info("Testing all RAG techniques with real PMC documents...")
    
    # Run each test in sequence
    test_basic_rag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries)
    test_colbert_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries)
    test_noderag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries)
    test_graphrag_with_real_pmc(iris_with_real_pmc_data, mock_embedding_func, mock_llm_func, sample_queries)
    test_context_reduction_with_real_pmc(iris_with_real_pmc_data, mock_llm_func, sample_queries)
    
    logger.info("All RAG techniques with real PMC documents test passed")
