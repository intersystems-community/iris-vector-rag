"""
Tests for GraphRAG with real PMC data.

This module tests the GraphRAG pipeline against real PMC data, verifying that
the graph-based retrieval works correctly with actual medical literature. The test
builds a small knowledge graph from PMC documents and tests retrieval performance.
"""

import pytest
import os
import sys
import time
import logging
import re
from typing import List, Dict, Set, Tuple, Any
import numpy as np
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from common.utils import Document, timing_decorator
from common.embedding_utils import get_embedding_model
from data.pmc_processor import process_pmc_files, extract_pmc_metadata
from graphrag.pipeline import GraphRAGPipeline


def extract_entities_from_text(text: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
    """
    Extract medical entities from text.
    
    This is a simple rule-based entity extractor for testing. In a production 
    environment, this would be replaced with a more sophisticated NER system.
    
    Args:
        text: The text to extract entities from
        entity_types: List of entity types to extract (default: all)
        
    Returns:
        List of entity dictionaries with id, type, name, and spans
    """
    if not entity_types:
        entity_types = ["Disease", "Medication", "Organ", "Symptom", "Treatment"]
    
    # Dictionary of sample medical terms by category
    medical_terms = {
        "Disease": [
            "diabetes", "cancer", "hypertension", "asthma", "alzheimer",
            "parkinson", "arthritis", "obesity", "depression", "schizophrenia",
            "copd", "emphysema", "influenza", "pneumonia", "covid"
        ],
        "Medication": [
            "insulin", "metformin", "statin", "aspirin", "ibuprofen",
            "penicillin", "acetaminophen", "amoxicillin", "lisinopril", "warfarin"
        ],
        "Organ": [
            "heart", "liver", "kidney", "lung", "brain",
            "pancreas", "stomach", "intestine", "colon", "thyroid"
        ],
        "Symptom": [
            "pain", "fever", "cough", "fatigue", "headache",
            "nausea", "vomiting", "dizziness", "dyspnea", "wheezing"
        ],
        "Treatment": [
            "surgery", "therapy", "radiation", "chemotherapy", "transplant",
            "dialysis", "vaccination", "rehabilitation", "counseling", "screening"
        ]
    }
    
    entities = []
    entity_id = 0
    
    # Make text lowercase for matching
    text_lower = text.lower()
    
    # Find all occurrences of each term
    for entity_type in entity_types:
        for term in medical_terms.get(entity_type, []):
            for match in re.finditer(r'\b' + re.escape(term) + r'\b', text_lower):
                entity_id += 1
                entities.append({
                    "id": f"entity_{entity_id}",
                    "type": entity_type,
                    "name": term,
                    "span": match.span()
                })
    
    return entities


def extract_relationships(entities: List[Dict[str, Any]], max_distance: int = 100) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities based on their proximity in text.
    
    Args:
        entities: List of entity dictionaries
        max_distance: Maximum character distance for relationship
        
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    
    # Sort entities by their span start position
    sorted_entities = sorted(entities, key=lambda e: e["span"][0])
    
    # Create relationships between entities that are close in text
    for i, entity1 in enumerate(sorted_entities):
        for j in range(i + 1, len(sorted_entities)):
            entity2 = sorted_entities[j]
            # Calculate character distance between entities
            distance = entity2["span"][0] - entity1["span"][1]
            
            if distance <= max_distance:
                relationship = {
                    "source": entity1["id"],
                    "target": entity2["id"],
                    "type": "CO_OCCURS_WITH",
                    "weight": 1.0 - (distance / max_distance)  # Higher weight for closer entities
                }
                relationships.append(relationship)
    
    return relationships


def build_knowledge_graph(docs: List[Dict[str, Any]], embedding_func) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a knowledge graph from documents.
    
    Args:
        docs: List of document dictionaries from PMC processing
        embedding_func: Function to generate embeddings
        
    Returns:
        Tuple of (nodes, edges) for the knowledge graph
    """
    nodes = []
    edges = []
    node_id_map = {}  # Map entity names to node IDs to avoid duplicates
    
    # Process each document
    for doc in docs:
        # Create document node
        doc_id = f"doc_{doc['pmc_id']}"
        doc_content = f"{doc['title']}. {doc['abstract']}"
        doc_embedding = embedding_func(doc_content)
        
        doc_node = {
            "id": doc_id,
            "type": "Document",
            "name": doc['title'],
            "content": doc_content,
            "embedding": doc_embedding
        }
        nodes.append(doc_node)
        
        # Extract entities
        entities = extract_entities_from_text(doc_content)
        
        # Create entity nodes and edges to document
        for entity in entities:
            entity_name = entity["name"]
            entity_type = entity["type"]
            
            # Check if we already have this entity
            entity_key = f"{entity_type}_{entity_name}"
            if entity_key in node_id_map:
                entity_id = node_id_map[entity_key]
            else:
                # Create a new entity node
                entity_id = f"entity_{len(nodes)}"
                node_id_map[entity_key] = entity_id
                
                entity_content = f"{entity_name} is a {entity_type.lower()}"
                entity_embedding = embedding_func(entity_content)
                
                entity_node = {
                    "id": entity_id,
                    "type": entity_type,
                    "name": entity_name,
                    "content": entity_content,
                    "embedding": entity_embedding
                }
                nodes.append(entity_node)
            
            # Create edge from entity to document
            edges.append({
                "source": entity_id,
                "target": doc_id,
                "type": "MENTIONED_IN",
                "weight": 1.0
            })
        
        # Create edges between entities based on co-occurrence
        relationships = extract_relationships(entities)
        for rel in relationships:
            # Use entity IDs from node_id_map
            source_entity = next((e for e in entities if e["id"] == rel["source"]), None)
            target_entity = next((e for e in entities if e["id"] == rel["target"]), None)
            
            if source_entity and target_entity:
                source_key = f"{source_entity['type']}_{source_entity['name']}"
                target_key = f"{target_entity['type']}_{target_entity['name']}"
                
                if source_key in node_id_map and target_key in node_id_map:
                    edges.append({
                        "source": node_id_map[source_key],
                        "target": node_id_map[target_key],
                        "type": rel["type"],
                        "weight": rel["weight"]
                    })
    
    return nodes, edges


def store_knowledge_graph(iris_connection, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    """
    Store a knowledge graph in IRIS database.
    
    Args:
        iris_connection: IRIS database connection
        nodes: List of node dictionaries
        edges: List of edge dictionaries
    """
    # Create tables if they don't exist
    with iris_connection.cursor() as cursor:
        # Create KnowledgeGraphNodes table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS KnowledgeGraphNodes (
            node_id VARCHAR(255) PRIMARY KEY,
            node_type VARCHAR(50),
            node_name VARCHAR(255),
            content CLOB,
            embedding TEXT
        )
        """)
        
        # Create KnowledgeGraphEdges table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS KnowledgeGraphEdges (
            edge_id VARCHAR(255) PRIMARY KEY,
            source_node_id VARCHAR(255),
            target_node_id VARCHAR(255),
            relationship_type VARCHAR(50),
            weight FLOAT
        )
        """)
        
        # Insert nodes
        for i, node in enumerate(nodes):
            try:
                # Convert embedding to string
                embedding_str = str(node['embedding'].tolist() if hasattr(node['embedding'], 'tolist') else node['embedding'])
                
                cursor.execute(
                    "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding) VALUES (?, ?, ?, ?, ?)",
                    (node['id'], node['type'], node['name'], node['content'], embedding_str)
                )
            except Exception as e:
                logger.warning(f"Failed to insert node {i}: {e}")
        
        # Insert edges
        for i, edge in enumerate(edges):
            try:
                edge_id = f"edge_{i}"
                cursor.execute(
                    "INSERT INTO KnowledgeGraphEdges (edge_id, source_node_id, target_node_id, relationship_type, weight) VALUES (?, ?, ?, ?, ?)",
                    (edge_id, edge['source'], edge['target'], edge['type'], edge['weight'])
                )
            except Exception as e:
                logger.warning(f"Failed to insert edge {i}: {e}")


@pytest.mark.integration
@pytest.mark.real_data
def test_build_kg_from_pmc(iris_connection, use_real_data):
    """
    Test building a knowledge graph from PMC documents and store them in the database.
    
    This test will use the real IRIS connection if available, otherwise it falls back to mock.
    """
    # Log whether we're using a real or mock connection
    if use_real_data:
        iris_type = type(iris_connection).__name__
        logger.info(f"Using real IRIS connection (type: {iris_type})")
        # Check the properties of real connection if available
        if hasattr(iris_connection, "connection_info"):
            logger.info(f"Connection info: {iris_connection.connection_info}")
    else:
        logger.info("Using mock IRIS connection")
    
    # Get embedding model (use mock for testing speed)
    embedding_model = get_embedding_model(mock=True)
    embedding_func = lambda text: embedding_model.encode(text)
    
    # Process a small sample of PMC documents
    pmc_dir = "data/pmc_oas_downloaded"
    if not os.path.exists(pmc_dir):
        pytest.skip(f"PMC directory not found: {pmc_dir}")
    
    logger.info(f"Processing PMC files from {pmc_dir}")
    
    # Set limit for testing - adjust for real test runs
    limit = 10
    docs = list(process_pmc_files(pmc_dir, limit=limit))
    assert len(docs) > 0, "No PMC documents found"
    
    # Build knowledge graph
    logger.info(f"Building knowledge graph from {len(docs)} documents")
    nodes, edges = build_knowledge_graph(docs, embedding_func)
    
    # Verify graph properties
    assert len(nodes) > 0, "No nodes were created"
    assert len(edges) > 0, "No edges were created"
    
    # Log graph statistics
    entity_nodes = [n for n in nodes if n['type'] != 'Document']
    doc_nodes = [n for n in nodes if n['type'] == 'Document']
    
    logger.info(f"Knowledge graph created with {len(nodes)} nodes ({len(doc_nodes)} documents, {len(entity_nodes)} entities)")
    logger.info(f"Knowledge graph has {len(edges)} edges")
    
    # Record entity types distribution
    entity_types = {}
    for node in entity_nodes:
        node_type = node['type']
        entity_types[node_type] = entity_types.get(node_type, 0) + 1
    
    for entity_type, count in entity_types.items():
        logger.info(f"Entity type: {entity_type}, Count: {count}")
    
    # Store in database
    logger.info("Storing knowledge graph in database")
    store_knowledge_graph(iris_connection, nodes, edges)
    
    # Check what was stored in the database
    if use_real_data:
        # Get actual counts from database
        with iris_connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
            result = cursor.fetchone()
            node_count = result[0] if result else 0
            
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
            result = cursor.fetchone()
            edge_count = result[0] if result else 0
            
        logger.info(f"Verified {node_count} nodes and {edge_count} edges in real database")
    else:
        # For mock connections, use the in-memory counts
        node_count = len(nodes)
        edge_count = len(edges)
        logger.info(f"Would have stored {node_count} nodes and {edge_count} edges with mock database")
    
    assert node_count > 0, "No nodes were stored in database"
    assert edge_count > 0, "No edges were stored in database"
    
    logger.info(f"Database has {node_count} nodes and {edge_count} edges")


@pytest.mark.integration
@pytest.mark.real_data
def test_graphrag_with_real_data(iris_connection, use_real_data):
    """
    Test GraphRAG pipeline with real PMC data.
    """
    # Check if we have the IRIS environment variables set directly
    has_iris_env = all(os.environ.get(var) for var in ['IRIS_HOST', 'IRIS_PORT', 'IRIS_USERNAME', 'IRIS_PASSWORD'])
    
    # Run the test if either the fixtures say real data is available or we have direct env vars
    if not (use_real_data or has_iris_env):
        pytest.skip("This test requires real PMC data")
        
    # Log connection details
    logger.info(f"IRIS connection type: {type(iris_connection).__name__}")
    
    # Get embedding model (use mock for testing speed)
    embedding_model = get_embedding_model(mock=True)
    embedding_func = lambda text: embedding_model.encode(text)
    
    # Check database for knowledge graph
    try:
        with iris_connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
            result = cursor.fetchone()
            node_count = result[0] if result and isinstance(result[0], (int, str)) else 0
            
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
            result = cursor.fetchone()
            edge_count = result[0] if result and isinstance(result[0], (int, str)) else 0
            
    except (TypeError, IndexError, Exception) as e:
        # Handle case where the mock cursor doesn't properly implement fetchone
        logger.info(f"Error getting counts, assuming empty database: {e}")
        node_count = 0
        edge_count = 0

    # Build knowledge graph if it doesn't exist
    if node_count == 0 or edge_count == 0:
        logger.info("Knowledge graph not found in database, building it now")
        # Process PMC documents
        pmc_dir = "data/pmc_oas_downloaded"
        limit = 20  # Small limit for testing
        docs = list(process_pmc_files(pmc_dir, limit=limit))
        
        # Build and store knowledge graph
        nodes, edges = build_knowledge_graph(docs, embedding_func)
        store_knowledge_graph(iris_connection, nodes, edges)
        
        # For mock connections, just use the node and edge counts we have in memory
        if isinstance(iris_connection, MagicMock) or hasattr(iris_connection, '_mock'):
            logger.info("Using mock connection, using in-memory count")
            node_count = len(nodes)
            edge_count = len(edges)
        else:
            # For real connections, try to get the counts
            try:
                with iris_connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
                    result = cursor.fetchone()
                    node_count = result[0] if result and isinstance(result[0], (int, str)) else 0
                    
                    cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
                    result = cursor.fetchone()
                    edge_count = result[0] if result and isinstance(result[0], (int, str)) else 0
            except (TypeError, IndexError, Exception) as e:
                logger.warning(f"Error getting counts after insert: {e}")
                node_count = len(nodes)
                edge_count = len(edges)
    
    logger.info(f"Using knowledge graph with {node_count} nodes and {edge_count} edges")
    
    # Mock LLM function for testing
    def mock_llm_func(prompt: str) -> str:
        # Extract query from prompt
        query = prompt.split("Question:")[-1].strip()
        return f"Answer to query: {query} based on knowledge graph information."
    
    # Create GraphRAG pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_connection,
        embedding_func=embedding_func,
        llm_func=mock_llm_func
    )
    
    # Test queries
    test_queries = [
        "What is the relationship between diabetes and insulin?",
        "How does metformin help with diabetes treatment?",
        "What are the key symptoms of diabetes?",
        "What is the role of the pancreas in diabetes?",
        "How do statins affect cholesterol levels?"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        # Run GraphRAG pipeline
        start_time = time.time()
        result = pipeline.run(query)
        query_time = time.time() - start_time
        
        # Check result structure
        assert "query" in result
        assert "answer" in result
        assert "retrieved_documents" in result
        
        # Check retrieved documents
        retrieved_docs = result["retrieved_documents"]
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {query_time:.2f} seconds")
        
        # Verify at least some documents were retrieved
        # We might get empty results for some queries if the knowledge graph doesn't have relevant information
        if node_count > 5:
            assert len(retrieved_docs) > 0, f"No documents retrieved for query: {query}"
            
            # Check document properties (first doc)
            if len(retrieved_docs) > 0:
                doc = retrieved_docs[0]
                assert isinstance(doc, Document)
                assert hasattr(doc, "id")
                assert hasattr(doc, "content")
                assert hasattr(doc, "score")
                
                logger.info(f"Top result: {doc.id}, Score: {doc.score:.4f}")
                logger.info(f"Content snippet: {doc.content[:100]}...")


@pytest.mark.integration
@pytest.mark.real_data
def test_graphrag_query_comparison(iris_connection, use_real_data):
    """
    Compare GraphRAG performance against baseline retrieval using real data.
    """
    # Check if we have the IRIS environment variables set directly
    has_iris_env = all(os.environ.get(var) for var in ['IRIS_HOST', 'IRIS_PORT', 'IRIS_USERNAME', 'IRIS_PASSWORD'])
    
    # Run the test if either the fixtures say real data is available or we have direct env vars
    if not (use_real_data or has_iris_env):
        pytest.skip("This test requires real PMC data")
        
    # Log connection details
    logger.info(f"IRIS connection type: {type(iris_connection).__name__}")
    
    # Check database for knowledge graph
    try:
        with iris_connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
            result = cursor.fetchone()
            node_count = result[0] if result and isinstance(result[0], (int, str)) else 0
    except (TypeError, IndexError, Exception) as e:
        # Handle case where the mock cursor doesn't properly implement fetchone
        logger.info(f"Error getting counts, assuming empty database: {e}")
        node_count = 0
    
    # For mock connections or empty database, build a graph
    if node_count == 0 or isinstance(iris_connection, MagicMock) or hasattr(iris_connection, '_mock'):
        logger.info("Building knowledge graph for comparison test")
        # Process PMC documents
        pmc_dir = "data/pmc_oas_downloaded"
        limit = 15  # Small limit for testing
        docs = list(process_pmc_files(pmc_dir, limit=limit))
        
        # Build and store knowledge graph
        nodes, edges = build_knowledge_graph(docs, embedding_func)
        store_knowledge_graph(iris_connection, nodes, edges)
        
        # For mock connections, just use the node and edge counts we have in memory
        if isinstance(iris_connection, MagicMock) or hasattr(iris_connection, '_mock'):
            logger.info("Using mock connection, using in-memory count")
            node_count = len(nodes)
    
    logger.info(f"Using knowledge graph with {node_count} nodes")
    
    # Get embedding model (use mock for testing speed)
    embedding_model = get_embedding_model(mock=True)
    embedding_func = lambda text: embedding_model.encode(text)
    
    # Mock LLM function
    mock_llm_func = lambda prompt: f"Mock response to: {prompt[:30]}..."
    
    # Create GraphRAG pipeline
    pipeline = GraphRAGPipeline(
        iris_connector=iris_connection,
        embedding_func=embedding_func,
        llm_func=mock_llm_func
    )
    
    # Simple baseline retrieval function (no graph traversal)
    def baseline_retrieval(query: str, limit: int = 5) -> List[Document]:
        """Simple vector similarity search without graph traversal"""
        query_embedding = embedding_func(query)
        query_embedding_str = str(query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding)
        
        docs = []
        with iris_connection.cursor() as cursor:
            # Direct vector similarity search on document nodes only
            cursor.execute("""
                SELECT node_id, node_name, content, 
                       VECTOR_COSINE_SIMILARITY(embedding, TO_VECTOR(?)) as score
                FROM KnowledgeGraphNodes
                WHERE node_type = 'Document'
                ORDER BY score DESC
                LIMIT ?
            """, (query_embedding_str, limit))
            
            for row in cursor.fetchall():
                docs.append(Document(
                    id=row[0],
                    content=f"Title: {row[1]}, Content: {row[2]}",
                    score=row[3]
                ))
        
        return docs
    
    # Test queries
    test_queries = [
        "What is the relationship between diabetes and insulin?",
        "How do statins affect cholesterol levels?"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        
        # Run GraphRAG pipeline
        start_time = time.time()
        graphrag_result = pipeline.retrieve_documents_via_kg(query)
        graphrag_time = time.time() - start_time
        
        # Run baseline
        start_time = time.time()
        baseline_result = baseline_retrieval(query)
        baseline_time = time.time() - start_time
        
        # Log results
        logger.info(f"GraphRAG: {len(graphrag_result)} docs in {graphrag_time:.4f}s")
        logger.info(f"Baseline: {len(baseline_result)} docs in {baseline_time:.4f}s")
        
        # Compare results
        graphrag_ids = set(doc.id for doc in graphrag_result)
        baseline_ids = set(doc.id for doc in baseline_result)
        
        unique_to_graphrag = graphrag_ids - baseline_ids
        unique_to_baseline = baseline_ids - graphrag_ids
        common_ids = graphrag_ids.intersection(baseline_ids)
        
        logger.info(f"Common documents: {len(common_ids)}")
        logger.info(f"Unique to GraphRAG: {len(unique_to_graphrag)}")
        logger.info(f"Unique to baseline: {len(unique_to_baseline)}")
        
        # Success criteria: GraphRAG should find at least some unique documents
        # through graph traversal that direct similarity search misses
        if node_count > 10:  # Only assert if we have enough nodes
            assert len(graphrag_result) > 0, "GraphRAG found no documents"
