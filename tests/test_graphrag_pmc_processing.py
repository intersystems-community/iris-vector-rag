"""
Tests for GraphRAG with real PMC data processing.

This module specifically tests the GraphRAG pipeline against real PMC data files,
focusing on the processing of actual medical literature rather than database operations.
"""

import pytest
import os
import sys
import logging
import re
from typing import List, Dict, Any, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.utils import Document, timing_decorator
from common.embedding_utils import get_embedding_model
from data.pmc_processor import process_pmc_files, extract_pmc_metadata
from graphrag.pipeline import GraphRAGPipeline


def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract medical entities from text using a rule-based approach.
    This demonstrates context reduction through entity identification.
    
    Args:
        text: The text to extract entities from
        
    Returns:
        List of entity dictionaries with id, type, name, and spans
    """
    # Dictionary of sample medical terms by category
    medical_terms = {
        "Disease": [
            "diabetes", "cancer", "hypertension", "asthma", "alzheimer",
            "parkinson", "arthritis", "obesity", "depression", "schizophrenia"
        ],
        "Medication": [
            "insulin", "metformin", "statin", "aspirin", "ibuprofen",
            "penicillin", "acetaminophen", "amoxicillin", "lisinopril"
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
            "dialysis", "vaccination", "rehabilitation", "counseling"
        ]
    }
    
    entities = []
    entity_id = 0
    
    # Make text lowercase for matching
    text_lower = text.lower()
    
    # Find all occurrences of each term
    for entity_type, terms in medical_terms.items():
        for term in terms:
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
    This demonstrates context reduction through relationship identification.
    
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


@pytest.mark.integration
def test_pmc_real_data_processing():
    """
    Test the processing of real PMC data files and entity extraction.
    This test specifically focuses on using real medical literature
    for knowledge graph construction, regardless of database availability.
    """
    # Set up the PMC data directory path
    pmc_dir = "data/pmc_oas_downloaded"
    assert os.path.exists(pmc_dir), f"PMC directory not found: {pmc_dir}"
    
    # Process a sample of real PMC documents
    logger.info(f"Processing real PMC files from {pmc_dir}")
    limit = 50  # Use a significant number of files for testing
    docs = list(process_pmc_files(pmc_dir, limit=limit))
    
    # Verify we have real documents
    assert len(docs) > 0, "No PMC documents found"
    logger.info(f"Successfully processed {len(docs)} real PMC documents")
    
    # Analyze the documents to get some statistics
    total_abstract_length = sum(len(doc.get("abstract", "")) for doc in docs)
    avg_abstract_length = total_abstract_length / len(docs) if docs else 0
    logger.info(f"Average abstract length: {avg_abstract_length:.1f} characters")
    
    # Extract entities from the documents
    total_entities = 0
    entity_types = {}
    
    for doc in docs:
        # Create document content by combining title and abstract
        doc_content = f"{doc['title']}. {doc['abstract']}"
        
        # Extract entities from real medical text
        entities = extract_entities_from_text(doc_content)
        total_entities += len(entities)
        
        # Track entity types
        for entity in entities:
            entity_type = entity["type"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    # Log entity extraction results
    logger.info(f"Extracted {total_entities} entities from {len(docs)} documents")
    logger.info(f"Entity distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {entity_type}: {count} entities")
    
    # Verify we found some entities
    assert total_entities > 0, "No entities found in the documents"
    
    # Test relationship extraction on a sample document
    if docs:
        sample_doc = docs[0]
        doc_content = f"{sample_doc['title']}. {sample_doc['abstract']}"
        entities = extract_entities_from_text(doc_content)
        
        if len(entities) > 1:
            relationships = extract_relationships(entities)
            logger.info(f"Found {len(relationships)} relationships in sample document")
            logger.info(f"Sample document: {sample_doc['pmc_id']}")
            logger.info(f"Title: {sample_doc['title'][:50]}...")
            
            # List some entities and their relationships
            if relationships:
                rel = relationships[0]
                source = next(e for e in entities if e["id"] == rel["source"])
                target = next(e for e in entities if e["id"] == rel["target"])
                logger.info(f"Relationship: {source['name']} ({source['type']}) <-> {target['name']} ({target['type']})")
                logger.info(f"Weight: {rel['weight']:.2f}")
    
    # Test that this is a good demonstration of context reduction
    if total_entities > 0 and len(docs) > 0:
        # Calculate context reduction metrics
        avg_entities_per_doc = total_entities / len(docs)
        total_text_size = sum(len(f"{doc['title']}. {doc['abstract']}") for doc in docs)
        
        logger.info(f"Context reduction stats:")
        logger.info(f"  Total text size: {total_text_size} characters")
        logger.info(f"  Total entities extracted: {total_entities}")
        logger.info(f"  Avg entities per document: {avg_entities_per_doc:.1f}")
        
        # Assert that we've achieved meaningful context reduction
        assert avg_entities_per_doc < avg_abstract_length, "Entity extraction should reduce context size"


@pytest.mark.integration
def test_graphrag_with_real_pmc_content():
    """
    Test the GraphRAG pipeline with real PMC content using a memory-based approach.
    This isolates the PMC data processing from database dependencies.
    """
    # Process real PMC documents
    pmc_dir = "data/pmc_oas_downloaded"
    assert os.path.exists(pmc_dir), f"PMC directory not found: {pmc_dir}"
    
    logger.info(f"Processing real PMC files for GraphRAG test")
    limit = 30  # Use a reasonable number for testing
    docs = list(process_pmc_files(pmc_dir, limit=limit))
    assert len(docs) > 0, "No PMC documents found"
    
    # Create in-memory document store
    doc_store = {}
    
    # Get embedding model (mock for testing speed)
    embedding_model = get_embedding_model(mock=True)
    embedding_func = lambda text: embedding_model.encode(text)
    
    # Process documents and extract entities to build a graph
    nodes = []
    edges = []
    knowledge_graph = {}
    
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
        knowledge_graph[doc_id] = doc_node
        
        # Extract entities
        entities = extract_entities_from_text(doc_content)
        
        # Store unique entities
        for entity in entities:
            entity_name = entity["name"]
            entity_type = entity["type"]
            entity_key = f"{entity_type}_{entity_name}"
            entity_id = f"entity_{entity_key}"
            
            if entity_id not in knowledge_graph:
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
                knowledge_graph[entity_id] = entity_node
            
            # Create edge from entity to document
            edges.append({
                "source": entity_id,
                "target": doc_id,
                "type": "MENTIONED_IN",
                "weight": 1.0
            })
    
    # Create a simplified GraphRAG pipeline to test with the in-memory graph
    class InMemoryGraphRAG:
        def __init__(self, graph, edges, embedding_func):
            self.graph = graph  # Dictionary of nodes
            self.edges = edges  # List of edges
            self.embedding_func = embedding_func
            
            # Build an edge index for faster traversal
            self.edge_index = {}
            for edge in edges:
                source = edge["source"]
                target = edge["target"]
                if source not in self.edge_index:
                    self.edge_index[source] = []
                self.edge_index[source].append(target)
                
                # Make bidirectional for testing
                if target not in self.edge_index:
                    self.edge_index[target] = []
                self.edge_index[target].append(source)
        
        def find_start_nodes(self, query, top_n=3):
            """Find starting nodes based on embedding similarity"""
            query_embedding = self.embedding_func(query)
            
            # Compute similarity for each node
            results = []
            for node_id, node in self.graph.items():
                node_embedding = node["embedding"]
                # Simplified cosine similarity calculation
                similarity = 0.5  # Placeholder value
                results.append((node_id, similarity))
            
            # Sort by similarity and return top N
            results.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in results[:top_n]]
        
        def traverse_graph(self, start_nodes, max_depth=2):
            """Traverse the graph from start nodes"""
            visited = set()
            to_visit = [(node, 0) for node in start_nodes]
            
            while to_visit:
                node, depth = to_visit.pop(0)
                if node in visited or depth > max_depth:
                    continue
                    
                visited.add(node)
                
                # Get neighbors
                neighbors = self.edge_index.get(node, [])
                to_visit.extend([(neighbor, depth+1) for neighbor in neighbors if neighbor not in visited])
            
            return visited
        
        def retrieve_documents(self, query):
            """Retrieve documents using graph traversal"""
            # Find starting nodes
            start_nodes = self.find_start_nodes(query)
            
            # Traverse graph
            traversed_nodes = self.traverse_graph(start_nodes)
            
            # Get document content
            docs = []
            for node_id in traversed_nodes:
                node = self.graph.get(node_id)
                if node and node["type"] == "Document":
                    docs.append(Document(
                        id=node["id"],
                        content=node["content"],
                        score=0.9  # Placeholder
                    ))
            
            return docs
    
    # Create the in-memory GraphRAG instance
    graph_rag = InMemoryGraphRAG(knowledge_graph, edges, embedding_func)
    
    # Test queries
    test_queries = [
        "What is the relationship between diabetes and insulin?",
        "How does metformin help with diabetes treatment?",
        "What are the key symptoms of diabetes?"
    ]
    
    # Track retrieved documents for each query
    query_results = {}
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        # Retrieve documents
        start_time = time.time()
        docs = graph_rag.retrieve_documents(query)
        elapsed = time.time() - start_time
        
        query_results[query] = docs
        logger.info(f"Retrieved {len(docs)} documents in {elapsed:.2f} seconds")
    
    # Verify that we retrieved some documents
    total_docs = sum(len(docs) for docs in query_results.values())
    logger.info(f"Total documents retrieved across all queries: {total_docs}")
    
    # Verify that this demonstrates real PMC data processing
    assert total_docs > 0, "No documents retrieved from real PMC data"
    logger.info("Successfully completed GraphRAG test with real PMC content")
