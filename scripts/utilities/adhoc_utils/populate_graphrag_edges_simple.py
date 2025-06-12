#!/usr/bin/env python3
"""
Simple GraphRAG edges population script
"""

import sys
import os
import logging
from common.iris_connector import get_iris_connection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_knowledge_graph_edges_simple():
    """
    Populate KnowledgeGraphEdges table using simple approach
    """
    iris = None
    cursor = None
    
    try:
        logger.info("Starting simple GraphRAG edges population...")
        
        # Connect to database
        iris = get_iris_connection()
        cursor = iris.cursor()
        
        # Check current state
        cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
        node_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
        edge_count = cursor.fetchone()[0]
        
        logger.info(f"Current state: {node_count} nodes, {edge_count} edges")
        
        if edge_count > 0:
            logger.info("Edges already exist, skipping population")
            return
        
        # Get all nodes - using correct column names
        cursor.execute("""
            SELECT node_id, node_type, content, metadata
            FROM RAG.KnowledgeGraphNodes
            ORDER BY node_id
        """)
        
        all_nodes = cursor.fetchall()
        logger.info(f"Retrieved {len(all_nodes)} nodes")
        
        # Parse metadata to get source document info
        doc_groups = {}
        for node in all_nodes:
            node_id, node_type, content, metadata = node
            
            # Extract document ID from metadata or use a default grouping
            source_doc_id = "default_doc"
            if metadata:
                # Try to parse JSON metadata for source_doc_id
                try:
                    import json
                    meta_dict = json.loads(metadata)
                    source_doc_id = meta_dict.get('source_doc_id', 'default_doc')
                except:
                    # If parsing fails, group by node_type
                    source_doc_id = node_type or "unknown"
            
            if source_doc_id not in doc_groups:
                doc_groups[source_doc_id] = []
            doc_groups[source_doc_id].append((node_id, content[:50] if content else node_id, node_type))
        
        logger.info(f"Found {len(doc_groups)} documents with entities")
        
        edges_created = 0
        
        # Create edges within each document
        for doc_id, doc_nodes in doc_groups.items():
            if len(doc_nodes) < 2:
                continue
            
            logger.info(f"Processing document {doc_id} with {len(doc_nodes)} entities")
            
            # Create edges between all pairs in the document
            for i, (node1_id, name1, type1) in enumerate(doc_nodes):
                for j, (node2_id, name2, type2) in enumerate(doc_nodes):
                    if i >= j:  # Avoid duplicates and self-loops
                        continue
                    
                    # Create edge ID
                    edge_id = f"edge_{edges_created + 1}"
                    edge_type = "co-occurrence"
                    weight = 0.8  # High weight for co-occurrence
                    
                    # Create metadata
                    metadata = f'{{"source_doc": "{doc_id}", "relationship": "co-occurs with"}}'
                    
                    # Insert edge using correct schema
                    try:
                        cursor.execute("""
                            INSERT INTO RAG.KnowledgeGraphEdges
                            (edge_id, source_node_id, target_node_id, edge_type, weight, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, [edge_id, node1_id, node2_id, edge_type, weight, metadata])
                        
                        edges_created += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to create edge {node1_id}->{node2_id}: {e}")
            
            # Commit after each document
            iris.commit()
        
        logger.info(f"GraphRAG edges population complete! Created {edges_created} edges")
        
        # Final verification
        cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
        final_edge_count = cursor.fetchone()[0]
        logger.info(f"Final edge count: {final_edge_count}")
        
    except Exception as e:
        logger.error(f"Error populating GraphRAG edges: {e}")
        if iris:
            iris.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if iris:
            iris.close()

if __name__ == "__main__":
    populate_knowledge_graph_edges_simple()