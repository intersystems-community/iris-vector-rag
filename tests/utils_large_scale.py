"""
Enhanced utilities for large-scale RAG testing

This module extends the standard test utilities to better handle large-scale
testing with 1000+ documents, including improved batch processing, better
error handling, and memory-efficient algorithms.
"""

import logging
import time
import os
import gc
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Generator, Union
from tqdm import tqdm

from common.utils import Document
from tests.utils import (
    load_pmc_documents as base_load_pmc_documents,
    process_pmc_files_in_batches,
    build_knowledge_graph as base_build_knowledge_graph,
    run_standardized_queries
)

logger = logging.getLogger(__name__)

def load_pmc_documents_large_scale(
    connection, 
    limit=1000, 
    pmc_dir="data/pmc_oas_downloaded", 
    batch_size=50,
    max_retries=3,
    collect_metrics=True
) -> Dict[str, Any]:
    """
    Enhanced version of load_pmc_documents optimized for large document sets.
    
    Args:
        connection: IRIS connection
        limit: Maximum number of documents to load
        pmc_dir: Directory containing PMC files
        batch_size: Number of documents to process and insert in each batch
        max_retries: Maximum number of retries for failed operations
        collect_metrics: Whether to collect detailed performance metrics
        
    Returns:
        Dict with document count and performance metrics
    """
    from data.pmc_processor import process_pmc_files
    
    # Process documents in batches with detailed metrics
    logger.info(f"Processing up to {limit} documents from {pmc_dir} (batch size: {batch_size})")
    start_time = time.time()
    
    # Initialize metrics collection
    metrics = {
        "total_documents": 0,
        "successful_inserts": 0,
        "failed_inserts": 0,
        "processing_time": 0,
        "database_time": 0,
        "peak_memory_mb": 0,
        "batches_processed": 0,
        "batch_times": [],
    }
    
    # Create table with updated schema that's compatible with IRIS
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS SourceDocuments (
                doc_id VARCHAR(255) PRIMARY KEY,
                title VARCHAR(1000),
                content LONGVARCHAR,
                authors VARCHAR(1000),
                keywords VARCHAR(1000),
                embedding VARCHAR(8000) NULL
            )
            """)
            connection.commit()
    except Exception as e:
        logger.error(f"Failed to create SourceDocuments table: {e}")
        raise
    
    # Process and insert documents in batches with retry logic
    count = 0
    total_batches = (limit + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_num, docs_batch in enumerate(tqdm(
        process_pmc_files_in_batches(pmc_dir, limit, batch_size),
        total=total_batches,
        desc="Loading documents",
        unit="batch"
    )):
        batch_start = time.time()
        metrics["batches_processed"] += 1
        
        # Track memory usage
        if collect_metrics:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], current_memory)
        
        # Insert batch with retry logic
        retry_count = 0
        batch_success = False
        
        while not batch_success and retry_count < max_retries:
            try:
                # Begin transaction
                if hasattr(connection, 'begin'):
                    connection.begin()
                
                db_start_time = time.time()
                successful_in_batch = 0
                
                with connection.cursor() as cursor:
                    for doc in docs_batch:
                        try:
                            # Convert authors and keywords to strings
                            authors = str(doc.get('authors', []))
                            keywords = str(doc.get('keywords', []))
                            abstract = doc.get('abstract', '')
                            
                            # Limit content length to avoid issues with very large documents
                            if len(abstract) > 32000:  # LONGVARCHAR might have limits
                                abstract = abstract[:32000]
                            
                            cursor.execute(
                                """
                                INSERT INTO SourceDocuments 
                                (doc_id, title, content, authors, keywords) 
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (doc['pmc_id'], doc['title'], abstract, authors, keywords)
                            )
                            successful_in_batch += 1
                        except Exception as e:
                            logger.warning(f"Failed to insert document {doc['pmc_id']}: {e}")
                            metrics["failed_inserts"] += 1
                
                # Commit transaction
                connection.commit()
                
                count += successful_in_batch
                metrics["successful_inserts"] += successful_in_batch
                batch_success = True
                
                db_time = time.time() - db_start_time
                metrics["database_time"] += db_time
                
            except Exception as e:
                # Error handling for the entire batch
                logger.warning(f"Batch {batch_num+1} failed (attempt {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                
                # Rollback transaction if possible
                try:
                    if hasattr(connection, 'rollback'):
                        connection.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
                
                # Wait before retrying
                if retry_count < max_retries:
                    time.sleep(retry_count * 2)  # Exponential backoff
        
        # Batch metrics
        batch_time = time.time() - batch_start
        metrics["processing_time"] += batch_time
        metrics["batch_times"].append(batch_time)
        metrics["total_documents"] += len(docs_batch)
        
        # Show progress
        progress = (batch_num + 1) / total_batches * 100
        eta = (time.time() - start_time) / (batch_num + 1) * (total_batches - batch_num - 1) if batch_num < total_batches - 1 else 0
        
        logger.info(f"Batch {batch_num+1}/{total_batches}: "
                    f"{successful_in_batch}/{len(docs_batch)} docs in {batch_time:.2f}s | "
                    f"Progress: {progress:.1f}% | "
                    f"ETA: {eta:.1f}s | "
                    f"Total loaded: {count}")
        
        # Force garbage collection after each batch to manage memory
        if batch_num % 5 == 0:
            gc.collect()
    
    # Final metrics
    total_time = time.time() - start_time
    
    metrics.update({
        "total_time_seconds": total_time,
        "docs_per_second": count / total_time if total_time > 0 else 0,
        "average_batch_time": sum(metrics["batch_times"]) / len(metrics["batch_times"]) if metrics["batch_times"] else 0,
        "document_count": count
    })
    
    logger.info(f"Large-scale document loading complete: {count}/{limit} documents loaded")
    logger.info(f"Performance: {count/total_time:.1f} docs/s, peak memory: {metrics['peak_memory_mb']:.1f} MB")
    
    return metrics

def build_knowledge_graph_large_scale(
    connection, 
    embedding_func, 
    limit=1000, 
    pmc_dir="data/pmc_oas_downloaded",
    batch_size=50,
    max_entities_per_doc=50,
    collect_metrics=True
) -> Dict[str, Any]:
    """
    Enhanced version of build_knowledge_graph optimized for large document sets.
    
    Args:
        connection: IRIS connection
        embedding_func: Function to generate embeddings
        limit: Maximum number of documents to process
        pmc_dir: Directory containing PMC XML files
        batch_size: Number of documents to process at once
        max_entities_per_doc: Maximum entities to extract per document
        collect_metrics: Whether to collect detailed performance metrics
        
    Returns:
        Dict with node/edge counts and performance metrics
    """
    from data.pmc_processor import process_pmc_files
    
    # Define local utility functions with optimizations for large document sets
    def extract_entities_from_text(text, max_entities=max_entities_per_doc):
        """Extract medical entities with limit on entities per document"""
        # Same medical terms dictionary as the original function
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
        import re
        text_lower = text.lower()
        
        # Process each entity type until we reach the maximum
        for entity_type, terms in medical_terms.items():
            if len(entities) >= max_entities:
                break
                
            for term in terms:
                for match in re.finditer(r'\b' + re.escape(term) + r'\b', text_lower):
                    entity_id += 1
                    entities.append({
                        "id": f"entity_{entity_id}",
                        "type": entity_type,
                        "name": term,
                        "span": match.span()
                    })
                    
                    if len(entities) >= max_entities:
                        break
                
                if len(entities) >= max_entities:
                    break
        
        return entities
    
    def store_kg_in_database_batched(connection, nodes, edges, batch_size=100):
        """Store knowledge graph with batched inserts for better performance"""
        # Make sure tables exist with correct schema for IRIS
        with connection.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS KnowledgeGraphNodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(50),
                node_name VARCHAR(255),
                content LONGVARCHAR,
                embedding VARCHAR(8000)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS KnowledgeGraphEdges (
                edge_id VARCHAR(255) PRIMARY KEY,
                source_node_id VARCHAR(255),
                target_node_id VARCHAR(255),
                relationship_type VARCHAR(50),
                weight FLOAT
            )
            """)
        
        # Insert nodes in batches with progress bar
        node_batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
        
        for i, node_batch in enumerate(tqdm(node_batches, desc="Inserting nodes", unit="batch")):
            try:
                with connection.cursor() as cursor:
                    for node in node_batch:
                        # Convert embedding to string
                        embedding_str = str(node['embedding'].tolist() if hasattr(node['embedding'], 'tolist') else node['embedding'])
                        
                        # Truncate content if needed
                        content = node['content']
                        if len(content) > 32000:  # LONGVARCHAR might have limits
                            content = content[:32000]
                        
                        cursor.execute(
                            """
                            INSERT INTO KnowledgeGraphNodes 
                            (node_id, node_type, node_name, content, embedding) 
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (node['id'], node['type'], node['name'], content, embedding_str)
                        )
                connection.commit()
                logger.info(f"Inserted node batch {i+1}/{len(node_batches)}: {len(node_batch)} nodes")
            except Exception as e:
                logger.error(f"Failed to insert node batch {i+1}: {e}")
                connection.rollback()
        
        # Insert edges in batches with progress bar
        edge_batches = [edges[i:i+batch_size] for i in range(0, len(edges), batch_size)]
        
        for i, edge_batch in enumerate(tqdm(edge_batches, desc="Inserting edges", unit="batch")):
            try:
                with connection.cursor() as cursor:
                    for j, edge in enumerate(edge_batch):
                        edge_id = f"edge_{i*batch_size + j}"
                        cursor.execute(
                            """
                            INSERT INTO KnowledgeGraphEdges 
                            (edge_id, source_node_id, target_node_id, relationship_type, weight) 
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (edge_id, edge['source'], edge['target'], edge['type'], edge['weight'])
                        )
                connection.commit()
                logger.info(f"Inserted edge batch {i+1}/{len(edge_batches)}: {len(edge_batch)} edges")
            except Exception as e:
                logger.error(f"Failed to insert edge batch {i+1}: {e}")
                connection.rollback()
        
        return True
    
    # Start metrics collection
    start_time = time.time()
    metrics = {
        "documents_processed": 0,
        "node_count": 0,
        "edge_count": 0,
        "unique_entity_count": 0,
        "document_node_count": 0,
        "processing_time": 0,
        "embedding_time": 0,
        "database_time": 0,
        "peak_memory_mb": 0
    }
    
    logger.info(f"Building large-scale knowledge graph from up to {limit} documents")
    
    # Process PMC documents in batches with progress tracking
    docs_generator = process_pmc_files(pmc_dir, limit=limit)
    doc_batches = []
    
    # Create batches manually for better control
    current_batch = []
    for doc in docs_generator:
        current_batch.append(doc)
        if len(current_batch) >= batch_size:
            doc_batches.append(current_batch)
            current_batch = []
    
    if current_batch:  # Don't forget the last partial batch
        doc_batches.append(current_batch)
    
    nodes = []
    edges = []
    node_id_map = {}  # Map entity names to node IDs to avoid duplicates
    
    # Process each batch
    for batch_idx, docs_batch in enumerate(tqdm(doc_batches, desc="Building knowledge graph", unit="batch")):
        batch_start = time.time()
        
        # Track memory usage if requested
        if collect_metrics:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], current_memory)
        
        # Update metrics
        metrics["documents_processed"] += len(docs_batch)
        
        # Process documents in this batch
        batch_nodes = []
        batch_edges = []
        
        for doc in docs_batch:
            # Create document node
            doc_id = f"doc_{doc['pmc_id']}"
            doc_content = f"{doc['title']}. {doc['abstract']}"
            
            # Generate embedding with timing
            embedding_start = time.time()
            doc_embedding = embedding_func(doc_content)
            metrics["embedding_time"] += time.time() - embedding_start
            
            doc_node = {
                "id": doc_id,
                "type": "Document",
                "name": doc['title'],
                "content": doc_content,
                "embedding": doc_embedding
            }
            batch_nodes.append(doc_node)
            metrics["document_node_count"] += 1
            
            # Extract entities with limit to prevent explosive growth
            entities = extract_entities_from_text(doc_content, max_entities=max_entities_per_doc)
            
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
                    entity_id = f"entity_{len(nodes) + len(batch_nodes)}"
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
                    batch_nodes.append(entity_node)
                    metrics["unique_entity_count"] += 1
                
                # Create edge from entity to document
                batch_edges.append({
                    "source": entity_id,
                    "target": doc_id,
                    "type": "MENTIONED_IN",
                    "weight": 1.0
                })
            
            # Add a limited number of entity-to-entity edges based on proximity
            # to prevent quadratic edge growth
            from tests.utils import extract_relationships
            relationships = extract_relationships(entities, max_distance=50)
            
            # Limit the number of relationships to prevent explosion
            max_relationships = min(len(relationships), 20)
            for rel in relationships[:max_relationships]:
                # Use entity IDs from node_id_map
                source_entity = next((e for e in entities if e["id"] == rel["source"]), None)
                target_entity = next((e for e in entities if e["id"] == rel["target"]), None)
                
                if source_entity and target_entity:
                    source_key = f"{source_entity['type']}_{source_entity['name']}"
                    target_key = f"{target_entity['type']}_{target_entity['name']}"
                    
                    if source_key in node_id_map and target_key in node_id_map:
                        batch_edges.append({
                            "source": node_id_map[source_key],
                            "target": node_id_map[target_key],
                            "type": rel["type"],
                            "weight": rel["weight"]
                        })
        
        # Extend the overall nodes and edges lists
        nodes.extend(batch_nodes)
        edges.extend(batch_edges)
        
        # Update metrics
        metrics["node_count"] += len(batch_nodes)
        metrics["edge_count"] += len(batch_edges)
        metrics["processing_time"] += time.time() - batch_start
        
        # Periodic status update
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(doc_batches) - 1:
            logger.info(f"Processed batch {batch_idx+1}/{len(doc_batches)}: "
                        f"{len(docs_batch)} documents, {len(batch_nodes)} nodes, {len(batch_edges)} edges")
            logger.info(f"Running totals: {metrics['documents_processed']} documents, "
                        f"{metrics['node_count']} nodes, {metrics['edge_count']} edges")
        
        # Force garbage collection periodically
        if batch_idx % 5 == 0:
            gc.collect()
    
    # Store in database with timing
    logger.info(f"Storing knowledge graph with {len(nodes)} nodes and {len(edges)} edges")
    db_start = time.time()
    store_kg_in_database_batched(connection, nodes, edges)
    metrics["database_time"] = time.time() - db_start
    
    # Final metrics
    total_time = time.time() - start_time
    metrics.update({
        "total_time_seconds": total_time,
        "nodes_per_second": metrics["node_count"] / total_time if total_time > 0 else 0,
        "edges_per_second": metrics["edge_count"] / total_time if total_time > 0 else 0
    })
    
    logger.info(f"Large-scale knowledge graph build complete: {metrics['node_count']} nodes, {metrics['edge_count']} edges")
    logger.info(f"Performance: {metrics['nodes_per_second']:.1f} nodes/s, {metrics['edges_per_second']:.1f} edges/s")
    
    return metrics

def validate_large_scale_graph(
    connection,
    expected_docs=1000,
    min_doc_percentage=0.8,
    min_nodes=2000,
    min_edges=1000
) -> Dict[str, Any]:
    """
    Validate a large-scale knowledge graph in the database.
    
    Args:
        connection: IRIS connection
        expected_docs: Expected number of documents
        min_doc_percentage: Minimum percentage of expected docs that must be present 
        min_nodes: Minimum number of nodes expected
        min_edges: Minimum number of edges expected
        
    Returns:
        Dict with validation results
    """
    validation = {
        "success": False,
        "document_count": 0,
        "node_count": 0,
        "edge_count": 0,
        "node_types": {},
        "edge_types": {},
        "errors": []
    }
    
    try:
        with connection.cursor() as cursor:
            # Check document count
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            doc_count = cursor.fetchone()[0]
            validation["document_count"] = doc_count
            
            min_docs = int(expected_docs * min_doc_percentage)
            if doc_count < min_docs:
                validation["errors"].append(
                    f"Insufficient documents loaded: {doc_count}/{expected_docs} "
                    f"(minimum required: {min_docs})"
                )
            
            # Check nodes
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
            node_count = cursor.fetchone()[0]
            validation["node_count"] = node_count
            
            if node_count < min_nodes:
                validation["errors"].append(
                    f"Insufficient nodes created: {node_count} (minimum required: {min_nodes})"
                )
            
            # Check node types
            cursor.execute("""
                SELECT node_type, COUNT(*) 
                FROM KnowledgeGraphNodes 
                GROUP BY node_type
            """)
            node_types = cursor.fetchall()
            
            for node_type, count in node_types:
                validation["node_types"][node_type] = count
            
            # Check edges
            cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
            edge_count = cursor.fetchone()[0]
            validation["edge_count"] = edge_count
            
            if edge_count < min_edges:
                validation["errors"].append(
                    f"Insufficient edges created: {edge_count} (minimum required: {min_edges})"
                )
            
            # Check edge types
            cursor.execute("""
                SELECT relationship_type, COUNT(*) 
                FROM KnowledgeGraphEdges 
                GROUP BY relationship_type
            """)
            edge_types = cursor.fetchall()
            
            for edge_type, count in edge_types:
                validation["edge_types"][edge_type] = count
            
            # Random sampling for content validation
            cursor.execute("""
                SELECT node_id, node_type, node_name, content
                FROM KnowledgeGraphNodes
                ORDER BY RAND()
                LIMIT 5
            """)
            sample_nodes = cursor.fetchall()
            
            validation["sample_nodes"] = []
            for node_id, node_type, node_name, content in sample_nodes:
                if not node_name or not content:
                    validation["errors"].append(f"Node {node_id} has empty name or content")
                else:
                    validation["sample_nodes"].append({
                        "id": node_id,
                        "type": node_type,
                        "name": node_name,
                        "content_preview": content[:100] + "..." if len(content) > 100 else content
                    })
    
    except Exception as e:
        validation["errors"].append(f"Error validating graph: {e}")
    
    # Set success flag
    validation["success"] = len(validation["errors"]) == 0
    
    return validation
