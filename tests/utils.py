"""
Test utilities for RAG templates

This module provides common utilities for testing RAG templates with real data,
including functions for knowledge graph creation, standardized query testing,
and performance comparison.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import os
import torch # Added for NaN/Inf check
from common.utils import Document
from colbert.doc_encoder import generate_token_embeddings_for_documents as colbert_generate_embeddings

logger = logging.getLogger(__name__)

def build_knowledge_graph(connection, embedding_func, limit=20, 
                         pmc_dir="data/pmc_oas_downloaded") -> Tuple[int, int]:
    """
    Build knowledge graph from PMC documents.
    
    Args:
        connection: IRIS connection
        embedding_func: Function to generate embeddings
        limit: Maximum number of documents to process
        pmc_dir: Directory containing PMC XML files
        
    Returns:
        Tuple of (node_count, edge_count)
    """
    from data.pmc_processor import process_pmc_files
    
    def extract_entities_from_text(text):
        medical_terms = {
            "Disease": ["diabetes", "cancer", "hypertension", "asthma", "alzheimer", "parkinson", "arthritis", "obesity", "depression", "schizophrenia"],
            "Medication": ["insulin", "metformin", "statin", "aspirin", "ibuprofen", "penicillin", "acetaminophen", "amoxicillin", "lisinopril"],
            "Organ": ["heart", "liver", "kidney", "lung", "brain", "pancreas", "stomach", "intestine", "colon", "thyroid"],
            "Symptom": ["pain", "fever", "cough", "fatigue", "headache", "nausea", "vomiting", "dizziness", "dyspnea", "wheezing"],
            "Treatment": ["surgery", "therapy", "radiation", "chemotherapy", "transplant", "dialysis", "vaccination", "rehabilitation", "counseling"]
        }
        entities = []
        entity_id_counter = 0 
        import re
        text_lower = text.lower() if text else ""
        for entity_type, terms in medical_terms.items():
            for term in terms:
                for match in re.finditer(r'\b' + re.escape(term) + r'\b', text_lower):
                    entity_id_counter += 1
                    entities.append({"id": f"entity_kg_{entity_id_counter}", "type": entity_type, "name": term, "span": match.span()})
        return entities
    
    def extract_relationships(entities, max_distance=100):
        relationships = []
        sorted_entities = sorted(entities, key=lambda e: e["span"][0])
        for i, entity1 in enumerate(sorted_entities):
            for j in range(i + 1, len(sorted_entities)):
                entity2 = sorted_entities[j]
                distance = entity2["span"][0] - entity1["span"][1]
                if distance <= max_distance:
                    relationships.append({"source": entity1["id"], "target": entity2["id"], "type": "CO_OCCURS_WITH", "weight": 1.0 - (distance / max_distance)})
        return relationships
    
    def create_kg_nodes_and_edges(docs, embedding_func_kg):
        nodes = []
        edges = []
        node_id_map = {} 
        node_counter = 0

        for doc_data in docs:
            doc_id_kg = f"doc_kg_{doc_data['pmc_id']}"
            doc_title = doc_data.get('title', '')
            doc_abstract = doc_data.get('abstract', '')
            doc_content_for_embed = f"{doc_title}. {doc_abstract}".strip()
            
            doc_embedding_list = None
            if doc_content_for_embed:
                try:
                    doc_embedding_list = embedding_func_kg([doc_content_for_embed])[0]
                except Exception as e:
                    logger.error(f"Error embedding KG document content for {doc_id_kg}: {e}")

            doc_node = {"id": doc_id_kg, "type": "Document", "name": doc_title, 
                        "description_text": doc_abstract, "embedding": doc_embedding_list}
            nodes.append(doc_node)
            node_counter +=1
            
            entities = extract_entities_from_text(doc_abstract)
            
            for entity_data in entities:
                entity_name = entity_data["name"]
                entity_type = entity_data["type"]
                entity_key = f"{entity_type}_{entity_name}"
                
                if entity_key in node_id_map:
                    entity_id = node_id_map[entity_key]
                else:
                    node_counter += 1
                    entity_id = f"entity_kg_{node_counter}"
                    node_id_map[entity_key] = entity_id
                    
                    entity_content_for_embed = f"{entity_name} is a {entity_type.lower()}"
                    entity_embedding_list = None
                    if entity_content_for_embed:
                        try:
                            entity_embedding_list = embedding_func_kg([entity_content_for_embed])[0]
                        except Exception as e:
                             logger.error(f"Error embedding KG entity content for {entity_id}: {e}")
                    
                    entity_node_obj = {"id": entity_id, "type": entity_type, "name": entity_name, 
                                       "description_text": entity_content_for_embed, "embedding": entity_embedding_list}
                    nodes.append(entity_node_obj)
                
                edges.append({"source": entity_id, "target": doc_id_kg, "type": "MENTIONED_IN", "weight": 1.0})
            
            relationships = extract_relationships(entities)
            for rel in relationships:
                source_entity_data = next((e for e in entities if e["id"] == rel["source"]), None)
                target_entity_data = next((e for e in entities if e["id"] == rel["target"]), None)
                
                if source_entity_data and target_entity_data:
                    source_key = f"{source_entity_data['type']}_{source_entity_data['name']}"
                    target_key = f"{target_entity_data['type']}_{target_entity_data['name']}"
                    if source_key in node_id_map and target_key in node_id_map:
                        edges.append({"source": node_id_map[source_key], "target": node_id_map[target_key], 
                                      "type": rel["type"], "weight": rel["weight"]})
        return nodes, edges
    
    def store_kg_in_database(connection, nodes, edges):
        with connection.cursor() as cursor:
            # The schema for KnowledgeGraphNodes and KnowledgeGraphEdges is defined in common/db_init.sql
            # and should have been created before this function is called.
            
            for i, node_data in enumerate(nodes):
                try:
                    embedding_list = node_data.get('embedding')
                    final_embedding_param_kg = None
                    if embedding_list is not None:
                        try:
                            converted_list_kg = [float(x) for x in embedding_list]
                            if len(converted_list_kg) == 768: # Dimension for e5-base-v2
                                final_embedding_param_kg = converted_list_kg
                            else:
                                logger.warning(f"KG Node {node_data.get('id', 'UNKNOWN')}: Incorrect embedding dimension {len(converted_list_kg)}. Expected 768. Skipping embedding.")
                                final_embedding_param_kg = None 
                        except (TypeError, ValueError) as e_conv_kg:
                            logger.error(f"Error converting KG node embedding for {node_data.get('id', 'UNKNOWN')}: {e_conv_kg}")
                            final_embedding_param_kg = None
                    
                    embedding_value_to_insert_kg = str(final_embedding_param_kg) if final_embedding_param_kg is not None else None
                    
                    cursor.execute(
                        "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding) VALUES (?, ?, ?, ?, TO_VECTOR(?))",
                        (node_data['id'], node_data['type'], node_data['name'], node_data.get('description_text', node_data.get('content')), embedding_value_to_insert_kg)
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert KG node {node_data.get('id', 'UNKNOWN')}: {e}")
            
            edge_id_counter = 0
            for edge_data in edges:
                try:
                    edge_id_counter += 1
                    edge_id = f"edge_kg_{edge_id_counter}"
                    cursor.execute(
                        "INSERT INTO KnowledgeGraphEdges (edge_id, source_node_id, target_node_id, relationship_type, weight) VALUES (?, ?, ?, ?, ?)",
                        (edge_id, edge_data['source'], edge_data['target'], edge_data['type'], edge_data['weight'])
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert KG edge {edge_id} between {edge_data.get('source')} and {edge_data.get('target')}: {e}")
        
        if hasattr(connection, 'commit'):
            try:
                connection.commit()
                logger.info("Committed KG nodes and edges.")
            except Exception as e_commit:
                logger.error(f"Error committing KG data: {e_commit}")
                if hasattr(connection, 'rollback'): connection.rollback()
        return True
    
    logger.info(f"Processing PMC files from {pmc_dir} (limit: {limit}) for KG build.")
    docs_for_kg = list(process_pmc_files(pmc_dir, limit=limit))
    
    logger.info(f"Building knowledge graph from {len(docs_for_kg)} documents using provided embedding_func.")
    nodes, edges = create_kg_nodes_and_edges(docs_for_kg, embedding_func)
    
    logger.info(f"Storing knowledge graph with {len(nodes)} nodes and {len(edges)} edges.")
    store_kg_in_database(connection, nodes, edges)
    
    return len(nodes), len(edges)

def run_standardized_queries(pipeline, queries: Optional[List[str]] = None, 
                            include_docs: bool = False) -> Dict[str, Any]:
    if queries is None:
        queries = [
            "What is the relationship between diabetes and insulin?",
            "How does metformin help with diabetes treatment?",
            "What are the key symptoms of diabetes?",
            "What is the role of the pancreas in diabetes?",
            "How do statins affect cholesterol levels?"
        ]
    results = {}
    total_time = 0
    total_docs = 0
    for query in queries:
        logger.info(f"Running query: {query}")
        start_time = time.time()
        result = pipeline.run(query)
        query_time = time.time() - start_time
        docs = result.get("retrieved_documents", [])
        total_docs += len(docs)
        query_result = {"answer": result.get("answer", ""), "doc_count": len(docs), "time_seconds": query_time}
        if include_docs:
            query_result["documents"] = [{"id": doc.id if hasattr(doc, "id") else "unknown", "score": float(doc.score) if hasattr(doc, "score") else 0.0, "content_preview": (doc.content[:100] + "...") if hasattr(doc, "content") and len(doc.content) > 100 else doc.content} for doc in docs[:5]]
        results[query] = query_result
        total_time += query_time
    results["_summary"] = {"total_queries": len(queries), "total_docs_retrieved": total_docs, "avg_docs_per_query": total_docs / len(queries) if queries else 0, "total_time_seconds": total_time, "avg_time_per_query": total_time / len(queries) if queries else 0}
    return results

def compare_rag_techniques(queries: List[str], techniques: Dict[str, Any]) -> Dict[str, Any]:
    comparison = {}
    for name, pipeline in techniques.items():
        logger.info(f"Running technique: {name}")
        results = run_standardized_queries(pipeline, queries)
        comparison[name] = results["_summary"]
        comparison[name]["individual_queries"] = {query: {"doc_count": results[query]["doc_count"], "time_seconds": results[query]["time_seconds"]} for query in queries}
    if len(techniques) > 1:
        fastest = min(techniques.keys(), key=lambda name: comparison[name]["avg_time_per_query"])
        most_docs = max(techniques.keys(), key=lambda name: comparison[name]["avg_docs_per_query"])
        comparison["_comparative"] = {"fastest_technique": fastest, "fastest_time": comparison[fastest]["avg_time_per_query"], "most_retrievals": most_docs, "most_retrievals_count": comparison[most_docs]["avg_docs_per_query"]}
    return comparison

def load_pmc_documents(connection, embedding_func: Callable, limit=50, pmc_dir="data/pmc_oas_downloaded", batch_size=20, 
                      show_progress=True):
    from data.pmc_processor import process_pmc_files
    import time
    logger.info(f"Processing up to {limit} documents from {pmc_dir} (batch size: {batch_size}), generating embeddings...")
    start_time = time.time()
    count = 0
    total_batches = (limit + batch_size - 1) // batch_size
    for batch_num, docs_batch in enumerate(process_pmc_files_in_batches(pmc_dir, limit, batch_size)):
        batch_start_time_loop = time.time() # Renamed to avoid conflict with outer batch_time
        with connection.cursor() as cursor:
            for doc in docs_batch:
                try:
                    authors = str(doc.get('authors', []))
                    keywords = str(doc.get('keywords', []))
                    content_to_embed = doc.get('abstract', doc.get('title', ''))
                    embedding_list = None
                    if not content_to_embed:
                        logger.warning(f"Document {doc['pmc_id']} has no abstract or title for embedding. Skipping embedding.")
                    else:
                        try:
                            embedding_list = embedding_func([content_to_embed])[0]
                        except Exception as e_embed:
                            logger.error(f"Error generating embedding for doc {doc['pmc_id']}: {e_embed}")
                    
                    final_embedding_param = None
                    if embedding_list is not None:
                        try:
                            converted_list = []
                            valid_embedding = True
                            for x_val in embedding_list: # Renamed x to x_val
                                val_float = float(x_val) # Renamed val to val_float
                                if torch.isnan(torch.tensor(val_float)) or torch.isinf(torch.tensor(val_float)):
                                    logger.warning(f"Doc {doc['pmc_id']} embedding contains NaN/Inf: {val_float}. Skipping embedding.")
                                    valid_embedding = False
                                    break
                                converted_list.append(val_float)
                            if valid_embedding:
                                if len(converted_list) == 768:
                                    final_embedding_param = converted_list
                                else:
                                    logger.warning(f"Doc {doc['pmc_id']}: Incorrect embedding dimension {len(converted_list)}. Expected 768. Skipping embedding.")
                                    final_embedding_param = None # Ensure it's None if not valid
                            else: # valid_embedding is False
                                final_embedding_param = None
                        except (TypeError, ValueError) as e_conv:
                            logger.error(f"Error converting embedding elements to float for doc {doc['pmc_id']}: {e_conv}. Embedding: {str(embedding_list)[:100]}...")
                            final_embedding_param = None
                    
                    embedding_value_to_insert = str(final_embedding_param) if final_embedding_param is not None else None
                    
                    # Escape single quotes in title and abstract for SQL insertion
                    title_to_insert = doc.get('title', '').replace("'", "''")
                    abstract_to_insert = doc.get('abstract', '').replace("'", "''")
                    # authors and keywords are str(list) representations.
                    # If they also cause issues, they might need more careful serialization.

                    # Ensure insertion into RAG.SourceDocuments
                    cursor.execute(
                        "INSERT INTO RAG.SourceDocuments (doc_id, title, abstract, text_content, authors, keywords, embedding) VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR(?))",
                        (doc['pmc_id'], title_to_insert, abstract_to_insert, abstract_to_insert, authors, keywords, embedding_value_to_insert) # Using abstract for text_content if not available
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert document {doc['pmc_id']}: {e}")
        if hasattr(connection, 'commit'):
            try:
                connection.commit()
                logger.info(f"Committed batch {batch_num+1}/{total_batches}.")
            except Exception as e_commit:
                logger.error(f"Error committing batch {batch_num+1}: {e_commit}")
                if hasattr(connection, 'rollback'):
                    logger.info(f"Attempting rollback for batch {batch_num+1} due to commit error.")
                    connection.rollback()
        if show_progress:
            batch_duration = time.time() - batch_start_time_loop # Use renamed variable
            total_elapsed_time = time.time() - start_time # Use renamed variable
            progress_percent = (batch_num + 1) / total_batches * 100
            eta_seconds = (total_elapsed_time / (batch_num + 1)) * (total_batches - (batch_num + 1)) if (batch_num + 1) < total_batches else 0
            logger.info(f"Batch {batch_num+1}/{total_batches} complete: {len(docs_batch)} docs in {batch_duration:.2f}s | Progress: {progress_percent:.1f}% | ETA: {eta_seconds:.1f}s | Total loaded: {count}")
    
    total_final_time = time.time() - start_time # Use renamed variable
    docs_per_sec = count / total_final_time if total_final_time > 0 else 0
    logger.info(f"Loaded {count} documents into database in {total_final_time:.2f}s ({docs_per_sec:.1f} docs/s)")
    return count

def process_pmc_files_in_batches(pmc_dir, limit, batch_size):
    from data.pmc_processor import process_pmc_files
    import itertools
    doc_generator = process_pmc_files(pmc_dir, limit=limit)
    while True:
        batch = list(itertools.islice(doc_generator, batch_size))
        if not batch:
            break
        yield batch

def load_colbert_token_embeddings(connection, limit=50, batch_size=10, colbert_model_name="colbert-ir/colbertv2.0", device="cpu", mock_colbert_encoder=False):
    """
    Generates and loads ColBERT token embeddings for documents already in SourceDocuments.
    """
    logger.info(f"Starting ColBERT token embedding generation and loading for up to {limit} documents.")
    
    source_docs_to_process = []
    with connection.cursor() as cursor:
        # Fetch doc_id, abstract, and text_content. COALESCE will be done in Python.
        cursor.execute(f"SELECT TOP {limit} doc_id, abstract, text_content FROM SourceDocuments")
        rows = cursor.fetchall()
        for row in rows:
            doc_id = row[0]
            abstract_content = row[1]
            text_content_val = row[2] # Renamed to avoid conflict
            
            # Perform COALESCE logic in Python
            content_for_colbert = abstract_content if abstract_content is not None and abstract_content.strip() != "" else text_content_val
            
            if content_for_colbert and content_for_colbert.strip() != "":
                source_docs_to_process.append({"id": doc_id, "content": content_for_colbert})
            else:
                logger.warning(f"Document {doc_id} has neither abstract nor text_content. Skipping for ColBERT.")
    
    if not source_docs_to_process:
        logger.info("No suitable documents found in SourceDocuments to process for ColBERT embeddings.")
        return 0

    logger.info(f"Fetched {len(source_docs_to_process)} documents from SourceDocuments for ColBERT processing.")

    # Generate token embeddings using ColBERT's doc encoder
    # Note: colbert_generate_embeddings expects a list of dicts with 'id' and 'content'
    enriched_docs_with_tokens = colbert_generate_embeddings(
        documents=source_docs_to_process,
        batch_size=batch_size, # Can be different from PMC loading batch size
        model_name=colbert_model_name,
        device=device,
        mock=mock_colbert_encoder 
    )

    total_tokens_inserted = 0
    with connection.cursor() as cursor:
        for doc_data in enriched_docs_with_tokens:
            doc_id = doc_data["id"]
            tokens = doc_data.get("tokens", [])
            token_embeddings = doc_data.get("token_embeddings", [])

            if not tokens or not token_embeddings or len(tokens) != len(token_embeddings):
                logger.warning(f"Skipping document {doc_id} due to missing or mismatched tokens/embeddings.")
                continue

            for i, (token_text, token_embedding) in enumerate(zip(tokens, token_embeddings)):
                try:
                    # Ensure embedding is a list of floats and has the correct dimension (128 for colbertv2.0)
                    final_token_embedding_param = None
                    if token_embedding is not None:
                        try:
                            converted_token_emb = [float(x) for x in token_embedding]
                            if len(converted_token_emb) == 128: # ColBERT default dimension
                                final_token_embedding_param = converted_token_emb
                            else:
                                logger.warning(f"Doc {doc_id}, Token {i}: Incorrect embedding dimension {len(converted_token_emb)}. Expected 128. Skipping token embedding.")
                        except (TypeError, ValueError) as e_conv_token:
                            logger.error(f"Error converting ColBERT token embedding for doc {doc_id}, token {i}: {e_conv_token}")
                    
                    embedding_value_to_insert = str(final_token_embedding_param) if final_token_embedding_param is not None else None

                    if embedding_value_to_insert: # Only insert if we have a valid embedding string
                        cursor.execute(
                            "INSERT INTO DocumentTokenEmbeddings (doc_id, token_sequence_index, token_text, token_embedding) VALUES (?, ?, ?, TO_VECTOR(?))",
                            (doc_id, i, token_text, embedding_value_to_insert)
                        )
                        total_tokens_inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert token embedding for doc {doc_id}, token index {i}, text '{token_text}': {e}")
        
        if hasattr(connection, 'commit'):
            try:
                connection.commit()
                logger.info(f"Committed {total_tokens_inserted} ColBERT token embeddings.")
            except Exception as e_commit:
                logger.error(f"Error committing ColBERT token embeddings: {e_commit}")
                if hasattr(connection, 'rollback'):
                    connection.rollback()
    
    logger.info(f"Finished ColBERT token embedding loading. Total tokens inserted: {total_tokens_inserted}.")
    return total_tokens_inserted
