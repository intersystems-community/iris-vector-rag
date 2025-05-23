"""
Data Loader for IRIS RAG Templates (Fixed Version)

This module handles loading processed documents into the IRIS database,
with support for batching, error handling, and performance metrics.

This version uses the vector_sql_utils.py module to properly handle
the TO_VECTOR function in IRIS SQL.
"""

import logging
import time
import json
from typing import List, Dict, Any, Generator, Optional, Tuple, Callable
import os
import sys

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from data.pmc_processor import process_pmc_files
from common.vector_sql_utils import validate_vector_string  # Import the validation function

# Define a more permissive vector string validation function
def validate_vector_string_extended(vector_string: str) -> bool:
    """
    Validates that a vector string contains only valid characters.
    This is a more permissive version that allows scientific notation and negative numbers.
    
    Args:
        vector_string: The vector string to validate, typically in format "[0.1,0.2,...]"
        
    Returns:
        bool: True if the vector string contains only valid characters, False otherwise
    """
    # Allow digits, dots, commas, square brackets, minus signs, 'e' for scientific notation
    allowed_chars = set("0123456789.[],e-+")
    return all(c in allowed_chars for c in vector_string)

logger = logging.getLogger(__name__)

def load_documents_to_iris(
    connection,
    documents: List[Dict[str, Any]],
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None,
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Load documents into IRIS database with batching, including embeddings.
    
    Args:
        connection: IRIS connection object
        documents: List of document dictionaries to load
        embedding_func: Optional function to generate embeddings for documents.
        colbert_doc_encoder_func: Optional function for ColBERT token embeddings
        batch_size: Number of documents to insert in a single batch
        
    Returns:
        Dictionary with loading statistics
    """
    start_time = time.time()
    loaded_doc_count = 0
    loaded_token_count = 0
    error_count = 0
    
    try:
        cursor = connection.cursor()
        
        # Prepare documents in batches
        doc_batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        
        logger.info(f"Loading {len(documents)} SourceDocuments in {len(doc_batches)} batches.")
        
        for batch_idx, current_doc_batch in enumerate(doc_batches):
            try:
                # Process each document in the batch
                for doc in current_doc_batch:
                    doc_id = doc.get("pmc_id")
                    title = doc.get("title")
                    text_content = doc.get("abstract")
                    authors = json.dumps(doc.get("authors", []))
                    keywords = json.dumps(doc.get("keywords", []))
                    
                    # Generate embedding if embedding_func is provided
                    embedding_vector_str = None
                    if embedding_func:
                        text_to_embed = doc.get("abstract") or doc.get("title", "")
                        if text_to_embed:
                            embedding = embedding_func([text_to_embed])[0]
                            embedding_vector_str = f"[{','.join(map(str, embedding))}]"
                            
                            # Validate vector string for security using our extended function
                            if not validate_vector_string_extended(embedding_vector_str):
                                logger.warning(f"Invalid vector string for document {doc_id}. Skipping embedding.")
                                embedding_vector_str = None
                        else:
                            logger.warning(f"Document {doc_id} has no abstract or title for sentence embedding.")
                    
                    # Use string interpolation for the entire SQL statement
                    if embedding_vector_str:
                        # Escape single quotes in string values
                        safe_title = title.replace("'", "''") if title else ""
                        safe_text = text_content.replace("'", "''") if text_content else ""
                        safe_authors = authors.replace("'", "''") if authors else "{}"
                        safe_keywords = keywords.replace("'", "''") if keywords else "{}"
                        
                        # Use string interpolation for the entire SQL statement
                        sql = f"INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, authors, keywords, embedding) VALUES ('{doc_id}', '{safe_title}', '{safe_text}', '{safe_authors}', '{safe_keywords}', TO_VECTOR('{embedding_vector_str}', 'double', 768))"
                        
                        # Print the SQL before executing
                        print(f"Executing SQL (with embedding): {sql}")
                        
                        cursor.execute(sql)
                    else:
                        # Escape single quotes in string values
                        safe_title = title.replace("'", "''") if title else ""
                        safe_text = text_content.replace("'", "''") if text_content else ""
                        safe_authors = authors.replace("'", "''") if authors else "{}"
                        safe_keywords = keywords.replace("'", "''") if keywords else "{}"
                        
                        # Use string interpolation for the entire SQL statement
                        sql = f"INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, authors, keywords, embedding) VALUES ('{doc_id}', '{safe_title}', '{safe_text}', '{safe_authors}', '{safe_keywords}', NULL)"
                        
                        # Print the SQL before executing
                        print(f"Executing SQL (without embedding): {sql}")
                        
                        cursor.execute(sql)
                    
                    loaded_doc_count += 1
                    
                    # Process ColBERT token embeddings if function is provided
                    if colbert_doc_encoder_func and doc_id and (text_content or title):
                        text_for_colbert = text_content or title
                        try:
                            token_data = colbert_doc_encoder_func(text_for_colbert)
                            for idx, (token_text, token_vec) in enumerate(token_data):
                                token_vec_str = f"[{','.join(map(str, token_vec))}]"
                                
                                # Validate vector string for security using our extended function
                                if not validate_vector_string_extended(token_vec_str):
                                    logger.warning(f"Invalid token vector string for document {doc_id}, token {idx}. Skipping token.")
                                    continue
                                
                                # Escape single quotes in string values
                                safe_token_text = token_text[:1000].replace("'", "''") if token_text else ""
                                
                                # Use string interpolation for the entire SQL statement
                                token_sql = f"INSERT INTO RAG.DocumentTokenEmbeddings (doc_id, token_sequence_index, token_text, token_embedding, metadata_json) VALUES ('{doc_id}', {idx}, '{safe_token_text}', TO_VECTOR('{token_vec_str}', 'double', 128), '{{}}')"
                                
                                # Print the SQL before executing (only for the first token to avoid flooding the console)
                                if idx == 0:
                                    print(f"Executing token SQL (first token only): {token_sql}")
                                
                                cursor.execute(token_sql)
                                loaded_token_count += 1
                        except Exception as colbert_e:
                            logger.error(f"Error generating ColBERT token embeddings for doc {doc_id}: {colbert_e}")
                
                connection.commit()
                
                if (batch_idx + 1) % 1 == 0 or batch_idx == len(doc_batches) - 1:
                    elapsed = time.time() - start_time
                    rate = loaded_doc_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Loaded {loaded_doc_count}/{len(documents)} SourceDocuments. Loaded {loaded_token_count} token embeddings. ({rate:.2f} docs/sec)")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_idx}: {e}")
                connection.rollback()
                error_count += len(current_doc_batch)
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error in document loading process: {e}")
        error_count = len(documents) - loaded_doc_count
    
    duration = time.time() - start_time
    
    return {
        "total_documents": len(documents),
        "loaded_doc_count": loaded_doc_count,
        "loaded_token_count": loaded_token_count,
        "error_count": error_count,
        "duration_seconds": duration,
        "documents_per_second": loaded_doc_count / duration if duration > 0 else 0
    }

def process_and_load_documents(
    pmc_directory: str, 
    connection=None, 
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None,
    limit: int = 1000, 
    batch_size: int = 50,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Process PMC XML files and load them into the IRIS database.
    
    Args:
        pmc_directory: Directory containing PMC XML files
        connection: IRIS connection object (if None, a new connection will be created)
        embedding_func: Optional function to generate embeddings for documents
        colbert_doc_encoder_func: Optional function for ColBERT token embeddings
        limit: Maximum number of documents to process
        batch_size: Number of documents to insert in a single batch
        use_mock: Whether to use a mock connection for testing
        
    Returns:
        Dictionary with processing and loading statistics
    """
    start_time = time.time()
    
    # Process PMC files
    logger.info(f"Processing up to {limit} documents from {pmc_directory}")
    # Convert generator to list so we can get its length
    processed_docs = list(process_pmc_files(pmc_directory, limit=limit))
    
    if not processed_docs:
        return {
            "success": False,
            "error": f"No documents processed from {pmc_directory}",
            "processed_count": 0,
            "loaded_doc_count": 0,
            "error_count": 0,
            "duration_seconds": time.time() - start_time,
            "documents_per_second": 0
        }
    
    # Create connection if not provided
    conn_created = False
    if connection is None:
        connection = get_iris_connection(use_mock=use_mock)
        conn_created = True
    
    if not connection:
        return {
            "success": False,
            "error": "Failed to establish database connection",
            "processed_count": len(processed_docs),
            "loaded_doc_count": 0,
            "error_count": len(processed_docs),
            "duration_seconds": time.time() - start_time,
            "documents_per_second": 0
        }
    
    # Load documents
    load_stats = load_documents_to_iris(
        connection=connection,
        documents=processed_docs,
        embedding_func=embedding_func,
        colbert_doc_encoder_func=colbert_doc_encoder_func,
        batch_size=batch_size
    )
    
    # Close connection if we created it
    if conn_created:
        try:
            connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    # Combine stats
    result = {
        "success": True,
        "processed_directory": pmc_directory,
        "processed_count": len(processed_docs),
        "duration_seconds": time.time() - start_time,
        "documents_per_second": len(processed_docs) / (time.time() - start_time) if time.time() > start_time else 0
    }
    result.update(load_stats)
    
    return result