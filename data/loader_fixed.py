"""
Data Loader for IRIS RAG Templates - FIXED VERSION

This module handles loading processed documents into the IRIS database,
with comprehensive fixes for NaN handling, vector format consistency, and data validation.
"""

import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import sys

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from data.pmc_processor import process_pmc_files

logger = logging.getLogger(__name__)

def validate_and_fix_embedding(embedding: List[float]) -> Optional[str]:
    """
    Validate and fix embedding vectors, handling NaN and inf values.
    
    Args:
        embedding: List of float values representing the embedding
        
    Returns:
        Comma-separated string representation or None if unfixable
    """
    if not embedding:
        logger.warning("Empty embedding provided")
        return None
    
    try:
        # Convert to numpy array for easier manipulation
        arr = np.array(embedding, dtype=np.float64)
        
        # Check for NaN or inf values
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            logger.warning(f"Found NaN/inf values in embedding, replacing with zeros")
            # Replace NaN and inf with 0.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure all values are finite
        if not np.all(np.isfinite(arr)):
            logger.warning("Non-finite values found after cleaning, using zero vector")
            arr = np.zeros_like(arr)
        
        # Convert to list and format as comma-separated string
        cleaned_embedding = arr.tolist()
        
        # Ensure all values are proper floats
        cleaned_embedding = [float(x) for x in cleaned_embedding]
        
        # Format as comma-separated string for IRIS VECTOR column (no brackets)
        embedding_str = ','.join(f"{x:.15g}" for x in cleaned_embedding)
        
        return embedding_str
        
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return None

def validate_and_fix_text_field(text: Any) -> str:
    """
    Validate and fix text fields to ensure they're proper strings.
    
    Args:
        text: Text field value
        
    Returns:
        Cleaned string value
    """
    if text is None:
        return ""
    
    if isinstance(text, (list, dict)):
        return json.dumps(text)
    
    try:
        # Convert to string and handle any encoding issues
        text_str = str(text)
        
        # Remove any null bytes that might cause issues
        text_str = text_str.replace('\x00', '')
        
        return text_str
        
    except Exception as e:
        logger.warning(f"Error processing text field: {e}")
        return ""

def load_documents_to_iris(
    connection,
    documents: List[Dict[str, Any]],
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    batch_size: int = 250,
    handle_chunks: bool = True
) -> Dict[str, Any]:
    """
    Load documents into IRIS database with comprehensive error handling and data validation.
    
    Args:
        connection: IRIS connection object
        documents: List of document dictionaries to load
        embedding_func: Optional function to generate embeddings for documents
        batch_size: Number of documents to insert in a single batch
        handle_chunks: Whether to process chunked documents separately
        
    Returns:
        Dictionary with loading statistics
    """
    start_time = time.time()
    loaded_doc_count = 0
    loaded_token_count = 0
    loaded_chunk_count = 0
    error_count = 0
    
    try:
        cursor = connection.cursor()
        
        # Separate chunked and non-chunked documents
        expanded_documents = []
        for doc in documents:
            if handle_chunks and doc.get("chunks"):
                # Process chunks as separate documents
                for chunk in doc["chunks"]:
                    chunk_doc = {
                        "id": chunk["chunk_id"],
                        "doc_id": chunk["chunk_id"],
                        "title": f"{doc.get('title', '')} [Chunk {chunk['chunk_index']}]",
                        "abstract": doc.get("abstract", "No abstract available"),
                        "content": chunk["text"],
                        "authors": doc.get("authors", []),
                        "keywords": doc.get("keywords", []),
                        "metadata": {
                            **doc.get("metadata", {}),
                            "is_chunk": True,
                            "parent_doc_id": doc["doc_id"],
                            "chunk_index": chunk["chunk_index"],
                            "chunk_metadata": chunk["metadata"]
                        }
                    }
                    expanded_documents.append(chunk_doc)
                    
                # Also add the original document with a flag indicating it has chunks
                original_doc = doc.copy()
                original_doc["metadata"] = original_doc.get("metadata", {}).copy()
                original_doc["metadata"]["has_chunks"] = True
                original_doc["metadata"]["chunk_count"] = len(doc["chunks"])
                expanded_documents.append(original_doc)
            else:
                expanded_documents.append(doc)
        
        # Prepare documents in batches
        doc_batches = [expanded_documents[i:i+batch_size] for i in range(0, len(expanded_documents), batch_size)]
        
        logger.info(f"Loading {len(expanded_documents)} documents ({len(documents)} original, expanded for chunks) in {len(doc_batches)} batches.")
        
        for batch_idx, current_doc_batch in enumerate(doc_batches):
            source_doc_batch_params = []
            docs_for_token_embedding = []

            for doc in current_doc_batch:
                try:
                    embedding_vector = None
                    if embedding_func:
                        # For chunks, use the chunk content; for regular docs, use abstract or title
                        if doc.get("metadata", {}).get("is_chunk"):
                            text_to_embed = doc.get("content", "")[:2000]  # Limit chunk size for embedding
                        else:
                            text_to_embed = doc.get("abstract") or doc.get("title", "")
                            
                        if text_to_embed:
                            try:
                                # Generate embedding with error handling
                                text_to_embed = validate_and_fix_text_field(text_to_embed)
                                embedding = embedding_func([text_to_embed])[0]
                                embedding_vector = validate_and_fix_embedding(embedding)
                                
                                if embedding_vector is None:
                                    logger.warning(f"Failed to process embedding for document {doc.get('doc_id')}")
                                    
                            except Exception as e:
                                logger.error(f"Error generating embedding for document {doc.get('doc_id')}: {e}")
                                embedding_vector = None
                        else:
                            logger.warning(f"Document {doc.get('doc_id')} has no content for embedding.")
                    
                    # Get document ID with validation
                    doc_id_value = doc.get("doc_id") or doc.get("pmc_id")
                    if not doc_id_value:
                        logger.error(f"Document missing doc_id: {doc}")
                        continue
                    
                    # Validate and clean all text fields
                    title = validate_and_fix_text_field(doc.get("title"))

                    # # For chunks, use content as abstract; for regular docs, use abstract
                    # if doc.get("metadata", {}).get("is_chunk"):
                    #     abstract = validate_and_fix_text_field(doc.get("content", ""))
                    # else:
                    #     abstract = validate_and_fix_text_field(doc.get("abstract"))

                    # Always use real abstract (from doc["abstract"])
                    abstract = validate_and_fix_text_field(doc.get("abstract"))

                    text_content = validate_and_fix_text_field(doc.get("content", ""))
                    
                    # Handle authors and keywords with validation
                    authors = doc.get("authors", [])
                    keywords = doc.get("keywords", [])
                    
                    try:
                        authors_json = json.dumps(authors) if authors else "[]"
                        keywords_json = json.dumps(keywords) if keywords else "[]"
                    except Exception as e:
                        logger.warning(f"Error serializing metadata for doc {doc_id_value}: {e}")
                        authors_json = "[]"
                        keywords_json = "[]"
                    
                    # Add chunking info to metadata if present
                    metadata = doc.get("metadata", {})
                    if doc.get("metadata", {}).get("is_chunk"):
                        loaded_chunk_count += 1
                    
                    doc_params = (
                        str(doc_id_value),
                        str(doc_id_value),
                        title,
                        abstract,
                        text_content,
                        authors_json,
                        keywords_json,
                        embedding_vector
                    )

                    source_doc_batch_params.append(doc_params)
                    docs_for_token_embedding.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('doc_id', 'unknown')}: {e}")
                    error_count += 1
                    continue
            
            if not source_doc_batch_params:
                logger.warning(f"No valid documents in batch {batch_idx}")
                continue
                
            try:
                # Use simple INSERT with proper VECTOR data
                sql_source_docs = """
                INSERT INTO RAG.SourceDocuments
                (id, doc_id, title, abstract, text_content, authors, keywords, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                logger.info(f"Executing batch {batch_idx} with {len(source_doc_batch_params)} documents")
                cursor.executemany(sql_source_docs, source_doc_batch_params)
                loaded_doc_count += len(source_doc_batch_params)

                connection.commit()
                
                if (batch_idx + 1) % 1 == 0 or batch_idx == len(doc_batches) - 1:
                    elapsed = time.time() - start_time
                    rate = loaded_doc_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Loaded {loaded_doc_count}/{len(expanded_documents)} total documents ({loaded_chunk_count} chunks). Loaded {loaded_token_count} token embeddings. ({rate:.2f} docs/sec)")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_idx}: {e}")
                connection.rollback()
                error_count += len(current_doc_batch)
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        print("Total documents in RAG.SourceDocuments after loading:", cursor.fetchone()[0])

        cursor.close()
        
    except Exception as e:
        logger.error(f"Error in document loading process: {e}")
        error_count = len(documents) - loaded_doc_count
    
    duration = time.time() - start_time
    
    success_flag = loaded_doc_count > 0

    return {
        "success": success_flag,
        "total_documents": len(documents),
        "total_expanded_documents": len(expanded_documents) if 'expanded_documents' in locals() else len(documents),
        "loaded_doc_count": loaded_doc_count,
        "loaded_chunk_count": loaded_chunk_count,
        "loaded_token_count": loaded_token_count,
        "error_count": error_count,
        "duration_seconds": duration,
        "documents_per_second": loaded_doc_count / duration if duration > 0 else 0
    }

def process_and_load_documents(
    pmc_directory: str, 
    connection=None, 
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    db_config: Optional[Dict[str, Any]] = None, # Added db_config parameter
    limit: int = 1000,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Process PMC XML files and load them into IRIS database with comprehensive error handling.
    """
    start_time = time.time()
    
    # Create connection if not provided
    conn_provided = connection is not None
    if not connection:
        # Pass db_config to get_iris_connection
        connection = get_iris_connection(config=db_config)
        if not connection:
            return {
                "success": False,
                "error": "Failed to establish database connection",
                "processed_count": 0,
                "loaded_count": 0,
                "duration_seconds": time.time() - start_time
            }
    
    try:
        # Process and collect documents
        logger.info(f"Processing up to {limit} documents from {pmc_directory}")
        documents = list(process_pmc_files(pmc_directory, limit))
        processed_count = len(documents)
        logger.info(f"Processed {processed_count} documents")
        
        # Load documents with fixed loader
        load_stats = load_documents_to_iris(
            connection, 
            documents, 
            embedding_func, 
            batch_size
        )
        
        # Close connection if we created it
        if not conn_provided:
            connection.close()
        
        # Combine stats
        return {
            "success": load_stats.get("success", False),
            "processed_count": processed_count,
            "processed_directory": pmc_directory,
            **load_stats
        }
        
    except Exception as e:
        logger.error(f"Error in processing and loading: {e}")
        
        # Close connection if we created it
        if not conn_provided:
            try:
                connection.close()
            except:
                pass
        
        return {
            "success": False,
            "error": str(e),
            "processed_count": 0,
            "loaded_count": 0,
            "duration_seconds": time.time() - start_time
        }