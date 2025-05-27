"""
Data Loader for IRIS RAG Templates - VARCHAR EMBEDDING FIXED VERSION

This module handles loading processed documents into the IRIS database,
with definitive fixes for LIST ERROR issues when using VARCHAR embedding columns.
"""

import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Generator, Optional, Tuple, Callable
import os
import sys

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from common.vector_format_fix import format_vector_for_iris, validate_vector_for_iris, VectorFormatError
from data.pmc_processor import process_pmc_files

logger = logging.getLogger(__name__)

def format_vector_for_varchar_column(vector: List[float]) -> str:
    """
    Format a vector as comma-separated string for VARCHAR embedding columns.
    
    This addresses the LIST ERROR by ensuring vectors are properly formatted
    as strings rather than being passed as Python lists to VARCHAR columns.
    
    Args:
        vector: Properly formatted vector (list of floats)
        
    Returns:
        Comma-separated string representation
    """
    try:
        # Ensure all values are finite floats
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in vector):
            raise VectorFormatError("Vector contains non-finite values")
        
        # Format as comma-separated string with controlled precision
        vector_str = ','.join(f"{float(x):.15g}" for x in vector)
        
        return vector_str
        
    except Exception as e:
        raise VectorFormatError(f"Error formatting vector for VARCHAR column: {e}")

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
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None,
    batch_size: int = 250
) -> Dict[str, Any]:
    """
    Load documents into IRIS database with definitive VARCHAR embedding fixes.
    
    Args:
        connection: IRIS connection object
        documents: List of document dictionaries to load
        embedding_func: Optional function to generate embeddings for documents
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
            source_doc_batch_params = []
            docs_for_token_embedding = []

            for doc in current_doc_batch:
                try:
                    embedding_vector_str = None
                    if embedding_func:
                        text_to_embed = doc.get("abstract") or doc.get("title", "")
                        if text_to_embed:
                            try:
                                # Generate embedding with error handling
                                text_to_embed = validate_and_fix_text_field(text_to_embed)
                                embedding = embedding_func([text_to_embed])[0]
                                
                                # CRITICAL FIX: Format vector properly for VARCHAR column
                                # Step 1: Clean the vector using our vector format fix
                                embedding_vector_clean = format_vector_for_iris(embedding)
                                
                                # Step 2: Convert to string for VARCHAR column
                                embedding_vector_str = format_vector_for_varchar_column(embedding_vector_clean)
                                
                                logger.debug(f"Generated embedding string of length {len(embedding_vector_str)} for doc {doc.get('doc_id')}")
                                    
                            except VectorFormatError as e:
                                logger.error(f"Vector format error for document {doc.get('doc_id')}: {e}")
                                embedding_vector_str = None
                            except Exception as e:
                                logger.error(f"Error generating embedding for document {doc.get('doc_id')}: {e}")
                                embedding_vector_str = None
                        else:
                            logger.warning(f"Document {doc.get('doc_id')} has no abstract or title for sentence embedding.")
                    
                    # Get document ID with validation
                    doc_id_value = doc.get("doc_id") or doc.get("pmc_id")
                    if not doc_id_value:
                        logger.error(f"Document missing doc_id: {doc}")
                        continue
                    
                    # Validate and clean all text fields
                    title = validate_and_fix_text_field(doc.get("title"))
                    abstract = validate_and_fix_text_field(doc.get("abstract"))
                    
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
                    
                    doc_params = (
                        str(doc_id_value),
                        title,
                        abstract,
                        authors_json,
                        keywords_json,
                        embedding_vector_str  # This is now a proper comma-separated string
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
                # Use INSERT with VARCHAR embedding column
                sql_source_docs = """
                INSERT INTO RAG.SourceDocuments
                (doc_id, title, text_content, authors, keywords, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                logger.info(f"Executing batch {batch_idx} with {len(source_doc_batch_params)} documents")
                cursor.executemany(sql_source_docs, source_doc_batch_params)
                loaded_doc_count += len(source_doc_batch_params)
                
                # Process ColBERT token embeddings with VARCHAR format fixes
                if colbert_doc_encoder_func:
                    token_embedding_batch_params = []
                    
                    for doc_for_tokens in docs_for_token_embedding:
                        try:
                            doc_id = doc_for_tokens.get("doc_id") or doc_for_tokens.get("pmc_id")
                            text_for_colbert = doc_for_tokens.get("abstract") or doc_for_tokens.get("title", "")
                            
                            if doc_id and text_for_colbert:
                                text_for_colbert = validate_and_fix_text_field(text_for_colbert)
                                
                                # Get ColBERT token embeddings with error handling
                                token_data = colbert_doc_encoder_func(text_for_colbert)
                                if token_data and len(token_data) == 2:
                                    tokens, embeddings = token_data
                                    if tokens and embeddings and len(tokens) == len(embeddings):
                                        for idx, (token_text, token_vec) in enumerate(zip(tokens, embeddings)):
                                            try:
                                                # CRITICAL FIX: Format token vector for VARCHAR column
                                                # Step 1: Clean the vector
                                                token_vec_clean = format_vector_for_iris(token_vec)
                                                
                                                # Step 2: Convert to string for VARCHAR column
                                                token_vec_str = format_vector_for_varchar_column(token_vec_clean)
                                                
                                                token_text_clean = validate_and_fix_text_field(token_text)[:1000]
                                                token_embedding_batch_params.append(
                                                    (str(doc_id), idx, token_text_clean, token_vec_str, "{}")
                                                )
                                                    
                                            except VectorFormatError as e:
                                                logger.error(f"Token vector format error for doc {doc_id}, token {idx}: {e}")
                                    else:
                                        logger.warning(f"Token/embedding length mismatch for doc {doc_id}")
                                else:
                                    logger.warning(f"No token embeddings generated for doc {doc_id}")
                                    
                        except Exception as colbert_e:
                            logger.error(f"Error generating ColBERT token embeddings for doc {doc_id}: {colbert_e}")
                    
                    if token_embedding_batch_params:
                        try:
                            sql_token_embeddings = """
                            INSERT INTO RAG.DocumentTokenEmbeddings
                            (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
                            VALUES (?, ?, ?, ?, ?)
                            """
                            cursor.executemany(sql_token_embeddings, token_embedding_batch_params)
                            loaded_token_count += len(token_embedding_batch_params)
                            logger.info(f"✅ Loaded {len(token_embedding_batch_params)} token embeddings for batch {batch_idx}")
                        except Exception as token_e:
                            logger.error(f"❌ Failed to insert token embeddings for batch {batch_idx}: {token_e}")

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
    Process PMC XML files and load them into IRIS database with VARCHAR embedding fixes.
    """
    start_time = time.time()
    
    # Create connection if not provided
    conn_provided = connection is not None
    if not connection:
        connection = get_iris_connection(use_mock=use_mock)
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
        
        # Load documents with VARCHAR-fixed loader
        load_stats = load_documents_to_iris(
            connection, 
            documents, 
            embedding_func, 
            colbert_doc_encoder_func,
            batch_size
        )
        
        # Close connection if we created it
        if not conn_provided:
            connection.close()
        
        # Combine stats
        return {
            "success": True,
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