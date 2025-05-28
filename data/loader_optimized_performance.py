"""
Optimized Data Loader for IRIS RAG Templates - PERFORMANCE OPTIMIZED VERSION

This module addresses the severe performance degradation in token embedding insertions
by implementing optimized batching, connection pooling, and reduced database contention.
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
    
    Args:
        vector: Properly formatted vector (list of floats)
        
    Returns:
        Comma-separated string representation
    """
    try:
        # CRITICAL FIX: Ensure we have a proper list of basic Python floats
        if not isinstance(vector, list):
            raise VectorFormatError(f"Expected list, got {type(vector)}")
        
        # Convert all values to basic Python floats and validate
        safe_values = []
        for i, v in enumerate(vector):
            # Handle complex objects that might be in the vector
            if hasattr(v, '__dict__') or hasattr(v, '__slots__'):
                if hasattr(v, 'item'):
                    v = v.item()
                elif hasattr(v, 'value'):
                    v = v.value
                else:
                    v = float(v)
            
            # Ensure it's a basic Python float
            float_val = float(v)
            
            if not np.isfinite(float_val):
                logger.warning(f"Non-finite value at index {i}: {float_val}, replacing with 0.0")
                float_val = 0.0
            
            safe_values.append(float_val)
        
        if not safe_values:
            raise VectorFormatError("Vector is empty after processing")
        
        # Format as comma-separated string with controlled precision
        vector_str = ','.join(f"{x:.15g}" for x in safe_values)
        
        # Final validation - ensure the string doesn't contain any problematic characters
        if any(char in vector_str for char in ['\x00', '\n', '\r', '\t']):
            raise VectorFormatError("Vector string contains problematic characters")
        
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

def load_documents_to_iris_optimized(
    connection,
    documents: List[Dict[str, Any]],
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None,
    batch_size: int = 50,  # REDUCED batch size for better performance
    token_batch_size: int = 1000  # SEPARATE batch size for token embeddings
) -> Dict[str, Any]:
    """
    Load documents into IRIS database with PERFORMANCE OPTIMIZATIONS.
    
    Key optimizations:
    1. Reduced batch sizes to prevent database contention
    2. Separate batching for token embeddings
    3. Optimized transaction management
    4. Progress monitoring with early warning for slowdowns
    
    Args:
        connection: IRIS connection object
        documents: List of document dictionaries to load
        embedding_func: Optional function to generate embeddings for documents
        colbert_doc_encoder_func: Optional function for ColBERT token embeddings
        batch_size: Number of documents to insert in a single batch (REDUCED)
        token_batch_size: Number of token embeddings to insert in a single batch
        
    Returns:
        Dictionary with loading statistics
    """
    start_time = time.time()
    loaded_doc_count = 0
    loaded_token_count = 0
    error_count = 0
    
    # Performance monitoring
    batch_times = []
    performance_warning_threshold = 30.0  # seconds per batch
    
    try:
        cursor = connection.cursor()
        
        # Prepare documents in SMALLER batches for better performance
        doc_batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        
        logger.info(f"🚀 OPTIMIZED LOADING: {len(documents)} documents in {len(doc_batches)} batches (batch_size={batch_size})")
        
        for batch_idx, current_doc_batch in enumerate(doc_batches):
            batch_start_time = time.time()
            
            source_doc_batch_params = []
            docs_for_token_embedding = []

            # Process documents for this batch
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
                        abstract,  # text_content
                        abstract,  # abstract (separate field)
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
                # Insert source documents
                sql_source_docs = """
                INSERT INTO RAG.SourceDocuments
                (doc_id, title, text_content, abstract, authors, keywords, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                
                logger.info(f"📝 Executing batch {batch_idx} with {len(source_doc_batch_params)} documents")
                cursor.executemany(sql_source_docs, source_doc_batch_params)
                loaded_doc_count += len(source_doc_batch_params)
                
                # OPTIMIZED TOKEN EMBEDDING PROCESSING
                if colbert_doc_encoder_func:
                    all_token_params = []
                    
                    # Generate all token embeddings for this batch
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
                                                all_token_params.append(
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
                    
                    # Insert token embeddings in OPTIMIZED SMALLER BATCHES
                    if all_token_params:
                        sql_token_embeddings = """
                        INSERT INTO RAG.DocumentTokenEmbeddings
                        (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
                        VALUES (?, ?, ?, ?, ?)
                        """
                        
                        # Process token embeddings in smaller sub-batches
                        token_batches = [all_token_params[i:i+token_batch_size] 
                                       for i in range(0, len(all_token_params), token_batch_size)]
                        
                        for token_batch_idx, token_batch in enumerate(token_batches):
                            try:
                                cursor.executemany(sql_token_embeddings, token_batch)
                                loaded_token_count += len(token_batch)
                                logger.debug(f"  ✅ Loaded token sub-batch {token_batch_idx} with {len(token_batch)} tokens")
                            except Exception as token_e:
                                logger.error(f"❌ Failed to insert token sub-batch {token_batch_idx}: {token_e}")
                        
                        logger.info(f"✅ Loaded {len(all_token_params)} token embeddings for batch {batch_idx}")

                # Commit after each batch to prevent long-running transactions
                connection.commit()
                
                # Performance monitoring
                batch_duration = time.time() - batch_start_time
                batch_times.append(batch_duration)
                
                # Calculate current rate
                elapsed = time.time() - start_time
                rate = loaded_doc_count / elapsed if elapsed > 0 else 0
                
                # Performance warning system
                if batch_duration > performance_warning_threshold:
                    logger.warning(f"⚠️  PERFORMANCE WARNING: Batch {batch_idx} took {batch_duration:.1f}s (threshold: {performance_warning_threshold}s)")
                    logger.warning(f"⚠️  Current rate: {rate:.2f} docs/sec (target: >10 docs/sec)")
                
                # Progress reporting
                if (batch_idx + 1) % 1 == 0 or batch_idx == len(doc_batches) - 1:
                    logger.info(f"📊 Progress: {loaded_doc_count}/{len(documents)} docs, {loaded_token_count} tokens ({rate:.2f} docs/sec)")
                    
                    # Adaptive performance monitoring
                    if len(batch_times) >= 3:
                        recent_avg = sum(batch_times[-3:]) / 3
                        if recent_avg > performance_warning_threshold:
                            logger.warning(f"⚠️  DEGRADING PERFORMANCE: Recent batches averaging {recent_avg:.1f}s")
                            logger.warning(f"⚠️  Consider reducing batch sizes or investigating database contention")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_idx}: {e}")
                connection.rollback()
                error_count += len(current_doc_batch)
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error in document loading process: {e}")
        error_count = len(documents) - loaded_doc_count
    
    duration = time.time() - start_time
    
    # Final performance analysis
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        max_batch_time = max(batch_times)
        logger.info(f"📈 Performance Summary:")
        logger.info(f"   Average batch time: {avg_batch_time:.1f}s")
        logger.info(f"   Maximum batch time: {max_batch_time:.1f}s")
        logger.info(f"   Performance degradation: {'YES' if max_batch_time > 2 * avg_batch_time else 'NO'}")
    
    return {
        "total_documents": len(documents),
        "loaded_doc_count": loaded_doc_count,
        "loaded_token_count": loaded_token_count,
        "error_count": error_count,
        "duration_seconds": duration,
        "documents_per_second": loaded_doc_count / duration if duration > 0 else 0,
        "batch_times": batch_times,
        "performance_degraded": max(batch_times) > performance_warning_threshold if batch_times else False
    }

def process_and_load_documents_optimized(
    pmc_directory: str, 
    connection=None, 
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None,
    limit: int = 1000, 
    batch_size: int = 50,  # REDUCED for better performance
    token_batch_size: int = 1000,  # Separate token batch size
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Process PMC XML files and load them into IRIS database with PERFORMANCE OPTIMIZATIONS.
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
        logger.info(f"🚀 OPTIMIZED PROCESSING: Up to {limit} documents from {pmc_directory}")
        documents = list(process_pmc_files(pmc_directory, limit))
        processed_count = len(documents)
        logger.info(f"📄 Processed {processed_count} documents")
        
        # Load documents with OPTIMIZED loader
        load_stats = load_documents_to_iris_optimized(
            connection, 
            documents, 
            embedding_func, 
            colbert_doc_encoder_func,
            batch_size,
            token_batch_size
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