"""
Data Loader for IRIS RAG Templates

This module handles loading processed documents into the IRIS database,
with support for batching, error handling, and performance metrics.
"""

import logging
import time
import json
from typing import List, Dict, Any, Generator, Optional, Tuple, Callable # Added Callable
import os
import sys

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from data.pmc_processor import process_pmc_files

logger = logging.getLogger(__name__)

def load_documents_to_iris(
    connection,
    documents: List[Dict[str, Any]],
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None, # For ColBERT token embeddings
    batch_size: int = 250
) -> Dict[str, Any]:
    """
    Load documents into IRIS database with batching, including embeddings.
    
    Args:
        connection: IRIS connection object
        documents: List of document dictionaries to load
        embedding_func: Optional function to generate embeddings for documents.
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
            docs_for_token_embedding = [] # Store docs in this batch for subsequent token embedding

            for doc in current_doc_batch:
                embedding_vector_str = None
                if embedding_func:
                    text_to_embed = doc.get("abstract") or doc.get("title", "")
                    if text_to_embed:
                        embedding = embedding_func([text_to_embed])[0]
                        # Store as comma-separated values (no brackets) for VARCHAR storage
                        embedding_vector_str = ','.join(map(str, embedding))
                    else:
                        logger.warning(f"Document {doc.get('pmc_id')} has no abstract or title for sentence embedding.")
                
                # FIXED: Use doc_id instead of pmc_id - the PMC processor returns 'doc_id', not 'pmc_id'
                doc_id_value = doc.get("doc_id") or doc.get("pmc_id")
                
                doc_params = (
                    doc_id_value,
                    doc.get("title"),
                    doc.get("abstract"),
                    json.dumps(doc.get("authors", [])),
                    json.dumps(doc.get("keywords", [])),
                    embedding_vector_str
                )
                source_doc_batch_params.append(doc_params)
                docs_for_token_embedding.append(doc) # Save for token processing
            
            try:
                # Use simple INSERT without TO_VECTOR - store embeddings as VARCHAR comma-separated values
                # This is the working approach based on schema investigation
                sql_source_docs = """
                INSERT INTO RAG.SourceDocuments
                (doc_id, title, text_content, authors, keywords, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                # Print the SQL and parameters before executing
                print(f"Executing SQL: {sql_source_docs}")
                print(f"First row parameters: {source_doc_batch_params[0] if source_doc_batch_params else 'No parameters'}")
                
                cursor.executemany(sql_source_docs, source_doc_batch_params)
                loaded_doc_count += len(current_doc_batch)
                
                # Now process and insert ColBERT token embeddings - FIXED TO ACTUALLY WORK
                if colbert_doc_encoder_func:
                    token_embedding_batch_params = []
                    
                    for doc_for_tokens in docs_for_token_embedding:
                        # Use doc_id instead of pmc_id - same fix as above
                        doc_id = doc_for_tokens.get("doc_id") or doc_for_tokens.get("pmc_id")
                        text_for_colbert = doc_for_tokens.get("abstract") or doc_for_tokens.get("title", "")
                        if doc_id and text_for_colbert:
                            try:
                                # Get ColBERT token embeddings
                                token_data = colbert_doc_encoder_func(text_for_colbert)
                                if token_data and len(token_data) == 2:  # Should be (tokens, embeddings)
                                    tokens, embeddings = token_data
                                    if tokens and embeddings and len(tokens) == len(embeddings):
                                        for idx, (token_text, token_vec) in enumerate(zip(tokens, embeddings)):
                                            # Store as comma-separated values for VARCHAR storage
                                            token_vec_str = ','.join(map(str, token_vec))
                                            token_embedding_batch_params.append(
                                                (doc_id, idx, token_text[:1000], token_vec_str, "{}")
                                            )
                                    else:
                                        logger.warning(f"Token/embedding length mismatch for doc {doc_id}: {len(tokens) if tokens else 0} tokens, {len(embeddings) if embeddings else 0} embeddings")
                                else:
                                    logger.warning(f"No token embeddings generated for doc {doc_id}")
                            except Exception as colbert_e:
                                logger.error(f"Error generating ColBERT token embeddings for doc {doc_id}: {colbert_e}")
                    
                    if token_embedding_batch_params:
                        try:
                            # Insert token embeddings as VARCHAR comma-separated values
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
                    else:
                        logger.warning(f"⚠️ No token embeddings to insert for batch {batch_idx}")

                connection.commit() 
                
                if (batch_idx + 1) % 1 == 0 or batch_idx == len(doc_batches) - 1: # Log more frequently
                    elapsed = time.time() - start_time
                    rate = loaded_doc_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Loaded {loaded_doc_count}/{len(documents)} SourceDocuments. Loaded {loaded_token_count} token embeddings. ({rate:.2f} docs/sec)")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_idx} (SourceDocs or Tokens): {e}")
                connection.rollback()
                error_count += len(current_doc_batch) # Count doc errors, token errors are harder to attribute here
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error in document loading process: {e}")
        error_count = len(documents) - loaded_doc_count # Approximate
    
    duration = time.time() - start_time
    
    return {
        "total_documents": len(documents),
        "loaded_doc_count": loaded_doc_count,
        "loaded_token_count": loaded_token_count,
        "error_count": error_count, # This primarily counts document load errors
        "duration_seconds": duration,
        "documents_per_second": loaded_doc_count / duration if duration > 0 else 0
    }

def process_and_load_documents(
    pmc_directory: str, 
    connection=None, 
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    colbert_doc_encoder_func: Optional[Callable[[str], List[Tuple[str, List[float]]]]] = None, # Added
    limit: int = 1000, 
    batch_size: int = 50,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Process PMC XML files and load them into IRIS database, including embeddings.
    
    Args:
        pmc_directory: Directory containing PMC XML files
        connection: Optional IRIS connection (will be created if None)
        embedding_func: Optional function to generate embeddings.
        limit: Maximum number of documents to process
        batch_size: Number of documents to insert in a single batch
        use_mock: Whether to use a mock connection if connection is None
        
    Returns:
        Dictionary with processing and loading statistics
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
        
        # Load documents
        load_stats = load_documents_to_iris(
            connection, 
            documents, 
            embedding_func, 
            colbert_doc_encoder_func, # Pass ColBERT encoder
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

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and load PMC documents into IRIS")
    parser.add_argument("--dir", type=str, default="data/pmc_oas_downloaded", help="Directory containing PMC XML files")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of documents to process")
    parser.add_argument("--batch", type=int, default=50, help="Batch size for database inserts")
    parser.add_argument("--mock", action="store_true", help="Use mock database connection")
    args = parser.parse_args()
    
    # Process and load documents
    # For standalone script execution, get real embedding functions
    from common.utils import get_embedding_func as get_real_embedding_func
    from common.utils import get_colbert_doc_encoder_func # Assuming this will be created
    
    real_embed_func = None
    real_colbert_doc_encoder_func = None

    if not args.mock: # Only get real embedders if not mocking DB
        try:
            real_embed_func = get_real_embedding_func()
        except Exception as e:
            logger.error(f"Failed to initialize real sentence embedding function: {e}")
            logger.info("Proceeding without sentence embeddings for standalone run.")
        try:
            # This function needs to be implemented in common/utils.py
            # It should return a function: (text: str) -> List[Tuple[str, List[float]]]
            real_colbert_doc_encoder_func = get_colbert_doc_encoder_func() 
        except Exception as e:
            logger.error(f"Failed to initialize real ColBERT document encoder function: {e}")
            logger.info("Proceeding without ColBERT token embeddings for standalone run.")

    stats = process_and_load_documents(
        args.dir,
        embedding_func=real_embed_func,
        colbert_doc_encoder_func=real_colbert_doc_encoder_func, # Pass ColBERT encoder
        limit=args.limit,
        batch_size=args.batch,
        use_mock=args.mock
    )
    
    # Print results
    if stats["success"]:
        print("\n=== Processing and Loading Results ===")
        print(f"Processed {stats['processed_count']} documents from {stats['processed_directory']}")
        print(f"Successfully loaded {stats['loaded_count']} documents")
        if stats['error_count'] > 0:
            print(f"Failed to load {stats['error_count']} documents")
        print(f"Total time: {stats['duration_seconds']:.2f} seconds")
        print(f"Loading rate: {stats['documents_per_second']:.2f} documents per second")
    else:
        print(f"\n❌ Error: {stats['error']}")
