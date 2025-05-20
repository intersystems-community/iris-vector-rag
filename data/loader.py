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
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None, # Added embedding_func
    batch_size: int = 50
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
    loaded_count = 0
    error_count = 0
    
    try:
        cursor = connection.cursor()
        
        # Prepare documents in batches
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        
        logger.info(f"Loading {len(documents)} documents in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            batch_params = []
            
            for doc in batch:
                # Generate embedding if function is provided
                embedding_vector_str = None
                if embedding_func:
                    # Use abstract for embedding, or title if abstract is empty
                    text_to_embed = doc.get("abstract") or doc.get("title", "")
                    if text_to_embed:
                        embedding = embedding_func([text_to_embed])[0]
                        embedding_vector_str = f"[{','.join(map(str, embedding))}]"
                    else:
                        logger.warning(f"Document {doc.get('pmc_id')} has no abstract or title for embedding.")
                
                doc_params = (
                    doc.get("pmc_id"),
                    doc.get("title"),
                    doc.get("abstract"), # text_content in DB
                    json.dumps(doc.get("authors", [])),
                    json.dumps(doc.get("keywords", [])),
                    embedding_vector_str # embedding
                )
                batch_params.append(doc_params)
            
            try:
                # Insert batch using executemany
                cursor.executemany(
                    """
                    INSERT INTO SourceDocuments
                    (doc_id, title, text_content, authors, keywords, embedding)
                    VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
                    """, 
                    batch_params
                )
                connection.commit()
                loaded_count += len(batch)
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
                    elapsed = time.time() - start_time
                    rate = loaded_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Loaded {loaded_count}/{len(documents)} documents ({rate:.2f} docs/sec)")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_idx}: {e}")
                connection.rollback()
                error_count += len(batch)
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error in document loading: {e}")
        error_count = len(documents) - loaded_count
    
    duration = time.time() - start_time
    
    return {
        "total_documents": len(documents),
        "loaded_count": loaded_count,
        "error_count": error_count,
        "duration_seconds": duration,
        "documents_per_second": loaded_count / duration if duration > 0 else 0
    }

def process_and_load_documents(
    pmc_directory: str, 
    connection=None, 
    embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None, # Added embedding_func
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
        load_stats = load_documents_to_iris(connection, documents, embedding_func, batch_size) # Pass embedding_func
        
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
    # For standalone script execution, get a real embedding function
    from common.utils import get_embedding_func as get_real_embedding_func
    
    real_embed_func = None
    if not args.mock: # Only get real embedder if not mocking DB
        try:
            real_embed_func = get_real_embedding_func()
        except Exception as e:
            logger.error(f"Failed to initialize real embedding function for standalone loader: {e}")
            logger.error("Proceeding without embeddings for standalone run.")

    stats = process_and_load_documents(
        args.dir,
        embedding_func=real_embed_func, # Pass embedding function
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
