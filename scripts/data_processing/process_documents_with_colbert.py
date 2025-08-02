#!/usr/bin/env python3
"""
Enhanced Document Processing with ColBERT Token Embeddings

This script processes PMC documents and generates both document-level embeddings
and ColBERT token embeddings, ensuring all RAG techniques have the required data.
"""

import os
import sys
import logging
import time
import json
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.core.connection import ConnectionManager
from data.pmc_processor import process_pmc_files
from common.db_vector_utils import insert_vector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_functions_with_schema_manager(schema_manager: SchemaManager):
    """Get embedding functions using schema manager for proper configuration."""
    try:
        from common.utils import get_embedding_func, get_colbert_doc_encoder_func
        
        # Get standard embedding function for document-level embeddings
        embedding_func = get_embedding_func()
        
        # Get ColBERT document encoder using schema manager's configuration
        colbert_encoder = get_colbert_doc_encoder_func()
        
        logger.info("Successfully initialized embedding functions with schema manager")
        return embedding_func, colbert_encoder
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding functions: {e}")
        raise

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
        import numpy as np
        
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
        
        # Format as comma-separated string for IRIS VECTOR column
        embedding_str = ','.join(f"{x:.15g}" for x in cleaned_embedding)
        
        return embedding_str
        
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return None

def store_document_with_embeddings(
    connection,
    doc: Dict[str, Any],
    embedding_func: Callable,
    colbert_encoder: Callable,
    schema_manager: SchemaManager
) -> bool:
    """
    Store a single document with both document-level and token-level embeddings.
    
    Args:
        connection: Database connection
        doc: Document dictionary
        embedding_func: Function for document-level embeddings
        colbert_encoder: Function for ColBERT token embeddings
        
    Returns:
        bool: True if successful, False otherwise
    """
    doc_id = doc.get("doc_id") or doc.get("pmc_id")
    if not doc_id:
        logger.error(f"Document missing doc_id: {doc}")
        return False
    
    cursor = connection.cursor()
    
    try:
        # Prepare document text content
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        text_content = doc.get("content", "") or doc.get("text_content", "")
        
        # Use full content for embedding
        text_for_embedding = text_content or abstract or title
        
        if not text_for_embedding:
            logger.warning(f"Document {doc_id} has no usable text content")
            return False
        
        # Generate document-level embedding
        doc_embedding = None
        if embedding_func:
            try:
                embedding = embedding_func([text_for_embedding])[0]
                doc_embedding = validate_and_fix_embedding(embedding)
            except Exception as e:
                logger.error(f"Error generating document embedding for {doc_id}: {e}")
        
        # Insert document into SourceDocuments
        authors_json = json.dumps(doc.get("authors", []))
        keywords_json = json.dumps(doc.get("keywords", []))
        
        # Use insert_vector utility for proper vector handling
        if doc_embedding:
            # Convert doc_embedding string back to list for insert_vector
            doc_embedding_list = [float(x) for x in doc_embedding.split(',')]
            
            # Get document dimension from schema manager (single source of truth)
            doc_dimension = schema_manager.get_vector_dimension("SourceDocuments")
            
            insert_vector(
                cursor=cursor,
                table_name="RAG.SourceDocuments",
                vector_column_name="embedding",
                vector_data=doc_embedding_list,
                target_dimension=doc_dimension,  # Schema manager authority
                key_columns={
                    "doc_id": str(doc_id)
                },
                additional_data={
                    "title": title,
                    "text_content": text_content,
                    "abstract": abstract,
                    "authors": authors_json,
                    "keywords": keywords_json
                }
            )
        else:
            # Insert without embedding
            cursor.execute("""
                INSERT INTO RAG.SourceDocuments
                (doc_id, title, text_content, abstract, authors, keywords)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (str(doc_id), title, text_content, abstract, authors_json, keywords_json))
        
        # Generate and store ColBERT token embeddings
        if colbert_encoder:
            try:
                # Use full content for ColBERT, fallback to abstract/title  
                colbert_text = text_content or abstract or title
                
                # Generate token embeddings
                # The mock colbert_encoder returns List[Tuple[str, List[float]]]
                token_data_tuples = colbert_encoder(colbert_text)
                
                if token_data_tuples:
                    tokens = [item[0] for item in token_data_tuples]
                    token_embeddings_list_of_lists = [item[1] for item in token_data_tuples]
                    
                    if tokens and token_embeddings_list_of_lists and len(tokens) == len(token_embeddings_list_of_lists):
                        successful_token_insertions = 0
                        failed_token_insertions = 0
                        # Store each token embedding
                        for token_idx, (token_text, single_token_embedding_list) in enumerate(zip(tokens, token_embeddings_list_of_lists)):
                            # single_token_embedding_list is already List[float] from the mock encoder
                            token_embedding_str = validate_and_fix_embedding(single_token_embedding_list)
                            
                            if token_embedding_str:
                                # insert_vector expects List[float], so convert the string back if valid
                                try:
                                    # Ensure token_embedding_str is a valid comma-separated list of numbers
                                    # validate_and_fix_embedding should already ensure this.
                                    # If validate_and_fix_embedding returned None, this will be skipped.
                                    final_token_embedding_list = [float(x) for x in token_embedding_str.split(',')]

                                    # Get ColBERT token dimension from schema manager (single source of truth)
                                    token_dimension = schema_manager.get_vector_dimension("DocumentTokenEmbeddings")
                                    
                                    if insert_vector(
                                        cursor=cursor,
                                        table_name="RAG.DocumentTokenEmbeddings",
                                        vector_column_name="token_embedding",
                                        vector_data=final_token_embedding_list, # This should be List[float]
                                        target_dimension=token_dimension,  # Schema manager authority
                                        key_columns={
                                            "doc_id": str(doc_id),
                                            "token_index": token_idx
                                        },
                                        additional_data={
                                            "token_text": token_text[:500]  # Limit token text length
                                        }
                                    ):
                                        successful_token_insertions += 1
                                    else:
                                        failed_token_insertions +=1
                                        logger.error(f"Failed to insert token embedding for doc {doc_id}, token_index {token_idx}")
                                except ValueError as ve:
                                    logger.error(f"Skipping token embedding for doc {doc_id}, token '{token_text}' due to invalid numeric string: {token_embedding_str}. Error: {ve}")
                                    failed_token_insertions +=1
                                    continue # Skip this token if conversion fails
                            else:
                                logger.warning(f"Skipping token embedding for doc {doc_id}, token_index {token_idx} due to invalid/empty embedding string after validation.")
                                failed_token_insertions += 1
                        
                        logger.info(f"For document {doc_id}: Attempted to store {len(tokens)} tokens. Successful: {successful_token_insertions}, Failed: {failed_token_insertions}")
                    else:
                        logger.warning(f"Token/embedding length mismatch or empty lists for document {doc_id}")
                else:
                    logger.warning(f"No token data returned by ColBERT encoder for document {doc_id}")
                    
            except Exception as e:
                logger.error(f"Error generating ColBERT token embeddings for {doc_id}: {e}")
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"Error storing document {doc_id}: {e}")
        cursor.close()
        return False

def process_and_load_documents_with_colbert(
    pmc_directory: str,
    limit: int = 1000,
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Process PMC documents and load them with both document and token embeddings.
    
    Args:
        pmc_directory: Directory containing PMC XML files
        limit: Maximum number of documents to process
        batch_size: Number of documents to process in each batch
        
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    
    logger.info(f"Starting enhanced document processing with ColBERT token embeddings")
    logger.info(f"Processing up to {limit} documents from {pmc_directory}")
    
    try:
        # Initialize schema manager and configuration
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Ensure all required tables are ready
        logger.info("Schema manager ensuring required tables are ready...")
        schema_manager.ensure_table_schema("SourceDocuments")
        schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
        
        # Get database connection through schema manager's connection manager
        connection = connection_manager.get_connection()
        if not connection:
            return {
                "success": False,
                "error": "Failed to establish database connection",
                "processed_count": 0,
                "duration_seconds": time.time() - start_time
            }
        
        # Initialize embedding functions with schema manager
        embedding_func, colbert_encoder = get_embedding_functions_with_schema_manager(schema_manager)
        
        # Process documents
        documents = list(process_pmc_files(pmc_directory, limit))
        processed_count = len(documents)
        
        logger.info(f"Processed {processed_count} documents from XML files")
        
        # Load documents in batches
        loaded_count = 0
        error_count = 0
        
        doc_batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        
        for batch_idx, batch in enumerate(doc_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(doc_batches)} ({len(batch)} documents)")
            
            batch_success_count = 0
            for doc in batch:
                if store_document_with_embeddings(connection, doc, embedding_func, colbert_encoder, schema_manager):
                    batch_success_count += 1
                else:
                    error_count += 1
            
            # Commit after each batch
            connection.commit()
            loaded_count += batch_success_count
            
            logger.info(f"Batch {batch_idx + 1} completed: {batch_success_count}/{len(batch)} documents loaded successfully")
        
        # Verify results
        cursor = connection.cursor()
        
        # Check document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        
        # Check token embeddings count
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        total_tokens = cursor.fetchone()[0]
        
        # Check documents with token embeddings
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
        docs_with_tokens = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        duration = time.time() - start_time
        
        result = {
            "success": True,
            "processed_count": processed_count,
            "loaded_count": loaded_count,
            "error_count": error_count,
            "total_documents_in_db": total_docs,
            "total_token_embeddings": total_tokens,
            "documents_with_token_embeddings": docs_with_tokens,
            "duration_seconds": duration,
            "documents_per_second": loaded_count / duration if duration > 0 else 0
        }
        
        logger.info("="*60)
        logger.info("ENHANCED DOCUMENT PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Documents processed from XML: {processed_count}")
        logger.info(f"Documents loaded to database: {loaded_count}")
        logger.info(f"Errors encountered: {error_count}")
        logger.info(f"Total documents in database: {total_docs}")
        logger.info(f"Total token embeddings: {total_tokens}")
        logger.info(f"Documents with token embeddings: {docs_with_tokens}")
        logger.info(f"Processing rate: {loaded_count / duration:.2f} docs/sec")
        
        if docs_with_tokens > 0:
            logger.info("✅ ColBERT token embeddings successfully generated!")
            logger.info("✅ All RAG techniques should now work properly")
        else:
            logger.warning("⚠️  No ColBERT token embeddings were generated")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced document processing: {e}")
        connection.close()
        
        return {
            "success": False,
            "error": str(e),
            "processed_count": 0,
            "loaded_count": 0,
            "duration_seconds": time.time() - start_time
        }

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents with ColBERT token embeddings")
    parser.add_argument("--directory", default="data/pmc_oas_downloaded", 
                       help="Directory containing PMC XML files")
    parser.add_argument("--limit", type=int, default=1000,
                       help="Maximum number of documents to process")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        sys.exit(1)
    
    # Process documents
    result = process_and_load_documents_with_colbert(
        pmc_directory=args.directory,
        limit=args.limit,
        batch_size=args.batch_size
    )
    
    if result["success"]:
        logger.info("✅ Enhanced document processing completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"❌ Enhanced document processing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()