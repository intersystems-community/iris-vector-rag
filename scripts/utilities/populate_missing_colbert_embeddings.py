#!/usr/bin/env python3
"""
Populate Missing ColBERT Token Embeddings Script

This script populates missing ColBERT token embeddings for documents that don't have them yet.
It follows the specification provided in the ColBERT Token Embedding Population & RAGAS Evaluation Specification.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Tuple

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
# Import proven vector formatting utilities
from common.vector_format_fix import format_vector_for_iris, create_iris_vector_string, validate_vector_for_iris, VectorFormatError
# Try to import the real ColBERT encoder first, fall back to mock
try:
    import sys
    import os
    # Add the archive colbert path to sys.path
    archive_colbert_path = os.path.join(os.path.dirname(__file__), '..', 'archive', 'colbert')
    if archive_colbert_path not in sys.path:
        sys.path.insert(0, archive_colbert_path)
    from doc_encoder import ColBERTDocEncoder
    REAL_COLBERT_AVAILABLE = True
except ImportError:
    from common.utils import get_colbert_doc_encoder_func
    REAL_COLBERT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """
    Check for required IRIS connection environment variables.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    required_vars = ['IRIS_HOST', 'IRIS_PORT', 'IRIS_NAMESPACE', 'IRIS_USERNAME', 'IRIS_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("Environment validation passed")
    return True


def initialize_connections() -> ConnectionManager:
    """
    Initialize and test the IRIS database connection using ConnectionManager.
    
    Returns:
        ConnectionManager: Configured connection manager
        
    Raises:
        ConnectionError: If connection cannot be established
    """
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Test the connection
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        
        logger.info("IRIS database connection established successfully")
        return connection_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize IRIS connection: {e}")
        raise ConnectionError(f"Database connection failed: {e}")


def initialize_colbert_encoder():
    """
    Initialize the ColBERT document encoder.
    
    Returns:
        Callable: ColBERT document encoder function
    """
    try:
        # Get the correct embedding dimension from config
        config_manager = ConfigurationManager()
        token_embedding_dim = config_manager.get('colbert.token_embedding_dimension', 384)
        logger.info(f"Using token embedding dimension: {token_embedding_dim}")
        
        if REAL_COLBERT_AVAILABLE:
            # Use the real ColBERT encoder
            logger.info("Initializing real ColBERT document encoder")
            encoder = ColBERTDocEncoder(
                model_name="fjmgAI/reason-colBERT-150M-GTE-ModernColBERT",
                device="cpu",
                embedding_dim=token_embedding_dim,
                mock=False  # Try real first, will fall back to mock if needed
            )
            logger.info("Real ColBERT encoder initialized successfully")
            return encoder.encode  # Return the encode method that returns (tokens, embeddings)
        else:
            # Use the mock utility function
            logger.info("Using mock ColBERT encoder from common.utils")
            encoder_func = get_colbert_doc_encoder_func()
            logger.info("Mock ColBERT encoder initialized successfully")
            return encoder_func
        
    except Exception as e:
        logger.error(f"Failed to initialize ColBERT encoder: {e}")
        # Fall back to mock encoder
        logger.warning("Falling back to mock ColBERT encoder")
        try:
            encoder_func = get_colbert_doc_encoder_func()
            logger.info("Fallback mock ColBERT encoder initialized successfully")
            return encoder_func
        except Exception as fallback_error:
            logger.error(f"Failed to initialize fallback encoder: {fallback_error}")
            raise


def identify_missing_documents(iris_connector) -> List[Dict[str, Any]]:
    """
    Execute SQL query to identify documents missing ColBERT token embeddings.
    
    Args:
        iris_connector: Database connection
        
    Returns:
        List of dictionaries containing doc_id and text fields
    """
    try:
        cursor = iris_connector.cursor()
        
        # Query to find documents without token embeddings
        sql = """
            SELECT sd.doc_id, sd.text_content, sd.abstract, sd.title 
            FROM RAG.SourceDocuments sd 
            LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.doc_id = dte.doc_id 
            WHERE dte.doc_id IS NULL
        """
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        documents = []
        for row in results:
            doc_id, text_content, abstract, title = row
            documents.append({
                'doc_id': doc_id,
                'text_content': text_content,
                'abstract': abstract,
                'title': title
            })
        
        cursor.close()
        logger.info(f"Found {len(documents)} documents missing ColBERT token embeddings")
        return documents
        
    except Exception as e:
        logger.error(f"Error identifying missing documents: {e}")
        return []


def process_batch_embeddings(doc_batch: List[Dict[str, Any]], iris_connector, colbert_encoder, batch_size: int = 10):
    """
    Process a batch of documents to generate and store token embeddings.
    
    Args:
        doc_batch: List of document dictionaries
        iris_connector: Database connection
        colbert_encoder: ColBERT encoder function
        batch_size: Size of processing batch
    """
    logger.info(f"Processing batch of {len(doc_batch)} documents")
    
    processed_count = 0
    for doc in doc_batch:
        try:
            success = process_single_document(doc, iris_connector, colbert_encoder)
            if success:
                processed_count += 1
        except Exception as e:
            logger.error(f"Error processing document {doc.get('doc_id', 'unknown')}: {e}")
            continue
    
    # Commit after processing the batch
    try:
        iris_connector.commit()
        logger.info(f"Batch processing completed: {processed_count}/{len(doc_batch)} documents processed successfully")
    except Exception as e:
        logger.error(f"Error committing batch: {e}")


def process_single_document(doc: Dict[str, Any], iris_connector, colbert_encoder) -> bool:
    """
    Process a single document to generate and store token embeddings.
    
    Args:
        doc: Document dictionary with doc_id and text fields
        iris_connector: Database connection
        colbert_encoder: ColBERT encoder function
        
    Returns:
        bool: True if successful, False otherwise
    """
    doc_id = doc['doc_id']
    
    # Determine the text to encode (prefer text_content, fallback to abstract, then title)
    text_to_encode = None
    if doc.get('text_content') and doc['text_content'].strip():
        text_to_encode = doc['text_content']
    elif doc.get('abstract') and doc['abstract'].strip():
        text_to_encode = doc['abstract']
    elif doc.get('title') and doc['title'].strip():
        text_to_encode = doc['title']
    
    if not text_to_encode:
        logger.warning(f"Skipping document {doc_id} - no usable content")
        return False
    
    try:
        # Generate token embeddings using ColBERT encoder
        logger.debug(f"Encoding document {doc_id} with text length: {len(text_to_encode)}")
        encoder_output = colbert_encoder(text_to_encode)
        
        # Debug: Log the raw encoder output to understand the format
        logger.debug(f"Raw encoder output type: {type(encoder_output)}")
        
        if not encoder_output:
            logger.warning(f"No token embeddings generated for document {doc_id}")
            return False
        
        # Handle different encoder output formats
        if isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            # Tuple format: (tokens, token_embeddings)
            tokens, token_embeddings = encoder_output
            
            # Validate the format
            if not isinstance(tokens, list) or not isinstance(token_embeddings, list):
                logger.error(f"Invalid encoder output format for document {doc_id}: expected (List[str], List[List[float]]), got ({type(tokens)}, {type(token_embeddings)})")
                return False
            
            if len(tokens) != len(token_embeddings):
                logger.error(f"Token count mismatch for document {doc_id}: {len(tokens)} tokens vs {len(token_embeddings)} embeddings")
                return False
            
            # Convert to list of (token, embedding) pairs for storage
            token_data = list(zip(tokens, token_embeddings))
            
        elif isinstance(encoder_output, list):
            # List format: just embeddings without explicit tokens
            logger.debug(f"Encoder returned list of {len(encoder_output)} embeddings for document {doc_id}")
            
            # Validate that it's a list of embeddings
            if not encoder_output:
                logger.warning(f"Empty encoder output for document {doc_id}")
                return False
            
            # Check if first element is an embedding (list of floats)
            if not isinstance(encoder_output[0], (list, tuple)):
                logger.error(f"Invalid encoder output format for document {doc_id}: expected list of embeddings, got list of {type(encoder_output[0])}")
                return False
            
            # Create token names and pair with embeddings
            token_data = [(f"token_{i}", embedding) for i, embedding in enumerate(encoder_output)]
            
        else:
            logger.error(f"Invalid encoder output format for document {doc_id}: expected tuple (tokens, embeddings) or list of embeddings, got {type(encoder_output)}")
            return False
        
        # Validate token data format
        if not token_data:
            logger.warning(f"No valid token data for document {doc_id}")
            return False
        
        # Validate first token for format checking
        if len(token_data) > 0:
            token_text, token_embedding = token_data[0]
            if not isinstance(token_text, str):
                logger.error(f"Invalid token text format for document {doc_id}: expected str, got {type(token_text)}")
                return False
            
            if not isinstance(token_embedding, (list, tuple)):
                logger.error(f"Invalid token embedding format for document {doc_id}: expected list/tuple, got {type(token_embedding)}")
                logger.error(f"Token embedding value: {token_embedding}")
                return False
            
            # Check if embedding contains numbers
            if len(token_embedding) > 0 and not isinstance(token_embedding[0], (int, float)):
                logger.error(f"Invalid embedding values for document {doc_id}: expected numbers, got {type(token_embedding[0])}")
                logger.error(f"First embedding value: {token_embedding[0]}")
                return False
        
        # Store the token embeddings
        success = store_token_embeddings(doc_id, token_data, iris_connector)
        
        if success:
            logger.debug(f"Successfully processed document {doc_id} with {len(token_data)} tokens")
        
        return success
        
    except Exception as e:
        logger.error(f"Error encoding document {doc_id}: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def store_token_embeddings(doc_id: str, token_data: List[Tuple[str, List[float]]], iris_connector) -> bool:
    """
    Store token embeddings in the database.
    
    Args:
        doc_id: Document ID
        token_data: List of (token_text, token_embedding) tuples
        iris_connector: Database connection
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = iris_connector.cursor()
        
        # Prepare data for insertion
        insert_data = []
        for token_index, (token_text, token_embedding) in enumerate(token_data):
            try:
                # Debug: Log the raw embedding data
                logger.debug(f"Raw token_embedding type: {type(token_embedding)}")
                logger.debug(f"Raw token_embedding first 5 values: {token_embedding[:5] if hasattr(token_embedding, '__getitem__') and len(token_embedding) > 0 else 'Cannot slice'}")
                
                # Convert embedding to IRIS native vector string format
                embedding_vector_str = convert_to_iris_vector(token_embedding)
                
                # Debug: Log the converted string
                logger.info(f"CRITICAL DEBUG - Token {token_index} for doc {doc_id}:")
                logger.info(f"  Input type: {type(token_embedding)}")
                logger.info(f"  Input first 5: {token_embedding[:5] if hasattr(token_embedding, '__getitem__') and len(token_embedding) > 0 else 'Cannot slice'}")
                logger.info(f"  Output string: {embedding_vector_str[:100]}...")
                logger.info(f"  Contains @$vector: {'@$vector' in embedding_vector_str}")
                
                insert_data.append((doc_id, token_index, token_text, embedding_vector_str))
                
                # Log first few embeddings for debugging
                # Always log vector dimensions for debugging
                vector_length = len(token_embedding) if hasattr(token_embedding, '__len__') else 'unknown'
                logger.info(f"Token {token_index}: '{token_text}' -> vector length {vector_length}")
                
                if token_index < 3:
                    logger.debug(f"Token {token_index}: '{token_text}' -> vector length {vector_length}")
                    
            except Exception as e:
                logger.error(f"Error converting token {token_index} embedding for document {doc_id}: {e}")
                logger.error(f"Token text: '{token_text}', embedding type: {type(token_embedding)}")
                if hasattr(token_embedding, '__len__') and len(token_embedding) > 0:
                    logger.error(f"First embedding value: {token_embedding[0]} (type: {type(token_embedding[0])})")
                raise
        
        if not insert_data:
            logger.warning(f"No valid token embeddings to store for document {doc_id}")
            return False
        
        # Insert token embeddings directly into native VECTOR column
        insert_sql = """
            INSERT INTO RAG.DocumentTokenEmbeddings
            (doc_id, token_index, token_text, token_embedding)
            VALUES (?, ?, ?, ?)
        """
        
        logger.debug(f"Executing SQL with {len(insert_data)} token embeddings for document {doc_id}")
        # Use individual execute() calls for better error handling and debugging
        for data_row in insert_data:
            cursor.execute(insert_sql, data_row)
        cursor.close()
        
        logger.debug(f"Successfully stored {len(token_data)} token embeddings for document {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing token embeddings for document {doc_id}: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def convert_to_iris_vector(embedding_list: List[float]) -> str:
    """
    Convert a Python list of floats to IRIS vector string format using proven utilities.
    
    Args:
        embedding_list: List of float values
        
    Returns:
        str: String representation for IRIS native VECTOR column (comma-separated, no brackets)
        
    Raises:
        ValueError: If embedding_list is not a valid list of numbers
    """
    try:
        # Step 1: Format the vector using proven utilities
        formatted_vector = format_vector_for_iris(embedding_list)
        
        # Step 2: Validate the formatted vector
        if not validate_vector_for_iris(formatted_vector):
            raise ValueError("Vector validation failed after formatting")
        
        # Step 3: Create the IRIS vector string (comma-separated, no brackets for native VECTOR column)
        vector_str = create_iris_vector_string(formatted_vector)
        
        logger.debug(f"Converted embedding to IRIS vector format: {vector_str[:100]}...")
        return vector_str
        
    except VectorFormatError as e:
        logger.error(f"Vector format error: {e}")
        raise ValueError(f"Failed to convert embedding to IRIS vector format: {e}")
    except Exception as e:
        logger.error(f"Error converting embedding to IRIS vector format: {e}")
        logger.error(f"Embedding list: {embedding_list}")
        raise ValueError(f"Failed to convert embedding to IRIS vector format: {e}")


def verify_completion(iris_connector) -> Dict[str, Any]:
    """
    Verify that all missing embeddings were populated.
    
    Args:
        iris_connector: Database connection
        
    Returns:
        Dict with verification results
    """
    try:
        # Re-run the query to check for remaining missing documents
        missing_docs = identify_missing_documents(iris_connector)
        
        cursor = iris_connector.cursor()
        
        # Get total count of token embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        total_embeddings = cursor.fetchone()[0]
        
        # Get count of documents with embeddings
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
        docs_with_embeddings = cursor.fetchone()[0]
        
        cursor.close()
        
        verification_result = {
            'remaining_missing_docs': len(missing_docs),
            'total_token_embeddings': total_embeddings,
            'documents_with_embeddings': docs_with_embeddings,
            'completion_status': 'complete' if len(missing_docs) == 0 else 'incomplete'
        }
        
        if len(missing_docs) == 0:
            logger.info("✅ All missing embeddings have been populated")
        else:
            logger.warning(f"⚠️  {len(missing_docs)} documents still missing embeddings")
        
        return verification_result
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return {'error': str(e)}


def generate_completion_report(processed_docs: int, errors: int, verification_result: Dict[str, Any]):
    """
    Generate and print a completion report.
    
    Args:
        processed_docs: Number of documents processed
        errors: Number of errors encountered
        verification_result: Results from verification
    """
    logger.info("\n" + "="*60)
    logger.info("COLBERT TOKEN EMBEDDING POPULATION REPORT")
    logger.info("="*60)
    logger.info(f"Documents processed: {processed_docs}")
    logger.info(f"Errors encountered: {errors}")
    logger.info(f"Total token embeddings: {verification_result.get('total_token_embeddings', 'N/A')}")
    logger.info(f"Documents with embeddings: {verification_result.get('documents_with_embeddings', 'N/A')}")
    logger.info(f"Remaining missing docs: {verification_result.get('remaining_missing_docs', 'N/A')}")
    logger.info(f"Status: {verification_result.get('completion_status', 'unknown')}")
    logger.info("="*60)


def populate_missing_colbert_embeddings(batch_size: int = 10, dry_run: bool = False) -> Dict[str, Any]:
    """
    Main function to populate missing ColBERT token embeddings.
    
    Args:
        batch_size: Number of documents to process in each batch
        dry_run: If True, only identify missing documents without processing
        
    Returns:
        Dict with execution results
    """
    logger.info("Starting ColBERT token embedding population")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return {'success': False, 'error': 'Environment validation failed'}
    
    try:
        # Initialize connections and encoder
        connection_manager = initialize_connections()
        iris_connector = connection_manager.get_connection()
        colbert_encoder = initialize_colbert_encoder()
        
        # Identify missing documents
        missing_docs = identify_missing_documents(iris_connector)
        
        if not missing_docs:
            logger.info("No missing documents found")
            return {'success': True, 'processed_docs': 0, 'missing_docs': 0}
        
        logger.info(f"Found {len(missing_docs)} documents needing embeddings")
        
        if dry_run:
            logger.info("Dry run mode - not processing documents")
            return {'success': True, 'processed_docs': 0, 'missing_docs': len(missing_docs)}
        
        # Process documents in batches
        processed_docs = 0
        errors = 0
        
        for i in range(0, len(missing_docs), batch_size):
            batch = missing_docs[i:i + batch_size]
            try:
                process_batch_embeddings(batch, iris_connector, colbert_encoder, batch_size)
                processed_docs += len(batch)
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                errors += 1
        
        # Verify completion
        verification_result = verify_completion(iris_connector)
        
        # Generate completion report
        generate_completion_report(processed_docs, errors, verification_result)
        
        # Close connections
        connection_manager.close_all_connections()
        
        return {
            'success': True,
            'processed_docs': processed_docs,
            'errors': errors,
            'verification': verification_result
        }
        
    except Exception as e:
        logger.error(f"Fatal error during embedding population: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Populate missing ColBERT token embeddings')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing documents')
    parser.add_argument('--dry-run', action='store_true', help='Only identify missing documents without processing')
    
    args = parser.parse_args()
    
    result = populate_missing_colbert_embeddings(batch_size=args.batch_size, dry_run=args.dry_run)
    
    if result['success']:
        logger.info("Script completed successfully")
        sys.exit(0)
    else:
        logger.error(f"Script failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()