#!/usr/bin/env python
"""
Generate Embeddings Script

This script generates embeddings for documents in the IRIS database.
It supports both document-level embeddings for basic RAG and 
token-level embeddings for ColBERT.
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

# Import our modules
from common.iris_connector import get_iris_connection
from common.utils import Document
from common.embedding_utils import (
    generate_document_embeddings,
    generate_token_embeddings,
    get_embedding_model,
    get_colbert_model,
    create_tables_if_needed
)

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the script"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for documents in IRIS"
    )
    
    parser.add_argument(
        "--token-level", 
        action="store_true", 
        help="Generate token-level embeddings for ColBERT"
    )
    
    parser.add_argument(
        "--doc-level", 
        action="store_true", 
        help="Generate document-level embeddings for Basic RAG"
    )
    
    parser.add_argument(
        "--mock", 
        action="store_true", 
        help="Use mock database connection"
    )
    
    parser.add_argument(
        "--batch", 
        type=int, 
        default=32, 
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Maximum number of documents to process"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="sentence-transformers/all-MiniLM-L6-v2", 
        help="Model to use for document embeddings"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Set logging level"
    )
    
    return parser.parse_args()

def main() -> int:
    """Main function"""
    args = parse_args()
    
    # Default to document-level if neither is specified
    if not args.token_level and not args.doc_level:
        args.doc_level = True
        
    setup_logging(args.log_level)
    
    logger.info(f"Starting embedding generation process")
    if args.doc_level:
        logger.info(f"Will generate document-level embeddings")
    if args.token_level:
        logger.info(f"Will generate token-level embeddings")
    logger.info(f"Using mock connection: {args.mock}")
    
    start_time = time.time()
    
    # Get connection
    connection = get_iris_connection(use_mock=args.mock)
    if not connection:
        logger.error("Failed to establish database connection")
        return 1
    
    try:
        # Create tables if needed
        create_tables_if_needed(connection)
        
        stats = []
        
        # Generate document-level embeddings if requested
        if args.doc_level:
            logger.info("Generating document-level embeddings")
            embedding_model = get_embedding_model(args.model)
            doc_stats = generate_document_embeddings(
                connection,
                embedding_model,
                batch_size=args.batch,
                limit=args.limit
            )
            stats.append(doc_stats)
            
        # Generate token-level embeddings if requested
        if args.token_level:
            logger.info("Generating token-level embeddings")
            token_encoder = get_colbert_model()
            token_stats = generate_token_embeddings(
                connection,
                token_encoder,
                batch_size=max(1, args.batch // 3),  # Token encoding is slower, use smaller batch
                limit=args.limit
            )
            stats.append(token_stats)
            
        # Print results
        total_duration = time.time() - start_time
        print("\n=== Embedding Generation Results ===")
        
        for stat in stats:
            if stat["type"] == "document_embeddings":
                print(f"\nDocument-level embeddings:")
                print(f"  Processed {stat['processed_count']}/{stat['total_documents']} documents")
                print(f"  Errors: {stat['error_count']}")
                print(f"  Duration: {stat['duration_seconds']:.2f} seconds")
                print(f"  Rate: {stat['documents_per_second']:.2f} documents per second")
            elif stat["type"] == "token_embeddings":
                print(f"\nToken-level embeddings:")
                print(f"  Processed {stat['processed_count']}/{stat['total_documents']} documents")
                print(f"  Generated embeddings for {stat['tokens_count']} tokens")
                print(f"  Errors: {stat['error_count']}")
                print(f"  Duration: {stat['duration_seconds']:.2f} seconds")
                print(f"  Rate: {stat['documents_per_second']:.2f} documents per second")
                print(f"  Token rate: {stat['tokens_per_second']:.2f} tokens per second")
                
        print(f"\nTotal duration: {total_duration:.2f} seconds")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        return 1
        
    finally:
        # Close connection
        try:
            connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")

if __name__ == "__main__":
    sys.exit(main())
