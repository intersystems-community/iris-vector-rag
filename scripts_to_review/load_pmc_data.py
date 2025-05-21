#!/usr/bin/env python
"""
Load PMC Data Script

This script processes PMC XML files and loads them into the IRIS database.
It serves as a command-line interface to the data loader module.
"""

import os
import sys

# Add project root to sys.path to allow importing 'data' and 'common' packages
# This is necessary because this script is in a subdirectory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
# 'os' and 'sys' were imported above for path manipulation
import time
from typing import Dict, Any

from data.loader import process_and_load_documents
from common.iris_connector import get_iris_connection
from common.db_init import initialize_database

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
        description="Process PMC XML files and load them into the IRIS database"
    )
    
    parser.add_argument(
        "--dir", 
        type=str, 
        default="data/pmc_oas_downloaded",
        help="Directory containing PMC XML files"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=1000, 
        help="Maximum number of documents to process"
    )
    
    parser.add_argument(
        "--batch", 
        type=int, 
        default=50, 
        help="Batch size for database inserts"
    )
    
    parser.add_argument(
        "--mock", 
        action="store_true", 
        help="Use mock database connection"
    )
    
    parser.add_argument(
        "--init-db", 
        action="store_true", 
        help="Initialize database schema before loading"
    )
    
    parser.add_argument(
        "--force-recreate", 
        action="store_true", 
        help="Force recreate database tables (use with --init-db)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Set logging level"
    )

    parser.add_argument(
        "--load-colbert",
        action="store_true",
        help="Generate and load ColBERT token embeddings into RAG.DocumentTokenEmbeddings"
    )
    
    return parser.parse_args()

def main() -> int:
    """Main function"""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger("load_pmc_data")
    logger.info(f"Starting PMC data loading process")
    logger.info(f"Processing directory: {args.dir}")
    logger.info(f"Document limit: {args.limit}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Using mock connection: {args.mock}")
    
    start_time = time.time()
    
    # Get connection
    connection = get_iris_connection(use_mock=args.mock)
    if not connection:
        logger.error("Failed to establish database connection")
        return 1

    # Get embedding functions
    # Standard sentence embedding function
    from common.utils import get_embedding_func
    embedding_function = None
    if not args.mock: # Only get real embedder if not mocking DB
        try:
            embedding_function = get_embedding_func() # Uses default model
        except Exception as e:
            logger.error(f"Failed to initialize sentence embedding function: {e}")
            logger.info("Proceeding without sentence embeddings.")

    # ColBERT document token encoder function (optional)
    colbert_doc_encoder_function = None
    if args.load_colbert and not args.mock:
        try:
            from common.utils import get_colbert_doc_encoder_func
            colbert_doc_encoder_function = get_colbert_doc_encoder_func() # Uses mock for now
            logger.info("ColBERT document encoder function loaded (mock).")
        except Exception as e:
            logger.error(f"Failed to initialize ColBERT document encoder function: {e}")
            logger.info("Proceeding without ColBERT token embeddings.")
    elif args.load_colbert:
        logger.info("ColBERT loading requested but using mock DB, so ColBERT encoder will not be used by loader.")


    try:
        # Initialize database if requested
        if args.init_db:
            logger.info("Initializing database schema")
            initialize_database(connection, force_recreate=args.force_recreate)
        
        # Process and load documents
        stats = process_and_load_documents(
            pmc_directory=args.dir,
            connection=connection,
            embedding_func=embedding_function, # Pass sentence embedder
            colbert_doc_encoder_func=colbert_doc_encoder_function, # Pass ColBERT doc encoder
            limit=args.limit,
            batch_size=args.batch
        )
        
        # Print results
        if stats["success"]:
            logger.info("\n=== Processing and Loading Results ===")
            logger.info(f"Processed {stats['processed_count']} documents from {stats['processed_directory']}")
            logger.info(f"Successfully loaded {stats['loaded_doc_count']} documents") # Changed to loaded_doc_count
            if stats.get('loaded_token_count', 0) > 0 : # Check if token count exists
                logger.info(f"Successfully loaded {stats['loaded_token_count']} ColBERT token embeddings")
            if stats['error_count'] > 0:
                logger.warning(f"Failed to load {stats['error_count']} documents (or batches containing them)")
            logger.info(f"Total time: {stats['duration_seconds']:.2f} seconds")
            logger.info(f"Loading rate: {stats['documents_per_second']:.2f} documents per second")
            
            # Print a summary table
            print("\n=== Summary ===")
            print(f"{'Metric':<30} {'Value':<20}")
            print(f"{'-'*30} {'-'*20}")
            print(f"{'Documents Processed':<30} {stats['processed_count']:<20}")
            print(f"{'Source Documents Loaded':<30} {stats.get('loaded_doc_count', 0):<20}") # Use .get for safety
            print(f"{'Token Embeddings Loaded':<30} {stats.get('loaded_token_count', 0):<20}") # Use .get for safety
            print(f"{'Load Errors (Batches)':<30} {stats['error_count']:<20}")
            print(f"{'Processing Time (s)':<30} {stats['duration_seconds']:.2f}")
            print(f"{'Documents per Second':<30} {stats['documents_per_second']:.2f}")
            
            return 0
        else:
            logger.error(f"Error: {stats['error']}")
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
