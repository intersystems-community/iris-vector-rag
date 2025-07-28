#!/usr/bin/env python3
"""
Fixed ColBERT Token Embedding Population Script.

This script populates missing ColBERT token embeddings using the proper
ColBERT interface with 768D embeddings instead of the incorrect 384D.

Key fixes:
- Uses proper ColBERT interface for 768D token embeddings
- Integrates with TokenEmbeddingService for centralized management
- Uses SchemaManager for consistent database operations
- Proper error handling and logging
- Batch processing for efficiency
"""

import logging
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.services.token_embedding_service import TokenEmbeddingService
from common.iris_connection_manager import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to populate ColBERT token embeddings."""
    parser = argparse.ArgumentParser(
        description="Populate missing ColBERT token embeddings with proper 768D dimensions"
    )
    parser.add_argument(
        "--doc-ids",
        nargs="*",
        help="Specific document IDs to process (if not provided, processes all missing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing documents (default: 50)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting ColBERT token embedding population")
        logger.info(f"Using configuration: {args.config}")
        
        # Initialize configuration manager
        config_manager = ConfigurationManager(args.config)
        
        # Override batch size if provided
        if args.batch_size != 50:
            config_manager.config["colbert"] = config_manager.config.get("colbert", {})
            config_manager.config["colbert"]["batch_size"] = args.batch_size
        
        # Create connection manager wrapper
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda: get_iris_connection()
        })()
        
        # Initialize token embedding service
        token_service = TokenEmbeddingService(config_manager, connection_manager)
        
        # Get current statistics
        logger.info("Checking current token embedding status...")
        stats = token_service.get_token_embedding_stats()
        
        logger.info(f"Current status:")
        logger.info(f"  Total documents: {stats['total_documents']}")
        logger.info(f"  Documents with token embeddings: {stats['documents_with_token_embeddings']}")
        logger.info(f"  Documents missing token embeddings: {stats['documents_missing_token_embeddings']}")
        logger.info(f"  Total token embeddings: {stats['total_token_embeddings']}")
        logger.info(f"  Token dimension: {stats['token_dimension']}D")
        logger.info(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        
        if stats['documents_missing_token_embeddings'] == 0:
            logger.info("All documents already have token embeddings. Nothing to do.")
            return 0
        
        if args.dry_run:
            logger.info(f"DRY RUN: Would process {stats['documents_missing_token_embeddings']} documents")
            if args.doc_ids:
                logger.info(f"DRY RUN: Would filter to specific doc IDs: {args.doc_ids}")
            return 0
        
        # Process token embeddings
        logger.info(f"Processing {stats['documents_missing_token_embeddings']} documents...")
        
        processing_stats = token_service.ensure_token_embeddings_exist(args.doc_ids)
        
        # Report results
        logger.info("Token embedding population completed!")
        logger.info(f"Results:")
        logger.info(f"  Documents processed: {processing_stats.documents_processed}")
        logger.info(f"  Token embeddings generated: {processing_stats.tokens_generated}")
        logger.info(f"  Processing time: {processing_stats.processing_time:.2f} seconds")
        logger.info(f"  Errors: {processing_stats.errors}")
        
        if processing_stats.errors > 0:
            logger.warning(f"Completed with {processing_stats.errors} errors")
            return 1
        
        # Get final statistics
        final_stats = token_service.get_token_embedding_stats()
        logger.info(f"Final coverage: {final_stats['coverage_percentage']:.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Token embedding population failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())