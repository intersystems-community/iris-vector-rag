#!/usr/bin/env python3
"""
Source Documents Deletion Script

This script deletes specified document records from the RAG.SourceDocuments table 
based on a list of doc_ids. It includes a dry-run capability to simulate deletions
without actually modifying the database.

Usage:
    # Dry run deletion
    python scripts/delete_source_documents.py --doc-ids "PMC123,PMC456" --dry-run
    
    # Actual deletion
    python scripts/delete_source_documents.py --doc-ids "PMC123,PMC456"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager


def setup_logging() -> logging.Logger:
    """Set up standard logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_doc_ids(doc_ids_str: str) -> List[str]:
    """
    Parse comma-separated document IDs string into a list.
    
    Args:
        doc_ids_str: Comma-separated string of document IDs
        
    Returns:
        List of document ID strings
    """
    return [doc_id.strip() for doc_id in doc_ids_str.split(',') if doc_id.strip()]


def delete_source_documents(
    doc_ids: List[str], 
    connection_manager: ConnectionManager, 
    logger: logging.Logger,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Delete specified document IDs from RAG.SourceDocuments table.
    
    Args:
        doc_ids: List of document IDs to delete
        connection_manager: ConnectionManager instance for database operations
        logger: Logger instance for logging
        dry_run: If True, simulate deletion without actually modifying database
        
    Returns:
        Tuple of (successful_deletions, errors_count)
    """
    successful_deletions = 0
    errors_count = 0
    would_be_deleted = 0
    
    try:
        # Get database connection
        logger.info("Establishing database connection...")
        connection = connection_manager.get_connection("iris")
        cursor = connection.cursor()
        
        # Begin transaction (disable autocommit for transaction control)
        connection.autocommit = False
        logger.info("Transaction started")
        
        # Process each document ID
        for doc_id in doc_ids:
            try:
                if dry_run:
                    # For dry run, check if document exists without deleting
                    logger.info(f"[DRY RUN] Would delete document: {doc_id}")
                    check_query = "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id = ?"
                    cursor.execute(check_query, [doc_id])
                    result = cursor.fetchone()
                    if result and result[0] > 0:
                        would_be_deleted += 1
                        logger.info(f"[DRY RUN] Document {doc_id} exists and would be deleted")
                    else:
                        logger.warning(f"[DRY RUN] Document {doc_id} not found - would skip")
                else:
                    # Actual deletion
                    logger.info(f"Deleting document: {doc_id}")
                    delete_query = "DELETE FROM RAG.SourceDocuments WHERE doc_id = ?"
                    cursor.execute(delete_query, [doc_id])
                    
                    # Check affected row count
                    affected_rows = cursor.rowcount
                    if affected_rows > 0:
                        successful_deletions += 1
                        logger.info(f"Successfully deleted document {doc_id} ({affected_rows} row(s) affected)")
                    else:
                        logger.warning(f"Document {doc_id} not found for deletion")
                        
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                errors_count += 1
                # Continue processing other documents
                continue
        
        if not dry_run:
            if errors_count == 0:
                # Commit transaction if no errors occurred
                connection.commit()
                logger.info("Transaction committed successfully")
            else:
                # Rollback if there were errors
                connection.rollback()
                logger.warning("Transaction rolled back due to errors")
        else:
            # For dry run, always rollback (though no changes were made)
            connection.rollback()
            logger.info("Dry run completed - no changes made to database")
        
        cursor.close()
        
        if dry_run:
            return would_be_deleted, 0
        else:
            return successful_deletions, errors_count
            
    except Exception as e:
        logger.error(f"Error during deletion process: {e}")
        try:
            # Attempt to rollback on major error
            if 'connection' in locals():
                connection.rollback()
                logger.info("Transaction rolled back due to major error")
        except Exception as rollback_error:
            logger.error(f"Error during rollback: {rollback_error}")
        raise


def main():
    """Main function to run the document deletion script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Delete source documents from RAG.SourceDocuments table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run deletion
    python scripts/delete_source_documents.py --doc-ids "PMC123,PMC456" --dry-run
    
    # Actual deletion
    python scripts/delete_source_documents.py --doc-ids "PMC123,PMC456"
        """
    )
    parser.add_argument(
        '--doc-ids',
        required=True,
        help='Comma-separated string of document IDs to delete (e.g., "PMC123,PMC456,PMC789")'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate deletion without actually modifying the database'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting source documents deletion script")
    
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Parse document IDs
        doc_ids = parse_doc_ids(args.doc_ids)
        if not doc_ids:
            logger.error("No valid document IDs provided")
            sys.exit(1)
        
        logger.info(f"Parsed {len(doc_ids)} document IDs: {', '.join(doc_ids)}")
        
        if args.dry_run:
            logger.info("DRY RUN MODE: No actual deletions will be performed")
        else:
            logger.info("DELETION MODE: Documents will be permanently deleted")
        
        # Initialize configuration and connection managers
        logger.info("Initializing configuration manager...")
        config_manager = ConfigurationManager()
        
        logger.info("Initializing connection manager...")
        connection_manager = ConnectionManager(config_manager)
        
        # Delete documents
        successful_count, error_count = delete_source_documents(
            doc_ids, connection_manager, logger, dry_run=args.dry_run
        )
        
        # Report results
        if args.dry_run:
            print(f"\nDry run complete. Would delete {successful_count} document(s).")
            if error_count > 0:
                print(f"Encountered {error_count} error(s) during dry run.")
        else:
            print(f"\nDeletion process complete. Successfully deleted: {successful_count}. Not found/Failed to delete: {error_count}.")
        
        logger.info("Document deletion script completed successfully")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
    finally:
        # Clean up connections
        try:
            if 'connection_manager' in locals():
                connection_manager.close_all_connections()
                logger.info("Database connections closed")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


if __name__ == "__main__":
    main()