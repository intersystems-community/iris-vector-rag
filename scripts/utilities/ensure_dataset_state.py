#!/usr/bin/env python3
"""
Dataset State Management Script - Full Functionality Version

This script manages dataset state by identifying and healing missing data states,
specifically for token embeddings and other pipeline requirements.

Usage:
    # Validate token embeddings state for 1000 documents
    python scripts/ensure_dataset_state.py --target-state token-embeddings-ready --doc-count 1000 --validate-only
    
    # Auto-fix missing token embeddings for 1000 documents
    python scripts/ensure_dataset_state.py --target-state token-embeddings-ready --doc-count 1000 --auto-fix
    
    # Force regenerate all token embeddings for 1000 documents
    python scripts/ensure_dataset_state.py --target-state token-embeddings-ready --doc-count 1000 --auto-fix --force-regenerate
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.validation.orchestrator import SetupOrchestrator


def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage dataset state by identifying and healing missing data states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target States:
  token-embeddings-ready    Ensure token embeddings exist for ColBERT pipeline
  document-embeddings-ready Ensure document embeddings exist for basic pipelines
  chunks-ready             Ensure document chunks and embeddings exist
  
Examples:
  # Validate token embeddings state for 1000 documents
  python scripts/ensure_dataset_state.py --target-state token-embeddings-ready --doc-count 1000 --validate-only
  
  # Auto-fix missing token embeddings for 1000 documents
  python scripts/ensure_dataset_state.py --target-state token-embeddings-ready --doc-count 1000 --auto-fix
  
  # Force regenerate all token embeddings for 1000 documents
  python scripts/ensure_dataset_state.py --target-state token-embeddings-ready --doc-count 1000 --auto-fix --force-regenerate
  
  # Validate document embeddings for 5000 documents
  python scripts/ensure_dataset_state.py --target-state document-embeddings-ready --doc-count 5000 --validate-only
        """
    )
    
    parser.add_argument(
        '--target-state',
        type=str,
        required=True,
        choices=['token-embeddings-ready', 'document-embeddings-ready', 'chunks-ready'],
        help="Target state to ensure for the dataset"
    )
    
    parser.add_argument(
        '--doc-count',
        type=int,
        default=1000,
        help="Target number of documents to process (default: 1000)"
    )
    
    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help="Automatically fix missing data states"
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help="Only validate current state, don't perform any fixes"
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help="Force regeneration of all data for target documents (requires --auto-fix)"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help="Output format for results (default: text)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate argument combinations."""
    if args.force_regenerate and not args.auto_fix:
        raise ValueError("--force-regenerate requires --auto-fix")
    
    if args.auto_fix and args.validate_only:
        raise ValueError("--auto-fix and --validate-only are mutually exclusive")
    
    if args.doc_count <= 0:
        raise ValueError("Document count must be positive")


def validate_token_embeddings_state(orchestrator: SetupOrchestrator, doc_count: int) -> Dict[str, Any]:
    """Validate token embeddings state for specified document count."""
    connection = orchestrator.connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Get target document set
        target_docs_with_content = orchestrator._get_target_document_set(cursor, doc_count)
        target_doc_ids = [item['doc_id'] for item in target_docs_with_content]
        
        if not target_doc_ids:
            return {
                "status": "no_documents",
                "target_doc_count": doc_count,
                "found_doc_count": 0,
                "missing_embeddings_count": 0,
                "missing_doc_ids": []
            }
        
        # Identify documents missing token embeddings
        missing_doc_ids = orchestrator._identify_missing_token_embeddings(cursor, target_doc_ids)
        
        return {
            "status": "complete" if len(missing_doc_ids) == 0 else "missing_embeddings",
            "target_doc_count": doc_count,
            "found_doc_count": len(target_doc_ids),
            "missing_embeddings_count": len(missing_doc_ids),
            "missing_doc_ids": missing_doc_ids
        }
        
    finally:
        cursor.close()
        connection.close()


def validate_document_embeddings_state(orchestrator: SetupOrchestrator, doc_count: int) -> Dict[str, Any]:
    """Validate document embeddings state for specified document count."""
    connection = orchestrator.connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Check for documents without embeddings in the target set
        cursor.execute("""
            SELECT TOP ? doc_id FROM RAG.SourceDocuments
            WHERE embedding IS NULL
            ORDER BY doc_id
        """, [doc_count])
        
        missing_doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Get total count of target documents
        cursor.execute("SELECT TOP ? doc_id FROM RAG.SourceDocuments ORDER BY doc_id", [doc_count])
        target_doc_ids = [row[0] for row in cursor.fetchall()]
        
        return {
            "status": "complete" if len(missing_doc_ids) == 0 else "missing_embeddings",
            "target_doc_count": doc_count,
            "found_doc_count": len(target_doc_ids),
            "missing_embeddings_count": len(missing_doc_ids),
            "missing_doc_ids": missing_doc_ids
        }
        
    finally:
        cursor.close()
        connection.close()


def heal_token_embeddings_state(orchestrator: SetupOrchestrator, doc_count: int, force_regenerate: bool) -> Dict[str, Any]:
    """Heal token embeddings state using the orchestrator."""
    return orchestrator.heal_token_embeddings(
        target_doc_count=doc_count,
        force_regenerate=force_regenerate
    )


def heal_document_embeddings_state(orchestrator: SetupOrchestrator, doc_count: int) -> Dict[str, Any]:
    """Heal document embeddings state using the orchestrator."""
    # Use the orchestrator's document embedding generation
    orchestrator._ensure_document_embeddings()
    
    # Validate the result
    return validate_document_embeddings_state(orchestrator, doc_count)


def format_output(results: Dict[str, Any], output_format: str, logger) -> None:
    """Format and display results."""
    if output_format == 'json':
        print(json.dumps(results, indent=2))
    else:
        # Text format
        logger.info("=== Dataset State Results ===")
        logger.info(f"Status: {results.get('status', 'unknown')}")
        logger.info(f"Target document count: {results.get('target_doc_count', 'N/A')}")
        logger.info(f"Found document count: {results.get('found_doc_count', 'N/A')}")
        
        if 'missing_embeddings_count' in results:
            logger.info(f"Missing embeddings count: {results['missing_embeddings_count']}")
        
        if 'processed' in results:
            logger.info(f"Processed: {results['processed']}")
        
        if 'failed' in results:
            logger.info(f"Failed: {results['failed']}")
        
        if 'still_missing' in results:
            logger.info(f"Still missing: {results['still_missing']}")
        
        if 'duration' in results:
            logger.info(f"Duration: {results['duration']:.1f}s")
        
        if 'skipped_doc_ids_bad_content' in results and results['skipped_doc_ids_bad_content']:
            logger.info(f"Skipped due to bad content: {len(results['skipped_doc_ids_bad_content'])}")
        
        # Show sample missing doc_ids if validation only
        if 'missing_doc_ids' in results and results['missing_doc_ids']:
            sample_size = min(10, len(results['missing_doc_ids']))
            logger.info(f"Sample missing doc_ids ({sample_size}/{len(results['missing_doc_ids'])}): {results['missing_doc_ids'][:sample_size]}")


def main():
    """Main script execution."""
    # Load environment variables
    load_dotenv()
    
    # Parse and validate arguments
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting dataset state management script")
    logger.info(f"Target state: {args.target_state}")
    logger.info(f"Document count: {args.doc_count}")
    logger.info(f"Auto-fix: {args.auto_fix}")
    logger.info(f"Validate only: {args.validate_only}")
    logger.info(f"Force regenerate: {args.force_regenerate}")
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        setup_orchestrator = SetupOrchestrator(connection_manager, config_manager)
        
        logger.info("Components initialized successfully")
        
        # Execute based on target state and mode
        if args.target_state == "token-embeddings-ready":
            if args.validate_only:
                logger.info("Validating token embeddings state...")
                results = validate_token_embeddings_state(setup_orchestrator, args.doc_count)
            else:
                logger.info("Healing token embeddings state...")
                results = heal_token_embeddings_state(setup_orchestrator, args.doc_count, args.force_regenerate)
                
        elif args.target_state == "document-embeddings-ready":
            if args.validate_only:
                logger.info("Validating document embeddings state...")
                results = validate_document_embeddings_state(setup_orchestrator, args.doc_count)
            else:
                logger.info("Healing document embeddings state...")
                results = heal_document_embeddings_state(setup_orchestrator, args.doc_count)
                
        elif args.target_state == "chunks-ready":
            logger.error("Chunks state management not yet implemented")
            sys.exit(1)
        
        # Format and display results
        format_output(results, args.output_format, logger)
        
        # Set exit code based on results
        if results.get('status') == 'complete':
            logger.info("Dataset state management completed successfully")
            sys.exit(0)
        elif results.get('status') == 'error':
            logger.error("Dataset state management failed")
            sys.exit(1)
        else:
            logger.warning("Dataset state management completed with issues")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error during dataset state management: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()