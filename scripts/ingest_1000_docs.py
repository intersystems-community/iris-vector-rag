#!/usr/bin/env python3
"""
Document Ingestion Script for RAG Templates

This script ingests at least 1000 documents into the IRIS database using the
RAG class from rag_templates.simple. It loads documents from a specified directory,
handles potential errors during ingestion, and logs the number of documents
successfully ingested.

Usage:
    uv run python scripts/ingest_1000_docs.py <directory_path> [--limit LIMIT]

Example:
    uv run python scripts/ingest_1000_docs.py data/sample_10_docs
    uv run python scripts/ingest_1000_docs.py /path/to/pmc/data --limit 2000
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_templates.simple import RAG
from data.pmc_processor import process_pmc_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingest_1000_docs.log')
    ]
)

logger = logging.getLogger(__name__)


def validate_directory(directory_path: str) -> Path:
    """
    Validate that the provided directory exists and contains files.
    
    Args:
        directory_path: Path to the directory containing documents
        
    Returns:
        Path object for the validated directory
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If directory is empty or invalid
    """
    path = Path(directory_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Check if directory contains any XML files (recursively)
    xml_files = list(path.rglob("*.xml"))
    if not xml_files:
        logger.warning(f"No XML files found in directory: {directory_path}")
    else:
        logger.info(f"Found {len(xml_files)} XML files in directory: {directory_path}")
    
    return path


def process_documents_from_directory(directory_path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Process documents from the specified directory using PMC processor.
    
    Args:
        directory_path: Path to directory containing PMC XML files
        limit: Maximum number of documents to process
        
    Returns:
        List of processed document dictionaries
    """
    logger.info(f"Processing documents from {directory_path} with limit {limit}")
    
    try:
        documents = []
        processed_count = 0
        
        for doc_metadata in process_pmc_files(str(directory_path), limit):
            # Convert PMC metadata to format expected by RAG class
            # Use pmc_id as the document ID to ensure it's preserved in the database
            pmc_id = doc_metadata.get('pmc_id', f'doc_{processed_count}')
            
            doc_content = {
                'page_content': doc_metadata.get('content', ''),
                'metadata': {
                    'pmc_id': pmc_id,
                    'title': doc_metadata.get('title', ''),
                    'abstract': doc_metadata.get('abstract', ''),
                    'authors': doc_metadata.get('authors', []),
                    'keywords': doc_metadata.get('keywords', []),
                    'source': 'PMC',
                    'file_path': doc_metadata.get('metadata', {}).get('file_path', ''),
                    'document_id': pmc_id  # Keep for backward compatibility
                },
                'id': pmc_id  # This is the key fix - set the Document.id field directly
            }
            
            documents.append(doc_content)
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} documents so far...")
        
        logger.info(f"Successfully processed {len(documents)} documents from directory")
        return documents
        
    except Exception as e:
        logger.error(f"Error processing documents from directory: {e}")
        raise


def ingest_documents_with_rag(documents: List[Dict[str, Any]]) -> int:
    """
    Ingest documents using the RAG class from rag_templates.simple.
    
    Args:
        documents: List of document dictionaries to ingest
        
    Returns:
        Number of documents successfully ingested
        
    Raises:
        Exception: If ingestion fails
    """
    logger.info(f"Starting ingestion of {len(documents)} documents using RAG class")
    
    try:
        # Initialize RAG instance
        rag = RAG()
        logger.info("RAG instance initialized successfully")
        
        # Get initial document count
        initial_count = rag.get_document_count()
        logger.info(f"Initial document count in knowledge base: {initial_count}")
        
        # Add documents to the RAG knowledge base
        start_time = time.time()
        rag.add_documents(documents)
        end_time = time.time()
        
        # Get final document count
        final_count = rag.get_document_count()
        ingested_count = final_count - initial_count
        
        duration = end_time - start_time
        logger.info(f"Successfully ingested {ingested_count} documents in {duration:.2f} seconds")
        logger.info(f"Total documents in knowledge base: {final_count}")
        
        if ingested_count >= 1000:
            logger.info(f"✅ Successfully ingested at least 1000 documents ({ingested_count} total)")
        else:
            logger.warning(f"⚠️  Only ingested {ingested_count} documents, which is less than 1000")
        
        return ingested_count
        
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise


def main(directory_path: str, limit: int = 1000) -> None:
    """
    Main function to orchestrate document ingestion process.
    
    Args:
        directory_path: Path to directory containing documents
        limit: Maximum number of documents to process
    """
    logger.info("=" * 60)
    logger.info("Starting Document Ingestion Script")
    logger.info("=" * 60)
    
    try:
        # Validate directory
        validated_path = validate_directory(directory_path)
        logger.info(f"Directory validated: {validated_path}")
        
        # Process documents from directory
        documents = process_documents_from_directory(validated_path, limit)
        
        if not documents:
            logger.error("No documents were processed. Exiting.")
            sys.exit(1)
        
        # Ingest documents using RAG
        ingested_count = ingest_documents_with_rag(documents)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("Document Ingestion Complete")
        logger.info(f"Total documents processed: {len(documents)}")
        logger.info(f"Total documents ingested: {ingested_count}")
        logger.info("=" * 60)
        
        if ingested_count < 1000:
            logger.warning("Warning: Less than 1000 documents were ingested")
            sys.exit(1)
        
    except FileNotFoundError as e:
        logger.error(f"Directory error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest at least 1000 documents into IRIS database using RAG templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/sample_10_docs
  %(prog)s /path/to/pmc/data --limit 2000
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory path containing PMC XML files to ingest'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Maximum number of documents to process (default: 1000)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run main function
    main(args.directory, args.limit)