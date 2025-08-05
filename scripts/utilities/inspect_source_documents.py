#!/usr/bin/env python3
"""
Source Documents Inspection Script

This script queries the RAG.SourceDocuments table for specified doc_ids and prints
relevant details, including doc_id, title, text_content, and any available source/file
path information. This helps diagnose why their text_content might be '-1' and why
their source XMLs were not found by scripts/reprocess_documents.py.

Usage:
    python scripts/inspect_source_documents.py --doc-ids "PMC11586160,PMC11587494"
    python scripts/inspect_source_documents.py --doc-ids "PMC123,PMC456,PMC789"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

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


def build_query(doc_ids: List[str]) -> tuple[str, List[str]]:
    """
    Build SQL query and parameters for fetching document details.
    
    Args:
        doc_ids: List of document IDs to query
        
    Returns:
        Tuple of (SQL query string, list of parameters)
    """
    # Create placeholders for parameterized query
    placeholders = ','.join(['?' for _ in doc_ids])
    
    # Build the query - start with basic columns that should always exist
    query = f"""
    SELECT doc_id, title, text_content
    FROM RAG.SourceDocuments
    WHERE doc_id IN ({placeholders})
    ORDER BY doc_id
    """
    
    return query, doc_ids


def build_extended_query(doc_ids: List[str]) -> tuple[str, List[str]]:
    """
    Build extended SQL query that attempts to fetch additional columns.
    This is a fallback that tries to get more information if available.
    
    Args:
        doc_ids: List of document IDs to query
        
    Returns:
        Tuple of (SQL query string, list of parameters)
    """
    # Create placeholders for parameterized query
    placeholders = ','.join(['?' for _ in doc_ids])
    
    # Try to get additional columns that might exist
    query = f"""
    SELECT doc_id, title, text_content, file_path, source_url, ingestion_date
    FROM RAG.SourceDocuments
    WHERE doc_id IN ({placeholders})
    ORDER BY doc_id
    """
    
    return query, doc_ids


def convert_stream_to_string(value: Any) -> str:
    """
    Convert IRISInputStream or other stream objects to Python strings.
    
    Args:
        value: The value from database (could be string, stream, or other type)
        
    Returns:
        String representation of the value
    """
    if value is None:
        return "NULL"
    
    # Check if it's a stream-like object (IRISInputStream, CLOB, etc.)
    if hasattr(value, 'read') and callable(getattr(value, 'read')):
        try:
            # Read the entire stream
            stream_content = value.read()
            # Decode if it's bytes
            if isinstance(stream_content, bytes):
                return stream_content.decode('utf-8', errors='replace')
            else:
                return str(stream_content)
        except Exception as e:
            return f"[Error Reading Stream: {e}]"
    
    # For non-stream objects, convert to string
    return str(value)


def format_text_content(text_content: Any, max_length: int = 200) -> str:
    """
    Format text content for display, handling CLOBs and long text.
    
    Args:
        text_content: The text content value from database
        max_length: Maximum length to display before truncating
        
    Returns:
        Formatted text content string
    """
    if text_content is None:
        return "NULL"
    
    # Convert stream to string first
    content = convert_stream_to_string(text_content)
    
    # Handle error cases from stream conversion
    if content.startswith("[Error Reading Stream"):
        return content
    
    # Handle special case of '-1' content
    if content == '-1':
        return "'-1' (indicates processing error)"
    
    # Truncate if too long
    if len(content) > max_length:
        return content[:100] + "..."
    
    return content


def process_row_data(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process row data to convert any IRISInputStream objects to strings.
    
    Args:
        row_data: Dictionary containing raw row data from database
        
    Returns:
        Dictionary with all stream objects converted to strings
    """
    processed_data = {}
    
    for key, value in row_data.items():
        processed_data[key] = convert_stream_to_string(value)
    
    return processed_data


def print_document_details(doc_data: Dict[str, Any]) -> None:
    """
    Print document details in a readable format.
    
    Args:
        doc_data: Dictionary containing processed document data (all strings)
    """
    print("=" * 60)
    print(f"Doc ID: {doc_data.get('doc_id', 'N/A')}")
    
    # Handle title - now it's already a string
    title = doc_data.get('title', 'N/A')
    if title == 'NULL':
        print("Title: N/A")
    else:
        print(f"Title: {title}")
    
    # Format text content - now it's already a string
    text_content = doc_data.get('text_content', 'N/A')
    if text_content == 'NULL':
        text_display = "N/A"
    elif text_content == '-1':
        text_display = "'-1' (indicates processing error)"
    elif len(text_content) > 200:
        text_display = text_content[:200] + "..."
    else:
        text_display = text_content
    
    print(f"Text Content: {text_display}")
    
    # Handle file path - now it's already a string
    file_path = doc_data.get('file_path', 'NULL')
    if file_path == 'NULL':
        print("File Path: N/A")
    else:
        print(f"File Path: {file_path}")
    
    # Handle source URL if available - now it's already a string
    source_url = doc_data.get('source_url')
    if source_url is not None and source_url != 'NULL':
        print(f"Source URL: {source_url}")
    
    # Handle ingestion date if available - now it's already a string
    ingestion_date = doc_data.get('ingestion_date')
    if ingestion_date is not None and ingestion_date != 'NULL':
        print(f"Ingestion Date: {ingestion_date}")
    
    print("=" * 60)


def inspect_source_documents(doc_ids: List[str], connection_manager: ConnectionManager, logger: logging.Logger) -> None:
    """
    Query and display details for specified document IDs.
    
    Args:
        doc_ids: List of document IDs to inspect
        connection_manager: ConnectionManager instance for database operations
        logger: Logger instance for logging
    """
    try:
        # Get database connection
        logger.info("Establishing database connection...")
        connection = connection_manager.get_connection("iris")
        cursor = connection.cursor()
        
        # Try extended query first, fall back to basic query if it fails
        query, params = build_extended_query(doc_ids)
        logger.info(f"Querying for {len(doc_ids)} document(s): {', '.join(doc_ids)}")
        
        try:
            cursor.execute(query, params)
            columns = [desc[0].lower() for desc in cursor.description]
            results = cursor.fetchall()
            logger.info("Extended query successful")
        except Exception as e:
            logger.warning(f"Extended query failed ({e}), trying basic query...")
            # Fall back to basic query
            query, params = build_query(doc_ids)
            cursor.execute(query, params)
            columns = [desc[0].lower() for desc in cursor.description]
            results = cursor.fetchall()
            logger.info("Basic query successful")
        
        logger.info(f"Found {len(results)} document(s)")
        
        if not results:
            print("\nNo documents found for the specified doc_ids.")
            print("This could indicate:")
            print("- The doc_ids don't exist in the database")
            print("- There's a typo in the doc_ids")
            print("- The documents haven't been ingested yet")
            return
        
        # Display results
        print(f"\nDocument Details ({len(results)} found):")
        print()
        
        for row in results:
            # Convert row to dictionary
            raw_doc_data = dict(zip(columns, row))
            # Process the data to convert any stream objects to strings
            doc_data = process_row_data(raw_doc_data)
            print_document_details(doc_data)
        
        # Check for missing documents
        found_doc_ids = {row[0] for row in results}  # Assuming doc_id is first column
        missing_doc_ids = set(doc_ids) - found_doc_ids
        
        if missing_doc_ids:
            print(f"\nMissing Documents ({len(missing_doc_ids)}):")
            for missing_id in sorted(missing_doc_ids):
                print(f"- {missing_id}")
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        raise


def main():
    """Main function to run the document inspection script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Inspect source documents in RAG.SourceDocuments table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/inspect_source_documents.py --doc-ids "PMC11586160,PMC11587494"
    python scripts/inspect_source_documents.py --doc-ids "PMC123"
        """
    )
    parser.add_argument(
        '--doc-ids',
        required=True,
        help='Comma-separated string of document IDs to inspect (e.g., "PMC123,PMC456,PMC789")'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting source documents inspection script")
    
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
        
        # Initialize configuration and connection managers
        logger.info("Initializing configuration manager...")
        config_manager = ConfigurationManager()
        
        logger.info("Initializing connection manager...")
        connection_manager = ConnectionManager(config_manager)
        
        # Inspect documents
        inspect_source_documents(doc_ids, connection_manager, logger)
        
        logger.info("Document inspection completed successfully")
        
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