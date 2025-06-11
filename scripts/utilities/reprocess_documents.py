#!/usr/bin/env python3
"""
Document Reprocessing Script

This script re-processes specified XML documents by their doc_ids, updates their entries 
in the RAG.SourceDocuments table, and logs the outcomes. This is intended to fix 
documents that may have incorrect text_content (e.g., '-1').

Usage:
    python scripts/reprocess_documents.py --doc-ids "PMC123,PMC456,PMC789"
    python scripts/reprocess_documents.py --doc-ids "PMC123" --xml-dir "path/to/xmls"
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from data.pmc_processor import extract_pmc_metadata


class PmcProcessor:
    """
    PMC Document Processor for reprocessing individual documents.
    
    This class provides functionality to reprocess individual PMC XML files
    and update their corresponding database records.
    """
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager):
        """
        Initialize the PmcProcessor.
        
        Args:
            connection_manager: ConnectionManager instance for database operations
            config_manager: ConfigurationManager instance for configuration access
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def process_single_file(self, file_path: str, doc_id_override: Optional[str] = None) -> bool:
        """
        Process a single PMC XML file and update the database record.
        
        Args:
            file_path: Path to the XML file to process
            doc_id_override: Optional doc_id to use instead of extracting from filename
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Extract metadata from the XML file
            self.logger.info(f"Extracting metadata from: {file_path}")
            metadata = extract_pmc_metadata(file_path)
            
            # Use override doc_id if provided
            if doc_id_override:
                metadata['doc_id'] = doc_id_override
                self.logger.info(f"Using doc_id override: {doc_id_override}")
            
            # Check if extraction was successful
            if metadata.get("title") == "Error" and "Failed to process" in metadata.get("content", ""):
                self.logger.error(f"Failed to extract valid metadata from {file_path}")
                return False
            
            # Update the database record
            return self._update_database_record(metadata)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def _update_database_record(self, metadata: Dict[str, Any]) -> bool:
        """
        Update the database record with the extracted metadata.
        
        Args:
            metadata: Dictionary containing the extracted metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            connection = self.connection_manager.get_connection("iris")
            cursor = connection.cursor()
            
            # Prepare the update SQL
            update_sql = """
            UPDATE RAG.SourceDocuments 
            SET title = ?, 
                text_content = ?, 
                authors = ?, 
                keywords = ?
            WHERE doc_id = ?
            """
            
            # Prepare parameters
            doc_id = metadata.get('doc_id')
            title = metadata.get('title', 'Unknown Title')
            text_content = metadata.get('content', '')
            authors = json.dumps(metadata.get('authors', []))
            keywords = json.dumps(metadata.get('keywords', []))
            
            params = (title, text_content, authors, keywords, doc_id)
            
            self.logger.debug(f"Executing update for doc_id: {doc_id}")
            self.logger.debug(f"Title: {title[:50]}...")
            self.logger.debug(f"Content length: {len(text_content)} characters")
            
            # Execute the update
            cursor.execute(update_sql, params)
            
            # Check if any rows were affected
            if cursor.rowcount == 0:
                self.logger.warning(f"No rows updated for doc_id: {doc_id}. Document may not exist in database.")
                return False
            
            # Commit the transaction
            connection.commit()
            self.logger.info(f"Successfully updated doc_id: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database update failed for doc_id {metadata.get('doc_id')}: {e}")
            try:
                connection.rollback()
            except:
                pass
            return False


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reprocess specified XML documents by their doc_ids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reprocess specific documents
  python scripts/reprocess_documents.py --doc-ids "PMC11586160,PMC11587494"
  
  # Specify custom XML directory
  python scripts/reprocess_documents.py --doc-ids "PMC123" --xml-dir "path/to/my/xmls"
        """
    )
    
    parser.add_argument(
        '--doc-ids',
        required=True,
        help='Comma-separated string of document IDs to reprocess (e.g., "PMC123,PMC456,PMC789")'
    )
    
    parser.add_argument(
        '--xml-dir',
        help='Directory containing the XML source files. If not provided, uses configuration.'
    )
    
    return parser.parse_args()


def get_xml_source_directory(config_manager: ConfigurationManager, xml_dir_arg: Optional[str]) -> str:
    """
    Determine the XML source directory from arguments or configuration.
    
    Args:
        config_manager: ConfigurationManager instance
        xml_dir_arg: XML directory from command line arguments
        
    Returns:
        Path to the XML source directory
        
    Raises:
        ValueError: If no XML directory can be determined
    """
    if xml_dir_arg:
        return xml_dir_arg
    
    # Try to get from configuration
    xml_dir = config_manager.get('data_paths:xml_input_dir')
    if xml_dir:
        return xml_dir
    
    # Fallback to common locations
    common_paths = [
        'data/pmc_oas_downloaded',
        'data/pmc_100k_downloaded',
        'data/sample_10_docs'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise ValueError(
        "No XML directory specified and none found in configuration. "
        "Please specify --xml-dir or configure data_paths.xml_input_dir"
    )


def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting document reprocessing script")
    
    # Parse arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize configuration and connection managers
        logger.info("Initializing configuration and connection managers")
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Initialize PMC processor
        pmc_processor = PmcProcessor(connection_manager, config_manager)
        
        # Parse doc_ids
        doc_ids = [doc_id.strip() for doc_id in args.doc_ids.split(',') if doc_id.strip()]
        logger.info(f"Processing {len(doc_ids)} document IDs: {doc_ids}")
        
        # Determine XML source directory
        xml_source_dir = get_xml_source_directory(config_manager, args.xml_dir)
        logger.info(f"Using XML source directory: {xml_source_dir}")
        
        # Process each document
        successful_count = 0
        failed_count = 0
        
        for doc_id in doc_ids:
            logger.info(f"Processing doc_id: {doc_id}")
            
            # Construct expected XML file path
            xml_file_path = os.path.join(xml_source_dir, f"{doc_id}.xml")
            logger.info(f"Looking for XML file: {xml_file_path}")
            
            # Check if file exists
            if not os.path.exists(xml_file_path):
                logger.warning(f"Source XML file not found for doc_id: {doc_id} at {xml_file_path}")
                failed_count += 1
                continue
            
            try:
                # Process the file
                success = pmc_processor.process_single_file(xml_file_path, doc_id_override=doc_id)
                
                if success:
                    logger.info(f"Successfully reprocessed doc_id: {doc_id}")
                    successful_count += 1
                else:
                    logger.error(f"Failed to reprocess doc_id: {doc_id}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Exception while processing doc_id {doc_id}: {e}")
                failed_count += 1
        
        # Print summary report
        logger.info("=" * 60)
        logger.info("REPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total documents requested: {len(doc_ids)}")
        logger.info(f"Successfully reprocessed: {successful_count}")
        logger.info(f"Failed/Skipped: {failed_count}")
        logger.info("=" * 60)
        
        if successful_count > 0:
            logger.info("Reprocessing completed with some successes.")
        else:
            logger.warning("No documents were successfully reprocessed.")
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)
    
    finally:
        # Clean up connections
        try:
            connection_manager.close_all_connections()
        except:
            pass


if __name__ == "__main__":
    main()