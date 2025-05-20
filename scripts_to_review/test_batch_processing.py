#!/usr/bin/env python
"""
Simple test script to verify batch processing implementation for large document sets.
"""

import logging
import time
import sys
from tests.utils import process_pmc_files_in_batches, load_pmc_documents
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_process_pmc_files_in_batches():
    """Test the batch processing function with a small document count."""
    pmc_dir = "data/pmc_oas_downloaded"
    document_count = 3
    batch_size = 2
    
    logger.info(f"Testing batch processing with {document_count} documents in batches of {batch_size}")
    
    # Verify documents exist
    xml_files = list(Path(pmc_dir).glob("**/*.xml"))
    available_count = len(xml_files)
    
    if available_count < document_count:
        logger.warning(f"Not enough documents: found {available_count}, needed {document_count}")
        document_count = available_count
    
    logger.info(f"Found {available_count} PMC XML files, will process {document_count}")
    
    # Process in batches
    batch_count = 0
    doc_count = 0
    
    start_time = time.time()
    
    for batch in process_pmc_files_in_batches(pmc_dir, document_count, batch_size):
        batch_count += 1
        batch_size = len(batch)
        doc_count += batch_size
        
        logger.info(f"Batch {batch_count}: Retrieved {batch_size} documents")
        
        # Print information about the first document in each batch
        if batch:
            first_doc = batch[0]
            logger.info(f"  First document: ID={first_doc.get('pmc_id')}, Title={first_doc.get('title')[:30]}...")
    
    elapsed = time.time() - start_time
    logger.info(f"Processed {doc_count} documents in {batch_count} batches ({elapsed:.2f} seconds)")
    
    return doc_count, batch_count

def test_mock_db_load():
    """Test the document loading with a mock database connection."""
    
    class MockCursor:
        def __init__(self):
            self.executed_queries = []
            self.params = []
        
        def execute(self, query, params=None):
            self.executed_queries.append(query)
            self.params.append(params)
    
    class MockConnection:
        def __init__(self):
            self.cursor_obj = MockCursor()
        
        def cursor(self):
            return self
        
        def __enter__(self):
            return self.cursor_obj
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    mock_conn = MockConnection()
    
    logger.info("Testing document loading with mock database")
    
    # Attempt to load documents
    try:
        document_count = 2
        batch_size = 1
        count = load_pmc_documents(
            connection=mock_conn,
            limit=document_count,
            batch_size=batch_size,
            show_progress=True
        )
        
        # Check if the mock cursor was called correctly
        cursor = mock_conn.cursor_obj
        
        if cursor.executed_queries:
            logger.info(f"Database was called with {len(cursor.executed_queries)} queries")
            # Check for CREATE TABLE query
            create_queries = [q for q in cursor.executed_queries if "CREATE TABLE" in q]
            logger.info(f"Found {len(create_queries)} CREATE TABLE queries")
            
            # Check for INSERT queries
            insert_queries = [q for q in cursor.executed_queries if "INSERT INTO" in q]
            logger.info(f"Found {len(insert_queries)} INSERT queries")
            
            return len(insert_queries)
        else:
            logger.error("No queries were executed on the mock database")
            return 0
    
    except Exception as e:
        logger.error(f"Error during mock database test: {e}")
        return 0

if __name__ == "__main__":
    logger.info("Running batch processing tests")
    
    # Test batch processing
    doc_count, batch_count = test_process_pmc_files_in_batches()
    
    # Test mock database loading
    insert_count = test_mock_db_load()
    
    # Print summary
    logger.info("Test Summary:")
    logger.info(f"  Documents processed in batches: {doc_count}")
    logger.info(f"  Number of batches: {batch_count}")
    logger.info(f"  Insert queries on mock database: {insert_count}")
    
    if doc_count > 0 and batch_count > 0 and insert_count > 0:
        logger.info("SUCCESS: Batch processing implementation working correctly!")
        sys.exit(0)
    else:
        logger.error("FAILED: Batch processing implementation not working correctly")
        sys.exit(1)
