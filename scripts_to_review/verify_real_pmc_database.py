#!/usr/bin/env python3
"""
Verification script to ensure real PMC documents are loaded into a real database.

This script verifies that:
1. We have real PMC documents in the data directory
2. A real IRIS database is used (not mocks)
3. The database contains at least 1000 real documents

This helps ensure compliance with the .clinerules requirement that 
"Tests must use real PMC documents, not synthetic data. At least 1000 documents should be used."
"""

import os
import sys
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

# --- Debug Print ---
print(f"DEBUG: Executing script: {__file__}")
try:
    with open(__file__, 'r') as f:
        first_lines = ''.join(f.readlines()[:10])
        print(f"DEBUG: First 10 lines of script:\n---\n{first_lines}\n---")
except Exception as e:
    print(f"DEBUG: Could not read script file: {e}")
# --- End Debug Print ---


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_real_pmc_documents() -> int:
    """Count real PMC XML documents in the data directory."""
    pmc_dir = os.path.join(os.getcwd(), "data", "pmc_oas_downloaded")
    logger.info(f"Counting PMC documents in {pmc_dir}")
    
    if not os.path.exists(pmc_dir):
        logger.error(f"PMC directory does not exist: {pmc_dir}")
        return 0
    
    xml_files = []
    for dirpath, dirnames, filenames in os.walk(pmc_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_files.append(os.path.join(dirpath, filename))
    
    logger.info(f"Found {len(xml_files)} PMC XML files")
    
    # Verify a sample of files are actually PMC XML
    sample_size = min(10, len(xml_files))
    if sample_size > 0:
        logger.info(f"Verifying random sample of {sample_size} files are valid PMC XML")
        import random
        valid_count = 0
        
        for xml_file in random.sample(xml_files, sample_size):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Look for PMC-specific elements
                if (root.tag == 'article' or 
                    root.find(".//article-id[@pub-id-type='pmc']") is not None or
                    root.find(".//journal-id") is not None):
                    valid_count += 1
                else:
                    logger.warning(f"File doesn't appear to be PMC XML: {xml_file}")
            except Exception as e:
                logger.warning(f"Error parsing XML file {xml_file}: {e}")
        
        logger.info(f"Verified {valid_count}/{sample_size} PMC XML files are valid")
    
    return len(xml_files)

def verify_iris_database(required_docs: int = 1000) -> bool:
    """Verify IRIS database is real and contains required number of documents."""
    # Import the IRIS connector from the common module
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from common.iris_connector import get_iris_connection
        
        logger.info("Connecting to IRIS database")
        conn = get_iris_connection()
        
        # Check if connection is a mock
        conn_str = str(conn)
        if "mock" in conn_str.lower():
            logger.error("Using mock IRIS connection, not a real database!")
            return False
        logger.info("Connected to real IRIS database")
        
        # Check document count
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            count = cursor.fetchone()[0]
            logger.info(f"Database contains {count} documents")
            
            if count < required_docs:
                logger.warning(f"Insufficient documents: {count} < {required_docs}")
                return False
            
            # Check a sample document to verify content
            cursor.execute("SELECT TOP 1 doc_id, text_content FROM SourceDocuments") # Use text_content column
            doc = cursor.fetchone()
            if doc:
                doc_id, content = doc
                logger.info(f"Sample document {doc_id} has {len(content)} characters of content")
                # Real content should be substantial in length
                if len(content) < 100:
                    logger.warning("Document content seems suspiciously short")
        
        return count >= required_docs
    
    except Exception as e:
        logger.error(f"Error verifying IRIS database: {e}")
        return False

def main():
    """Main verification function."""
    logger.info("=" * 80)
    logger.info("Verifying real PMC documents and database")
    logger.info("=" * 80)
    
    # 1. Count real PMC documents
    real_docs = count_real_pmc_documents()
    
    # 2. Verify IRIS database
    db_valid = verify_iris_database(required_docs=1000)
    
    # 3. Summary
    logger.info("=" * 80)
    logger.info("Verification Summary:")
    logger.info(f"- Real PMC documents found: {real_docs}")
    logger.info(f"- Real IRIS database with 1000+ documents: {'✅ Yes' if db_valid else '❌ No'}")
    
    if real_docs >= 1000 and db_valid:
        logger.info("✅ COMPLIANT: Using real database with real PMC documents")
        return 0
    else:
        logger.error("❌ NON-COMPLIANT: Requirements not met")
        if real_docs < 1000:
            logger.error(f"   - Need at least 1000 real PMC documents, found only {real_docs}")
        if not db_valid:
            logger.error("   - Real IRIS database with 1000+ documents not verified")
        return 1

if __name__ == "__main__":
    sys.exit(main())
