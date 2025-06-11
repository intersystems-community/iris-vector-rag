#!/usr/bin/env python3
"""
Simple debug script to check what's actually in the database.
"""

import os
import sys
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required components
from iris_rag.core.connection import ConnectionManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_simple_content():
    """Debug what's actually stored in the database."""
    
    # Initialize connection
    connection_manager = ConnectionManager()
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Check total document count
        logger.info("=== CHECKING DOCUMENT COUNT ===")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Total documents in RAG.SourceDocuments: {total_docs}")
        
        # Sample a few documents to see what's in text_content (without CAST)
        logger.info("\n=== SAMPLING DOCUMENT CONTENT (RAW) ===")
        sample_sql = """
            SELECT TOP 5 doc_id, text_content, title
            FROM RAG.SourceDocuments
            ORDER BY doc_id
        """
        cursor.execute(sample_sql)
        sample_results = cursor.fetchall()
        
        logger.info("Sample documents:")
        for i, row in enumerate(sample_results):
            doc_id, text_content, title = row
            logger.info(f"  Document {i+1}:")
            logger.info(f"    doc_id: {doc_id}")
            logger.info(f"    text_content type: {type(text_content)}")
            logger.info(f"    text_content value: {repr(text_content)}")
            logger.info(f"    title type: {type(title)}")
            logger.info(f"    title value: {repr(title)}")
            
        # Check specific documents that were returned in our test
        logger.info("\n=== CHECKING SPECIFIC DOCUMENTS FROM COLBERT TEST ===")
        test_doc_ids = ['PMC1748215532', 'PMC11650388', 'PMC1748704564']
        
        for doc_id in test_doc_ids:
            specific_sql = """
                SELECT doc_id, text_content, title
                FROM RAG.SourceDocuments
                WHERE doc_id = ?
            """
            cursor.execute(specific_sql, [doc_id])
            result = cursor.fetchone()
            
            if result:
                doc_id_result, text_content, title_content = result
                logger.info(f"Document {doc_id}:")
                logger.info(f"  text_content: {repr(text_content)}")
                logger.info(f"  title: {repr(title_content)}")
            else:
                logger.info(f"Document {doc_id}: NOT FOUND")
            
    except Exception as e:
        logger.error(f"Database investigation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()

if __name__ == "__main__":
    debug_simple_content()