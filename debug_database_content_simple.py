#!/usr/bin/env python3
"""
Debug script to investigate what's actually stored in the database.
Uses IRIS-compatible queries without CAST operations.
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
from iris_rag.storage.clob_handler import convert_clob_to_string

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_database_content():
    """Debug what's actually stored in the database."""
    
    # Initialize connection
    connection_manager = ConnectionManager()
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # 1. Check total document count
        logger.info("=== CHECKING DOCUMENT COUNT ===")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Total documents in RAG.SourceDocuments: {total_docs}")
        
        # 2. Sample a few documents to see what's in text_content (without CAST)
        logger.info("\n=== SAMPLING DOCUMENT CONTENT (Raw) ===")
        sample_sql = """
            SELECT TOP 3 doc_id, text_content, title, metadata
            FROM RAG.SourceDocuments
            ORDER BY doc_id
        """
        cursor.execute(sample_sql)
        sample_results = cursor.fetchall()
        
        logger.info("Sample documents (raw):")
        for i, row in enumerate(sample_results):
            doc_id, text_content, title, metadata = row
            
            # Convert CLOBs to strings
            text_str = convert_clob_to_string(text_content)
            title_str = convert_clob_to_string(title)
            metadata_str = convert_clob_to_string(metadata)
            
            logger.info(f"  Document {i+1}:")
            logger.info(f"    doc_id: {doc_id}")
            logger.info(f"    text_content type: {type(text_content)}")
            logger.info(f"    text_content length: {len(text_str)}")
            logger.info(f"    text_content preview: {text_str[:200]}...")
            logger.info(f"    title: {title_str}")
            logger.info(f"    metadata preview: {metadata_str[:100]}...")
        
        # 3. Check specific documents that were returned in our ColBERT test
        logger.info("\n=== CHECKING SPECIFIC DOCUMENTS FROM COLBERT TEST ===")
        test_doc_ids = ['PMC1748215532', 'PMC11650388', 'PMC1748704564', 'PMC1748704557', 'PMC11651618']
        
        for doc_id in test_doc_ids:
            specific_sql = """
                SELECT doc_id, text_content, title
                FROM RAG.SourceDocuments
                WHERE doc_id = ?
            """
            cursor.execute(specific_sql, [doc_id])
            result = cursor.fetchone()
            
            if result:
                doc_id_result, text_content, title = result
                
                # Convert CLOBs to strings
                text_str = convert_clob_to_string(text_content)
                title_str = convert_clob_to_string(title)
                
                logger.info(f"Document {doc_id}:")
                logger.info(f"  text_content type: {type(text_content)}")
                logger.info(f"  text_content length: {len(text_str)}")
                logger.info(f"  text_content preview: {text_str[:300]}...")
                logger.info(f"  title: {title_str}")
            else:
                logger.info(f"Document {doc_id}: NOT FOUND")
        
        # 4. Test the exact query used by ColBERT pipeline
        logger.info("\n=== TESTING COLBERT PIPELINE QUERY ===")
        colbert_sql = """
            SELECT TOP 5 doc_id, text_content, title, metadata
            FROM RAG.SourceDocuments
            WHERE doc_id IN (?, ?, ?, ?, ?)
        """
        cursor.execute(colbert_sql, test_doc_ids)
        colbert_results = cursor.fetchall()
        
        logger.info("ColBERT pipeline query results:")
        for row in colbert_results:
            doc_id, text_content, title, metadata = row
            
            # Convert CLOBs to strings using the same logic as ColBERT pipeline
            text_str = convert_clob_to_string(text_content)
            title_str = convert_clob_to_string(title)
            
            logger.info(f"  {doc_id}:")
            logger.info(f"    Raw text_content type: {type(text_content)}")
            logger.info(f"    Converted text_content type: {type(text_str)}")
            logger.info(f"    Converted text_content length: {len(text_str)}")
            logger.info(f"    Converted text_content: '{text_str}'")
            logger.info(f"    Title: '{title_str}'")
            
    except Exception as e:
        logger.error(f"Database investigation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()

if __name__ == "__main__":
    debug_database_content()