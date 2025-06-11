#!/usr/bin/env python3
"""
Debug script to investigate what's actually stored in the database.

This script will:
1. Check the schema of RAG.SourceDocuments
2. Query a few sample documents to see what's in text_content
3. Identify why we're getting numeric IDs instead of actual content
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

def debug_database_content():
    """Debug what's actually stored in the database."""
    
    # Initialize connection
    connection_manager = ConnectionManager()
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # 1. Check the schema of RAG.SourceDocuments
        logger.info("=== CHECKING RAG.SourceDocuments SCHEMA ===")
        schema_sql = """
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'
            ORDER BY ORDINAL_POSITION
        """
        cursor.execute(schema_sql)
        schema_results = cursor.fetchall()
        
        logger.info("Schema of RAG.SourceDocuments:")
        for row in schema_results:
            column_name, data_type, max_length = row
            logger.info(f"  {column_name}: {data_type} ({max_length})")
        
        # 2. Check total document count
        logger.info("\n=== CHECKING DOCUMENT COUNT ===")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Total documents in RAG.SourceDocuments: {total_docs}")
        
        # 3. Sample a few documents to see what's in text_content
        logger.info("\n=== SAMPLING DOCUMENT CONTENT ===")
        sample_sql = """
            SELECT TOP 5 doc_id, 
                   CASE 
                       WHEN text_content IS NULL THEN 'NULL'
                       ELSE SUBSTRING(CAST(text_content AS VARCHAR(200)), 1, 200)
                   END as text_preview,
                   CASE 
                       WHEN title IS NULL THEN 'NULL'
                       ELSE CAST(title AS VARCHAR(100))
                   END as title_preview,
                   CASE 
                       WHEN metadata IS NULL THEN 'NULL'
                       ELSE SUBSTRING(CAST(metadata AS VARCHAR(200)), 1, 200)
                   END as metadata_preview
            FROM RAG.SourceDocuments
            ORDER BY doc_id
        """
        cursor.execute(sample_sql)
        sample_results = cursor.fetchall()
        
        logger.info("Sample documents:")
        for i, row in enumerate(sample_results):
            doc_id, text_preview, title_preview, metadata_preview = row
            logger.info(f"  Document {i+1}:")
            logger.info(f"    doc_id: {doc_id}")
            logger.info(f"    text_content preview: {text_preview}")
            logger.info(f"    title preview: {title_preview}")
            logger.info(f"    metadata preview: {metadata_preview}")
        
        # 4. Check specific documents that were returned in our test
        logger.info("\n=== CHECKING SPECIFIC DOCUMENTS FROM COLBERT TEST ===")
        test_doc_ids = ['PMC1748215532', 'PMC11650388', 'PMC1748704564', 'PMC1748704557', 'PMC11651618']
        
        for doc_id in test_doc_ids:
            specific_sql = """
                SELECT doc_id,
                       CASE 
                           WHEN text_content IS NULL THEN 'NULL'
                           ELSE SUBSTRING(CAST(text_content AS VARCHAR(500)), 1, 500)
                       END as text_content_preview,
                       CASE 
                           WHEN title IS NULL THEN 'NULL'
                           ELSE CAST(title AS VARCHAR(200))
                       END as title_content
                FROM RAG.SourceDocuments
                WHERE doc_id = ?
            """
            cursor.execute(specific_sql, [doc_id])
            result = cursor.fetchone()
            
            if result:
                doc_id_result, text_content, title_content = result
                logger.info(f"Document {doc_id}:")
                logger.info(f"  text_content length: {len(text_content) if text_content != 'NULL' else 0}")
                logger.info(f"  text_content preview: {text_content[:200] if text_content != 'NULL' else 'NULL'}...")
                logger.info(f"  title: {title_content}")
            else:
                logger.info(f"Document {doc_id}: NOT FOUND")
        
        # 5. Check if there are any documents with substantial text content
        logger.info("\n=== CHECKING FOR DOCUMENTS WITH SUBSTANTIAL CONTENT ===")
        substantial_sql = """
            SELECT TOP 5 doc_id,
                   LENGTH(CAST(text_content AS VARCHAR(MAX))) as content_length,
                   SUBSTRING(CAST(text_content AS VARCHAR(500)), 1, 200) as content_preview
            FROM RAG.SourceDocuments
            WHERE text_content IS NOT NULL
            ORDER BY LENGTH(CAST(text_content AS VARCHAR(MAX))) DESC
        """
        cursor.execute(substantial_sql)
        substantial_results = cursor.fetchall()
        
        logger.info("Documents with longest content:")
        for row in substantial_results:
            doc_id, content_length, content_preview = row
            logger.info(f"  {doc_id}: {content_length} chars - {content_preview}...")
            
    except Exception as e:
        logger.error(f"Database investigation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()

if __name__ == "__main__":
    debug_database_content()