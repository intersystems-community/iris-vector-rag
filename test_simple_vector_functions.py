#!/usr/bin/env python3
"""
Simple test of IRIS vector functions
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import logging
from common.utils import get_embedding_func
from common.iris_connector_jdbc import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_vector_functions():
    """Test basic vector functions"""
    logger.info("=== Testing Simple Vector Functions ===")
    
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    cursor = None
    
    try:
        cursor = iris_connector.cursor()
        
        # Test 1: Check what we have in the database
        logger.info("Test 1: Checking embedding storage format")
        check_sql = """
            SELECT TOP 3 doc_id, 
                   LENGTH(embedding) as embedding_length,
                   SUBSTRING(embedding, 1, 50) as embedding_sample
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
            AND embedding NOT LIKE '0.1,0.1,0.1%'
            ORDER BY doc_id
        """
        cursor.execute(check_sql)
        results = cursor.fetchall()
        
        logger.info(f"Found {len(results)} documents with embeddings:")
        for row in results:
            logger.info(f"  Doc {row[0]}: Length={row[1]}, Sample='{row[2]}...'")
        
        if not results:
            logger.error("No documents with embeddings found!")
            return False
        
        # Test 2: Try TO_VECTOR function with parameters
        logger.info("Test 2: Testing TO_VECTOR function with parameters")
        sample_embedding = "0.1,0.2,0.3"
        
        try:
            test_sql = "SELECT TO_VECTOR(?, 'FLOAT', 3) as test_vector"
            cursor.execute(test_sql, [sample_embedding])
            result = cursor.fetchone()
            logger.info(f"  ✓ TO_VECTOR with parameters works: {result}")
        except Exception as e:
            logger.error(f"  ✗ TO_VECTOR with parameters failed: {e}")
            return False
        
        # Test 3: Try vector similarity with parameters
        logger.info("Test 3: Testing VECTOR_COSINE with parameters")
        
        # Get a real embedding from the database
        sample_sql = """
            SELECT TOP 1 embedding 
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
            AND embedding NOT LIKE '0.1,0.1,0.1%'
        """
        cursor.execute(sample_sql)
        sample_result = cursor.fetchone()
        
        if sample_result:
            sample_embedding_str = sample_result[0]
            
            # Generate a query embedding
            query_embedding = embedding_func(["test query"])[0]
            query_embedding_str = ','.join([f'{x:.6f}' for x in query_embedding])
            
            try:
                similarity_sql = """
                    SELECT VECTOR_COSINE(
                        TO_VECTOR(?, 'FLOAT', 384), 
                        TO_VECTOR(?, 'FLOAT', 384)
                    ) as similarity
                """
                cursor.execute(similarity_sql, [sample_embedding_str, query_embedding_str])
                result = cursor.fetchone()
                logger.info(f"  ✓ VECTOR_COSINE with parameters works: {result[0]}")
            except Exception as e:
                logger.error(f"  ✗ VECTOR_COSINE with parameters failed: {e}")
                return False
        
        # Test 4: Try the full query with parameters
        logger.info("Test 4: Testing full vector search with parameters")
        
        query_embedding = embedding_func(["diabetes symptoms"])[0]
        query_embedding_str = ','.join([f'{x:.6f}' for x in query_embedding])
        
        try:
            full_sql = """
                SELECT TOP 3 
                    doc_id, 
                    title, 
                    VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 384), TO_VECTOR(?, 'FLOAT', 384)) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND embedding NOT LIKE '0.1,0.1,0.1%'
                ORDER BY similarity_score DESC
            """
            
            cursor.execute(full_sql, [query_embedding_str])
            results = cursor.fetchall()
            
            logger.info(f"  ✓ Full vector search with parameters works! Found {len(results)} documents")
            for i, row in enumerate(results):
                doc_id = row[0]
                title = row[1] or "No title"
                similarity = row[2]
                logger.info(f"    {i+1}. Doc {doc_id}: {title[:30]}... (similarity: {similarity:.4f})")
                
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Full vector search with parameters failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error in simple vector testing: {e}")
        return False
        
    finally:
        if cursor:
            cursor.close()

def main():
    """Run simple vector tests"""
    logger.info("Starting Simple IRIS Vector Function Testing")
    logger.info("=" * 50)
    
    success = test_simple_vector_functions()
    
    logger.info("=" * 50)
    logger.info("SIMPLE VECTOR TEST SUMMARY:")
    logger.info(f"Vector functions work with parameters: {success}")
    
    if success:
        logger.info("✓ Vector functions are available and working!")
        logger.info("  The issue is likely in the SQL string formatting in the pipeline")
    else:
        logger.info("✗ Vector functions are not working properly")
        logger.info("  Fallback mechanism is necessary")

if __name__ == "__main__":
    main()