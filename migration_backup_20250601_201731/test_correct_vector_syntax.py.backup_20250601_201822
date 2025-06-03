#!/usr/bin/env python3
"""
Test Correct Vector Syntax for IRIS
Based on the diagnostic results, we need to test the proper vector syntax
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import logging
from common.utils import get_embedding_func
from common.iris_connector_jdbc import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_correct_vector_syntax():
    """Test the correct IRIS vector syntax"""
    logger.info("=== Testing Correct IRIS Vector Syntax ===")
    
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    cursor = None
    
    try:
        cursor = iris_connector.cursor()
        
        # Generate a test embedding
        test_query = "diabetes symptoms"
        query_embedding = embedding_func([test_query])[0]
        logger.info(f"Generated embedding with {len(query_embedding)} dimensions")
        
        # Test 1: Check if we have any documents with vector embeddings
        logger.info("Test 1: Checking for documents with vector embeddings")
        check_sql = """
            SELECT TOP 5 doc_id, 
                   CASE WHEN embedding IS NULL THEN 'NULL' 
                        WHEN DATATYPE(embedding) = 'VECTOR' THEN 'VECTOR_TYPE'
                        ELSE 'STRING_TYPE' END as embedding_type,
                   LENGTH(embedding) as embedding_length
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
            ORDER BY doc_id
        """
        cursor.execute(check_sql)
        results = cursor.fetchall()
        
        logger.info(f"Found {len(results)} documents with embeddings:")
        for row in results:
            logger.info(f"  Doc {row[0]}: Type={row[1]}, Length={row[2]}")
        
        # Test 2: Try to convert string embeddings to vectors
        logger.info("Test 2: Testing string to vector conversion")
        if results:
            # Get a sample embedding string
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
                logger.info(f"Sample embedding (first 100 chars): {sample_embedding_str[:100]}...")
                
                # Try different vector conversion approaches
                test_cases = [
                    {
                        "name": "TO_VECTOR with DOUBLE type",
                        "sql": f"SELECT TO_VECTOR(?, 'DOUBLE', 384) as test_vector",
                        "params": [sample_embedding_str]
                    },
                    {
                        "name": "TO_VECTOR with REAL type", 
                        "sql": f"SELECT TO_VECTOR(?, 'REAL', 384) as test_vector",
                        "params": [sample_embedding_str]
                    },
                    {
                        "name": "Direct VECTOR() constructor",
                        "sql": f"SELECT VECTOR(?) as test_vector",
                        "params": [sample_embedding_str]
                    }
                ]
                
                for test_case in test_cases:
                    try:
                        logger.info(f"  Testing: {test_case['name']}")
                        cursor.execute(test_case['sql'], test_case['params'])
                        result = cursor.fetchone()
                        logger.info(f"    ✓ Success: {type(result[0])}")
                    except Exception as e:
                        logger.info(f"    ✗ Failed: {e}")
        
        # Test 3: Test vector similarity functions with proper syntax
        logger.info("Test 3: Testing vector similarity functions")
        
        # Create a simple test vector
        test_vector_str = ','.join([f'{x:.6f}' for x in query_embedding])
        
        similarity_tests = [
            {
                "name": "VECTOR_COSINE with TO_VECTOR",
                "sql": """
                    SELECT TOP 1 doc_id,
                           VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 384), TO_VECTOR(?, 'DOUBLE', 384)) as similarity
                    FROM RAG.SourceDocuments 
                    WHERE embedding IS NOT NULL
                    AND embedding NOT LIKE '0.1,0.1,0.1%'
                """,
                "params": [test_vector_str]
            },
            {
                "name": "VECTOR_DOT_PRODUCT with TO_VECTOR",
                "sql": """
                    SELECT TOP 1 doc_id,
                           VECTOR_DOT_PRODUCT(TO_VECTOR(embedding, 'DOUBLE', 384), TO_VECTOR(?, 'DOUBLE', 384)) as similarity
                    FROM RAG.SourceDocuments 
                    WHERE embedding IS NOT NULL
                    AND embedding NOT LIKE '0.1,0.1,0.1%'
                """,
                "params": [test_vector_str]
            }
        ]
        
        for test in similarity_tests:
            try:
                logger.info(f"  Testing: {test['name']}")
                cursor.execute(test['sql'], test['params'])
                result = cursor.fetchone()
                if result:
                    logger.info(f"    ✓ Success: Doc {result[0]}, Similarity: {result[1]}")
                else:
                    logger.info(f"    ✓ Success but no results")
            except Exception as e:
                logger.info(f"    ✗ Failed: {e}")
        
        # Test 4: Test the working vector search query
        logger.info("Test 4: Testing complete vector search query")
        try:
            working_sql = """
                SELECT TOP 5 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 384), TO_VECTOR(?, 'DOUBLE', 384)) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND embedding NOT LIKE '0.1,0.1,0.1%'
                AND VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 384), TO_VECTOR(?, 'DOUBLE', 384)) > 0.1
                ORDER BY similarity_score DESC
            """
            
            cursor.execute(working_sql, [test_vector_str, test_vector_str])
            results = cursor.fetchall()
            
            logger.info(f"    ✓ Complete vector search succeeded! Found {len(results)} documents")
            for i, row in enumerate(results):
                doc_id = row[0]
                title = row[1] or "No title"
                similarity = row[3]
                logger.info(f"      {i+1}. Doc {doc_id}: {title[:50]}... (similarity: {similarity:.4f})")
                
            return True, len(results)
            
        except Exception as e:
            logger.error(f"    ✗ Complete vector search failed: {e}")
            return False, str(e)
        
    except Exception as e:
        logger.error(f"Error in vector syntax testing: {e}")
        return False, str(e)
        
    finally:
        if cursor:
            cursor.close()

def main():
    """Run vector syntax tests"""
    logger.info("Starting IRIS Vector Syntax Testing")
    logger.info("=" * 50)
    
    success, result = test_correct_vector_syntax()
    
    logger.info("=" * 50)
    logger.info("VECTOR SYNTAX TEST SUMMARY:")
    logger.info(f"Vector search works: {success}")
    if success:
        logger.info(f"Found {result} documents using vector search")
    else:
        logger.info(f"Error: {result}")

if __name__ == "__main__":
    main()