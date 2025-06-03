#!/usr/bin/env python3
"""
Test direct SQL execution to isolate the CRAG issue
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

def test_direct_sql():
    """Test direct SQL execution"""
    print("Testing direct SQL execution...")
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    
    # Generate a test embedding
    query = "What are the symptoms of diabetes?"
    query_embedding = embedding_func([query])[0]
    query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
    
    print(f"Query: {query}")
    print(f"Embedding string length: {len(query_embedding_str)}")
    
    # Test the exact SQL that CRAG would use
    sql_query = f"""
        SELECT TOP 5 
            doc_id, 
            title, 
            text_content,
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) as similarity_score
        FROM RAG.SourceDocuments
        WHERE document_embedding_vector IS NOT NULL
        AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) > 0.0
        ORDER BY similarity_score DESC
    """
    
    cursor = None
    try:
        cursor = iris_conn.cursor()
        print("Executing SQL directly...")
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        print(f"Retrieved {len(results)} documents")
        
        for i, row in enumerate(results[:3]):
            doc_id, title, content, score = row
            print(f"  Doc {i+1}: ID={doc_id}, Score={score:.4f}, Title={title[:50]}...")
            
    except Exception as e:
        print(f"Error executing SQL: {e}")
        print(f"SQL was: {sql_query[:500]}...")
        
    finally:
        if cursor:
            cursor.close()
    
    # Test with a simpler approach - use parameterized query
    print("\n--- Testing with parameterized query ---")
    try:
        cursor = iris_conn.cursor()
        
        # Try using ? parameters instead of string formatting
        param_sql = """
            SELECT TOP 5 
                doc_id, 
                title, 
                text_content,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE', 384)) as similarity_score
            FROM RAG.SourceDocuments
            WHERE document_embedding_vector IS NOT NULL
            AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE', 384)) > 0.0
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(param_sql, (query_embedding_str, query_embedding_str))
        results = cursor.fetchall()
        
        print(f"Parameterized query retrieved {len(results)} documents")
        
        for i, row in enumerate(results[:3]):
            doc_id, title, content, score = row
            print(f"  Doc {i+1}: ID={doc_id}, Score={score:.4f}, Title={title[:50]}...")
            
    except Exception as e:
        print(f"Error with parameterized query: {e}")
        
    finally:
        if cursor:
            cursor.close()

if __name__ == "__main__":
    test_direct_sql()