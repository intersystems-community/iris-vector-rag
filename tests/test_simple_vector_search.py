#!/usr/bin/env python3
"""
Test simple vector search to find what actually works
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

def test_simple_vector_search():
    """Test simple vector search to find what works"""
    print("Testing simple vector search...")
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    
    # Generate a test embedding
    query = "diabetes"
    query_embedding = embedding_func([query])[0]
    query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
    
    print(f"Query: {query}")
    print(f"Embedding string length: {len(query_embedding_str)}")
    
    cursor = None
    try:
        cursor = iris_conn.cursor()
        
        # Test 1: Check what tables exist
        print("\n--- Checking available tables ---")
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'RAG'")
        tables = cursor.fetchall()
        print("Available tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Test 2: Check SourceDocuments structure
        print("\n--- Checking SourceDocuments structure ---")
        cursor.execute("SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'")
        columns = cursor.fetchall()
        print("SourceDocuments columns:")
        for col in columns:
            print(f"  - {col[0]}: {col[1]}")
        
        # Test 3: Check if we have data
        print("\n--- Checking data availability ---")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL AND LENGTH(embedding) > 1000")
        count = cursor.fetchone()[0]
        print(f"Documents with embeddings: {count}")
        
        if count > 0:
            # Test 4: Try simple vector search without threshold
            print("\n--- Testing simple vector search ---")
            cursor.execute("""
                SELECT TOP 5 doc_id, title,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND LENGTH(embedding) > 1000
                ORDER BY similarity_score DESC
            """, [query_embedding_str])
            
            results = cursor.fetchall()
            print(f"Retrieved {len(results)} documents")
            
            for i, row in enumerate(results):
                doc_id, title, score = row
                print(f"  Doc {i+1}: ID={doc_id}, Score={score:.4f}, Title={title[:50]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if cursor:
            cursor.close()

if __name__ == "__main__":
    test_simple_vector_search()