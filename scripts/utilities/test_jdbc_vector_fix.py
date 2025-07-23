#!/usr/bin/env python3
"""
Test JDBC Vector Fix - Verify vector operations work with direct SQL
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_jdbc_vector_operations():
    """Test various vector operations with JDBC"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    print("🔍 Testing JDBC Vector Operations")
    print("=" * 60)
    
    # Test 1: Basic vector similarity
    print("\n1. Testing basic vector similarity...")
    try:
        test_vector = "1,2,3,4,5"
        query = f"""
            SELECT VECTOR_COSINE(TO_VECTOR('{test_vector}'), TO_VECTOR('{test_vector}')) as similarity
        """
        cursor.execute(query)
        result = cursor.fetchone()
        print(f"✅ Self-similarity: {result[0]} (should be ~1.0)")
    except Exception as e:
        print(f"❌ Basic vector test failed: {e}")
    
    # Test 2: Vector search with direct SQL (no parameters)
    print("\n2. Testing vector search with direct SQL...")
    try:
        # Generate a real embedding
        test_text = "diabetes treatment"
        embedding = embedding_func([test_text])[0]
        vector_str = ','.join(map(str, embedding))
        
        query = f"""
            SELECT TOP 5 
                doc_id,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) as score
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
              AND LENGTH(embedding) > 1000
            ORDER BY score DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"✅ Found {len(results)} documents")
        for doc_id, score in results[:3]:
            print(f"   - {doc_id}: {score:.4f}")
    except Exception as e:
        print(f"❌ Direct SQL vector search failed: {e}")
    
    # Test 3: Vector search with threshold (direct SQL)
    print("\n3. Testing vector search with threshold...")
    try:
        threshold = 0.1
        query = f"""
            SELECT TOP 5 
                doc_id,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) as score
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
              AND LENGTH(embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) > {threshold}
            ORDER BY score DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"✅ Found {len(results)} documents above threshold {threshold}")
    except Exception as e:
        print(f"❌ Threshold vector search failed: {e}")
    
    # Test 4: Chunk retrieval with direct SQL
    print("\n4. Testing chunk retrieval...")
    try:
        query = f"""
            SELECT TOP 5
                chunk_id,
                doc_id,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) as score
            FROM RAG.DocumentChunks
            WHERE embedding IS NOT NULL
              AND chunk_type IN ('content', 'mixed')
            ORDER BY score DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"✅ Found {len(results)} chunks")
        for chunk_id, doc_id, score in results[:3]:
            print(f"   - Chunk {chunk_id} from {doc_id}: {score:.4f}")
    except Exception as e:
        print(f"❌ Chunk retrieval failed: {e}")
    
    # Test 5: Parameter binding attempt (expected to fail)
    print("\n5. Testing parameter binding (expected to fail)...")
    try:
        query = """
            SELECT TOP 1 
                doc_id,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as score
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
        """
        cursor.execute(query, (vector_str,))
        print("❓ Parameter binding unexpectedly succeeded!")
    except Exception as e:
        print(f"✅ Parameter binding failed as expected: {e}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("📌 Conclusion: Use direct SQL with string interpolation for vector operations")
    print("📌 Avoid parameter binding with vector functions in JDBC")

if __name__ == "__main__":
    test_jdbc_vector_operations()