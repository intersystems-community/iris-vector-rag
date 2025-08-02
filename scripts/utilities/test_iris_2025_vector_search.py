#!/usr/bin/env python3
"""
Test IRIS 2025.1 Vector Search capabilities with licensed version.
This script validates:
1. VECTOR data type support
2. HNSW index creation and functionality
3. Vector search performance
4. Complete RAG schema with native vector support
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

def test_vector_data_type():
    """Test that VECTOR data type is properly supported."""
    print("=" * 60)
    print("TESTING VECTOR DATA TYPE SUPPORT")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Drop test table if exists
        cursor.execute("DROP TABLE IF EXISTS test_vector_table")
        
        # Create table with VECTOR column
        create_sql = """
        CREATE TABLE test_vector_table (
            id INTEGER PRIMARY KEY,
            content VARCHAR(1000),
            embedding VECTOR(FLOAT, 768)
        )
        """
        cursor.execute(create_sql)
        print("✓ Successfully created table with VECTOR(FLOAT, 768) column")
        
        # Insert test vector
        test_vector = np.random.random(768).tolist()
        insert_sql = """
        INSERT INTO test_vector_table (id, content, embedding) 
        VALUES (?, ?, TO_VECTOR(?))
        """
        cursor.execute(insert_sql, (1, "Test document", str(test_vector)))
        print("✓ Successfully inserted vector data")
        
        # Verify the column type remains VECTOR
        cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'TEST_VECTOR_TABLE' AND COLUMN_NAME = 'EMBEDDING'
        """)
        result = cursor.fetchone()
        
        if result:
            col_name, data_type, max_length = result
            print(f"✓ Column type verification: {col_name} = {data_type}")
            if data_type == 'VECTOR':
                print("✓ VECTOR data type is properly supported!")
                return True
            else:
                print(f"✗ Expected VECTOR, got {data_type}")
                return False
        else:
            print("✗ Could not verify column type")
            return False
            
    except Exception as e:
        print(f"✗ Vector data type test failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_hnsw_index_creation():
    """Test HNSW index creation and functionality."""
    print("\n" + "=" * 60)
    print("TESTING HNSW INDEX CREATION")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Create HNSW index
        index_sql = """
        CREATE INDEX idx_test_vector_hnsw 
        ON test_vector_table (embedding) 
        AS HNSW(Distance='Cosine')
        """
        cursor.execute(index_sql)
        print("✓ Successfully created HNSW index with Cosine distance")
        
        # Verify index exists
        cursor.execute("""
        SELECT INDEX_NAME, INDEX_TYPE 
        FROM INFORMATION_SCHEMA.INDEXES 
        WHERE TABLE_NAME = 'TEST_VECTOR_TABLE' 
        AND INDEX_NAME = 'IDX_TEST_VECTOR_HNSW'
        """)
        result = cursor.fetchone()
        
        if result:
            index_name, index_type = result
            print(f"✓ Index verification: {index_name} = {index_type}")
            return True
        else:
            print("✗ HNSW index not found in system catalog")
            return False
            
    except Exception as e:
        print(f"✗ HNSW index creation failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_vector_search_functionality():
    """Test vector search with HNSW index."""
    print("\n" + "=" * 60)
    print("TESTING VECTOR SEARCH FUNCTIONALITY")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Insert more test vectors
        for i in range(2, 11):
            test_vector = np.random.random(768).tolist()
            cursor.execute("""
            INSERT INTO test_vector_table (id, content, embedding) 
            VALUES (?, ?, TO_VECTOR(?))
            """, (i, f"Test document {i}", str(test_vector)))
        
        print("✓ Inserted 10 test vectors")
        
        # Perform vector similarity search
        query_vector = np.random.random(768).tolist()
        search_sql = """
        SELECT TOP 5 id, content, 
               VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
        FROM test_vector_table 
        ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?)) DESC
        """
        
        start_time = time.time()
        cursor.execute(search_sql, (str(query_vector), str(query_vector)))
        results = cursor.fetchall()
        search_time = time.time() - start_time
        
        print(f"✓ Vector search completed in {search_time:.4f} seconds")
        print(f"✓ Retrieved {len(results)} results")
        
        for i, (doc_id, content, similarity) in enumerate(results):
            print(f"  {i+1}. ID: {doc_id}, Similarity: {similarity:.4f}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"✗ Vector search test failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_rag_schema_with_vectors():
    """Test complete RAG schema with native vector support."""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE RAG SCHEMA WITH VECTORS")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Create documents table with vector support
        cursor.execute("DROP TABLE IF EXISTS rag_documents_vector")
        cursor.execute("""
        CREATE TABLE rag_documents_vector (
            doc_id VARCHAR(50) PRIMARY KEY,
            title VARCHAR(500),
            content TEXT,
            embedding VECTOR(FLOAT, 768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        print("✓ Created rag_documents_vector table")
        
        # Create HNSW index
        cursor.execute("""
        CREATE INDEX idx_rag_documents_vector_hnsw 
        ON rag_documents_vector (embedding) 
        AS HNSW(Distance='Cosine')
        """)
        print("✓ Created HNSW index on rag_documents_vector")
        
        # Create chunks table with vector support
        cursor.execute("DROP TABLE IF EXISTS rag_chunks_vector")
        cursor.execute("""
        CREATE TABLE rag_chunks_vector (
            chunk_id VARCHAR(100) PRIMARY KEY,
            doc_id VARCHAR(50),
            chunk_text TEXT,
            chunk_index INTEGER,
            embedding VECTOR(FLOAT, 768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES rag_documents_vector(doc_id)
        )
        """)
        print("✓ Created rag_chunks_vector table")
        
        # Create HNSW index on chunks
        cursor.execute("""
        CREATE INDEX idx_rag_chunks_vector_hnsw 
        ON rag_chunks_vector (embedding) 
        AS HNSW(Distance='Cosine')
        """)
        print("✓ Created HNSW index on rag_chunks_vector")
        
        # Test inserting sample data
        embedding_model = get_embedding_model(mock=True)
        
        # Insert sample document
        sample_text = "This is a sample document for testing vector search capabilities in IRIS 2025.1"
        sample_embedding = embedding_model.encode([sample_text])[0]
        
        cursor.execute("""
        INSERT INTO rag_documents_vector (doc_id, title, content, embedding)
        VALUES (?, ?, ?, TO_VECTOR(?))
        """, ("DOC001", "Sample Document", sample_text, str(sample_embedding.tolist())))
        
        # Insert sample chunks
        chunks = [
            "This is a sample document for testing",
            "vector search capabilities in IRIS 2025.1"
        ]
        
        for i, chunk in enumerate(chunks):
            chunk_embedding = embedding_model.encode([chunk])[0]
            cursor.execute("""
            INSERT INTO rag_chunks_vector (chunk_id, doc_id, chunk_text, chunk_index, embedding)
            VALUES (?, ?, ?, ?, TO_VECTOR(?))
            """, (f"DOC001_CHUNK_{i}", "DOC001", chunk, i, str(chunk_embedding.tolist())))
        
        print("✓ Inserted sample documents and chunks with embeddings")
        
        # Test vector search on the RAG schema
        query = "testing vector search"
        query_embedding = embedding_model.encode([query])[0]
        
        cursor.execute("""
        SELECT TOP 3 chunk_id, chunk_text, 
               VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
        FROM rag_chunks_vector 
        ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?)) DESC
        """, (str(query_embedding.tolist()), str(query_embedding.tolist())))
        
        results = cursor.fetchall()
        print(f"✓ RAG vector search returned {len(results)} results")
        
        for chunk_id, chunk_text, similarity in results:
            print(f"  - {chunk_id}: {similarity:.4f} - {chunk_text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ RAG schema test failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_license_verification():
    """Verify that Vector Search is enabled in the license."""
    print("\n" + "=" * 60)
    print("TESTING LICENSE VERIFICATION")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Check license information
        cursor.execute("SELECT $SYSTEM.License.GetFeature('Vector Search')")
        result = cursor.fetchone()
        
        if result and result[0] == 1:
            print("✓ Vector Search is enabled in the license")
            return True
        else:
            print("✗ Vector Search is not enabled in the license")
            return False
            
    except Exception as e:
        print(f"✗ License verification failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    """Run all vector search tests."""
    print("IRIS 2025.1 Vector Search Validation")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        ("License Verification", test_license_verification),
        ("Vector Data Type", test_vector_data_type),
        ("HNSW Index Creation", test_hnsw_index_creation),
        ("Vector Search Functionality", test_vector_search_functionality),
        ("RAG Schema with Vectors", test_rag_schema_with_vectors)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! IRIS 2025.1 Vector Search is working correctly!")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)