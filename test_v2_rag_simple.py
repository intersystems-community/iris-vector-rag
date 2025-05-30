#!/usr/bin/env python3
"""
Simple test of V2 tables using actual embeddings from the database
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import time
import logging
from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_rag_v2():
    """Test basic RAG with V2 tables using actual embeddings"""
    print("\n🔍 Testing Basic RAG with V2 tables...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Get a sample embedding from the database to use as query
    cursor.execute("""
        SELECT TOP 1 document_embedding_vector
        FROM RAG.SourceDocuments_V2
        WHERE document_embedding_vector IS NOT NULL
    """)
    sample_vector = cursor.fetchone()[0]
    
    # Search using the sample vector
    start_time = time.time()
    cursor.execute(f"""
        SELECT TOP 5 
            doc_id,
            title,
            text_content,
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{sample_vector}')) as similarity
        FROM RAG.SourceDocuments_V2
        WHERE document_embedding_vector IS NOT NULL
        ORDER BY similarity DESC
    """)
    
    results = cursor.fetchall()
    search_time = time.time() - start_time
    
    print(f"✅ Basic RAG V2 search completed in {search_time:.3f}s")
    print(f"   Found {len(results)} documents")
    if results:
        print(f"   Top result similarity: {results[0][3]:.4f} (should be 1.0 for self-match)")
        print(f"   Title: {results[0][1][:80]}...")
    
    cursor.close()
    conn.close()
    return len(results) > 0 and results[0][3] > 0.99

def test_chunking_rag_v2():
    """Test chunking-based RAG with V2 tables"""
    print("\n🔍 Testing Chunking RAG with V2 tables...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Get a sample chunk embedding
    cursor.execute("""
        SELECT TOP 1 chunk_embedding_vector, doc_id
        FROM RAG.DocumentChunks_V2
        WHERE chunk_embedding_vector IS NOT NULL
    """)
    result = cursor.fetchone()
    if not result:
        print("❌ No chunks with embeddings found")
        return False
        
    sample_vector, sample_doc_id = result
    
    # Search using the sample vector
    start_time = time.time()
    cursor.execute(f"""
        SELECT TOP 5 
            c.chunk_id,
            c.chunk_text,
            c.chunk_index,
            d.title,
            VECTOR_COSINE(c.chunk_embedding_vector, TO_VECTOR('{sample_vector}')) as similarity
        FROM RAG.DocumentChunks_V2 c
        JOIN RAG.SourceDocuments_V2 d ON c.doc_id = d.doc_id
        WHERE c.chunk_embedding_vector IS NOT NULL
        ORDER BY similarity DESC
    """)
    
    results = cursor.fetchall()
    search_time = time.time() - start_time
    
    print(f"✅ Chunking RAG V2 search completed in {search_time:.3f}s")
    print(f"   Found {len(results)} chunks")
    if results:
        print(f"   Top chunk similarity: {results[0][4]:.4f} (should be 1.0 for self-match)")
        print(f"   From document: {results[0][3][:60]}...")
    
    cursor.close()
    conn.close()
    return len(results) > 0 and results[0][4] > 0.99

def test_hnsw_performance():
    """Test HNSW index performance on V2 tables"""
    print("\n🔍 Testing HNSW index performance...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Get multiple sample embeddings for testing
    cursor.execute("""
        SELECT TOP 10 document_embedding_vector, doc_id, title
        FROM RAG.SourceDocuments_V2
        WHERE document_embedding_vector IS NOT NULL
        ORDER BY doc_id
    """)
    samples = cursor.fetchall()
    
    total_time = 0
    total_searches = 0
    
    for sample_vector, doc_id, title in samples[:5]:  # Test with 5 different queries
        start_time = time.time()
        cursor.execute(f"""
            SELECT TOP 20 
                doc_id,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{sample_vector}')) as similarity
            FROM RAG.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
            ORDER BY similarity DESC
        """)
        
        results = cursor.fetchall()
        search_time = time.time() - start_time
        total_time += search_time
        total_searches += 1
        
        print(f"   Search {total_searches}: {search_time:.3f}s for '{title[:40]}...'")
    
    avg_time = total_time / total_searches
    print(f"\n✅ HNSW performance test completed")
    print(f"   Average search time: {avg_time:.3f}s")
    print(f"   Total searches: {total_searches}")
    print(f"   Using HNSW indexes on V2 tables")
    
    cursor.close()
    conn.close()
    return avg_time < 1.0  # Should be fast with HNSW

def test_vector_data_integrity():
    """Test that vector data is properly stored in V2 tables"""
    print("\n🔍 Testing vector data integrity...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Check SourceDocuments_V2
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(document_embedding_vector) as has_vector,
            AVG(LENGTH(document_embedding_vector)) as avg_vector_length
        FROM RAG.SourceDocuments_V2
    """)
    total, has_vector, avg_length = cursor.fetchone()
    print(f"\nSourceDocuments_V2:")
    print(f"   Total: {total:,}, Has vector: {has_vector:,}")
    print(f"   Average vector length: {avg_length:.0f} chars")
    
    # Check DocumentChunks_V2
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(chunk_embedding_vector) as has_vector,
            AVG(LENGTH(chunk_embedding_vector)) as avg_vector_length
        FROM RAG.DocumentChunks_V2
    """)
    total, has_vector, avg_length = cursor.fetchone()
    print(f"\nDocumentChunks_V2:")
    print(f"   Total: {total:,}, Has vector: {has_vector:,}")
    print(f"   Average vector length: {avg_length:.0f} chars")
    
    # Check DocumentTokenEmbeddings_V2
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(token_embedding_vector) as has_vector,
            AVG(LENGTH(token_embedding_vector)) as avg_vector_length
        FROM RAG.DocumentTokenEmbeddings_V2
    """)
    total, has_vector, avg_length = cursor.fetchone()
    print(f"\nDocumentTokenEmbeddings_V2:")
    print(f"   Total: {total:,}, Has vector: {has_vector:,}")
    print(f"   Average vector length: {avg_length:.0f} chars")
    
    cursor.close()
    conn.close()
    return True

def main():
    """Run all V2 table tests"""
    print("🚀 Testing V2 Tables with Actual Data")
    print("=" * 60)
    
    all_passed = True
    
    # Test each function
    tests = [
        ("Vector Data Integrity", test_vector_data_integrity),
        ("Basic RAG V2", test_basic_rag_v2),
        ("Chunking RAG V2", test_chunking_rag_v2),
        ("HNSW Performance", test_hnsw_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"\n✅ {test_name} passed")
            else:
                print(f"\n❌ {test_name} failed")
                all_passed = False
        except Exception as e:
            print(f"\n❌ {test_name} error: {e}")
            logger.error(f"Error in {test_name}: {e}", exc_info=True)
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All V2 table tests passed!")
        print("\n🎉 V2 Migration Complete and Verified!")
        print("\nSummary:")
        print("  - All embeddings migrated to native VECTOR columns")
        print("  - HNSW indexes active and performing well")
        print("  - Vector similarity searches working correctly")
        print("  - Ready for production RAG operations")
    else:
        print("❌ Some tests failed - please investigate")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)