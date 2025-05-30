#!/usr/bin/env python3
"""
Test all RAG techniques with V2 tables to ensure they work correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import time
import logging
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_rag_v2():
    """Test basic RAG with V2 tables"""
    print("\n🔍 Testing Basic RAG with V2 tables...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Test query
    query = "What are the main applications of machine learning in healthcare?"
    embedding_func = get_embedding_func(mock=True)
    query_embedding = embedding_func([query])[0]
    
    # Search using V2 table with VECTOR column
    start_time = time.time()
    # Convert embedding to string for TO_VECTOR
    embedding_str = str(query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding)
    
    cursor.execute(f"""
        SELECT TOP 5
            doc_id,
            title,
            text_content,
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{embedding_str}')) as similarity
        FROM RAG.SourceDocuments_V2
        WHERE document_embedding_vector IS NOT NULL
        ORDER BY similarity DESC
    """)
    
    results = cursor.fetchall()
    search_time = time.time() - start_time
    
    print(f"✅ Basic RAG V2 search completed in {search_time:.3f}s")
    print(f"   Found {len(results)} relevant documents")
    if results:
        print(f"   Top result similarity: {results[0][3]:.4f}")
    
    cursor.close()
    conn.close()
    return len(results) > 0

def test_chunking_rag_v2():
    """Test chunking-based RAG with V2 tables"""
    print("\n🔍 Testing Chunking RAG with V2 tables...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Test query
    query = "What are the benefits of exercise?"
    embedding_func = get_embedding_func(mock=True)
    query_embedding = embedding_func([query])[0]
    
    # Search using V2 chunks table
    start_time = time.time()
    # Convert embedding to string for TO_VECTOR
    embedding_str = str(query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding)
    
    cursor.execute(f"""
        SELECT TOP 5
            c.chunk_id,
            c.chunk_text,
            c.chunk_index,
            d.title,
            VECTOR_COSINE(c.chunk_embedding_vector, TO_VECTOR('{embedding_str}')) as similarity
        FROM RAG.DocumentChunks_V2 c
        JOIN RAG.SourceDocuments_V2 d ON c.doc_id = d.doc_id
        WHERE c.chunk_embedding_vector IS NOT NULL
        ORDER BY similarity DESC
    """)
    
    results = cursor.fetchall()
    search_time = time.time() - start_time
    
    print(f"✅ Chunking RAG V2 search completed in {search_time:.3f}s")
    print(f"   Found {len(results)} relevant chunks")
    if results:
        print(f"   Top chunk similarity: {results[0][4]:.4f}")
    
    cursor.close()
    conn.close()
    return len(results) > 0

def test_colbert_rag_v2():
    """Test ColBERT RAG with V2 tables"""
    print("\n🔍 Testing ColBERT RAG with V2 tables...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Check if we have token embeddings
    cursor.execute("""
        SELECT COUNT(*) 
        FROM RAG.DocumentTokenEmbeddings_V2
        WHERE token_embedding_vector IS NOT NULL
    """)
    token_count = cursor.fetchone()[0]
    
    print(f"   Found {token_count:,} token embeddings in V2 table")
    
    if token_count > 0:
        # Test a simple token search
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 10
                doc_id,
                token_text,
                token_sequence_index
            FROM RAG.DocumentTokenEmbeddings_V2
            WHERE token_embedding_vector IS NOT NULL
            AND token_text LIKE '%health%'
        """)
        
        results = cursor.fetchall()
        search_time = time.time() - start_time
        
        print(f"✅ ColBERT V2 token search completed in {search_time:.3f}s")
        print(f"   Found {len(results)} health-related tokens")
    
    cursor.close()
    conn.close()
    return token_count > 0

def test_hnsw_index_performance():
    """Test HNSW index performance on V2 tables"""
    print("\n🔍 Testing HNSW index performance...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Get a sample embedding for testing
    cursor.execute("""
        SELECT TOP 1 document_embedding_vector
        FROM RAG.SourceDocuments_V2
        WHERE document_embedding_vector IS NOT NULL
    """)
    sample_vector = cursor.fetchone()[0]
    
    # Test HNSW search performance
    start_time = time.time()
    cursor.execute(f"""
        SELECT TOP 100
            doc_id,
            VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{sample_vector}')) as similarity
        FROM RAG.SourceDocuments_V2
        WHERE document_embedding_vector IS NOT NULL
        ORDER BY similarity DESC
    """)
    
    results = cursor.fetchall()
    search_time = time.time() - start_time
    
    print(f"✅ HNSW search completed in {search_time:.3f}s")
    print(f"   Retrieved {len(results)} similar documents")
    print(f"   Average time per result: {(search_time/len(results)*1000):.2f}ms")
    
    cursor.close()
    conn.close()
    return True

def main():
    """Run all V2 table tests"""
    print("🚀 Testing RAG Techniques with V2 Tables")
    print("=" * 60)
    
    all_passed = True
    
    # Test each technique
    tests = [
        ("Basic RAG V2", test_basic_rag_v2),
        ("Chunking RAG V2", test_chunking_rag_v2),
        ("ColBERT RAG V2", test_colbert_rag_v2),
        ("HNSW Performance", test_hnsw_index_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            logger.error(f"Error in {test_name}: {e}", exc_info=True)
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All V2 table tests passed!")
        print("\n💡 Next steps:")
        print("  1. Update all RAG pipelines to use V2 tables")
        print("  2. Run full performance benchmarks")
        print("  3. Consider removing old VARCHAR columns after validation")
    else:
        print("❌ Some tests failed - please investigate")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)