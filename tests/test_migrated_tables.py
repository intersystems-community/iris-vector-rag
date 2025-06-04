#!/usr/bin/env python3
"""
Test the migrated VECTOR tables
"""

import sys
import time
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

def test_migrated_tables():
    """Test performance of migrated tables"""
    print("🧪 Testing Migrated VECTOR Tables")
    print("=" * 60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create test query
        query_text = "diabetes treatment"
        query_embedding = embedding_model.encode([query_text])[0]
        query_embedding_str = str(query_embedding.tolist())
        
        print(f"\n📝 Test query: '{query_text}'")
        
        # Test 1: Entities_V2 performance
        print("\n1️⃣ Testing RAG.Entities_V2 (VECTOR type)...")
        
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities_V2 WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"   Total entities: {count:,}")
        
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 10 
                entity_id,
                entity_name,
                VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
            FROM RAG.Entities_V2
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
        """, [query_embedding_str])
        
        results = cursor.fetchall()
        elapsed = (time.time() - start_time) * 1000
        
        print(f"   ⏱️  Query time: {elapsed:.2f} ms")
        print(f"   ✅ Found {len(results)} results")
        
        if elapsed < 100:
            print("   🚀 EXCELLENT: HNSW index is working!")
        elif elapsed < 500:
            print("   ✅ GOOD: Reasonable performance")
        else:
            print("   ⚠️  SLOW: Performance needs improvement")
        
        # Test 2: SourceDocuments_V3 performance
        print("\n2️⃣ Testing RAG.SourceDocuments_V3 (VECTOR type)...")
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V3 WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"   Total documents: {count:,}")
        
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 10 
                doc_id,
                title,
                VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
            FROM RAG.SourceDocuments_V3
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
        """, [query_embedding_str])
        
        results = cursor.fetchall()
        elapsed = (time.time() - start_time) * 1000
        
        print(f"   ⏱️  Query time: {elapsed:.2f} ms")
        print(f"   ✅ Found {len(results)} results")
        
        if elapsed < 100:
            print("   🚀 EXCELLENT: HNSW index is working!")
        elif elapsed < 500:
            print("   ✅ GOOD: Reasonable performance")
        else:
            print("   ⚠️  SLOW: Performance needs improvement")
        
        # Compare with old tables
        print("\n3️⃣ Comparing with old VARCHAR tables...")
        
        # Test old Entities table
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 10 
                entity_id,
                entity_name,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
            FROM RAG.Entities
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
        """, [query_embedding_str])
        
        results = cursor.fetchall()
        old_elapsed = (time.time() - start_time) * 1000
        
        print(f"   Old Entities query time: {old_elapsed:.2f} ms")
        print(f"   New Entities_V2 query time: {elapsed:.2f} ms")
        print(f"   🎯 Performance improvement: {old_elapsed/elapsed:.1f}x faster")
        
        print("\n✅ Migration test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    test_migrated_tables()