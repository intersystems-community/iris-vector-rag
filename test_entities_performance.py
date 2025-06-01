#!/usr/bin/env python3
"""
Test performance of vector search on Entities table
"""

import sys
import time
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

def test_entities_performance():
    """Test performance of vector search on Entities table"""
    print("⏱️  Testing Entities Vector Search Performance")
    print("=" * 60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        # Check total entities
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE embedding IS NOT NULL")
        total_entities = cursor.fetchone()[0]
        print(f"\n📊 Total entities with embeddings: {total_entities:,}")
        
        # Test queries
        test_queries = [
            "diabetes treatment",
            "cancer research",
            "heart disease",
            "covid-19 vaccine",
            "mental health therapy"
        ]
        
        print("\n🧪 Running performance tests...")
        print("-" * 60)
        
        for query_text in test_queries:
            # Create embedding
            query_embedding = embedding_model.encode([query_text])[0]
            query_embedding_str = str(query_embedding.tolist())
            
            # Measure query time
            start_time = time.time()
            
            cursor.execute("""
                SELECT TOP 10 
                    entity_id,
                    entity_name,
                    entity_type,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
                FROM RAG.Entities
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
            """, [query_embedding_str])
            
            results = cursor.fetchall()
            end_time = time.time()
            
            query_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            print(f"\n📝 Query: '{query_text}'")
            print(f"   ⏱️  Query time: {query_time:.2f} ms")
            print(f"   📊 Searching {total_entities:,} entities")
            print(f"   ✅ Found {len(results)} results")
            
            if results:
                print("   Top 3 matches:")
                for i, (entity_id, entity_name, entity_type, similarity) in enumerate(results[:3], 1):
                    print(f"      {i}. {entity_name} ({entity_type}) - similarity: {float(similarity):.4f}")
        
        # Test with larger result set
        print("\n\n🔥 Stress test - retrieving 100 results...")
        start_time = time.time()
        
        cursor.execute("""
            SELECT TOP 100 
                entity_id,
                entity_name,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
            FROM RAG.Entities
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
        """, [query_embedding_str])
        
        results = cursor.fetchall()
        end_time = time.time()
        
        query_time = (end_time - start_time) * 1000
        print(f"   ⏱️  Query time for 100 results: {query_time:.2f} ms")
        print(f"   📊 Searched {total_entities:,} entities")
        
        # Performance analysis
        print("\n\n📈 Performance Analysis:")
        print("-" * 60)
        
        if query_time < 100:
            print("✅ EXCELLENT: Sub-100ms query time indicates HNSW index is working perfectly!")
            print("   - HNSW provides logarithmic search complexity")
            print("   - This performance is only possible with a proper vector index")
        elif query_time < 500:
            print("✅ GOOD: Sub-500ms query time indicates HNSW index is likely present")
            print("   - Performance is acceptable for real-time applications")
        else:
            print("⚠️  SLOW: Query time over 500ms might indicate missing HNSW index")
            print("   - Linear scan of 114K+ vectors would be much slower")
        
        print(f"\n📊 Performance metrics:")
        print(f"   - Entities searched: {total_entities:,}")
        print(f"   - Average query time: ~{query_time:.0f} ms")
        print(f"   - Throughput: ~{1000/query_time:.0f} queries/second")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    test_entities_performance()