#!/usr/bin/env python3
"""
Compare performance between old Entities (VARCHAR) and new Entities_V2 (VECTOR with HNSW)
"""

import sys
import time
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

def test_performance_comparison():
    """Compare query performance between old and new Entities tables"""
    print("⚡ Entities Performance Comparison: VARCHAR vs VECTOR+HNSW")
    print("=" * 60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test queries
        test_queries = [
            "diabetes treatment",
            "cancer research", 
            "heart disease",
            "covid-19 vaccine",
            "mental health"
        ]
        
        print("\n📊 Table Information:")
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE embedding IS NOT NULL")
        old_count = cursor.fetchone()[0]
        print(f"   RAG.Entities (VARCHAR): {old_count:,} rows")
        
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities_V2 WHERE embedding IS NOT NULL")
        new_count = cursor.fetchone()[0]
        print(f"   RAG.Entities_V2 (VECTOR): {new_count:,} rows")
        
        # Check for HNSW index
        cursor.execute("""
            SELECT Name FROM %Dictionary.CompiledIndex 
            WHERE Parent = 'RAG.Entities_V2' AND Name LIKE '%hnsw%'
        """)
        hnsw_indexes = cursor.fetchall()
        if hnsw_indexes:
            print(f"   ✅ HNSW indexes on Entities_V2: {[idx[0] for idx in hnsw_indexes]}")
        else:
            print("   ⚠️  No HNSW index found on Entities_V2")
        
        print("\n🏁 Running Performance Tests...")
        print("-" * 60)
        
        old_times = []
        new_times = []
        
        for query_text in test_queries:
            # Create embedding
            query_embedding = embedding_model.encode([query_text])[0]
            query_embedding_str = str(query_embedding.tolist())
            
            print(f"\n📝 Query: '{query_text}'")
            
            # Test OLD table (VARCHAR with TO_VECTOR)
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
            
            old_results = cursor.fetchall()
            old_time = (time.time() - start_time) * 1000
            old_times.append(old_time)
            
            # Test NEW table (VECTOR with HNSW)
            start_time = time.time()
            cursor.execute("""
                SELECT TOP 10
                    entity_id,
                    entity_name,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
                FROM RAG.Entities_V2
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
            """, [query_embedding_str])
            
            new_results = cursor.fetchall()
            new_time = (time.time() - start_time) * 1000
            new_times.append(new_time)
            
            print(f"   Old table (VARCHAR): {old_time:>8.2f} ms")
            print(f"   New table (VECTOR):  {new_time:>8.2f} ms")
            print(f"   🎯 Speedup: {old_time/new_time:.1f}x faster")
            
            # Show top result
            if new_results:
                entity_name = new_results[0][1]
                similarity = float(new_results[0][2])
                print(f"   Top match: {entity_name} (similarity: {similarity:.4f})")
        
        # Summary statistics
        avg_old = sum(old_times) / len(old_times)
        avg_new = sum(new_times) / len(new_times)
        
        print("\n\n📈 Performance Summary:")
        print("=" * 60)
        print(f"Average query times across {len(test_queries)} queries:")
        print(f"   Old table (VARCHAR): {avg_old:>8.2f} ms")
        print(f"   New table (VECTOR):  {avg_new:>8.2f} ms")
        print(f"   🚀 Average speedup:  {avg_old/avg_new:.1f}x faster")
        
        print("\n📊 Performance Analysis:")
        if avg_new < 100:
            print("   ✅ EXCELLENT: Sub-100ms average query time!")
            print("   → HNSW index is working perfectly")
            print("   → Ready for production use")
        elif avg_new < 500:
            print("   ✅ GOOD: Sub-500ms average query time")
            print("   → Acceptable for most applications")
        else:
            print("   ⚠️  SLOW: Performance needs investigation")
        
        print(f"\n💡 Recommendation:")
        if avg_new < avg_old / 5:  # At least 5x faster
            print("   The new VECTOR table with HNSW index provides significant performance gains.")
            print("   Consider migrating GraphRAG to use Entities_V2 for production.")
        else:
            print("   Performance improvement is modest. Check if HNSW index is properly created.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    test_performance_comparison()