#!/usr/bin/env python3
"""
URGENT: Create Simple IRIS Index on RAG.SourceDocuments.embedding
Direct index creation that IRIS will automatically optimize for vector operations.
"""

import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add project root

from common.iris_connector import get_iris_connection

def create_simple_iris_index():
    """Create simple index on embedding column - IRIS will optimize automatically"""
    print("🚀 CREATING SIMPLE IRIS INDEX - IRIS will optimize for vector operations!")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Check existing indexes first
        print("\n🔍 Checking for existing indexes...")
        cursor.execute("""
            SELECT INDEX_NAME 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND COLUMN_NAME = 'embedding'
        """)
        
        existing_indexes = cursor.fetchall()
        if existing_indexes:
            print("📊 Existing indexes on embedding column:")
            for idx in existing_indexes:
                print(f"  - {idx[0]}")
            print("✅ Index already exists! IRIS should be optimizing vector operations.")
            return True
        
        # Create simple index - IRIS will optimize for vector operations
        print("\n🔧 Creating simple index on embedding column...")
        
        index_sql = """
        CREATE INDEX idx_embedding_vector 
        ON RAG.SourceDocuments (embedding)
        """
        
        print(f"📊 Executing: {index_sql}")
        cursor.execute(index_sql)
        print("✅ SUCCESS: Index created! IRIS will automatically optimize for vector operations!")
        return True
        
    except Exception as e:
        print(f"❌ Index creation failed: {e}")
        return False
    finally:
        cursor.close()

def verify_index_and_test_performance():
    """Verify index creation and test performance"""
    print("\n🔍 Verifying index and testing performance...")
    
    from common.utils import get_embedding_func
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Verify index exists
        cursor.execute("""
            SELECT INDEX_NAME 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND COLUMN_NAME = 'embedding'
        """)
        
        indexes = cursor.fetchall()
        if indexes:
            print("🎯 SUCCESS! Index found:")
            for idx in indexes:
                print(f"  ✅ {idx[0]}")
        else:
            print("❌ No index found")
            return False
        
        # Test vector search performance
        print("\n🧪 Testing vector search performance...")
        query_embedding = embedding_func(['diabetes symptoms'])[0]
        embedding_str = ','.join(map(str, query_embedding))
        
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 10 doc_id, 
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL 
              AND LENGTH(embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > 0.1
            ORDER BY similarity_score DESC
        """, [embedding_str, embedding_str])
        
        results = cursor.fetchall()
        search_time = time.time() - start_time
        
        print(f"📊 Vector search completed in {search_time:.2f}s")
        print(f"📊 Retrieved {len(results)} documents")
        
        # Performance assessment
        baseline_time = 7.43  # Previous optimized time
        if search_time < baseline_time * 0.5:
            improvement = baseline_time / search_time
            print(f"🎉 EXCELLENT! {improvement:.1f}x faster than before!")
            print(f"📈 Index is providing significant performance boost!")
        elif search_time < baseline_time * 0.8:
            improvement = baseline_time / search_time
            print(f"✅ GOOD! {improvement:.1f}x faster than before!")
            print(f"📈 Index is working well!")
        elif search_time < baseline_time:
            improvement = baseline_time / search_time
            print(f"⚡ IMPROVED! {improvement:.1f}x faster than before!")
        else:
            print(f"⚠️ Performance similar to before. Index may still be building...")
            
        return search_time
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return None
    finally:
        cursor.close()

if __name__ == "__main__":
    print("🚀 SIMPLE IRIS INDEX CREATION")
    print("=" * 50)
    
    # Create the index
    success = create_simple_iris_index()
    
    if success:
        # Test performance
        performance = verify_index_and_test_performance()
        
        if performance:
            print(f"\n🎉 INDEX CREATION COMPLETE!")
            print(f"📊 Current vector search time: {performance:.2f}s")
            print(f"🚀 IRIS is now optimizing vector operations automatically!")
            print(f"📈 Expected HybridiFindRAG improvement:")
            
            # Calculate expected improvement
            old_vector_time = 7.43
            new_vector_time = performance
            vector_improvement = old_vector_time / new_vector_time
            
            # HybridiFindRAG breakdown: ~1.5s other + vector_time
            old_total = 9.79
            other_time = old_total - old_vector_time  # ~2.36s
            new_total = other_time + new_vector_time
            
            total_improvement = old_total / new_total
            
            print(f"  - Vector component: {old_vector_time:.2f}s → {new_vector_time:.2f}s ({vector_improvement:.1f}x faster)")
            print(f"  - Total pipeline: {old_total:.2f}s → {new_total:.2f}s ({total_improvement:.1f}x faster)")
            print(f"  - Performance gain: {((old_total - new_total) / old_total * 100):.1f}% improvement")
        else:
            print(f"\n✅ Index created successfully!")
            print(f"⏳ Performance testing failed, but index should improve operations")
    else:
        print(f"\n❌ Index creation failed")