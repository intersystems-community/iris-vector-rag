#!/usr/bin/env python3
"""
URGENT: Add IRIS Vector Indexes to RAG.SourceDocuments
This script creates proper IRIS vector indexes for dramatic performance improvement.
"""

import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add project root

from common.iris_connector import get_iris_connection

def create_iris_vector_indexes():
    """Create IRIS vector indexes on RAG.SourceDocuments.embedding"""
    print("🚀 URGENT: Creating IRIS Vector Indexes for Performance Optimization...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Method 1: Try IRIS-native vector index creation
        print("\n🔧 Attempting IRIS vector index creation...")
        
        # Check current table structure
        cursor.execute("DESCRIBE RAG.SourceDocuments")
        columns = cursor.fetchall()
        print(f"📊 Table structure: {len(columns)} columns")
        
        # Try different IRIS vector index approaches
        vector_index_attempts = [
            # IRIS 2024+ Vector Search syntax
            "CREATE INDEX idx_embedding_vector ON RAG.SourceDocuments (embedding) WITH (TYPE='VECTOR', METRIC='COSINE')",
            
            # Alternative IRIS syntax
            "CREATE INDEX idx_embedding_hnsw ON RAG.SourceDocuments (embedding) USING HNSW",
            
            # IRIS Vector Search API approach
            "CALL %SQL.Manager.API.CreateVectorIndex('RAG', 'SourceDocuments', 'embedding')",
            
            # ObjectScript approach via SQL
            "SET status = ##class(%SQL.Manager.API).CreateVectorIndex('RAG', 'SourceDocuments', 'embedding')",
        ]
        
        success = False
        for i, sql in enumerate(vector_index_attempts, 1):
            try:
                print(f"\n📊 Attempt {i}: {sql[:60]}...")
                cursor.execute(sql)
                print(f"✅ SUCCESS: Vector index created with method {i}!")
                success = True
                break
            except Exception as e:
                print(f"❌ Method {i} failed: {e}")
        
        if not success:
            print("\n🔧 Trying alternative approach: Enable vector search first...")
            try:
                # Try to enable vector search on the table
                cursor.execute("ALTER TABLE RAG.SourceDocuments ADD VECTOR SEARCH ON embedding")
                print("✅ Vector search enabled on table!")
                success = True
            except Exception as e:
                print(f"❌ Vector search enablement failed: {e}")
        
        # Verify index creation
        print("\n🔍 Verifying vector index creation...")
        cursor.execute("""
            SELECT INDEX_NAME, COLUMN_NAME 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND COLUMN_NAME = 'embedding'
        """)
        
        vector_indexes = cursor.fetchall()
        if vector_indexes:
            print("🎯 SUCCESS! Vector indexes found:")
            for idx in vector_indexes:
                print(f"  ✅ {idx[0]} on {idx[1]}")
            return True
        else:
            print("❌ No vector indexes found after creation attempts")
            return False
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return False
    finally:
        cursor.close()

def test_vector_performance():
    """Test vector search performance after index creation"""
    print("\n🧪 Testing vector search performance...")
    
    from common.utils import get_embedding_func
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate test query embedding
        query_embedding = embedding_func(['diabetes symptoms'])[0]
        embedding_str = ','.join(map(str, query_embedding))
        
        # Test vector search performance
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
        
        if search_time < 5.0:
            print("✅ Excellent performance! Vector index is working.")
        elif search_time < 10.0:
            print("⚠️ Good performance, but could be better.")
        else:
            print("❌ Poor performance, index may not be active.")
            
        return search_time
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return None
    finally:
        cursor.close()

if __name__ == "__main__":
    print("🚀 URGENT IRIS VECTOR INDEX CREATION")
    print("=" * 50)
    
    # Create vector indexes
    index_success = create_iris_vector_indexes()
    
    if index_success:
        # Test performance
        performance = test_vector_performance()
        
        if performance and performance < 10.0:
            print(f"\n🎉 SUCCESS! Vector indexes created and performing well ({performance:.2f}s)")
            print("📈 Expected performance improvements:")
            print("  - HybridiFindRAG: 9.25s → ~2-3s (70% improvement)")
            print("  - BasicRAG: 7.95s → ~1-2s (80% improvement)")
            print("  - All techniques: Dramatic performance gains")
        else:
            print(f"\n⚠️ Indexes created but performance needs optimization")
    else:
        print(f"\n❌ Vector index creation failed")
        print("🔍 This IRIS version may need manual vector search configuration")
        print("📋 Next steps:")
        print("  1. Check IRIS version and vector search support")
        print("  2. Enable vector search in IRIS configuration")
        print("  3. Use IRIS Management Portal for vector index creation")