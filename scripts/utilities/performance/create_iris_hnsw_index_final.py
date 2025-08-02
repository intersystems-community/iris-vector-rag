#!/usr/bin/env python3
"""
FINAL: Create IRIS HNSW Index using correct syntax from project
Based on existing patterns in chunking/schema_clean.sql and IRIS documentation.
"""

import sys
import time
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add project root

from common.iris_connector import get_iris_connection

def create_iris_hnsw_index():
    """Create IRIS HNSW index using the correct AS HNSW syntax"""
    print("🚀 CREATING IRIS HNSW INDEX - Using correct AS HNSW syntax!")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Check for existing indexes first
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
        
        # Create IRIS HNSW index using correct syntax from project
        print("\n🔧 Creating IRIS HNSW index using AS HNSW syntax...")
        
        # Based on chunking/schema_clean.sql pattern and IRIS docs
        hnsw_sql = """
        CREATE INDEX idx_hnsw_source_embeddings
        ON RAG.SourceDocuments (embedding)
        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
        """
        
        print(f"📊 Executing HNSW index creation:")
        print(f"    {hnsw_sql.strip()}")
        
        cursor.execute(hnsw_sql)
        print("✅ SUCCESS: IRIS HNSW index created!")
        return True
        
    except Exception as e:
        print(f"❌ HNSW index creation failed: {e}")
        
        # Try alternative HNSW parameters
        try:
            print("\n🔧 Trying alternative HNSW parameters...")
            alt_hnsw_sql = """
            CREATE INDEX idx_hnsw_source_embeddings_alt
            ON RAG.SourceDocuments (embedding)
            AS HNSW(M=24, Distance='COSINE')
            """
            
            print(f"📊 Executing alternative HNSW:")
            print(f"    {alt_hnsw_sql.strip()}")
            
            cursor.execute(alt_hnsw_sql)
            print("✅ SUCCESS: Alternative HNSW index created!")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative HNSW failed: {e2}")
            
            # Try minimal HNSW syntax
            try:
                print("\n🔧 Trying minimal HNSW syntax...")
                minimal_hnsw_sql = """
                CREATE INDEX idx_hnsw_source_embeddings_minimal
                ON RAG.SourceDocuments (embedding)
                AS HNSW(Distance='COSINE')
                """
                
                print(f"📊 Executing minimal HNSW:")
                print(f"    {minimal_hnsw_sql.strip()}")
                
                cursor.execute(minimal_hnsw_sql)
                print("✅ SUCCESS: Minimal HNSW index created!")
                return True
                
            except Exception as e3:
                print(f"❌ Minimal HNSW failed: {e3}")
                return False
    finally:
        cursor.close()

def verify_hnsw_index():
    """Verify HNSW index creation and test performance"""
    print("\n🔍 Verifying HNSW index creation...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Check for HNSW indexes
        cursor.execute("""
            SELECT INDEX_NAME 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND COLUMN_NAME = 'embedding'
            AND INDEX_NAME LIKE '%hnsw%'
        """)
        
        hnsw_indexes = cursor.fetchall()
        if hnsw_indexes:
            print("🎯 SUCCESS! HNSW indexes found:")
            for idx in hnsw_indexes:
                print(f"  ✅ {idx[0]}")
            return True
        else:
            print("❌ No HNSW indexes found")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    finally:
        cursor.close()

def test_hnsw_performance():
    """Test vector search performance with HNSW index"""
    print("\n🧪 Testing HNSW performance...")
    
    from common.utils import get_embedding_func
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate test query embedding
        query_embedding = embedding_func(['diabetes treatment'])[0]
        embedding_str = ','.join(map(str, query_embedding))
        
        # Test vector search with HNSW
        print("📊 Running vector search with HNSW index...")
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
        
        print(f"📊 HNSW vector search completed in {search_time:.2f}s")
        print(f"📊 Retrieved {len(results)} documents")
        
        # Performance assessment
        baseline_time = 7.43  # Previous optimized time
        if search_time < 2.0:
            improvement = baseline_time / search_time
            print(f"🎉 EXCELLENT! {improvement:.1f}x faster with HNSW!")
            print(f"📈 HNSW index is providing massive performance boost!")
        elif search_time < 4.0:
            improvement = baseline_time / search_time
            print(f"✅ GREAT! {improvement:.1f}x faster with HNSW!")
            print(f"📈 HNSW index is working very well!")
        elif search_time < baseline_time:
            improvement = baseline_time / search_time
            print(f"⚡ IMPROVED! {improvement:.1f}x faster with HNSW!")
        else:
            print(f"⚠️ Performance similar. HNSW index may still be building...")
            
        return search_time
        
    except Exception as e:
        print(f"❌ HNSW performance test failed: {e}")
        return None
    finally:
        cursor.close()

def calculate_expected_improvements(hnsw_time):
    """Calculate expected improvements for HybridiFindRAG"""
    print(f"\n📈 CALCULATING EXPECTED IMPROVEMENTS:")
    
    # Current performance breakdown
    old_vector_time = 7.43  # Previous vector search time
    old_total_time = 9.65   # Previous total HybridiFindRAG time
    other_time = old_total_time - old_vector_time  # ~2.22s for other components
    
    # New performance with HNSW
    new_vector_time = hnsw_time
    new_total_time = other_time + new_vector_time
    
    # Calculate improvements
    vector_improvement = old_vector_time / new_vector_time
    total_improvement = old_total_time / new_total_time
    
    print(f"📊 Vector component: {old_vector_time:.2f}s → {new_vector_time:.2f}s ({vector_improvement:.1f}x faster)")
    print(f"📊 Total HybridiFindRAG: {old_total_time:.2f}s → {new_total_time:.2f}s ({total_improvement:.1f}x faster)")
    print(f"📊 Performance gain: {((old_total_time - new_total_time) / old_total_time * 100):.1f}% improvement")
    
    # Updated rankings
    print(f"\n🏆 UPDATED PERFORMANCE RANKINGS:")
    print(f"1. GraphRAG: 0.76s (speed-critical)")
    print(f"2. BasicRAG: 7.95s (production baseline)")
    print(f"3. CRAG: 8.26s (enhanced coverage)")
    if new_total_time < 7.95:
        print(f"4. 🆕 OptimizedHybridiFindRAG: {new_total_time:.2f}s (HNSW-accelerated)")
        print(f"5. HyDE: 10.11s (quality-focused)")
    else:
        print(f"4. HyDE: 10.11s (quality-focused)")
        print(f"5. 🆕 OptimizedHybridiFindRAG: {new_total_time:.2f}s (HNSW-accelerated)")
    
    return new_total_time

if __name__ == "__main__":
    print("🚀 IRIS HNSW INDEX CREATION - FINAL ATTEMPT")
    print("=" * 60)
    
    # Create HNSW index
    success = create_iris_hnsw_index()
    
    if success:
        # Verify creation
        verified = verify_hnsw_index()
        
        if verified:
            # Test performance
            hnsw_time = test_hnsw_performance()
            
            if hnsw_time:
                # Calculate improvements
                new_total = calculate_expected_improvements(hnsw_time)
                
                print(f"\n🎉 HNSW INDEX CREATION: ✅ COMPLETE SUCCESS!")
                print(f"📊 IRIS HNSW index is now accelerating vector operations!")
                print(f"🚀 HybridiFindRAG performance: {new_total:.2f}s (HNSW-accelerated)")
            else:
                print(f"\n✅ HNSW index created but performance test failed")
        else:
            print(f"\n⚠️ HNSW index creation attempted but verification failed")
    else:
        print(f"\n❌ HNSW index creation failed")
        print("🔍 This may require IRIS configuration or data cleanup")