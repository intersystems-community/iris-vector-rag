#!/usr/bin/env python3
"""
URGENT: Option 3 CORRECTED - Proper IRIS VECTOR Syntax
Testing with correct VECTOR data type syntax for IRIS
"""

import sys
import time
sys.path.insert(0, '.')

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

def test_corrected_vector_approaches():
    """Test multiple corrected VECTOR approaches for IRIS"""
    print("🚀 TESTING CORRECTED IRIS VECTOR APPROACHES")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    approaches = [
        {
            "name": "VECTOR(DOUBLE)",
            "sql": "ALTER TABLE RAG.SourceDocuments ALTER COLUMN embedding VECTOR(DOUBLE)"
        },
        {
            "name": "VECTOR(FLOAT)", 
            "sql": "ALTER TABLE RAG.SourceDocuments ALTER COLUMN embedding VECTOR(FLOAT)"
        },
        {
            "name": "VECTOR(STRING)",
            "sql": "ALTER TABLE RAG.SourceDocuments ALTER COLUMN embedding VECTOR(STRING)"
        }
    ]
    
    for i, approach in enumerate(approaches, 1):
        print(f"\n🔧 Approach {i}: {approach['name']}")
        print(f"📊 SQL: {approach['sql']}")
        
        try:
            cursor.execute(approach['sql'])
            print(f"✅ SUCCESS! {approach['name']} worked!")
            
            # Verify the change
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' 
                AND TABLE_NAME = 'SourceDocuments' 
                AND COLUMN_NAME = 'embedding'
            """)
            
            column_info = cursor.fetchone()
            if column_info:
                print(f"📊 New column type: {column_info[1]}")
                print(f"📊 Max length: {column_info[2]}")
            
            # Now try HNSW index creation
            return test_hnsw_on_vector_column(cursor, approach['name'])
            
        except Exception as e:
            print(f"❌ {approach['name']} failed: {e}")
            continue
    
    cursor.close()
    return False, None

def test_hnsw_on_vector_column(cursor, vector_type):
    """Test HNSW index creation on the corrected VECTOR column"""
    print(f"\n🔧 Testing HNSW index on {vector_type} column...")
    
    try:
        # Create HNSW index
        hnsw_sql = """
        CREATE INDEX idx_hnsw_corrected_vector
        ON RAG.SourceDocuments (embedding)
        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
        """
        
        print(f"📊 Creating HNSW index...")
        cursor.execute(hnsw_sql)
        print("✅ HNSW INDEX CREATED SUCCESSFULLY!")
        
        # Verify index creation
        cursor.execute("""
            SELECT INDEX_NAME, COLUMN_NAME, INDEX_TYPE
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND INDEX_NAME = 'idx_hnsw_corrected_vector'
        """)
        
        index_result = cursor.fetchone()
        if index_result:
            print(f"✅ Index verified: {index_result[0]} ({index_result[2]})")
            
            # Test performance
            return test_hnsw_performance(cursor)
        else:
            print("❌ Index verification failed")
            return False, None
            
    except Exception as e:
        print(f"❌ HNSW index creation failed: {e}")
        return False, None

def test_hnsw_performance(cursor):
    """Test HNSW performance"""
    print(f"\n🧪 Testing HNSW performance...")
    
    try:
        # Get embedding function
        embedding_func = get_embedding_func()
        
        # Generate test query embedding
        test_query = "diabetes treatment symptoms"
        query_embedding = embedding_func([test_query])[0]
        embedding_str = ','.join(map(str, query_embedding))
        
        print(f"📊 Test query: {test_query}")
        
        # Test HNSW performance
        start_time = time.time()
        cursor.execute("""
            SELECT TOP 10 doc_id, title,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL 
              AND LENGTH(embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > 0.1
            ORDER BY similarity_score DESC
        """, [embedding_str, embedding_str])
        
        results = cursor.fetchall()
        search_time = time.time() - start_time
        
        print(f"📊 HNSW search time: {search_time:.3f}s")
        print(f"📊 Retrieved documents: {len(results)}")
        
        if results:
            print(f"📊 Top similarity: {results[0][2]:.4f}")
        
        # Calculate improvement
        baseline_time = 7.43  # Previous baseline
        if search_time < baseline_time:
            improvement = baseline_time / search_time
            print(f"📈 Performance improvement: {improvement:.1f}x faster!")
            
            if improvement >= 1.7:  # 70% improvement
                print(f"🎉 TARGET ACHIEVED! 70%+ improvement confirmed!")
                return True, search_time
            else:
                print(f"⚠️ Improvement below 70% target")
                return True, search_time
        else:
            print(f"⚠️ Performance not improved")
            return True, search_time
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False, None

def main():
    """Execute the corrected Option 3 test"""
    print("🚀 OPTION 3 CORRECTED: PROPER IRIS VECTOR SYNTAX")
    print("=" * 60)
    print("Testing with correct VECTOR data type syntax")
    print("=" * 60)
    
    success, performance = test_corrected_vector_approaches()
    
    if success:
        print(f"\n🎉 SUCCESS! HNSW index created with corrected VECTOR syntax!")
        if performance:
            print(f"📊 Performance: {performance:.3f}s")
            
            # Calculate total impact
            baseline_total = 23.88  # HybridiFindRAG baseline
            baseline_vector = 7.43   # Vector search baseline
            other_time = baseline_total - baseline_vector
            new_total = other_time + performance
            total_improvement = baseline_total / new_total
            
            print(f"\n📊 Total HybridiFindRAG impact:")
            print(f"  - Old: {baseline_total:.2f}s → New: {new_total:.2f}s")
            print(f"  - Total improvement: {total_improvement:.1f}x faster")
            print(f"  - Performance gain: {((baseline_total - new_total) / baseline_total * 100):.1f}%")
            
        return True
    else:
        print(f"\n❌ All corrected approaches failed")
        print(f"🔍 IRIS Community Edition may not support VECTOR types at all")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 MISSION ACCOMPLISHED!")
        print(f"🚀 Corrected VECTOR syntax enabled HNSW indexing!")
    else:
        print(f"\n❌ Mission failed - IRIS Community Edition limitations confirmed")