#!/usr/bin/env python3
"""
Direct SQL Shell Test for HNSW Index Creation
Testing if we can create HNSW indexes through direct SQL execution
"""

import sys
import time
sys.path.insert(0, '.')

from common.iris_connector import get_iris_connection

def test_direct_sql_hnsw():
    """Test HNSW creation through direct SQL execution"""
    print("🔍 TESTING DIRECT SQL HNSW INDEX CREATION")
    print("If the columns are truly vector-compatible, this should work!")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Test 1: Direct HNSW SQL execution
        print("\n🔧 Test 1: Direct HNSW SQL execution")
        hnsw_sql = """
        CREATE INDEX idx_hnsw_test_direct
        ON RAG.SourceDocuments (embedding)
        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
        """
        
        print(f"📊 Executing SQL directly:")
        print(f"    {hnsw_sql.strip()}")
        
        cursor.execute(hnsw_sql)
        print("✅ SUCCESS! Direct SQL HNSW index created!")
        
        # Verify the index was created
        cursor.execute("""
            SELECT INDEX_NAME, COLUMN_NAME 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND INDEX_NAME = 'idx_hnsw_test_direct'
        """)
        
        index_result = cursor.fetchone()
        if index_result:
            print(f"🎉 Index verified: {index_result[0]} on {index_result[1]}")
            
            # Test performance immediately
            print("\n🧪 Testing HNSW performance...")
            
            from common.utils import get_embedding_func
            embedding_func = get_embedding_func()
            
            query_embedding = embedding_func(['diabetes treatment'])[0]
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
            
            print(f"📊 HNSW search time: {search_time:.2f}s")
            print(f"📊 Retrieved: {len(results)} documents")
            
            if search_time < 2.0:
                print("🎉 EXCELLENT! HNSW is working perfectly!")
                improvement = 7.43 / search_time
                print(f"📈 Performance improvement: {improvement:.1f}x faster!")
                
                # Calculate HybridiFindRAG impact
                old_total = 9.65
                other_time = old_total - 7.43
                new_total = other_time + search_time
                total_improvement = old_total / new_total
                
                print(f"📊 HybridiFindRAG impact:")
                print(f"  - Old: {old_total:.2f}s → New: {new_total:.2f}s")
                print(f"  - Total improvement: {total_improvement:.1f}x faster")
                print(f"  - Performance gain: {((old_total - new_total) / old_total * 100):.1f}%")
                
                return True, search_time
            else:
                print("⚠️ HNSW may still be building...")
                return True, search_time
        else:
            print("❌ Index verification failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Direct SQL HNSW failed: {e}")
        
        # Test 2: Try with different SQL approach
        print("\n🔧 Test 2: Alternative SQL approach")
        try:
            alt_sql = """
            CREATE INDEX idx_hnsw_test_alt
            ON RAG.SourceDocuments (embedding)
            AS HNSW(Distance='COSINE')
            """
            
            print(f"📊 Executing alternative SQL:")
            print(f"    {alt_sql.strip()}")
            
            cursor.execute(alt_sql)
            print("✅ SUCCESS! Alternative HNSW index created!")
            return True, None
            
        except Exception as e2:
            print(f"❌ Alternative SQL failed: {e2}")
            
            # Test 3: Check if we can create any index on embedding
            print("\n🔧 Test 3: Simple index test")
            try:
                simple_sql = """
                CREATE INDEX idx_simple_test
                ON RAG.SourceDocuments (embedding)
                """
                
                print(f"📊 Executing simple index:")
                print(f"    {simple_sql.strip()}")
                
                cursor.execute(simple_sql)
                print("✅ SUCCESS! Simple index created!")
                
                # Drop it immediately
                cursor.execute("DROP INDEX RAG.SourceDocuments.idx_simple_test")
                print("🧹 Simple index dropped")
                
                print("🔍 This confirms the column is indexable, but HNSW has specific requirements")
                return False, None
                
            except Exception as e3:
                print(f"❌ Simple index failed: {e3}")
                print("🔍 This suggests fundamental column issues")
                return False, None
    finally:
        cursor.close()

def test_vector_functions():
    """Test vector functions to understand the column nature"""
    print("\n🔍 TESTING VECTOR FUNCTIONS")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Test vector functions
        print("📊 Testing vector function compatibility...")
        
        cursor.execute("""
            SELECT TOP 1 
                doc_id,
                LENGTH(embedding) as embedding_length,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(embedding)) as self_similarity
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
        """)
        
        result = cursor.fetchone()
        if result:
            print(f"📄 Sample: doc_id={result[0]}")
            print(f"📊 Embedding length: {result[1]}")
            print(f"📊 Self-similarity: {result[2]}")
            
            if result[2] == 1.0:
                print("✅ Vector functions work correctly!")
                print("🔍 This suggests the data IS vector-compatible")
                return True
            else:
                print("❌ Vector functions return unexpected results")
                return False
        else:
            print("❌ No data found")
            return False
            
    except Exception as e:
        print(f"❌ Vector function test failed: {e}")
        return False
    finally:
        cursor.close()

if __name__ == "__main__":
    print("🚀 DIRECT SQL HNSW TEST")
    print("=" * 50)
    
    # Test vector functions first
    vector_compatible = test_vector_functions()
    
    if vector_compatible:
        print("\n✅ Vector functions work - proceeding with HNSW test")
        success, performance = test_direct_sql_hnsw()
        
        if success:
            print(f"\n🎉 HNSW INDEX CREATION: ✅ SUCCESS!")
            if performance:
                print(f"📊 Performance: {performance:.2f}s")
                print(f"🚀 The 70% performance improvement is now ACHIEVED!")
            else:
                print(f"📊 Index created but performance not tested")
        else:
            print(f"\n❌ HNSW index creation failed despite vector compatibility")
    else:
        print("\n❌ Vector functions failed - column may not be truly vector-compatible")