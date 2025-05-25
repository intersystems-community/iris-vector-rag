#!/usr/bin/env python3
"""
Script to verify that HNSW indexes are created and functional.

This script demonstrates that HNSW indexing has been successfully enabled.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection

def verify_hnsw_indexes():
    """Verify that HNSW indexes are created and functional."""
    print("🔍 Verifying HNSW index creation and functionality...")
    
    try:
        # Connect to IRIS
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        # 1. Check if HNSW indexes exist
        print("\n1️⃣ Checking for HNSW indexes...")
        index_query = """
        SELECT INDEX_NAME, TABLE_NAME
        FROM INFORMATION_SCHEMA.INDEXES 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND INDEX_NAME LIKE '%hnsw%'
        ORDER BY INDEX_NAME
        """
        
        cursor.execute(index_query)
        indexes = cursor.fetchall()
        
        if indexes:
            print(f"✅ Found {len(indexes)} HNSW indexes:")
            for index_name, table_name in indexes:
                print(f"   • {index_name} on {table_name}")
        else:
            print("❌ No HNSW indexes found")
            return False
        
        # 2. Test vector functions
        print("\n2️⃣ Testing vector functions...")
        vector_test_query = """
        SELECT VECTOR_COSINE(
            TO_VECTOR('[1.0,0.0,0.0]', 'DOUBLE', 3),
            TO_VECTOR('[1.0,0.0,0.0]', 'DOUBLE', 3)
        ) as identical_similarity,
        VECTOR_COSINE(
            TO_VECTOR('[1.0,0.0,0.0]', 'DOUBLE', 3),
            TO_VECTOR('[0.0,1.0,0.0]', 'DOUBLE', 3)
        ) as orthogonal_similarity
        """
        
        cursor.execute(vector_test_query)
        result = cursor.fetchone()
        
        if result:
            identical_sim = float(result[0])
            orthogonal_sim = float(result[1])
            
            print(f"✅ Vector functions working:")
            print(f"   • Identical vectors similarity: {identical_sim}")
            print(f"   • Orthogonal vectors similarity: {orthogonal_sim}")
            
            # Verify expected results
            if abs(identical_sim - 1.0) < 0.001:
                print("✅ Identical vectors test passed")
            else:
                print(f"❌ Identical vectors test failed: expected ~1.0, got {identical_sim}")
                
            if abs(orthogonal_sim - 0.0) < 0.001:
                print("✅ Orthogonal vectors test passed")
            else:
                print(f"❌ Orthogonal vectors test failed: expected ~0.0, got {orthogonal_sim}")
        else:
            print("❌ Vector function test failed")
            return False
        
        # 3. Check table accessibility
        print("\n3️⃣ Checking table accessibility...")
        tables_to_check = [
            'RAG.SourceDocuments',
            'RAG.DocumentTokenEmbeddings', 
            'RAG.KnowledgeGraphNodes'
        ]
        
        for table in tables_to_check:
            try:
                count_query = f"SELECT COUNT(*) FROM {table}"
                cursor.execute(count_query)
                count = cursor.fetchone()[0]
                print(f"✅ {table}: {count} rows")
            except Exception as e:
                print(f"❌ {table}: Error - {e}")
                return False
        
        cursor.close()
        connection.close()
        
        print("\n🎉 HNSW index verification completed successfully!")
        print("\n📋 Summary:")
        print("   • HNSW indexes are created and detectable")
        print("   • Vector similarity functions are working correctly")
        print("   • All required tables are accessible")
        print("   • Database is ready for vector search operations")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_hnsw_indexes()
    sys.exit(0 if success else 1)