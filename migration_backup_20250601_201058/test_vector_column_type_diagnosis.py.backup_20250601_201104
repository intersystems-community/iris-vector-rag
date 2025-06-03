#!/usr/bin/env python3
"""
Diagnose the actual VECTOR column type and test different query approaches
"""

import sys
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
import json

def main():
    print("🔍 VECTOR Column Type Diagnosis")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate test embedding
        query = "diabetes treatment"
        query_embedding = embedding_func([query])[0]
        query_embedding_str = ','.join(map(str, query_embedding))
        
        print(f"📊 Query: '{query}'")
        print(f"📊 Embedding dimensions: {len(query_embedding)}")
        
        # Check how the vector is actually stored
        print("\n🧪 Checking how vectors are stored in the table...")
        cursor.execute("""
            SELECT TOP 1 doc_id, 
                   LENGTH(document_embedding_vector) as vec_length,
                   SUBSTRING(document_embedding_vector, 1, 100) as vec_preview
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL
        """)
        
        result = cursor.fetchone()
        if result:
            print(f"📄 Doc ID: {result[0]}")
            print(f"📊 Vector storage length: {result[1]}")
            print(f"📊 Vector preview: {result[2]}...")
            
        # Test 1: Try with JSON array format
        print("\n🧪 Test 1: JSON array format")
        try:
            json_embedding = json.dumps(query_embedding.tolist())
            sql = f"""
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, '{json_embedding}'::VECTOR(DOUBLE, 384)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ JSON format worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ JSON format failed: {e}")
            
        # Test 2: Try with CAST syntax
        print("\n🧪 Test 2: CAST syntax")
        try:
            sql = f"""
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, CAST('{query_embedding_str}' AS VECTOR(DOUBLE, 384))) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ CAST syntax worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ CAST syntax failed: {e}")
            
        # Test 3: Try with escaped string
        print("\n🧪 Test 3: Escaped string format")
        try:
            # Replace any problematic characters
            escaped_embedding = query_embedding_str.replace("'", "''")
            sql = f"""
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{escaped_embedding}', 'DOUBLE')) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Escaped string worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ Escaped string failed: {e}")
            
        # Test 4: Try with parameter binding
        print("\n🧪 Test 4: Parameter binding")
        try:
            sql = """
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE')) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql, (query_embedding_str,))
            results = cursor.fetchall()
            print(f"✅ Parameter binding worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ Parameter binding failed: {e}")
            
        # Test 5: Try without TO_VECTOR (direct vector comparison)
        print("\n🧪 Test 5: Direct vector string")
        try:
            # Get a sample vector from the table to see its format
            cursor.execute("SELECT TOP 1 document_embedding_vector FROM RAG.SourceDocuments_V2 WHERE document_embedding_vector IS NOT NULL")
            sample_vec = cursor.fetchone()[0]
            
            # Use the same format
            sql = f"""
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, '{sample_vec}') AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Direct vector string worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ Direct vector string failed: {e}")
            
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()