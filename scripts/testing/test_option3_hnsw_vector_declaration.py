#!/usr/bin/env python3
"""
Test Option 3: Try declaring the vector differently
"""

import sys
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
import json

def main():
    print("🔍 Testing HNSW Vector Declaration Options")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate test embedding
        query = "diabetes treatment"
        query_embedding = embedding_func([query])[0]
        
        print(f"📊 Query: '{query}'")
        print(f"📊 Embedding dimensions: {len(query_embedding)}")
        
        # Option 1: Try with JSON array format
        print("\n🧪 Option 1: JSON array format with VECTOR_FROM_JSON")
        try:
            json_embedding = json.dumps(query_embedding.tolist())
            sql = f"""
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, VECTOR_FROM_JSON('{json_embedding}')) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ VECTOR_FROM_JSON worked! Got {len(results)} results")
            if results:
                print(f"   First result: {results[0][0]}, score: {results[0][1]:.4f}")
        except Exception as e:
            print(f"❌ VECTOR_FROM_JSON failed: {e}")
            
        # Option 2: Try with array literal syntax
        print("\n🧪 Option 2: Array literal syntax")
        try:
            # Format as array literal
            array_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            sql = f"""
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, '{array_str}'::VECTOR(DOUBLE, 384)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Array literal worked! Got {len(results)} results")
            if results:
                print(f"   First result: {results[0][0]}, score: {results[0][1]:.4f}")
        except Exception as e:
            print(f"❌ Array literal failed: {e}")
            
        # Option 3: Try without any quotes (direct vector)
        print("\n🧪 Option 3: Direct vector without quotes")
        try:
            # Get a sample vector to see the exact format
            cursor.execute("SELECT TOP 1 doc_id, document_embedding_vector FROM RAG.SourceDocuments_V2 WHERE document_embedding_vector IS NOT NULL")
            sample_doc_id, sample_vec = cursor.fetchone()
            print(f"📄 Sample doc_id: {sample_doc_id}")
            
            # Try to use it in a query
            sql = f"""
                SELECT doc_id,
                VECTOR_COSINE(document_embedding_vector, document_embedding_vector) AS self_score
                FROM RAG.SourceDocuments_V2
                WHERE doc_id = '{sample_doc_id}'
            """
            cursor.execute(sql)
            result = cursor.fetchone()
            if result:
                print(f"✅ Self-similarity test worked! Score: {result[1]:.4f} (should be 1.0)")
            
        except Exception as e:
            print(f"❌ Direct vector test failed: {e}")
            
        # Option 4: Try with parameter binding and different formats
        print("\n🧪 Option 4: Parameter binding with different formats")
        
        # Try CSV format with parameter
        try:
            csv_embedding = ','.join(map(str, query_embedding.tolist()))
            sql = """
                SELECT TOP 5 doc_id,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE', 384)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql, (csv_embedding,))
            results = cursor.fetchall()
            print(f"✅ Parameter with dimensions worked! Got {len(results)} results")
            if results:
                print(f"   First result: {results[0][0]}, score: {results[0][1]:.4f}")
        except Exception as e:
            print(f"❌ Parameter with dimensions failed: {e}")
            
        # Option 5: Try the simplest possible query
        print("\n🧪 Option 5: Simplest possible vector query")
        try:
            sql = """
                SELECT TOP 5 doc_id
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY VECTOR_COSINE(document_embedding_vector, 
                    (SELECT document_embedding_vector FROM RAG.SourceDocuments_V2 WHERE document_embedding_vector IS NOT NULL LIMIT 1)) DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Subquery approach worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ Subquery approach failed: {e}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()