#!/usr/bin/env python3
"""
Test parameter binding approach for IRIS vector search
"""

import sys
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

def main():
    print("🔍 Testing Parameter Binding for Vector Search")
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
        print(f"📊 Embedding string contains colons: {':' in query_embedding_str}")
        
        # Test 1: Current approach (direct embedding) - we know this fails with colons
        print("\n🧪 Test 1: Direct embedding (current approach)")
        try:
            sql = f"""
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'FLOAT')) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Direct embedding worked! Got {len(results)} results")
        except Exception as e:
            print(f"❌ Direct embedding failed (expected): {str(e)[:100]}...")
            
        # Test 2: Using parameter binding with TO_VECTOR
        print("\n🧪 Test 2: Parameter binding with TO_VECTOR")
        try:
            sql = """
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'FLOAT', 384)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql, (query_embedding_str,))
            results = cursor.fetchall()
            print(f"✅ Parameter binding worked! Got {len(results)} results")
            for i, (doc_id, title, score) in enumerate(results[:3]):
                print(f"   {i+1}. {doc_id}: {title[:50]}... (score: {score:.4f})")
        except Exception as e:
            print(f"❌ Parameter binding failed: {e}")
            
        # Test 3: Using CAST with parameter binding
        print("\n🧪 Test 3: CAST with parameter binding")
        try:
            sql = """
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, CAST(? AS VECTOR(FLOAT, 384))) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql, (query_embedding_str,))
            results = cursor.fetchall()
            print(f"✅ CAST with parameter worked! Got {len(results)} results")
            for i, (doc_id, title, score) in enumerate(results[:3]):
                print(f"   {i+1}. {doc_id}: {title[:50]}... (score: {score:.4f})")
        except Exception as e:
            print(f"❌ CAST with parameter failed: {e}")
            
        # Test 4: Using escaped quotes
        print("\n🧪 Test 4: Escaped quotes approach")
        try:
            # Escape single quotes in the embedding string
            escaped_embedding = query_embedding_str.replace("'", "''")
            sql = f"""
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{escaped_embedding}', 'FLOAT', 384)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Escaped quotes worked! Got {len(results)} results")
            for i, (doc_id, title, score) in enumerate(results[:3]):
                print(f"   {i+1}. {doc_id}: {title[:50]}... (score: {score:.4f})")
        except Exception as e:
            print(f"❌ Escaped quotes failed: {e}")
            
        # Test 5: Check if we can use JSON format
        print("\n🧪 Test 5: JSON format with parameter binding")
        try:
            import json
            json_embedding = json.dumps(query_embedding)
            sql = """
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, VECTOR_FROM_JSON(?)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC
            """
            cursor.execute(sql, (json_embedding,))
            results = cursor.fetchall()
            print(f"✅ JSON format worked! Got {len(results)} results")
            for i, (doc_id, title, score) in enumerate(results[:3]):
                print(f"   {i+1}. {doc_id}: {title[:50]}... (score: {score:.4f})")
        except Exception as e:
            print(f"❌ JSON format failed: {e}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()