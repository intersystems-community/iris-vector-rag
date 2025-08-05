#!/usr/bin/env python3
"""
Test IRIS Vector Search Workaround
Demonstrates a working approach for vector search in IRIS
"""

import sys
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
import time
import uuid

def main():
    print("🔍 IRIS Vector Search Workaround Test")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate test query embedding
        query = "diabetes treatment"
        query_embedding = embedding_func([query])[0]
        query_embedding_str = ','.join(map(str, query_embedding))
        
        print(f"📊 Query: '{query}'")
        print(f"📊 Embedding dimensions: {len(query_embedding)}")
        
        # WORKAROUND: Insert query vector as a temporary document
        print("\n🔧 Workaround: Using temporary document approach")
        
        # Generate unique temporary doc_id
        temp_doc_id = f"__TEMP_QUERY_{uuid.uuid4().hex[:8]}__"
        
        try:
            # Step 1: Insert query vector as temporary document
            print(f"📝 Inserting temporary query vector with doc_id: {temp_doc_id}")
            
            # Build the SQL with the vector string directly embedded (no parameters for TO_VECTOR)
            insert_sql = f"""
                INSERT INTO RAG.SourceDocuments_V2
                (doc_id, title, document_embedding_vector)
                VALUES ('{temp_doc_id}', 'Temporary Query Vector', TO_VECTOR('{query_embedding_str}', 'FLOAT', 384))
            """
            cursor.execute(insert_sql)
            conn.commit()
            
            # Step 2: Perform vector search using direct comparison
            print("🔍 Performing vector search...")
            
            search_sql = f"""
                SELECT s.doc_id, s.title, s.text_content,
                       VECTOR_COSINE(s.document_embedding_vector, q.document_embedding_vector) AS similarity_score
                FROM RAG.SourceDocuments_V2 s,
                     RAG.SourceDocuments_V2 q
                WHERE q.doc_id = '{temp_doc_id}'
                  AND s.doc_id != '{temp_doc_id}'
                  AND s.document_embedding_vector IS NOT NULL
                  AND q.document_embedding_vector IS NOT NULL
                ORDER BY similarity_score DESC
                LIMIT 5
            """
            
            start_time = time.time()
            cursor.execute(search_sql)
            results = cursor.fetchall()
            search_time = time.time() - start_time
            
            print(f"✅ Search completed in {search_time:.3f} seconds")
            print(f"📊 Found {len(results)} results\n")
            
            # Display results
            if results:
                print("🏆 Top Results:")
                for i, (doc_id, title, content, score) in enumerate(results, 1):
                    print(f"\n{i}. Document: {doc_id}")
                    print(f"   Title: {title}")
                    print(f"   Score: {score:.4f}")
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"   Content: {preview}")
            else:
                print("❌ No results found")
                
        finally:
            # Step 3: Clean up temporary document
            print(f"\n🧹 Cleaning up temporary document {temp_doc_id}")
            cursor.execute("DELETE FROM RAG.SourceDocuments_V2 WHERE doc_id = ?", (temp_doc_id,))
            conn.commit()
            
        # Alternative approach: Using a dedicated query table
        print("\n" + "=" * 60)
        print("🔧 Alternative: Using dedicated query table")
        
        # Create a dedicated table for query vectors if it doesn't exist
        try:
            create_query_table = """
                CREATE TABLE IF NOT EXISTS RAG.QueryVectors (
                    query_id VARCHAR(255) PRIMARY KEY,
                    query_text VARCHAR(1000),
                    query_vector VECTOR(FLOAT, 384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            cursor.execute(create_query_table)
            
            # Create index on query vectors
            try:
                cursor.execute("""
                    CREATE INDEX idx_query_vectors 
                    ON RAG.QueryVectors (query_vector) 
                    AS HNSW(Distance='COSINE')
                """)
                print("✅ Created HNSW index on QueryVectors table")
            except:
                pass  # Index might already exist
                
            # Insert query into dedicated table
            query_id = f"QUERY_{uuid.uuid4().hex[:8]}"
            insert_query_sql = f"""
                INSERT INTO RAG.QueryVectors (query_id, query_text, query_vector)
                VALUES ('{query_id}', '{query}', TO_VECTOR('{query_embedding_str}', 'FLOAT', 384))
            """
            cursor.execute(insert_query_sql)
            conn.commit()
            
            # Search using the query table
            search_sql2 = f"""
                SELECT s.doc_id, s.title,
                       VECTOR_COSINE(s.document_embedding_vector, q.query_vector) AS similarity_score
                FROM RAG.SourceDocuments_V2 s,
                     RAG.QueryVectors q
                WHERE q.query_id = '{query_id}'
                  AND s.document_embedding_vector IS NOT NULL
                ORDER BY similarity_score DESC
                LIMIT 3
            """
            
            cursor.execute(search_sql2)
            results2 = cursor.fetchall()
            
            print(f"✅ Found {len(results2)} results using query table approach")
            for i, (doc_id, title, score) in enumerate(results2, 1):
                print(f"   {i}. {doc_id}: {title} (score: {score:.4f})")
                
            # Optional: Clean up old queries
            cursor.execute("""
                DELETE FROM RAG.QueryVectors 
                WHERE created_at < DATEADD('hour', -1, CURRENT_TIMESTAMP)
            """)
            conn.commit()
            
        except Exception as e:
            print(f"⚠️  Query table approach error: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()