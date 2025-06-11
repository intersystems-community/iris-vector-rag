#!/usr/bin/env python3
"""
Test the working vector solution based on our findings
"""

import sys
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
import json

def main():
    print("🔍 Testing Working Vector Solution")
    print("=" * 60)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate test embedding
        query = "diabetes treatment"
        query_embedding = embedding_func([query])[0]  # This returns a list
        
        print(f"📊 Query: '{query}'")
        print(f"📊 Embedding type: {type(query_embedding)}")
        print(f"📊 Embedding dimensions: {len(query_embedding)}")
        
        # Solution 1: Store query embedding in a temp table
        print("\n🧪 Solution 1: Using a temporary table for query vector")
        try:
            # Create a temp table with the query vector
            cursor.execute("DROP TABLE IF EXISTS RAG.TempQueryVector")
            cursor.execute("""
                CREATE TABLE RAG.TempQueryVector (
                    id INTEGER,
                    query_vector VECTOR(FLOAT, 384)
                )
            """)
            
            # Insert the query vector
            query_vector_str = ','.join(map(str, query_embedding))
            cursor.execute(f"""
                INSERT INTO RAG.TempQueryVector (id, query_vector) 
                VALUES (1, TO_VECTOR('{query_vector_str}', 'FLOAT', 384))
            """)
            
            # Now use it in the query
            sql = """
                SELECT TOP 5 s.doc_id, s.title,
                VECTOR_COSINE(s.document_embedding_vector, t.query_vector) AS score
                FROM RAG.SourceDocuments_V2 s, RAG.TempQueryVector t
                WHERE s.document_embedding_vector IS NOT NULL
                AND t.id = 1
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Temp table approach worked! Got {len(results)} results")
            for i, (doc_id, title, score) in enumerate(results[:3]):
                print(f"   {i+1}. {doc_id}: {title[:50]}... (score: {score:.4f})")
                
        except Exception as e:
            print(f"❌ Temp table approach failed: {e}")
        finally:
            cursor.execute("DROP TABLE IF EXISTS RAG.TempQueryVector")
            
        # Solution 2: Use a stored procedure
        print("\n🧪 Solution 2: Creating a stored procedure for vector search")
        try:
            # Drop existing procedure if exists
            cursor.execute("DROP PROCEDURE IF EXISTS RAG.VectorSearch")
            
            # Create stored procedure
            create_proc = """
            CREATE PROCEDURE RAG.VectorSearch(
                IN query_vector_str VARCHAR(50000),
                IN top_k INTEGER DEFAULT 5
            )
            BEGIN
                DECLARE query_vec VECTOR(FLOAT, 384);
                SET query_vec = TO_VECTOR(query_vector_str, 'FLOAT', 384);
                
                SELECT TOP :top_k doc_id, title, text_content,
                       VECTOR_COSINE(document_embedding_vector, query_vec) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC;
            END
            """
            cursor.execute(create_proc)
            print("✅ Stored procedure created successfully")
            
            # Test the stored procedure
            cursor.execute("CALL RAG.VectorSearch(?, ?)", (query_vector_str, 5))
            results = cursor.fetchall()
            print(f"✅ Stored procedure worked! Got {len(results)} results")
            
        except Exception as e:
            print(f"❌ Stored procedure approach failed: {e}")
            
        # Solution 3: Use dynamic SQL
        print("\n🧪 Solution 3: Using dynamic SQL")
        try:
            # Build the query dynamically
            query_vector_str = ','.join(map(str, query_embedding))
            
            # Use EXECUTE IMMEDIATE
            dynamic_sql = f"""
                EXECUTE IMMEDIATE '
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, TO_VECTOR(''{query_vector_str}'', ''DOUBLE'', 384)) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY score DESC'
            """
            cursor.execute(dynamic_sql)
            results = cursor.fetchall()
            print(f"✅ Dynamic SQL worked! Got {len(results)} results")
            
        except Exception as e:
            print(f"❌ Dynamic SQL failed: {e}")
            
        # Solution 4: Use the working subquery approach
        print("\n🧪 Solution 4: Using the working subquery approach (baseline)")
        try:
            # Get a random document's vector to use as query
            cursor.execute("""
                SELECT TOP 1 doc_id, document_embedding_vector 
                FROM RAG.SourceDocuments_V2 
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY RAND()
            """)
            query_doc_id, _ = cursor.fetchone()
            
            # Find similar documents
            sql = f"""
                SELECT TOP 5 doc_id, title,
                VECTOR_COSINE(document_embedding_vector, 
                    (SELECT document_embedding_vector FROM RAG.SourceDocuments_V2 WHERE doc_id = '{query_doc_id}')
                ) AS score
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                AND doc_id != '{query_doc_id}'
                ORDER BY score DESC
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"✅ Subquery approach worked! Got {len(results)} results similar to {query_doc_id}")
            for i, (doc_id, title, score) in enumerate(results[:3]):
                print(f"   {i+1}. {doc_id}: {title[:50]}... (score: {score:.4f})")
                
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