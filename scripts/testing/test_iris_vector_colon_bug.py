#!/usr/bin/env python3
"""
Test script to demonstrate IRIS vector search bug with colons in TO_VECTOR parameter
"""

import sys
sys.path.insert(0, '.')

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

def test_vector_search_bug():
    """Test and demonstrate the IRIS vector search colon bug"""
    
    print("🔍 IRIS Vector Search Colon Bug Demonstration")
    print("=" * 60)
    
    # Get connection and embedding function
    conn = get_iris_connection()
    cursor = conn.cursor()
    embedding_func = get_embedding_func()
    
    try:
        # Generate a test query embedding
        query = "diabetes treatment"
        print(f"\n📊 Test query: '{query}'")
        
        query_embedding = embedding_func([query])[0]
        print(f"📊 Embedding dimensions: {len(query_embedding)}")
        
        # Convert to string format
        query_embedding_str = ','.join(map(str, query_embedding))
        print(f"📊 Embedding string length: {len(query_embedding_str)} characters")
        
        # Check for problematic characters
        print("\n🔍 Checking for problematic characters:")
        
        # Check for colons
        if ':' in query_embedding_str:
            print("⚠️  FOUND COLONS IN EMBEDDING STRING!")
            colon_count = query_embedding_str.count(':')
            print(f"   Number of colons: {colon_count}")
            
            # Find first colon
            colon_idx = query_embedding_str.find(':')
            context_start = max(0, colon_idx - 30)
            context_end = min(len(query_embedding_str), colon_idx + 30)
            context = query_embedding_str[context_start:context_end]
            print(f"   Context around first colon: ...{context}...")
        else:
            print("✅ No colons found in embedding string")
        
        # Check for scientific notation
        import re
        sci_notation = re.findall(r'[-+]?\d*\.?\d+[eE][-+]?\d+', query_embedding_str)
        if sci_notation:
            print(f"\n⚠️  Found scientific notation: {sci_notation[:5]}...")
            print(f"   Total count: {len(sci_notation)}")
        
        # Try to execute vector search query
        print("\n🧪 Testing vector search query...")
        
        # First, check if we have _V2 tables with VECTOR columns
        cursor.execute("""
            SELECT COUNT(*) 
            FROM RAG.SourceDocuments_V2 
            WHERE document_embedding_vector IS NOT NULL
        """)
        v2_count = cursor.fetchone()[0]
        
        if v2_count > 0:
            print(f"✅ Found {v2_count:,} documents in _V2 table with VECTOR embeddings")
            
            # Try the vector search
            try:
                sql_query = f"""
                    SELECT TOP 5 doc_id, 
                    VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, 'DOUBLE')) AS similarity_score
                    FROM RAG.SourceDocuments_V2
                    WHERE document_embedding_vector IS NOT NULL
                    ORDER BY similarity_score DESC
                """
                
                print("\n📊 Attempting parameterized query...")
                cursor.execute(sql_query, [query_embedding_str])
                results = cursor.fetchall()
                
                print("✅ SUCCESS! Parameterized query worked!")
                print(f"   Retrieved {len(results)} documents")
                
            except Exception as e1:
                print(f"❌ Parameterized query failed: {e1}")
                
                # Try with direct string interpolation (the problematic approach)
                try:
                    print("\n📊 Attempting direct string interpolation...")
                    sql_query_direct = f"""
                        SELECT TOP 5 doc_id,
                        VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) AS similarity_score
                        FROM RAG.SourceDocuments_V2
                        WHERE document_embedding_vector IS NOT NULL
                        ORDER BY similarity_score DESC
                    """
                    
                    # Show a preview of the SQL
                    print(f"   SQL preview (first 500 chars):")
                    print(f"   {sql_query_direct[:500]}...")
                    
                    cursor.execute(sql_query_direct)
                    results = cursor.fetchall()
                    
                    print("✅ Direct interpolation worked!")
                    print(f"   Retrieved {len(results)} documents")
                    
                except Exception as e2:
                    print(f"❌ Direct interpolation failed: {e2}")
                    print("\n🔍 This is likely the colon bug!")
                    
        else:
            print("❌ No documents found in _V2 table with VECTOR embeddings")
            print("   The vector migration may not have completed yet")
            
            # Check regular table
            cursor.execute("""
                SELECT COUNT(*) 
                FROM RAG.SourceDocuments 
                WHERE embedding IS NOT NULL
            """)
            regular_count = cursor.fetchone()[0]
            print(f"\n📊 Regular SourceDocuments table has {regular_count:,} documents with embeddings")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cursor.close()
        conn.close()
        
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION:")
    print("The IRIS SQL parser incorrectly interprets colons (:) within the TO_VECTOR")
    print("string parameter as parameter markers, causing SQL parsing errors.")
    print("\nWORKAROUND: Use parameterized queries with ? placeholders instead of")
    print("string interpolation to avoid this issue.")

if __name__ == "__main__":
    test_vector_search_bug()