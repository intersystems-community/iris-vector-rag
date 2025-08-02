#!/usr/bin/env python3
"""
Check the actual vector format in the database
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection

def check_vector_format():
    """Check the actual vector format in the database"""
    print("🔍 Checking Vector Format in Database")
    print("=" * 50)
    
    iris_conn = get_iris_connection()
    cursor = iris_conn.cursor()
    
    try:
        # Check SourceDocuments vector format
        print(f"📊 Checking SourceDocuments table...")
        cursor.execute("""
            SELECT TOP 1 doc_id, embedding, LENGTH(embedding) as len
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
        """)
        result = cursor.fetchone()
        if result:
            doc_id, embedding_data, length = result
            print(f"   Document {doc_id}:")
            print(f"   Length: {length}")
            print(f"   Type: {type(embedding_data)}")
            print(f"   First 100 chars: {str(embedding_data)[:100]}...")
            
            # Try to understand the format
            if hasattr(embedding_data, 'read'):
                # It's a stream/blob
                print("   Format: Binary/Stream data")
            elif isinstance(embedding_data, str):
                print("   Format: String data")
                if embedding_data.startswith('['):
                    print("   Appears to be JSON array format")
                else:
                    print("   Unknown string format")
            else:
                print(f"   Format: {type(embedding_data)}")
        else:
            print("   No documents with embeddings found")
        
        # Check if we can use VECTOR_COSINE with existing data
        print(f"\n🧪 Testing VECTOR_COSINE with existing data...")
        cursor.execute("""
            SELECT TOP 1 doc_id, 
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(embedding)) as self_similarity
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
        """)
        result = cursor.fetchone()
        if result:
            doc_id, similarity = result
            print(f"   Document {doc_id} self-similarity: {similarity}")
            print("   ✅ VECTOR_COSINE works with existing format")
        else:
            print("   ❌ No documents to test with")
        
    except Exception as e:
        print(f"❌ Error checking vector format: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    check_vector_format()