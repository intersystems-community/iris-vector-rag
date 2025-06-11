#!/usr/bin/env python3
"""
Targeted test to diagnose and fix basic RAG retrieval issues
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_database_content():
    """Test what's actually in the database"""
    print("🔍 Testing database content...")
    
    try:
        import intersystems_iris.dbapi._DBAPI as iris
        
        # Connect to IRIS
        connection = iris.connect(
            hostname="localhost",
            port=1972,
            namespace="USER",
            username="_SYSTEM",
            password="SYS"
        )
        
        cursor = connection.cursor()
        
        # Check document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        print(f"📚 Total documents: {doc_count}")
        
        # Check documents with embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        embedded_count = cursor.fetchone()[0]
        print(f"🔢 Documents with embeddings: {embedded_count}")
        
        # Sample some documents to see what we have
        cursor.execute("""
            SELECT TOP 5 title, LEFT(content, 100) as content_preview
            FROM RAG.SourceDocuments
            WHERE content IS NOT NULL AND title IS NOT NULL
        """)
        
        print(f"\n📄 Sample documents:")
        for i, (title, content) in enumerate(cursor.fetchall()):
            print(f"  {i+1}. {title}")
            print(f"     Content: {content}...")
            
        # Test a simple query for diabetes/medical content
        print(f"\n🔍 Searching for diabetes-related content...")
        cursor.execute("""
            SELECT TOP 3 title, LEFT(content, 150) as content_preview
            FROM RAG.SourceDocuments
            WHERE UPPER(content) LIKE '%DIABETES%'
               OR UPPER(content) LIKE '%INSULIN%'
               OR UPPER(content) LIKE '%GLUCOSE%'
               OR UPPER(title) LIKE '%DIABETES%'
        """)
        
        diabetes_docs = cursor.fetchall()
        if diabetes_docs:
            print(f"✅ Found {len(diabetes_docs)} diabetes-related documents:")
            for i, (title, content) in enumerate(diabetes_docs):
                print(f"  {i+1}. {title}")
                print(f"     Content: {content}...")
        else:
            print("❌ No diabetes-related documents found!")
            
        connection.close()
        return doc_count, embedded_count, len(diabetes_docs)
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return 0, 0, 0

def test_basic_vector_search():
    """Test basic vector search functionality"""
    print("\n🔍 Testing vector search...")
    
    try:
        import intersystems_iris.dbapi._DBAPI as iris
        from sentence_transformers import SentenceTransformer
        
        # Connect to IRIS
        connection = iris.connect(
            hostname="localhost",
            port=1972,
            namespace="USER",
            username="_SYSTEM",
            password="SYS"
        )
        
        cursor = connection.cursor()
        
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test query
        query = "diabetes treatment insulin"
        query_embedding = model.encode(query).tolist()
        
        print(f"📝 Query: {query}")
        print(f"🔢 Query embedding length: {len(query_embedding)}")
        
        # Test vector search with VECTOR_DOT_PRODUCT
        print("\n🔍 Testing VECTOR_DOT_PRODUCT search...")
        cursor.execute("""
            SELECT TOP 5
                title,
                LEFT(content, 100) as content_preview,
                VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) as similarity_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
        """, [str(query_embedding)])
        
        results = cursor.fetchall()
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, (title, content, score) in enumerate(results):
                print(f"  {i+1}. Score: {score:.3f}")
                print(f"     Title: {title}")
                print(f"     Content: {content}...")
        else:
            print("❌ No vector search results!")
            
        connection.close()
        return len(results)
        
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    print("🚀 Basic RAG Retrieval Diagnostic Test")
    print("=" * 50)
    
    # Test 1: Database content
    doc_count, embedded_count, diabetes_count = test_database_content()
    
    # Test 2: Vector search
    vector_results = test_basic_vector_search()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY:")
    print(f"  📚 Total documents: {doc_count}")
    print(f"  🔢 Documents with embeddings: {embedded_count}")
    print(f"  🏥 Diabetes-related documents: {diabetes_count}")
    print(f"  🔍 Vector search results: {vector_results}")
    
    if doc_count == 0:
        print("\n❌ ISSUE: No documents in database!")
    elif embedded_count == 0:
        print("\n❌ ISSUE: No embeddings found!")
    elif diabetes_count == 0:
        print("\n❌ ISSUE: No relevant medical content!")
    elif vector_results == 0:
        print("\n❌ ISSUE: Vector search not working!")
    else:
        print("\n✅ Basic components seem to be working")

if __name__ == "__main__":
    main()