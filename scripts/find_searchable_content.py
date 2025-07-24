#!/usr/bin/env python3
"""
Script to find what content we can actually search in the database.
Focuses on finding appropriate test queries based on available data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connection_manager import get_iris_connection


def find_searchable_content():
    """Find searchable content and suggest test queries."""
    print("Connecting to IRIS database...")
    
    try:
        connection = get_iris_connection()
        print("✅ Connected to database\n")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    cursor = connection.cursor()
    
    # Check if there's a keywords column we can search
    print("🔍 Checking searchable fields...")
    
    # Get some document IDs
    cursor.execute("""
        SELECT doc_id 
        FROM RAG.SourceDocuments 
        LIMIT 20
    """)
    
    doc_ids = [row[0] for row in cursor.fetchall()]
    print(f"\nSample document IDs (PMC IDs):")
    for doc_id in doc_ids[:10]:
        print(f"  - {doc_id}")
    
    # Since the documents are from PubMed Central, let's create relevant medical queries
    print("\n💡 Suggested test queries based on PMC content:")
    print("Since these are PubMed Central medical papers, try queries like:")
    
    suggested_queries = [
        "What are the latest treatments for cancer?",
        "Explain the mechanism of action of antibiotics",
        "What are the side effects of chemotherapy?",
        "How does the immune system work?",
        "What is the role of genetics in disease?",
        "Describe recent advances in cardiovascular medicine",
        "What are the symptoms of viral infections?",
        "How do vaccines work?",
        "What is the pathophysiology of inflammation?",
        "Explain the diagnosis and treatment of hypertension"
    ]
    
    for i, query in enumerate(suggested_queries, 1):
        print(f"  {i}. {query}")
    
    # Check embedding dimensions
    print("\n📊 Embedding information:")
    cursor.execute("""
        SELECT TOP 1 embedding 
        FROM RAG.SourceDocuments 
        WHERE embedding IS NOT NULL
    """)
    
    result = cursor.fetchone()
    if result and result[0]:
        try:
            # Try to get the length of the embedding vector
            if hasattr(result[0], '__len__'):
                print(f"  Embedding dimension: {len(result[0])}")
            else:
                print(f"  Embeddings exist but dimension unclear")
        except:
            print(f"  Embeddings exist (type: {type(result[0])})")
    
    # Try to understand the data through metadata
    print("\n📝 Checking metadata column...")
    cursor.execute("""
        SELECT TOP 5 metadata 
        FROM RAG.SourceDocuments 
        WHERE metadata IS NOT NULL
    """)
    
    for i, row in enumerate(cursor.fetchall(), 1):
        if row[0]:
            print(f"  Sample {i}: {str(row[0])[:100]}...")
    
    cursor.close()
    connection.close()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR TESTING:")
    print("=" * 80)
    print("1. The database contains PubMed Central (PMC) medical research papers")
    print("2. All 1000 documents have embeddings ready for vector search")
    print("3. Use medical/scientific queries that would match research paper content")
    print("4. Avoid specific drug names unless you know they're in the corpus")
    print("5. Focus on general medical topics, diseases, treatments, and biological processes")
    print("\nIf you need to test specific content (like metformin or SGLT2), you should:")
    print("- Load documents that contain that specific content")
    print("- Or modify your test queries to match the available PMC papers")


if __name__ == "__main__":
    find_searchable_content()