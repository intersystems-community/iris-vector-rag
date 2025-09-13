#!/usr/bin/env python3
"""
Test vector search directly to understand why retrieval returns empty results.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func


def test_vector_search():
    """Test vector search directly against the database."""
    print("Testing direct vector search...\n")
    
    # Connect to database
    try:
        connection = get_iris_connection()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    cursor = connection.cursor()
    
    # Get embedding function
    embed_func = get_embedding_func()
    
    # Test queries
    test_queries = [
        "What are the latest treatments for cancer?",
        "How does the immune system work?",
        "What is diabetes?",
        "metformin cardiovascular benefits",  # This should fail
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        try:
            # Generate query embedding
            query_embedding = embed_func(query)
            print(f"✅ Generated query embedding (dim: {len(query_embedding)})")
            
            # Check embedding dimension in database
            cursor.execute("""
                SELECT TOP 1 embedding 
                FROM RAG.SourceDocuments 
                WHERE embedding IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result[0]:
                db_embedding = result[0]
                if hasattr(db_embedding, '__len__'):
                    print(f"📊 Database embedding dimension: {len(db_embedding)}")
                    
                    if len(query_embedding) != len(db_embedding):
                        print(f"❌ Dimension mismatch! Query: {len(query_embedding)}, DB: {len(db_embedding)}")
                        print("This is why retrieval is failing - embeddings have different dimensions!")
                        continue
                
            # Try vector search with IRIS SQL
            # Note: IRIS uses VECTOR_DOT_PRODUCT for similarity
            print("\n🔍 Attempting vector search...")
            
            # Convert embedding to string format for SQL
            embedding_str = ','.join(map(str, query_embedding))
            
            try:
                # This is the typical IRIS vector search syntax
                cursor.execute(f"""
                    SELECT TOP 5 
                        doc_id,
                        VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """, (embedding_str,))
                
                results = cursor.fetchall()
                
                if results:
                    print(f"\n✅ Found {len(results)} similar documents:")
                    for doc_id, score in results:
                        print(f"  - {doc_id}: similarity = {score:.4f}")
                else:
                    print("\n❌ No results from vector search")
                    
            except Exception as e:
                print(f"\n❌ Vector search failed: {e}")
                
                # Try alternative search syntax
                try:
                    print("\n🔍 Trying alternative search method...")
                    cursor.execute("""
                        SELECT TOP 5 doc_id
                        FROM RAG.SourceDocuments
                        WHERE embedding IS NOT NULL
                    """)
                    
                    results = cursor.fetchall()
                    print(f"✅ Found {len(results)} documents with embeddings")
                    
                except Exception as e2:
                    print(f"❌ Alternative search also failed: {e2}")
                    
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    cursor.close()
    connection.close()
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    print("The retrieval is likely failing due to:")
    print("1. Embedding dimension mismatch (query embeddings vs stored embeddings)")
    print("2. The stored embeddings might be in a different format")
    print("3. The vector search SQL syntax might need adjustment")
    print("\nThe unusual embedding dimension of 9156 suggests these might be:")
    print("- Concatenated embeddings from multiple models")
    print("- Token-level embeddings (like ColBERT)")
    print("- A custom embedding format")


if __name__ == "__main__":
    test_vector_search()