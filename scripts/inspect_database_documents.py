#!/usr/bin/env python3
"""
Script to inspect documents in the RAG.SourceDocuments table.
This helps diagnose retrieval issues by showing what's actually stored.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connection_manager import get_iris_connection


def inspect_documents():
    """Inspect documents in the database."""
    print("Connecting to IRIS database...")
    
    try:
        connection = get_iris_connection()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    # Get total document count
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    total_count = cursor.fetchone()[0]
    print(f"\n📊 Total documents in RAG.SourceDocuments: {total_count}")
    
    if total_count == 0:
        print("\n⚠️  No documents found in the database!")
        print("This explains why retrieval is returning empty results.")
        print("\nTo fix this, you need to:")
        print("1. Load documents into the database using the data loader")
        print("2. Generate embeddings for the documents")
        return
    
    # First, check what columns exist
    cursor.execute("""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND TABLE_NAME = 'SOURCEDOCUMENTS'
        ORDER BY ORDINAL_POSITION
    """)
    columns = [row[0] for row in cursor.fetchall()]
    print(f"\nAvailable columns: {', '.join(columns)}")
    
    # Get sample documents
    print(f"\n📄 First 5 documents:")
    print("-" * 80)
    
    # Build query based on available columns
    # Map common names to actual column names
    id_col = "doc_id" if "doc_id" in columns else "ID"
    title_col = "title" if "title" in columns else "Title"
    content_col = "text_content" if "text_content" in columns else "Content"
    
    cursor.execute(f"""
        SELECT TOP 5 
            {id_col},
            {title_col},
            SUBSTRING({content_col}, 1, 200) as ContentPreview
        FROM RAG.SourceDocuments
        ORDER BY {id_col}
    """)
    
    documents = cursor.fetchall()
    
    for i, (doc_id, title, content_preview) in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"  ID: {doc_id}")
        print(f"  Title: {title or 'N/A'}")
        print(f"  Content Preview: {content_preview}...")
        
        # Show additional metadata if available
        if "authors" in columns:
            cursor.execute(f"SELECT authors FROM RAG.SourceDocuments WHERE {id_col} = ?", (doc_id,))
            authors = cursor.fetchone()[0]
            if authors:
                print(f"  Authors: {authors}")
        
    # Check for embeddings
    print("\n🔍 Checking for embeddings...")
    
    # Check if embedding columns exist
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND TABLE_NAME = 'SOURCEDOCUMENTS' 
        AND COLUMN_NAME LIKE '%Embedding%'
    """)
    
    embedding_cols = cursor.fetchone()[0]
    if embedding_cols > 0:
        print(f"✅ Found {embedding_cols} embedding column(s)")
        
        # Check how many documents have embeddings
        embedding_col = "embedding" if "embedding" in columns else "Embedding"
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM RAG.SourceDocuments 
            WHERE {embedding_col} IS NOT NULL
        """)
        docs_with_embeddings = cursor.fetchone()[0]
        print(f"📊 Documents with embeddings: {docs_with_embeddings}/{total_count}")
    else:
        print("❌ No embedding columns found in the table")
    
    # Since we can't search stream fields directly, let's check titles and abstracts
    print("\n🏥 Searching for medical content in titles and abstracts...")
    
    # Check if abstract column exists
    abstract_col = "abstract" if "abstract" in columns else None
    
    if abstract_col:
        # Search for metformin in abstract
        try:
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM RAG.SourceDocuments 
                WHERE {abstract_col} LIKE '%metformin%'
            """)
            metformin_count = cursor.fetchone()[0]
            print(f"  Documents with 'metformin' in abstract: {metformin_count}")
        except:
            print("  Could not search abstracts for 'metformin'")
            
        # Search for SGLT2 in abstract
        try:
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM RAG.SourceDocuments 
                WHERE {abstract_col} LIKE '%SGLT2%' OR {abstract_col} LIKE '%SGLT-2%'
            """)
            sglt2_count = cursor.fetchone()[0]
            print(f"  Documents with 'SGLT2/SGLT-2' in abstract: {sglt2_count}")
        except:
            print("  Could not search abstracts for 'SGLT2'")
            
        # Search for diabetes in abstract
        try:
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM RAG.SourceDocuments 
                WHERE {abstract_col} LIKE '%diabetes%'
            """)
            diabetes_count = cursor.fetchone()[0]
            print(f"  Documents with 'diabetes' in abstract: {diabetes_count}")
        except:
            print("  Could not search abstracts for 'diabetes'")
    else:
        print("  No abstract column found for searching")
    
    # Try to understand what content we have
    print("\n📚 Sample document details:")
    try:
        # Get a random document to show more details
        cursor.execute(f"""
            SELECT TOP 1 
                {id_col},
                {title_col}
            FROM RAG.SourceDocuments
            WHERE {title_col} IS NOT NULL
            ORDER BY {id_col}
        """)
        sample_doc = cursor.fetchone()
        if sample_doc:
            doc_id, title_stream = sample_doc
            print(f"  Document ID: {doc_id}")
            # Title might be a stream, try to read it
            if hasattr(title_stream, 'read'):
                try:
                    title_content = title_stream.read()
                    if isinstance(title_content, bytes):
                        title_content = title_content.decode('utf-8')
                    print(f"  Title: {title_content}")
                except:
                    print(f"  Title: (Could not read stream)")
            else:
                print(f"  Title: {title_stream}")
    except Exception as e:
        print(f"  Could not get sample document: {e}")
    
    # Check chunk table if it exists
    print("\n🔍 Checking for chunk table...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND TABLE_NAME = 'CHUNKS'
    """)
    
    if cursor.fetchone()[0] > 0:
        cursor.execute("SELECT COUNT(*) FROM RAG.Chunks")
        chunk_count = cursor.fetchone()[0]
        print(f"✅ Found RAG.Chunks table with {chunk_count} chunks")
        
        if chunk_count > 0:
            # Check chunks for medical content
            # Check chunk columns
            cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' 
                AND TABLE_NAME = 'CHUNKS'
                ORDER BY ORDINAL_POSITION
            """)
            chunk_columns = [row[0] for row in cursor.fetchall()]
            print(f"  Chunk columns: {', '.join(chunk_columns)}")
            
            # Try to count medical chunks - adjust column name as needed
            chunk_text_col = next((col for col in chunk_columns if 'text' in col.lower() or 'content' in col.lower()), None)
            if chunk_text_col:
                try:
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM RAG.Chunks 
                        WHERE {chunk_text_col} LIKE '%metformin%' OR {chunk_text_col} LIKE '%SGLT2%'
                    """)
                    medical_chunks = cursor.fetchone()[0]
                    print(f"  Chunks mentioning metformin/SGLT2: {medical_chunks}")
                    
                    # Show a sample chunk
                    if medical_chunks > 0:
                        cursor.execute(f"""
                            SELECT TOP 1 {chunk_text_col}
                            FROM RAG.Chunks 
                            WHERE {chunk_text_col} LIKE '%metformin%' OR {chunk_text_col} LIKE '%SGLT2%'
                        """)
                        sample_chunk = cursor.fetchone()[0]
                        print(f"\n  Sample chunk with medical content:")
                        print(f"  {sample_chunk[:200]}...")
                except Exception as e:
                    print(f"  Could not search chunks: {e}")
    else:
        print("❌ No RAG.Chunks table found")
    
    cursor.close()
    connection.close()
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS:")
    print("=" * 80)
    
    if total_count == 0:
        print("❌ No documents in database - need to load data first")
    elif metformin_count == 0 and sglt2_count == 0:
        print("⚠️  Documents exist but don't contain expected medical content")
        print("   - The sample data might be different from what queries expect")
        print("   - Consider loading medical documents or adjusting test queries")
    elif embedding_cols == 0 or docs_with_embeddings == 0:
        print("⚠️  Documents exist but lack embeddings")
        print("   - Need to generate embeddings for vector search to work")
    else:
        print("✅ Documents and embeddings appear to be present")
        print("   - Check retrieval pipeline configuration")
        print("   - Verify vector search is properly configured")


if __name__ == "__main__":
    inspect_documents()