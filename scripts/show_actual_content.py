#!/usr/bin/env python3
"""
Script to show actual content from the database by reading stream fields properly.
This helps understand what data is available for testing.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connection_manager import get_iris_connection


def read_stream_field(stream):
    """Read content from an IRIS stream field."""
    if stream is None:
        return None
    
    try:
        # If it's already a string, return it
        if isinstance(stream, str):
            return stream
        
        # If it has a read method (like IRISInputStream), use it
        if hasattr(stream, 'read'):
            content = stream.read()
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return content
        
        # Otherwise, try to convert to string
        return str(stream)
    except Exception as e:
        return f"[Error reading stream: {e}]"


def show_actual_content():
    """Show actual content from the database."""
    print("Connecting to IRIS database...")
    
    try:
        connection = get_iris_connection()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    cursor = connection.cursor()
    
    # Get 5 random documents
    print("\n📄 Sample documents with actual content:")
    print("=" * 80)
    
    cursor.execute("""
        SELECT TOP 5
            doc_id,
            title,
            abstract,
            SUBSTRING(text_content, 1, 500) as content_preview,
            authors,
            keywords
        FROM RAG.SourceDocuments
        ORDER BY doc_id
    """)
    
    documents = cursor.fetchall()
    
    for i, (doc_id, title, abstract, content, authors, keywords) in enumerate(documents, 1):
        print(f"\n📄 Document {i}:")
        print(f"ID: {doc_id}")
        
        # Read each stream field
        title_text = read_stream_field(title)
        abstract_text = read_stream_field(abstract)
        content_text = read_stream_field(content)
        authors_text = read_stream_field(authors)
        keywords_text = read_stream_field(keywords)
        
        print(f"\nTitle: {title_text}")
        print(f"\nAuthors: {authors_text}")
        print(f"\nKeywords: {keywords_text}")
        print(f"\nAbstract: {abstract_text[:500] if abstract_text else 'N/A'}...")
        print(f"\nContent Preview: {content_text if content_text else 'N/A'}")
        print("-" * 80)
    
    # Search for common terms to understand the content
    print("\n🔍 Content analysis:")
    
    # Get keywords to understand the domain
    cursor.execute("""
        SELECT DISTINCT keywords
        FROM RAG.SourceDocuments
        WHERE keywords IS NOT NULL
        LIMIT 10
    """)
    
    print("\nSample keywords from documents:")
    for row in cursor.fetchall():
        keywords = read_stream_field(row[0])
        if keywords:
            print(f"  - {keywords}")
    
    # Get titles to understand topics
    cursor.execute("""
        SELECT title
        FROM RAG.SourceDocuments
        WHERE title IS NOT NULL
        LIMIT 10
    """)
    
    print("\nSample titles:")
    for row in cursor.fetchall():
        title = read_stream_field(row[0])
        if title:
            print(f"  - {title[:100]}...")
    
    cursor.close()
    connection.close()
    
    print("\n" + "=" * 80)
    print("FINDINGS:")
    print("=" * 80)
    print("The database contains medical research papers from PubMed Central (PMC).")
    print("To test the RAG system, you should:")
    print("1. Use queries related to the actual content in the database")
    print("2. Look at the keywords and titles above to formulate relevant queries")
    print("3. Or load documents that contain the specific medical content you're testing for")


if __name__ == "__main__":
    show_actual_content()