#!/usr/bin/env python3
"""
Debug script to check database state after chunking integration test.
"""

import sys
import json
from common.iris_connection_manager import get_iris_connection
from iris_rag.config.manager import ConfigurationManager

def check_database_state():
    """Check what's actually in the database."""
    
    try:
        # Get connection
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        # Check SourceDocuments table
        print("=== Checking RAG.SourceDocuments ===")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count = cursor.fetchone()[0]
        print(f"Total documents: {count}")
        
        if count > 0:
            # Check first few documents
            cursor.execute("SELECT TOP 3 ID, content, metadata, embedding FROM RAG.SourceDocuments")
            rows = cursor.fetchall()
            
            for i, row in enumerate(rows):
                doc_id, content, metadata, embedding = row
                print(f"\nDocument {i+1}:")
                print(f"  ID: {doc_id}")
                print(f"  Content length: {len(content) if content else 0}")
                print(f"  Has metadata: {metadata is not None}")
                print(f"  Has embedding: {embedding is not None}")
                
                if embedding is not None:
                    # Try to parse embedding
                    try:
                        if isinstance(embedding, str):
                            embedding_data = json.loads(embedding)
                            print(f"  Embedding dimension: {len(embedding_data)}")
                        else:
                            print(f"  Embedding type: {type(embedding)}")
                    except Exception as e:
                        print(f"  Embedding parse error: {e}")
                else:
                    print("  No embedding stored!")
        
        # Check DocumentChunks table
        print("\n=== Checking RAG.DocumentChunks ===")
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            chunk_count = cursor.fetchone()[0]
            print(f"Total chunks: {chunk_count}")
        except Exception as e:
            print(f"DocumentChunks table error: {e}")
        
        connection.close()
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Checking database state...")
    success = check_database_state()
    sys.exit(0 if success else 1)