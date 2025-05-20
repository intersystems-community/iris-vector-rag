#!/usr/bin/env python
# test_schema.py
# A simplified script to test the database schema initialization

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import get_iris_connector
from common.db_init import initialize_database

def main():
    print("Testing database schema initialization...")
    
    # Get connector
    connector = get_iris_connector()
    if not connector:
        print("Failed to connect to IRIS database.")
        return
    
    try:
        # Initialize database schema
        initialize_database(connector)
        print("Database schema initialized successfully.")
        
        # Verify tables were created
        cursor = connector.cursor()
        
        # Test SourceDocuments table
        print("\nVerifying tables...")
        tables = ["SourceDocuments", "DocumentTokenEmbeddings", "KnowledgeGraphNodes", "KnowledgeGraphEdges"]
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"Table {table} exists with {count} rows.")
            except Exception as e:
                print(f"Error verifying table {table}: {e}")
        
        # Test vector insertion with TO_VECTOR
        print("\nTesting vector insertion...")
        try:
            # Create a sample embedding
            sample_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            embedding_str = str(sample_embedding)
            
            # Insert a test document with embedding
            sql = "INSERT INTO SourceDocuments (doc_id, text_content, embedding) VALUES (?, ?, TO_VECTOR(?, FLOAT))"
            cursor.execute(sql, ("test_doc_1", "This is a test document", embedding_str))
            print("Successfully inserted test document with embedding.")
            
            # Verify the document was inserted
            cursor.execute("SELECT doc_id, text_content FROM SourceDocuments WHERE doc_id = 'test_doc_1'")
            result = cursor.fetchone()
            if result:
                print(f"Retrieved test document: {result}")
            else:
                print("Could not retrieve test document.")
        except Exception as e:
            print(f"Error testing vector insertion: {e}")
        
    except Exception as e:
        print(f"Error during database initialization: {e}")
    finally:
        if connector:
            # Close connection
            connector.close()
            print("\nConnection closed.")

if __name__ == "__main__":
    main()
