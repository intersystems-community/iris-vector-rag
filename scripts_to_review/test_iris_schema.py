#!/usr/bin/env python
# test_iris_schema.py
# A script to test the database schema with testcontainers-iris

import os
import sys
import sqlalchemy
from testcontainers.iris import IRISContainer
from common.db_init import initialize_database

def main():
    print("Testing IRIS schema using testcontainers...")
    
    # Use the appropriate image for the architecture - ARM64 for Apple Silicon
    is_arm64 = os.uname().machine == 'arm64'
    # Use latest tag since specific version tags might not be available
    default_image = "intersystemsdc/iris-community:latest"
    iris_image_tag = os.getenv("IRIS_DOCKER_IMAGE", default_image)
    print(f"Using IRIS Docker image: {iris_image_tag} on {'ARM64' if is_arm64 else 'x86_64'} architecture")
    
    # Let Docker handle architecture automatically without specifying platform
    with IRISContainer(iris_image_tag) as iris_container:
        connection_url = iris_container.get_connection_url()
        print(f"IRIS Testcontainer started. Connection URL: {connection_url}")
        
        engine = sqlalchemy.create_engine(connection_url)
        sa_connection = None
        raw_dbapi_connection = None
        
        try:
            # Get the raw DBAPI connection from SQLAlchemy
            sa_connection = engine.connect()
            raw_dbapi_connection = sa_connection.connection
            
            print(f"Raw DB-API connection obtained: {raw_dbapi_connection}")
            
            # Initialize the database schema
            print("Initializing database schema...")
            initialize_database(raw_dbapi_connection)
            print("Database schema initialized successfully.")
            
            # Test vector insertion
            print("\nTesting vector insertion...")
            cursor = raw_dbapi_connection.cursor()
            
            try:
                # Create a sample embedding for VECTOR storage
                sample_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
                
                # Insert a test document with vector embedding
                # We can directly pass the list for VECTOR columns in IRIS 2025.1
                sql = "INSERT INTO SourceDocuments (doc_id, text_content, embedding) VALUES (?, ?, ?)"
                cursor.execute(sql, ("test_doc_1", "This is a test document", sample_embedding))
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
            # Clean up connections
            if sa_connection:
                sa_connection.close()
            if engine:
                engine.dispose()
            
    print("\nTest complete.")

if __name__ == "__main__":
    main()
