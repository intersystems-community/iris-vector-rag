#!/usr/bin/env python
# run_validation_tests.py
# Script to run SQL validation tests using testcontainers-iris

import os
import sys
import pytest
import sqlalchemy
import time
from testcontainers.iris import IRISContainer
from common.db_init import initialize_database

def main():
    print("Starting IRIS container for SQL validation tests...")
    
    # Use the appropriate image for the architecture - ARM64 for Apple Silicon
    is_arm64 = os.uname().machine == 'arm64'
    default_image = "intersystemsdc/iris-community:latest"
    iris_image_tag = os.getenv("IRIS_DOCKER_IMAGE", default_image)
    print(f"Using IRIS Docker image: {iris_image_tag} on {'ARM64' if is_arm64 else 'x86_64'} architecture")
    
    # Start IRIS container
    with IRISContainer(iris_image_tag) as iris_container:
        connection_url = iris_container.get_connection_url()
        print(f"IRIS Testcontainer started. Connection URL: {connection_url}")
        
        # Parse connection URL to get components
        # Format is typically: iris://username:password@host:port/namespace
        url_parts = connection_url.replace("iris://", "").split("/")
        namespace = url_parts[-1]
        auth_host_port = url_parts[0].split("@")
        auth = auth_host_port[0].split(":")
        username = auth[0]
        password = auth[1]
        host_port = auth_host_port[1].split(":")
        host = host_port[0]
        port = host_port[1]
        
        # Get SQLAlchemy connection and raw DBAPI connection
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
            
            # Set environment variables for the tests
            os.environ["IRIS_HOST"] = host
            os.environ["IRIS_PORT"] = port
            os.environ["IRIS_NAMESPACE"] = namespace
            os.environ["IRIS_USERNAME"] = username
            os.environ["IRIS_PASSWORD"] = password
            
            # Allow a moment for initialization to complete
            time.sleep(1)
            
            # Run the SQL validation tests using Poetry
            print("\nRunning SQL validation tests with Poetry...")
            os.system(
                "poetry run pytest "
                "tests/test_colbert.py::test_colbert_sql_syntax_validation "
                "tests/test_noderag.py::test_noderag_sql_syntax_validation "
                "tests/test_graphrag.py::test_graphrag_sql_syntax_validation "
                "-v"
            )
            
        except Exception as e:
            print(f"Error during SQL validation tests: {e}")
        finally:
            # Clean up connections
            if sa_connection:
                sa_connection.close()
            if engine:
                engine.dispose()
            
    print("\nSQL validation tests complete.")

if __name__ == "__main__":
    main()
