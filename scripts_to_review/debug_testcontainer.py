#!/usr/bin/env python
"""
Debug script for IRIS testcontainer connections

This script isolates the testcontainer connection logic to help debug issues
with the test suite not properly running with 1000 documents.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main function to debug testcontainer connections"""
    logger.info("Starting testcontainer debug script")

    # Set environment variables
    os.environ["TEST_IRIS"] = "true"
    os.environ["TEST_DOCUMENT_COUNT"] = "20"  # Small value for quick testing

    try:
        # Import necessary modules
        logger.info("Importing testcontainer modules")
        try:
            from testcontainers.iris import IRISContainer
            import sqlalchemy
            from sqlalchemy import text
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Install testcontainers with: pip install testcontainers-iris sqlalchemy")
            return 1

        # Create and start container
        logger.info("Creating IRIS testcontainer")
        default_image = "intersystemsdc/iris-community:latest"
        image = os.environ.get("IRIS_DOCKER_IMAGE", default_image)
        
        container = IRISContainer(image)
        
        try:
            logger.info(f"Starting container with image: {image}")
            container.start()
            
            # Create connection URL manually
            host = container.get_container_host_ip()
            port = container.get_exposed_port(container.port)
            username = container.username
            password = container.password
            namespace = container.namespace
            
            connection_url = f"iris://{username}:{password}@{host}:{port}/{namespace}"
            logger.info(f"Connection URL: {connection_url}")
            
            # Create engine and connect
            logger.info("Creating SQLAlchemy engine")
            engine = sqlalchemy.create_engine(connection_url)
            
            logger.info("Connecting to database")
            connection = engine.connect()
            
            # Test the connection with simple queries
            logger.info("Testing connection with simple queries")
            
            # Test query 1: Simple SELECT
            try:
                logger.info("Running simple SELECT query")
                result = connection.execute(text("SELECT 1"))
                logger.info(f"Query result: {result.fetchone()}")
            except Exception as e:
                logger.error(f"Error running simple SELECT query: {e}")
            
            # Test query 2: Create table
            try:
                logger.info("Creating test table")
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS TestTable (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(1000),
                        description LONGVARCHAR
                    )
                """))
                logger.info("Table created successfully")
            except Exception as e:
                logger.error(f"Error creating table: {e}")
            
            # Test query 3: Insert data
            try:
                logger.info("Inserting test data")
                connection.execute(text("""
                    INSERT INTO TestTable (id, name, description)
                    VALUES ('test1', 'Test Name', 'This is a test description')
                """))
                connection.commit()
                logger.info("Data inserted successfully")
            except Exception as e:
                logger.error(f"Error inserting data: {e}")
            
            # Test query 4: Select data
            try:
                logger.info("Selecting test data")
                result = connection.execute(text("SELECT * FROM TestTable"))
                rows = result.fetchall()
                for row in rows:
                    logger.info(f"Row: {row}")
            except Exception as e:
                logger.error(f"Error selecting data: {e}")
            
            # Close connection
            logger.info("Closing connection")
            connection.close()
            engine.dispose()
            
        finally:
            # Stop container
            logger.info("Stopping container")
            container.stop()
            logger.info("Container stopped")
        
        logger.info("Testcontainer debug completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
