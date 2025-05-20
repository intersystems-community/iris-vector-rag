#!/usr/bin/env python
"""
Debug script for the conftest.py fixture failures

This script emulates the exact steps from conftest.py's testcontainer 
and database initialization to identify where tests are failing.
"""

import os
import sys
import logging
import time
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def initialize_database_debug(connection):
    """Copy of the database initialization function with detailed logging"""
    
    logger.info("Initializing database with our debugging version")
    
    # Get the path to the SQL file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    schema_sql_path = os.path.join(current_dir, "common", "db_init.sql")
    
    # Read the SQL file
    with open(schema_sql_path, 'r') as f:
        schema_sql_content = f.read()
    
    # Split into individual statements
    sql_statements = []
    current_statement = []
    
    for line in schema_sql_content.splitlines():
        # Skip empty lines and comments-only lines
        line = line.strip()
        if not line or line.startswith('--'):
            continue
            
        current_statement.append(line)
        if line.endswith(';'):
            sql_statements.append(' '.join(current_statement))
            current_statement = []
    
    # Add any remaining statement without semicolon
    if current_statement:
        sql_statements.append(' '.join(current_statement))
    
    # Execute each statement and log extensively
    cursor = connection.cursor()
    
    for i, stmt in enumerate(sql_statements):
        stmt = stmt.strip()
        if not stmt:  # Skip empty statements
            continue
        
        # Skip comments-only statements
        if stmt.startswith('--'):
            continue
            
        # Remove any inline comments that might cause issues
        clean_stmt = ''
        for line in stmt.split('\n'):
            comment_pos = line.find('--')
            if comment_pos >= 0:
                line = line[:comment_pos]
            clean_stmt += line + ' '
        
        clean_stmt = clean_stmt.strip()
        if not clean_stmt:
            continue
        
        # Log the exact SQL being executed - this is critical for debugging
        logger.info(f"Executing SQL statement {i+1}/{len(sql_statements)}: {clean_stmt}")
        
        try:
            cursor.execute(clean_stmt)
            logger.info(f"SQL statement executed successfully")
        except Exception as e:
            logger.error(f"Error executing SQL statement: {e}")
            logger.error(f"Full SQL statement was: {clean_stmt}")
    
    # Commit the transaction
    try:
        connection.commit()
        logger.info("Transaction committed successfully")
    except Exception as e:
        logger.error(f"Error committing transaction: {e}")
    
    logger.info("Database initialization complete")

def main():
    """Main function to debug testcontainer and database initialization"""
    logger.info("Starting conftest debug script")

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
            
            # Get connection details exactly as done in conftest.py
            host = container.get_container_host_ip()
            port = container.get_exposed_port(container.port)
            username = container.username
            password = container.password
            namespace = container.namespace
            
            connection_url = f"iris://{username}:{password}@{host}:{port}/{namespace}"
            logger.info(f"Connection URL: {connection_url}")
            
            # Create engine and connect using SQLAlchemy exactly as in conftest.py
            logger.info("Creating SQLAlchemy engine")
            engine = sqlalchemy.create_engine(connection_url)
            
            # Get the connection directly from the engine
            logger.info("Getting connection from engine as done in conftest.py")
            connection = engine.connect().connection
            
            # Now initialize the database using our debug version
            initialize_database_debug(connection)
            
            # Verify tables exist by running simple select queries
            logger.info("Verifying tables were created")
            tables_to_check = [
                "SourceDocuments", 
                "KnowledgeGraphNodes", 
                "KnowledgeGraphEdges", 
                "DocumentTokenEmbeddings"
            ]
            
            cursor = connection.cursor()
            for table in tables_to_check:
                try:
                    logger.info(f"Checking table: {table}")
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    logger.info(f"Table {table} exists with {count} rows")
                except Exception as e:
                    logger.error(f"Error checking table {table}: {e}")
            
            # Test with sample data insertion
            try:
                logger.info("Inserting sample document")
                cursor.execute("""
                    INSERT INTO SourceDocuments 
                    (doc_id, title, content) 
                    VALUES ('test1', 'Test Document', 'This is a test document')
                """)
                connection.commit()
                logger.info("Sample document inserted successfully")
            except Exception as e:
                logger.error(f"Error inserting sample document: {e}")
            
            # Close connection
            logger.info("Closing connection")
            connection.close()
            engine.dispose()
            
        finally:
            # Stop container
            logger.info("Stopping container")
            container.stop()
            logger.info("Container stopped")
        
        logger.info("Debug script completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
