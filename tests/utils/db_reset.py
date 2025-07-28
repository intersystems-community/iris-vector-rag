# tests/utils/db_reset.py

import logging

class DatabaseReset:
    """Utilities for resetting database state between tests."""
    
    def clean_test_tables(self, connection) -> None:
        """Clean test tables to ensure isolated test state."""
        cursor = connection.cursor()
        
        # List of tables to clean (preserve data, clean test artifacts)
        test_tables = [
            "RAG.TestResults",
            "RAG.TestDocuments", 
            "RAG.TestEmbeddings"
        ]
        
        for table in test_tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
                logging.debug(f"Cleaned table {table}")
            except Exception as e:
                # Table might not exist, which is fine
                logging.debug(f"Could not clean {table}: {e}")
        
        connection.commit()
        cursor.close()
    
    def ensure_test_schema(self, connection) -> None:
        """Ensure test schema exists."""
        cursor = connection.cursor()
        
        # Create schema if it doesn't exist
        try:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS RAG")
            connection.commit()
        except Exception as e:
            logging.debug(f"Schema creation note: {e}")
        
        cursor.close()