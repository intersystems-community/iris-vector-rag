#!/usr/bin/env python
"""
Simple script to test connection to IRIS 2025.1 using the updated driver.
"""

import os
import sys
import logging
import intersystems_iris

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    """Test connection to IRIS 2025.1."""
    try:
        # Connection parameters
        conn_params = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "SuperUser",
            "password": "SYS"
        }
        
        # Log connection attempt
        logger.info(f"Attempting to connect to IRIS at {conn_params['hostname']}:{conn_params['port']}/{conn_params['namespace']}")
        
        # Connect to IRIS
        conn = intersystems_iris.connect(**conn_params)
        
        # Test the connection
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        
        # Log success
        logger.info(f"Successfully connected to IRIS. Test query result: {result}")
        logger.info("Connection test passed!")
        
        # Close the connection
        conn.close()
        logger.info("Connection closed.")
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to IRIS: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)