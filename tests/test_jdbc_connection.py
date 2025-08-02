#!/usr/bin/env python3
"""Test JDBC connection to IRIS"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.iris_connection_manager import get_iris_jdbc_connection

def test_connection():
    """Test JDBC connection with different credentials"""
    
    # Try different credential combinations
    credentials = [
        {"username": "demo", "password": "demo"},
        {"username": "_SYSTEM", "password": "SYS"},
        {"username": "SuperUser", "password": "SYS"},
    ]
    
    for creds in credentials:
        print(f"\nTrying connection with username: {creds['username']}")
        try:
            conn = get_iris_jdbc_connection(
                host='localhost',
                port=1972,
                namespace='USER',
                username=creds['username'],
                password=creds['password']
            )
            print(f"✅ Successfully connected with {creds['username']}")
            
            # Test a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"   Test query result: {result}")
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed with {creds['username']}: {e}")
    
    return False

if __name__ == "__main__":
    if test_connection():
        print("\n✅ JDBC connection test successful!")
    else:
        print("\n❌ All JDBC connection attempts failed!")