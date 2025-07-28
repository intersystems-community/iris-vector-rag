#!/usr/bin/env python3
"""
Debug script to check the actual column mapping being used by IRISVectorStore.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_rag.storage.vector_store_iris import IRISVectorStore
from iris_rag.config.manager import ConfigurationManager

def main():
    """Debug the column mapping in IRISVectorStore."""
    print("🔍 Debugging IRISVectorStore column mapping...")
    
    # Initialize components
    config_manager = ConfigurationManager()
    vector_store = IRISVectorStore(config_manager)
    
    print(f"📋 Table name: {vector_store.table_name}")
    print(f"🆔 ID column: {vector_store.id_column}")
    print(f"📝 Content column: {vector_store.content_column}")
    print(f"🔢 Embedding column: {vector_store.embedding_column}")
    
    # Check the actual table schema
    connection = vector_store._get_connection()
    cursor = connection.cursor()
    
    try:
        # Get table structure
        cursor.execute(f"DESCRIBE {vector_store.table_name}")
        columns = cursor.fetchall()
        print(f"\n📊 Actual table structure for {vector_store.table_name}:")
        for col in columns:
            print(f"   - {col}")
    except Exception as e:
        print(f"❌ Error getting table structure: {e}")
        
        # Try alternative method
        try:
            cursor.execute(f"SELECT TOP 0 * FROM {vector_store.table_name}")
            column_names = [desc[0] for desc in cursor.description]
            print(f"\n📊 Column names from SELECT: {column_names}")
        except Exception as e2:
            print(f"❌ Error with SELECT method: {e2}")
    
    return 0

if __name__ == "__main__":
    exit(main())