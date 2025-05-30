#!/usr/bin/env python3
"""
Check the actual column types in IRIS using SQL Shell equivalent commands
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector_jdbc import get_iris_connection

def check_column_types():
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    print("Checking column types in IRIS...")
    print("=" * 60)
    
    # Method 1: Using INFORMATION_SCHEMA
    print("\n1. INFORMATION_SCHEMA view:")
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND COLUMN_NAME = 'embedding'
        ORDER BY TABLE_NAME
    """)
    
    for row in cursor.fetchall():
        print(f"   {row[0]}.{row[1]}: {row[2]} (max_length: {row[3]})")
    
    # Method 2: Check actual table definition
    print("\n2. Table definitions:")
    tables = ['SourceDocuments_V2', 'DocumentChunks']
    
    for table in tables:
        try:
            # Get table info
            cursor.execute(f"SELECT TOP 0 * FROM RAG.{table}")
            columns = cursor.description
            
            print(f"\n   RAG.{table}:")
            for col in columns:
                if col[0].lower() == 'embedding':
                    print(f"      {col[0]}: type_code={col[1]}, display_size={col[2]}")
        except Exception as e:
            print(f"   Error checking {table}: {e}")
    
    # Method 3: Test actual data type behavior
    print("\n3. Testing actual data type behavior:")
    
    # Test if we can use vector functions
    print("\n   Testing SourceDocuments.embedding:")
    try:
        cursor.execute("""
            SELECT TOP 1 
            LENGTH(embedding) as varchar_length,
            VECTOR_DOT_PRODUCT(TO_VECTOR(embedding), TO_VECTOR(embedding)) as dot_product
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
        """)
        result = cursor.fetchone()
        print(f"      VARCHAR length: {result[0]}")
        print(f"      Can use with TO_VECTOR: YES (dot product: {result[1]})")
    except Exception as e:
        print(f"      Error: {e}")
    
    # Check if it's stored as VARCHAR or VECTOR
    print("\n   Checking storage format:")
    try:
        cursor.execute("""
            SELECT TOP 1 
            SUBSTRING(embedding, 1, 50) as first_50_chars
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
        """)
        result = cursor.fetchone()
        print(f"      First 50 chars: {result[0]}")
        print(f"      => Stored as: VARCHAR (comma-separated values)")
    except Exception as e:
        print(f"      Error: {e}")
    
    # Method 4: Check system catalog
    print("\n4. System catalog information:")
    try:
        cursor.execute("""
            SELECT 
                parent->SqlTableName as table_name,
                SqlFieldName as column_name,
                Type as data_type
            FROM %Dictionary.CompiledProperty
            WHERE parent->SqlSchemaName = 'RAG'
            AND SqlFieldName = 'embedding'
        """)
        
        for row in cursor.fetchall():
            print(f"   {row[0]}.{row[1]}: {row[2]}")
    except Exception as e:
        print(f"   Error accessing system catalog: {e}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_column_types()