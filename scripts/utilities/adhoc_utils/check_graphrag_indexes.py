#!/usr/bin/env python3
"""
Check existing indexes on GraphRAG tables
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection

def check_indexes():
    """Check all indexes on GraphRAG tables"""
    print("🔍 Checking Indexes on GraphRAG Tables")
    print("=" * 60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    try:
        # Check all indexes on Entities table
        print("\n1️⃣ Indexes on RAG.Entities:")
        cursor.execute("""
            SELECT Name, Type, Properties
            FROM %Dictionary.CompiledIndex 
            WHERE Parent = 'RAG.Entities'
            ORDER BY Name
        """)
        
        entities_indexes = cursor.fetchall()
        if entities_indexes:
            for idx_name, idx_type, properties in entities_indexes:
                print(f"   - {idx_name} (Type: {idx_type}, Properties: {properties})")
        else:
            print("   No indexes found")
        
        # Check all indexes on SourceDocuments
        print("\n2️⃣ Indexes on RAG.SourceDocuments:")
        cursor.execute("""
            SELECT Name, Type, Properties
            FROM %Dictionary.CompiledIndex 
            WHERE Parent = 'RAG.SourceDocuments'
            ORDER BY Name
        """)
        
        source_indexes = cursor.fetchall()
        if source_indexes:
            for idx_name, idx_type, properties in source_indexes:
                print(f"   - {idx_name} (Type: {idx_type}, Properties: {properties})")
        else:
            print("   No indexes found")
        
        # Check if Entities has a vector column
        print("\n3️⃣ Checking Entities table structure:")
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'Entities'
            AND COLUMN_NAME = 'embedding'
        """)
        
        col_info = cursor.fetchone()
        if col_info:
            col_name, data_type, max_length = col_info
            print(f"   - Column: {col_name}")
            print(f"   - Type: {data_type}")
            print(f"   - Max Length: {max_length}")
        
        # Check if SourceDocuments has a vector column
        print("\n4️⃣ Checking SourceDocuments table structure:")
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            AND COLUMN_NAME = 'embedding'
        """)
        
        col_info = cursor.fetchone()
        if col_info:
            col_name, data_type, max_length = col_info
            print(f"   - Column: {col_name}")
            print(f"   - Type: {data_type}")
            print(f"   - Max Length: {max_length}")
        
        # Check for HNSW indexes specifically
        print("\n5️⃣ HNSW Indexes Summary:")
        cursor.execute("""
            SELECT Parent, Name
            FROM %Dictionary.CompiledIndex 
            WHERE Name LIKE '%hnsw%' OR Name LIKE '%HNSW%'
            ORDER BY Parent, Name
        """)
        
        hnsw_indexes = cursor.fetchall()
        if hnsw_indexes:
            for parent, idx_name in hnsw_indexes:
                print(f"   - {parent}.{idx_name}")
        else:
            print("   No HNSW indexes found")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    check_indexes()