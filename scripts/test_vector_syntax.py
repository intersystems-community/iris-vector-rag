#!/usr/bin/env python3
"""
Test VECTOR column syntax for IRIS 2025.1
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection

def test_vector_syntax():
    print('=== TESTING CORRECT VECTOR SYNTAX FOR IRIS 2025.1 ===')
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Test different VECTOR column syntaxes based on documentation
    test_syntaxes = [
        'VECTOR',
        'VECTOR(768)',
        'VECTOR(768, DOUBLE)',
        'VECTOR(FLOAT, 768)',
        'VECTOR(768, FLOAT)',
        'VECTOR(FLOAT, 768)'
    ]
    
    working_syntax = None
    
    for syntax in test_syntaxes:
        try:
            table_suffix = syntax.replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "_")
            test_table = f'RAG_HNSW.test_vector_syntax_{table_suffix}'
            
            cursor.execute(f'DROP TABLE IF EXISTS {test_table}')
            cursor.execute(f'CREATE TABLE {test_table} (id INT, vec {syntax})')
            
            # Check the actual column type
            cursor.execute(f"""
                SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG_HNSW' 
                AND TABLE_NAME = '{test_table.split('.')[1]}' 
                AND COLUMN_NAME = 'vec'
            """)
            actual_type = cursor.fetchone()[0]
            print(f'✅ {syntax} -> {actual_type}')
            
            # Test TO_VECTOR with this column - CORRECTED SYNTAX
            cursor.execute(f"INSERT INTO {test_table} (id, vec) VALUES (1, TO_VECTOR('1,2,3', double))")
            cursor.execute(f"SELECT vec FROM {test_table} WHERE id = 1")
            result = cursor.fetchone()[0]
            print(f'   TO_VECTOR test: {str(result)[:50]}...')
            
            cursor.execute(f'DROP TABLE {test_table}')
            working_syntax = syntax
            break  # Use the first working syntax
            
        except Exception as e:
            print(f'❌ {syntax} failed: {e}')
    
    cursor.close()
    conn.close()
    
    return working_syntax

if __name__ == "__main__":
    working_syntax = test_vector_syntax()
    if working_syntax:
        print(f"\n✅ WORKING VECTOR SYNTAX: {working_syntax}")
    else:
        print("\n❌ NO WORKING VECTOR SYNTAX FOUND")