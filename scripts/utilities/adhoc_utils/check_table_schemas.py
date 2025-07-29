#!/usr/bin/env python3
"""
Check the actual schemas of SourceDocuments and Entities tables
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection

def check_schemas():
    """Check table schemas"""
    print("📋 Checking Table Schemas")
    print("=" * 60)
    
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    try:
        # Check SourceDocuments columns
        print("\n1️⃣ RAG.SourceDocuments columns:")
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'SourceDocuments'
            ORDER BY ORDINAL_POSITION
        """)
        
        for col_name, data_type, max_length in cursor.fetchall():
            print(f"   - {col_name}: {data_type}" + (f"({max_length})" if max_length else ""))
        
        # Check Entities columns
        print("\n2️⃣ RAG.Entities columns:")
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' 
            AND TABLE_NAME = 'Entities'
            ORDER BY ORDINAL_POSITION
        """)
        
        for col_name, data_type, max_length in cursor.fetchall():
            print(f"   - {col_name}: {data_type}" + (f"({max_length})" if max_length else ""))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    check_schemas()