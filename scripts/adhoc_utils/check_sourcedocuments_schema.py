#!/usr/bin/env python3
"""
Check SourceDocuments table schema to match the vector datatype
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection

def check_sourcedocuments_schema():
    """Check the exact schema of SourceDocuments table"""
    print("🔍 Checking SourceDocuments Table Schema")
    print("=" * 50)
    
    iris_conn = get_iris_connection()
    cursor = iris_conn.cursor()
    
    try:
        # Get table schema
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        
        print("📊 SourceDocuments table schema:")
        for col_name, data_type, max_length, precision, scale in columns:
            if col_name == 'embedding':
                print(f"   🎯 {col_name}: {data_type}")
                if max_length:
                    print(f"      Max Length: {max_length}")
                if precision:
                    print(f"      Precision: {precision}")
                if scale:
                    print(f"      Scale: {scale}")
            else:
                print(f"   - {col_name}: {data_type}")
        
        # Also check the actual DDL
        print(f"\n📋 Getting table DDL...")
        cursor.execute("SHOW CREATE TABLE RAG.SourceDocuments")
        ddl_result = cursor.fetchone()
        if ddl_result:
            print(f"DDL: {ddl_result[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking schema: {e}")
        return False
    finally:
        cursor.close()

if __name__ == "__main__":
    check_sourcedocuments_schema()