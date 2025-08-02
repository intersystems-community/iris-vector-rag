#!/usr/bin/env python3
"""Check the exact status of SourceDocuments tables"""

import logging
from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("SOURCEDOCUMENTS TABLE STATUS CHECK")
    print("="*80 + "\n")
    
    # Check what tables exist
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'RAG'
        AND TABLE_NAME LIKE 'SourceDocuments%'
        ORDER BY TABLE_NAME
    """)
    
    tables = cursor.fetchall()
    print("📋 Found tables:")
    for table in tables:
        print(f"   - {table[0]}")
        
    # Check each table's structure and data
    for table_name in [t[0] for t in tables]:
        print(f"\n📊 {table_name}:")
        
        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
            count = cursor.fetchone()[0]
            print(f"   Total records: {count:,}")
        except Exception as e:
            print(f"   Error counting records: {e}")
            
        # Get columns
        try:
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG'
                AND TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """)
            columns = cursor.fetchall()
            print(f"   Columns:")
            for col_name, col_type in columns:
                print(f"     - {col_name}: {col_type}")
        except Exception as e:
            print(f"   Error getting columns: {e}")
            
        # Check for indexes
        try:
            cursor.execute(f"""
                SELECT INDEX_NAME, COLUMN_NAME
                FROM INFORMATION_SCHEMA.INDEXES
                WHERE TABLE_SCHEMA = 'RAG'
                AND TABLE_NAME = '{table_name}'
                ORDER BY INDEX_NAME
            """)
            indexes = cursor.fetchall()
            if indexes:
                print(f"   Indexes:")
                for idx_name, col_name in indexes:
                    print(f"     - {idx_name} on {col_name}")
        except Exception as e:
            print(f"   Error getting indexes: {e}")
    
    # Check if we need to do any renaming
    print("\n" + "="*80)
    print("MIGRATION STATUS:")
    print("="*80)
    
    has_v2 = any(t[0] == 'SourceDocuments_V2' for t in tables)
    has_original = any(t[0] == 'SourceDocuments' for t in tables)
    has_old = any(t[0] == 'SourceDocuments_OLD' for t in tables)
    
    if has_v2 and not has_original:
        print("✅ Migration appears complete - only SourceDocuments_V2 exists")
        print("⚠️  Need to rename SourceDocuments_V2 to SourceDocuments")
    elif has_v2 and has_original:
        print("⚠️  Both SourceDocuments and SourceDocuments_V2 exist")
        print("   Need to backup original and rename V2")
    elif has_original and not has_v2:
        print("❌ No V2 table found - migration not started")
    else:
        print("❓ Unexpected state")
        
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()