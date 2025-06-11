#!/usr/bin/env python3
"""
Quick database verification script to check IRIS connection and table contents.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection, IRISConnectionError

def verify_database():
    """Verify database connection and check table contents."""
    print("🔍 IRIS Database Verification")
    print("=" * 40)
    
    try:
        # Test connection
        print("🔌 Testing database connection...")
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"✅ Connection successful: {result[0]}")
        
        # Check available schemas
        print("\n📋 Available schemas:")
        cursor.execute("SELECT DISTINCT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA ORDER BY SCHEMA_NAME")
        schemas = cursor.fetchall()
        for schema in schemas:
            print(f"   - {schema[0]}")
        
        # Check RAG schema tables
        print("\n📊 RAG schema tables:")
        cursor.execute("""
            SELECT TABLE_NAME, TABLE_TYPE 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG' 
            ORDER BY TABLE_NAME
        """)
        tables = cursor.fetchall()
        
        if not tables:
            print("   ⚠️  No RAG tables found")
        else:
            for table_name, table_type in tables:
                print(f"   - {table_name} ({table_type})")
                
                # Get row count for each table
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
                    count = cursor.fetchone()[0]
                    print(f"     Rows: {count:,}")
                except Exception as e:
                    print(f"     Error counting rows: {e}")
        
        # Check for any other schemas with data
        print("\n🔍 Other schemas with tables:")
        cursor.execute("""
            SELECT TABLE_SCHEMA, COUNT(*) as table_count
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', '%SYS', 'SQLUSER')
            GROUP BY TABLE_SCHEMA
            ORDER BY TABLE_SCHEMA
        """)
        other_schemas = cursor.fetchall()
        
        for schema_name, table_count in other_schemas:
            print(f"   - {schema_name}: {table_count} tables")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Database verification completed successfully")
        return True
        
    except IRISConnectionError as e:
        print(f"❌ IRIS connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_database()
    sys.exit(0 if success else 1)