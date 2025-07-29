#!/usr/bin/env python3
"""
Verify that the DocumentChunks table fix is working properly.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection, IRISConnectionError

def verify_fix():
    """Verify that the DocumentChunks table is working properly."""
    print("🔍 Verifying DocumentChunks table fix...")
    
    try:
        # Connect to IRIS
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Test the exact query that was failing in monitoring
        print("📊 Testing DocumentChunks count query...")
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
        result = cursor.fetchone()
        count = result[0] if result else 0
        print(f"✅ DocumentChunks count: {count:,}")
        
        # Test table structure
        print("🏗️  Checking table structure...")
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks'
            ORDER BY ORDINAL_POSITION
        """)
        columns = cursor.fetchall()
        print("📋 Table columns:")
        for col_name, col_type in columns:
            print(f"   - {col_name}: {col_type}")
        
        # Test indexes
        print("🔍 Checking indexes...")
        cursor.execute("""
            SELECT INDEX_NAME 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks'
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        if indexes:
            print(f"📊 Indexes: {', '.join(indexes)}")
        else:
            print("⚠️  No indexes found")
        
        cursor.close()
        conn.close()
        
        print("\n✅ DocumentChunks table verification completed successfully!")
        print("🎉 The monitoring warning should now be resolved!")
        return True
        
    except IRISConnectionError as e:
        print(f"❌ Could not connect to IRIS: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)