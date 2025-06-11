#!/usr/bin/env python3
"""
Check what columns exist in RAG.SourceDocuments
"""

import intersystems_iris.dbapi._DBAPI as iris
from iris_rag.config.manager import ConfigurationManager

def check_columns():
    print("🔍 Checking columns in RAG.SourceDocuments...")
    
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        
        # Fetch database connection parameters from configuration
        db_host = config_manager.get("database:iris:host")
        db_port = config_manager.get("database:iris:port")
        db_namespace = config_manager.get("database:iris:namespace")
        db_username = config_manager.get("database:iris:username")
        db_password = config_manager.get("database:iris:password")
        
        # Connect to IRIS using configuration values
        connection = iris.connect(
            hostname=db_host,
            port=db_port,
            namespace=db_namespace,
            username=db_username,
            password=db_password
        )
        
        cursor = connection.cursor()
        
        # Check column names
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        if columns:
            print(f"✅ Found {len(columns)} columns:")
            for column_name, data_type, nullable in columns:
                print(f"  - {column_name} ({data_type}) {'NULL' if nullable == 'YES' else 'NOT NULL'}")
        else:
            print("❌ No columns found")
            
        # Sample some data to see what's actually there
        print(f"\n📄 Sample data from RAG.SourceDocuments:")
        cursor.execute("SELECT TOP 2 * FROM RAG.SourceDocuments")
        
        # Get column names from cursor description
        if cursor.description:
            col_names = [desc[0] for desc in cursor.description]
            print(f"  Columns: {col_names}")
            
            rows = cursor.fetchall()
            for i, row in enumerate(rows):
                print(f"\n  Row {i+1}:")
                for j, value in enumerate(row):
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"    {col_names[j]}: {value}")
        
        connection.close()
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_columns()