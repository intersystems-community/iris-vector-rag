#!/usr/bin/env python3
"""
Check what tables actually exist in IRIS
"""

import intersystems_iris.dbapi._DBAPI as iris
from iris_rag.config.manager import ConfigurationManager

def check_tables():
    print("🔍 Checking what tables exist in IRIS...")
    
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
        
        # Check all tables in different schemas
        schemas_to_check = ['RAG_TEMPLATES', 'RAG', 'USER', '%SYS']
        
        for schema in schemas_to_check:
            print(f"\n📊 Checking schema: {schema}")
            try:
                cursor.execute(f"""
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = '{schema}'
                    ORDER BY TABLE_NAME
                """)
                
                tables = cursor.fetchall()
                if tables:
                    print(f"  ✅ Found {len(tables)} tables:")
                    for (table_name,) in tables:
                        print(f"    - {schema}.{table_name}")
                else:
                    print(f"  ❌ No tables found in {schema}")
                    
            except Exception as e:
                print(f"  ❌ Error checking {schema}: {e}")
        
        # Also check for any table with 'source' or 'document' in the name
        print(f"\n🔍 Searching for tables with 'source' or 'document' in name...")
        try:
            cursor.execute("""
                SELECT TABLE_SCHEMA, TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE UPPER(TABLE_NAME) LIKE '%SOURCE%' 
                   OR UPPER(TABLE_NAME) LIKE '%DOCUMENT%'
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """)
            
            doc_tables = cursor.fetchall()
            if doc_tables:
                print(f"  ✅ Found {len(doc_tables)} document-related tables:")
                for schema, table_name in doc_tables:
                    print(f"    - {schema}.{table_name}")
            else:
                print(f"  ❌ No document-related tables found")
                
        except Exception as e:
            print(f"  ❌ Error searching for document tables: {e}")
        
        connection.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    check_tables()