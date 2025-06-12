#!/usr/bin/env python3
"""
Query and display the current database schema for the 100K document ingestion.
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_schema_info(conn) -> Dict[str, Any]:
    """Query comprehensive schema information from IRIS database."""
    cursor = conn.cursor()
    schema_info = {}
    
    try:
        # 1. Get all tables in the RAG schema
        logger.info("Querying RAG schema tables...")
        cursor.execute("""
            SELECT TABLE_NAME, TABLE_TYPE 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG'
            ORDER BY TABLE_NAME
        """)
        tables = cursor.fetchall()
        schema_info['tables'] = [{'name': row[0], 'type': row[1]} for row in tables]
        
        # 2. Get detailed column information for each table
        schema_info['table_details'] = {}
        for table in schema_info['tables']:
            table_name = table['name']
            logger.info(f"Querying details for table: RAG.{table_name}")
            
            # Get column information
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """, (table_name,))
            columns = cursor.fetchall()
            
            # Get indexes for this table using IRIS-specific system tables
            try:
                cursor.execute("""
                    SELECT Name, Properties
                    FROM %Dictionary.IndexDefinition
                    WHERE parent = ?
                """, (f"RAG.{table_name}",))
                indexes = cursor.fetchall()
            except Exception as e:
                logger.warning(f"Could not query indexes for {table_name}: {e}")
                indexes = []
            
            # Get row count
            try:
                cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
                row_count = cursor.fetchone()[0]
            except Exception as e:
                logger.warning(f"Could not get row count for {table_name}: {e}")
                row_count = "Unknown"
            
            schema_info['table_details'][table_name] = {
                'columns': [
                    {
                        'name': col[0],
                        'type': col[1],
                        'nullable': col[2],
                        'default': col[3],
                        'max_length': col[4]
                    } for col in columns
                ],
                'indexes': [
                    {
                        'name': idx[0],
                        'properties': idx[1] if len(idx) > 1 else 'N/A'
                    } for idx in indexes
                ],
                'row_count': row_count
            }
        
        # 3. Check for vector search related objects
        logger.info("Querying vector search objects...")
        try:
            cursor.execute("""
                SELECT NAME, TYPE 
                FROM %Dictionary.CompiledClass 
                WHERE NAME %STARTSWITH 'RAG.'
                ORDER BY NAME
            """)
            vector_objects = cursor.fetchall()
            schema_info['vector_objects'] = [{'name': row[0], 'type': row[1]} for row in vector_objects]
        except Exception as e:
            logger.warning(f"Could not query vector objects: {e}")
            schema_info['vector_objects'] = []
        
        # 4. Check for stored procedures
        logger.info("Querying stored procedures...")
        try:
            cursor.execute("""
                SELECT ROUTINE_NAME, ROUTINE_TYPE 
                FROM INFORMATION_SCHEMA.ROUTINES 
                WHERE ROUTINE_SCHEMA = 'RAG'
                ORDER BY ROUTINE_NAME
            """)
            routines = cursor.fetchall()
            schema_info['routines'] = [{'name': row[0], 'type': row[1]} for row in routines]
        except Exception as e:
            logger.warning(f"Could not query routines: {e}")
            schema_info['routines'] = []
            
    except Exception as e:
        logger.error(f"Error querying schema: {e}")
        raise
    finally:
        cursor.close()
    
    return schema_info

def print_schema_report(schema_info: Dict[str, Any]):
    """Print a formatted schema report."""
    print("\n" + "="*80)
    print("IRIS DATABASE SCHEMA REPORT - 100K DOCUMENT INGESTION")
    print("="*80)
    
    # Connection info
    print(f"\nDatabase: Licensed IRIS Instance")
    print(f"Host: localhost:1972")
    print(f"Namespace: USER")
    print(f"Schema: RAG")
    
    # Tables overview
    print(f"\n📊 TABLES OVERVIEW")
    print("-" * 40)
    if schema_info.get('tables'):
        for table in schema_info['tables']:
            row_count = schema_info['table_details'].get(table['name'], {}).get('row_count', 'Unknown')
            print(f"  • {table['name']} ({table['type']}) - {row_count} rows")
    else:
        print("  No tables found in RAG schema")
    
    # Detailed table information
    print(f"\n📋 DETAILED TABLE STRUCTURES")
    print("-" * 40)
    
    for table_name, details in schema_info.get('table_details', {}).items():
        print(f"\n🔹 RAG.{table_name}")
        print(f"   Rows: {details['row_count']}")
        
        print("   Columns:")
        for col in details['columns']:
            nullable = "NULL" if col['nullable'] == 'YES' else "NOT NULL"
            max_len = f"({col['max_length']})" if col['max_length'] else ""
            default = f" DEFAULT {col['default']}" if col['default'] else ""
            print(f"     - {col['name']}: {col['type']}{max_len} {nullable}{default}")
        
        if details['indexes']:
            print("   Indexes:")
            for idx in details['indexes']:
                print(f"     - {idx['name']}: {idx['properties']}")
    
    # Vector search objects
    if schema_info.get('vector_objects'):
        print(f"\n🔍 VECTOR SEARCH OBJECTS")
        print("-" * 40)
        for obj in schema_info['vector_objects']:
            print(f"  • {obj['name']} ({obj['type']})")
    
    # Stored procedures
    if schema_info.get('routines'):
        print(f"\n⚙️ STORED PROCEDURES")
        print("-" * 40)
        for routine in schema_info['routines']:
            print(f"  • {routine['name']} ({routine['type']})")
    
    print("\n" + "="*80)

def main():
    """Main function to query and display schema information."""
    try:
        # Set connection parameters for licensed IRIS instance
        config = {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "_SYSTEM",
            "password": "SYS"
        }
        
        logger.info("Connecting to licensed IRIS database...")
        conn = get_iris_connection(use_mock=False, use_testcontainer=False, config=config)
        
        logger.info("Querying schema information...")
        schema_info = query_schema_info(conn)
        
        # Print the schema report
        print_schema_report(schema_info)
        
        conn.close()
        logger.info("Schema query completed successfully.")
        
    except Exception as e:
        logger.error(f"Failed to query schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()