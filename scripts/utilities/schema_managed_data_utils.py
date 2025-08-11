#!/usr/bin/env python3
"""
Schema-Manager-Based Data Utilities

Provides data management operations using schema manager instead of hardcoded SQL.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.core.connection import ConnectionManager
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_status():
    """Check data status using schema manager."""
    logger.info("Checking data status using schema manager...")
    
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Check core table schema status
        core_tables = [
            "SourceDocuments", "DocumentChunks", "DocumentTokenEmbeddings",
            "Entities", "Relationships", "KnowledgeGraphNodes", "KnowledgeGraphEdges"
        ]

        for table in core_tables:
            needs_migration = schema_manager.needs_migration(table)
            logger.info(f"  {table}: {'✗ Needs migration' if needs_migration else '✓ Schema OK'}")
        
        # Get document count via safe query
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            print(f"Total documents: {doc_count}")
            return doc_count > 0
            
        finally:
            cursor.close()
            connection.close()
            
    except Exception as e:
        logger.error(f"Error checking data status: {e}")
        return False


def clear_rag_data():
    """Clear RAG data using schema manager."""
    logger.info("Clearing RAG data using schema manager...")
    
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Get connection
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Clear tables in dependency order (children first)
            tables_to_clear = [
                "SourceDocuments",
                "DocumentChunks",
                "DocumentTokenEmbeddings",
                "DocumentEntities",
                "EntityRelationships",
                "KnowledgeGraphNodes",
                "KnowledgeGraphEdges"
            ]
            
            total_cleared = 0
            for table in tables_to_clear:
                try:
                    # Schema manager has authority over all RAG tables, regardless of migration status
                    # First ensure the table schema is ready (migrate if needed)
                    logger.info(f"Schema manager ensuring {table} is ready for data operations...")
                    schema_manager.ensure_table_schema(table)
                    
                    # Now clear the data under schema manager authority
                    cursor.execute(f"DELETE FROM RAG.{table}")
                    rows_cleared = cursor.rowcount
                    print(f"{rows_cleared} rows deleted from RAG.{table}")
                    total_cleared += rows_cleared
                    
                except Exception as e:
                    logger.warning(f"Schema manager could not clear RAG.{table}: {e}")
                    # Still attempt basic clear if table exists but schema manager has issues
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
                        cursor.execute(f"DELETE FROM RAG.{table}")
                        rows_cleared = cursor.rowcount
                        print(f"{rows_cleared} rows deleted from RAG.{table} (fallback)")
                        total_cleared += rows_cleared
                    except:
                        logger.info(f"Table RAG.{table} does not exist or is inaccessible")
            
            connection.commit()
            logger.info(f"✓ Total rows cleared: {total_cleared}")
            return True
            
        finally:
            cursor.close()
            connection.close()
            
    except Exception as e:
        logger.error(f"Error clearing RAG data: {e}")
        return False


def sync_ifind_data():
    """Synchronize IFind tables using data sync manager."""
    logger.info("Synchronizing IFind tables using data sync manager...")
    
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Import and use data sync manager
        from iris_rag.validation.data_sync_manager import DataSyncManager
        data_sync_manager = DataSyncManager(connection_manager, schema_manager, config_manager)
        
        # Use data sync manager to handle IFind synchronization
        logger.info("Delegating IFind sync to data sync manager...")
        result = data_sync_manager._sync_ifind_data()
        
        if result.success:
            logger.info(f"✓ IFind sync successful: {result.message}")
            if result.rows_affected:
                logger.info(f"  Rows affected: {result.rows_affected}")
            return True
        else:
            logger.error(f"✗ IFind sync failed: {result.message}")
            return False
            
    except Exception as e:
        logger.error(f"Error syncing IFind data: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Schema-managed data utilities")
    parser.add_argument("--check", action="store_true", help="Check data status")
    parser.add_argument("--clear", action="store_true", help="Clear RAG data")
    parser.add_argument("--sync-ifind", action="store_true", help="Synchronize IFind tables")
    
    args = parser.parse_args()
    
    if args.check:
        success = check_data_status()
        sys.exit(0 if success else 1)
    elif args.clear:
        success = clear_rag_data()
        sys.exit(0 if success else 1)
    elif args.sync_ifind:
        success = sync_ifind_data()
        sys.exit(0 if success else 1)
    else:
        print("Usage: --check, --clear, or --sync-ifind")
        sys.exit(1)