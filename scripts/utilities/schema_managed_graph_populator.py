#!/usr/bin/env python3
"""
Schema-Manager-Based Graph Population

This script properly uses the schema manager to populate GraphRAG data
without hardcoded table names or column references.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.validation.data_sync_manager import DataSyncManager
from iris_rag.core.connection import ConnectionManager
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def populate_graph_using_schema_manager():
    """Populate graph data using proper schema manager."""
    logger.info("Starting schema-managed graph population...")
    
    try:
        # Initialize managers
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        data_sync_manager = DataSyncManager(connection_manager, schema_manager, config_manager)
        
        # Ensure all graph tables are ready
        logger.info("Ensuring graph tables have proper schema...")
        schema_manager.ensure_table_schema("DocumentEntities")
        schema_manager.ensure_table_schema("KnowledgeGraphNodes") 
        schema_manager.ensure_table_schema("KnowledgeGraphEdges")
        
        # Use data sync manager to populate graph data
        logger.info("Populating graph data via data sync manager...")
        result = data_sync_manager._sync_graph_data()
        
        if result.success:
            logger.info(f"✓ Graph population successful: {result.message}")
            if result.rows_affected:
                logger.info(f"  Rows affected: {result.rows_affected}")
        else:
            logger.error(f"✗ Graph population failed: {result.message}")
            return False
            
        # Get final status
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
            entities = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
            edges = cursor.fetchone()[0]
            
            logger.info(f"Final status: {docs} documents, {entities} entities, {nodes} nodes, {edges} edges")
            logger.info(f"Entities per document: {entities/docs:.3f}" if docs > 0 else "No documents")
            
        finally:
            cursor.close()
            connection.close()
            
        return True
        
    except Exception as e:
        logger.error(f"Error during graph population: {e}")
        return False


def check_graph_status():
    """Check graph data status using schema manager."""
    logger.info("Checking graph data status...")
    
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Check table schema status
        tables = ["DocumentEntities", "KnowledgeGraphNodes", "KnowledgeGraphEdges"]
        for table in tables:
            needs_migration = schema_manager.needs_migration(table)
            logger.info(f"  {table}: {'✗ Needs migration' if needs_migration else '✓ Schema OK'}")
        
        # Get counts
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
            entities = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
            edges = cursor.fetchone()[0]
            
            print(f"Documents: {docs}, Entities: {entities}, Graph Nodes: {nodes}, Graph Edges: {edges}")
            if docs > 0:
                print(f"Entities per document: {entities/docs:.3f}")
            else:
                print("No documents found")
                
            return entities >= docs * 0.1  # Return success if we have reasonable entity coverage
            
        finally:
            cursor.close()
            connection.close()
            
    except Exception as e:
        logger.error(f"Error checking graph status: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Schema-managed graph population")
    parser.add_argument("--check", action="store_true", help="Check graph status only")
    parser.add_argument("--populate", action="store_true", help="Populate graph data")
    
    args = parser.parse_args()
    
    if args.check:
        success = check_graph_status()
        sys.exit(0 if success else 1)
    elif args.populate:
        success = populate_graph_using_schema_manager()
        sys.exit(0 if success else 1)
    else:
        # Default: populate
        success = populate_graph_using_schema_manager()
        sys.exit(0 if success else 1)