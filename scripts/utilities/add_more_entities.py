#!/usr/bin/env python3
"""
Add More Entities for GraphRAG

Uses schema manager and data sync manager for proper entity population.
NO hardcoded SQL - delegates to proper data management authorities.
"""

import os
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_more_entities():
    """Add more entities using schema manager and data sync manager."""
    logger.info("Using schema manager and data sync manager for entity population...")
    
    try:
        # Initialize managers with proper authority
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        schema_manager = SchemaManager(connection_manager, config_manager)
        data_sync_manager = DataSyncManager(connection_manager, schema_manager, config_manager)
        
        # Use data sync manager to handle entity population
        logger.info("Delegating entity population to data sync manager...")
        result = data_sync_manager._sync_graph_data()
        
        if result.success:
            logger.info(f"✓ Entity population successful: {result.message}")
            if result.rows_affected:
                logger.info(f"  Rows affected: {result.rows_affected}")
            return True
        else:
            logger.error(f"✗ Entity population failed: {result.message}")
            return False
            
    except Exception as e:
        logger.error(f"Error during entity population: {e}")
        return False


if __name__ == "__main__":
    success = add_more_entities()
    sys.exit(0 if success else 1)