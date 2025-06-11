#!/usr/bin/env python3
"""
Schema management system for IRIS RAG tables with automatic migration support.

This module provides robust schema versioning, configuration tracking, and 
automatic migration capabilities for vector dimensions and other schema changes.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Manages database schema versions and automatic migrations.
    
    Features:
    - Tracks vector dimensions and other configuration parameters
    - Automatically detects configuration changes
    - Performs safe schema migrations
    - Maintains schema version history
    """
    
    def __init__(self, connection_manager, config_manager):
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.schema_version = "1.0.0"
        
    def ensure_schema_metadata_table(self):
        """Create schema metadata table if it doesn't exist."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            create_sql = """
            CREATE TABLE IF NOT EXISTS RAG.SchemaMetadata (
                table_name VARCHAR(255) NOT NULL,
                schema_version VARCHAR(50) NOT NULL,
                vector_dimension INTEGER,
                embedding_model VARCHAR(255),
                configuration VARCHAR(MAX),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (table_name)
            )
            """
            cursor.execute(create_sql)
            connection.commit()
            logger.info("✅ Schema metadata table ensured")
            
        except Exception as e:
            logger.error(f"Failed to create schema metadata table: {e}")
            raise
        finally:
            cursor.close()
    
    def get_current_schema_config(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get current schema configuration for a table."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute("""
                SELECT schema_version, vector_dimension, embedding_model, configuration
                FROM RAG.SchemaMetadata 
                WHERE table_name = ?
            """, [table_name])
            
            result = cursor.fetchone()
            if result:
                schema_version, vector_dim, embedding_model, config_json = result
                config = json.loads(config_json) if config_json else {}
                return {
                    "schema_version": schema_version,
                    "vector_dimension": vector_dim,
                    "embedding_model": embedding_model,
                    "configuration": config
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get schema config for {table_name}: {e}")
            return None
        finally:
            cursor.close()
    
    def _get_expected_schema_config(self, table_name: str) -> Dict[str, Any]:
        """Get expected schema configuration based on current system config."""
        if table_name == "DocumentEntities":
            # Get embedding model configuration
            embedding_config = self.config_manager.get_embedding_config()
            model_name = embedding_config.get("model", "all-MiniLM-L6-v2")
            
            # Get vector data type from configuration, default to FLOAT
            vector_data_type = self.config_manager.get("storage:iris:vector_data_type", "FLOAT")
            
            # Map model names to dimensions
            model_dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            
            expected_dim = model_dimensions.get(model_name, 384)  # Default to 384
            
            return {
                "schema_version": self.schema_version,
                "vector_dimension": expected_dim,
                "embedding_model": model_name,
                "vector_data_type": vector_data_type,
                "configuration": {
                    "table_type": "entity_storage",
                    "supports_vector_search": True,
                    "created_by": "GraphRAG",
                    "auto_migration": True
                }
            }
        
        # Add other table configurations as needed
        return {
            "schema_version": self.schema_version,
            "configuration": {}
        }
    
    def needs_migration(self, table_name: str) -> bool:
        """Check if table needs migration based on configuration changes."""
        current_config = self.get_current_schema_config(table_name)
        expected_config = self._get_expected_schema_config(table_name)
        
        if not current_config:
            logger.info(f"Table {table_name} has no schema metadata - migration needed")
            return True
        
        # Check vector dimension mismatch
        if current_config.get("vector_dimension") != expected_config.get("vector_dimension"):
            logger.info(f"Vector dimension mismatch for {table_name}: "
                       f"current={current_config.get('vector_dimension')}, "
                       f"expected={expected_config.get('vector_dimension')}")
            return True
        
        # Check embedding model change
        if current_config.get("embedding_model") != expected_config.get("embedding_model"):
            logger.info(f"Embedding model change for {table_name}: "
                       f"current={current_config.get('embedding_model')}, "
                       f"expected={expected_config.get('embedding_model')}")
            return True
        
        # Check schema version
        if current_config.get("schema_version") != expected_config.get("schema_version"):
            logger.info(f"Schema version change for {table_name}: "
                       f"current={current_config.get('schema_version')}, "
                       f"expected={expected_config.get('schema_version')}")
            return True
        
        return False
    
    def migrate_table(self, table_name: str, preserve_data: bool = False) -> bool:
        """
        Migrate table to match expected configuration.
        
        Args:
            table_name: Name of table to migrate
            preserve_data: Whether to attempt data preservation (not implemented yet)
            
        Returns:
            True if migration successful, False otherwise
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            expected_config = self._get_expected_schema_config(table_name)
            
            if table_name == "DocumentEntities":
                success = self._migrate_document_entities_table(cursor, expected_config, preserve_data)
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            
            # Add other table migrations as needed
            logger.warning(f"No migration handler for table {table_name}")
            return False
            
        except Exception as e:
            logger.error(f"Migration failed for {table_name}: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()
    
    def _migrate_document_entities_table(self, cursor, expected_config: Dict[str, Any], preserve_data: bool) -> bool:
        """Migrate DocumentEntities table."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")
            
            logger.info(f"🔧 Migrating DocumentEntities table to {vector_dim}-dimensional vectors with {vector_data_type} data type")
            
            # For now, we'll drop and recreate (data preservation can be added later)
            if preserve_data:
                logger.warning("Data preservation not yet implemented - data will be lost")
            
            # Check if table has data
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
                row_count = cursor.fetchone()[0]
                if row_count > 0:
                    logger.warning(f"Dropping table with {row_count} existing rows")
            except:
                pass  # Table might not exist
            
            # Drop existing table
            cursor.execute("DROP TABLE IF EXISTS RAG.DocumentEntities")
            
            # Create new table with correct dimension and data type
            create_sql = f"""
            CREATE TABLE RAG.DocumentEntities (
                entity_id VARCHAR(255) NOT NULL,
                document_id VARCHAR(255) NOT NULL,
                entity_text VARCHAR(1000) NOT NULL,
                entity_type VARCHAR(100),
                position INTEGER,
                embedding VECTOR({vector_data_type}, {vector_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (entity_id)
            )
            """
            cursor.execute(create_sql)
            
            # Create indexes
            indexes = [
                "CREATE INDEX idx_documententities_document_id ON RAG.DocumentEntities (document_id)",
                "CREATE INDEX idx_documententities_entity_type ON RAG.DocumentEntities (entity_type)",
                "CREATE INDEX idx_documententities_created_at ON RAG.DocumentEntities (created_at)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            # Update schema metadata
            self._update_schema_metadata(cursor, "DocumentEntities", expected_config)
            
            logger.info(f"✅ DocumentEntities table migrated to {vector_dim}-dimensional vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate DocumentEntities table: {e}")
            return False
    
    def _update_schema_metadata(self, cursor, table_name: str, config: Dict[str, Any]):
        """Update schema metadata for a table."""
        try:
            # Use MERGE or INSERT/UPDATE pattern
            cursor.execute("DELETE FROM RAG.SchemaMetadata WHERE table_name = ?", [table_name])
            
            cursor.execute("""
                INSERT INTO RAG.SchemaMetadata 
                (table_name, schema_version, vector_dimension, embedding_model, configuration, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                table_name,
                config["schema_version"],
                config.get("vector_dimension"),
                config.get("embedding_model"),
                json.dumps(config["configuration"])
            ])
            
            logger.info(f"✅ Updated schema metadata for {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to update schema metadata for {table_name}: {e}")
            raise
    
    def ensure_table_schema(self, table_name: str) -> bool:
        """
        Ensure table schema matches current configuration.
        Performs migration if needed.
        
        Returns:
            True if schema is correct or migration successful, False otherwise
        """
        try:
            # Ensure metadata table exists
            self.ensure_schema_metadata_table()
            
            # Check if migration is needed
            if self.needs_migration(table_name):
                logger.info(f"Schema migration needed for {table_name}")
                return self.migrate_table(table_name)
            else:
                logger.info(f"Schema for {table_name} is up to date")
                return True
                
        except Exception as e:
            logger.error(f"Failed to ensure schema for {table_name}: {e}")
            return False
    
    def get_schema_status(self) -> Dict[str, Any]:
        """Get status of all managed schemas."""
        tables = ["DocumentEntities"]  # Add other tables as needed
        status = {}
        
        for table in tables:
            current_config = self.get_current_schema_config(table)
            expected_config = self._get_expected_schema_config(table)
            needs_migration = self.needs_migration(table)
            
            status[table] = {
                "current_config": current_config,
                "expected_config": expected_config,
                "needs_migration": needs_migration,
                "status": "migration_needed" if needs_migration else "up_to_date"
            }
        
        return status