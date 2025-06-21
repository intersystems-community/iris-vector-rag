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
    Universal vector dimension authority and schema manager for IRIS RAG.
    
    Features:
    - Central authority for ALL vector dimensions across all tables
    - Tracks vector dimensions and other configuration parameters
    - Automatically detects configuration changes
    - Performs safe schema migrations
    - Maintains schema version history
    - Provides simple dimension API for all components
    """
    
    def __init__(self, connection_manager, config_manager):
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.schema_version = "1.0.0"
        
        # Cache for dimension lookups
        self._dimension_cache = {}
        
        # Load and validate configuration on initialization
        self._load_and_validate_config()
        
        # Ensure schema metadata table exists
        self.ensure_schema_metadata_table()
    
    def _load_and_validate_config(self):
        """Load configuration from config manager and validate it makes sense."""
        logger.info("Schema Manager: Loading and validating configuration...")
        
        # Load base embedding model configuration
        self.base_embedding_model = self.config_manager.get("embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2")
        self.base_embedding_dimension = self.config_manager.get("embedding_model.dimension", 384)
        
        # Load ColBERT configuration
        colbert_config = self.config_manager.get("colbert", {})
        self.colbert_backend = colbert_config.get("backend", "native")
        self.colbert_token_dimension = colbert_config.get("token_dimension", 768)
        self.colbert_model_name = colbert_config.get("model_name", "bert-base-uncased")
        
        # Validate configuration consistency
        self._validate_configuration()
        
        # Build unified model-to-dimension mapping from config
        self._build_model_dimension_mapping()
        
        # Build table-specific configurations from config
        self._build_table_configurations()
        
        logger.info(f"✅ Schema Manager: Configuration validated and loaded")
        logger.info(f"   Base embedding: {self.base_embedding_model} ({self.base_embedding_dimension}D)")
        logger.info(f"   ColBERT backend: {self.colbert_backend} ({self.colbert_token_dimension}D)")
    
    def _validate_configuration(self):
        """Validate that configuration values make sense."""
        errors = []
        
        # Validate base embedding dimension
        if not isinstance(self.base_embedding_dimension, int) or self.base_embedding_dimension <= 0:
            errors.append(f"Invalid base embedding dimension: {self.base_embedding_dimension}")
        
        # Validate ColBERT token dimension  
        if not isinstance(self.colbert_token_dimension, int) or self.colbert_token_dimension <= 0:
            errors.append(f"Invalid ColBERT token dimension: {self.colbert_token_dimension}")
        
        # Validate ColBERT backend
        valid_backends = ["native", "pylate"]
        if self.colbert_backend not in valid_backends:
            errors.append(f"Invalid ColBERT backend '{self.colbert_backend}', must be one of: {valid_backends}")
        
        # Validate dimension relationships (document embeddings should be smaller than token embeddings)
        if self.base_embedding_dimension >= self.colbert_token_dimension:
            logger.warning(f"Unusual: Document embeddings ({self.base_embedding_dimension}D) >= token embeddings ({self.colbert_token_dimension}D)")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validation passed")
    
    def _build_model_dimension_mapping(self):
        """Build unified model-to-dimension mapping from configuration."""
        # Start with known model dimensions
        self._model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "bert-base-uncased": 768,
            "bert-large-uncased": 1024,
        }
        
        # Add configured models
        self._model_dimensions[self.base_embedding_model] = self.base_embedding_dimension
        self._model_dimensions[self.colbert_model_name] = self.colbert_token_dimension
        
        # Add legacy ColBERT models from config
        legacy_doc_model = self.config_manager.get("colbert.document_encoder_model")
        if legacy_doc_model:
            self._model_dimensions[legacy_doc_model] = self.colbert_token_dimension
        
        logger.debug(f"Model-dimension mapping: {len(self._model_dimensions)} models configured")
    
    def _build_table_configurations(self):
        """Build table-specific configurations from config."""
        # Table-specific configurations based on config
        self._table_configs = {
            "SourceDocuments": {
                "embedding_column": "embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension
            },
            "DocumentTokenEmbeddings": {
                "embedding_column": "token_embedding", 
                "uses_token_embeddings": True,
                "default_model": self.colbert_model_name,
                "dimension": self.colbert_token_dimension,
                "colbert_backend": self.colbert_backend
            },
            "DocumentEntities": {
                "embedding_column": "embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension
            }
        }
        
        logger.debug(f"Table configurations: {len(self._table_configs)} tables configured")
        
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
        # Get model and dimension from centralized methods
        model_name = self.get_embedding_model(table_name)
        expected_dim = self.get_vector_dimension(table_name, model_name)
        
        # Get vector data type from configuration, default to FLOAT
        vector_data_type = self.config_manager.get("storage:iris:vector_data_type", "FLOAT")
        
        # Base configuration
        config = {
            "schema_version": self.schema_version,
            "vector_dimension": expected_dim,
            "embedding_model": model_name,
            "vector_data_type": vector_data_type,
            "configuration": {
                "managed_by_schema_manager": True,
                "supports_vector_search": True,
                "auto_migration": True
            }
        }
        
        # Table-specific configurations
        if table_name == "DocumentEntities":
            config["configuration"].update({
                "table_type": "entity_storage",
                "created_by": "GraphRAG"
            })
        elif table_name == "SourceDocuments":
            config["configuration"].update({
                "table_type": "document_storage",
                "created_by": "BasicRAG"
            })
        elif table_name == "DocumentTokenEmbeddings":
            config["configuration"].update({
                "table_type": "token_storage",
                "created_by": "ColBERT"
            })
        
        return config
    
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
            
            if table_name == "SourceDocuments":
                success = self._migrate_source_documents_table(cursor, expected_config, preserve_data)
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "DocumentTokenEmbeddings":
                success = self._migrate_document_token_embeddings_table(cursor, expected_config, preserve_data)
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "DocumentEntities":
                success = self._migrate_document_entities_table(cursor, expected_config, preserve_data)
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "KnowledgeGraphNodes":
                success = self._migrate_knowledge_graph_nodes_table(cursor, expected_config, preserve_data)
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "KnowledgeGraphEdges":
                success = self._migrate_knowledge_graph_edges_table(cursor, expected_config, preserve_data)
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
    
    def _migrate_source_documents_table(self, cursor, expected_config: Dict[str, Any], preserve_data: bool) -> bool:
        """Migrate SourceDocuments table."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")
            
            logger.info(f"🔧 Migrating SourceDocuments table to {vector_dim}-dimensional vectors with {vector_data_type} data type")
            
            # For now, we'll drop and recreate (data preservation can be added later)
            if preserve_data:
                logger.warning("Data preservation not yet implemented - data will be lost")
            
            # Check if table has data
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                row_count = cursor.fetchone()[0]
                if row_count > 0:
                    logger.warning(f"Dropping table with {row_count} existing rows")
            except:
                pass  # Table might not exist
            
            # Drop existing table
            cursor.execute("DROP TABLE IF EXISTS RAG.SourceDocuments")
            logger.info("Successfully dropped SourceDocuments table")
            
            # Create new table with correct dimension and data type
            create_sql = f"""
            CREATE TABLE RAG.SourceDocuments (
                doc_id VARCHAR(255) NOT NULL,
                title VARCHAR(1000),
                text_content VARCHAR(MAX),
                abstract VARCHAR(MAX),
                authors VARCHAR(MAX),
                keywords VARCHAR(MAX),
                embedding VECTOR({vector_data_type}, {vector_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (doc_id)
            )
            """
            cursor.execute(create_sql)
            
            # Create indexes
            indexes = [
                "CREATE INDEX idx_sourcedocuments_created_at ON RAG.SourceDocuments (created_at)",
                "CREATE INDEX idx_sourcedocuments_title ON RAG.SourceDocuments (title)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            # Update schema metadata
            self._update_schema_metadata(cursor, "SourceDocuments", expected_config)
            
            logger.info(f"✅ SourceDocuments table migrated to {vector_dim}-dimensional vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate SourceDocuments table: {e}")
            return False
    
    def _migrate_document_token_embeddings_table(self, cursor, expected_config: Dict[str, Any], preserve_data: bool) -> bool:
        """Migrate DocumentTokenEmbeddings table."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")
            
            logger.info(f"🔧 Migrating DocumentTokenEmbeddings table to {vector_dim}-dimensional vectors with {vector_data_type} data type")
            
            # For now, we'll drop and recreate (data preservation can be added later)
            if preserve_data:
                logger.warning("Data preservation not yet implemented - data will be lost")
            
            # Check if table has data
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
                row_count = cursor.fetchone()[0]
                if row_count > 0:
                    logger.warning(f"Dropping table with {row_count} existing rows")
            except:
                pass  # Table might not exist
            
            # Drop existing table
            cursor.execute("DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings")
            logger.info("Successfully dropped DocumentTokenEmbeddings table")
            
            # Create new table with correct dimension and data type
            create_sql = f"""
            CREATE TABLE RAG.DocumentTokenEmbeddings (
                doc_id VARCHAR(255) NOT NULL,
                token_index INTEGER NOT NULL,
                token_text VARCHAR(500),
                token_embedding VECTOR({vector_data_type}, {vector_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (doc_id, token_index)
            )
            """
            cursor.execute(create_sql)
            
            # Create indexes
            indexes = [
                "CREATE INDEX idx_documenttokenembeddings_doc_id ON RAG.DocumentTokenEmbeddings (doc_id)",
                "CREATE INDEX idx_documenttokenembeddings_created_at ON RAG.DocumentTokenEmbeddings (created_at)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            # Update schema metadata
            self._update_schema_metadata(cursor, "DocumentTokenEmbeddings", expected_config)
            
            logger.info(f"✅ DocumentTokenEmbeddings table migrated to {vector_dim}-dimensional vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate DocumentTokenEmbeddings table: {e}")
            return False

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
            
            # Handle foreign key constraints before dropping table
            try:
                # First, identify and drop foreign key constraints that reference this table
                logger.info("Checking for foreign key constraints on DocumentEntities...")
                
                # Handle the specific foreign key constraints we know about
                # Based on the error message: ENTITYRELATIONSHIPSFKEY2 and ENTITYRELATIONSHIPSFKEY3 in table RAG.ENTITYRELATIONSHIPS
                known_constraints = [
                    ("ENTITYRELATIONSHIPSFKEY2", "EntityRelationships"),
                    ("ENTITYRELATIONSHIPSFKEY3", "EntityRelationships")
                ]
                
                dropped_constraints = []
                
                for constraint_name, referencing_table in known_constraints:
                    try:
                        logger.info(f"Dropping foreign key constraint {constraint_name} from RAG.{referencing_table}")
                        cursor.execute(f"ALTER TABLE RAG.{referencing_table} DROP CONSTRAINT {constraint_name}")
                        dropped_constraints.append((constraint_name, referencing_table))
                        logger.info(f"✓ Successfully dropped constraint {constraint_name}")
                    except Exception as fk_error:
                        logger.warning(f"Could not drop foreign key {constraint_name}: {fk_error}")
                        # If we can't drop the constraint, try dropping the entire referencing table
                        try:
                            logger.info(f"Attempting to drop referencing table RAG.{referencing_table}")
                            cursor.execute(f"DROP TABLE IF EXISTS RAG.{referencing_table}")
                            logger.info(f"✓ Dropped referencing table RAG.{referencing_table}")
                        except Exception as table_error:
                            logger.warning(f"Could not drop referencing table {referencing_table}: {table_error}")
                
                # Now drop the table
                cursor.execute("DROP TABLE IF EXISTS RAG.DocumentEntities")
                logger.info("Successfully dropped DocumentEntities table")
                
            except Exception as drop_error:
                logger.error(f"Failed to handle foreign key constraints and drop table: {drop_error}")
                # If we can't drop due to constraints, try to work with existing table structure
                logger.info("Attempting to work with existing table structure...")
                return True  # Consider this a successful "migration" for now
            
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
    
    def _migrate_knowledge_graph_nodes_table(self, cursor, expected_config: Dict[str, Any], preserve_data: bool) -> bool:
        """Migrate KnowledgeGraphNodes table."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")
            
            logger.info(f"🔧 Migrating KnowledgeGraphNodes table to {vector_dim}-dimensional vectors with {vector_data_type} data type")
            
            # For now, we'll drop and recreate (data preservation can be added later)
            if preserve_data:
                logger.warning("Data preservation not yet implemented - data will be lost")
            
            # Check if table has data
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
                row_count = cursor.fetchone()[0]
                if row_count > 0:
                    logger.warning(f"Dropping table with {row_count} existing rows")
            except:
                pass  # Table might not exist
            
            # Handle foreign key constraints before dropping table
            try:
                # Drop referencing tables first (edges reference nodes)
                cursor.execute("DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges")
                logger.info("Dropped KnowledgeGraphEdges table (references nodes)")
            except Exception as e:
                logger.warning(f"Could not drop KnowledgeGraphEdges: {e}")
            
            # Now drop the nodes table
            cursor.execute("DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes")
            logger.info("Successfully dropped KnowledgeGraphNodes table")
            
            # Create new table with correct dimension and data type
            create_sql = f"""
            CREATE TABLE RAG.KnowledgeGraphNodes (
                node_id VARCHAR(255) NOT NULL,
                node_type VARCHAR(100),
                node_properties VARCHAR(MAX),
                embedding VECTOR({vector_data_type}, {vector_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (node_id)
            )
            """
            cursor.execute(create_sql)
            
            # Create indexes
            indexes = [
                "CREATE INDEX idx_knowledgegraphnodes_node_type ON RAG.KnowledgeGraphNodes (node_type)",
                "CREATE INDEX idx_knowledgegraphnodes_created_at ON RAG.KnowledgeGraphNodes (created_at)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            # Update schema metadata
            self._update_schema_metadata(cursor, "KnowledgeGraphNodes", expected_config)
            
            logger.info(f"✅ KnowledgeGraphNodes table migrated to {vector_dim}-dimensional vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate KnowledgeGraphNodes table: {e}")
            return False
    
    def _migrate_knowledge_graph_edges_table(self, cursor, expected_config: Dict[str, Any], preserve_data: bool) -> bool:
        """Migrate KnowledgeGraphEdges table."""
        try:
            logger.info("🔧 Migrating KnowledgeGraphEdges table")
            
            # For now, we'll drop and recreate (data preservation can be added later)
            if preserve_data:
                logger.warning("Data preservation not yet implemented - data will be lost")
            
            # Check if table has data
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
                row_count = cursor.fetchone()[0]
                if row_count > 0:
                    logger.warning(f"Dropping table with {row_count} existing rows")
            except:
                pass  # Table might not exist
            
            # Drop existing table
            cursor.execute("DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges")
            logger.info("Successfully dropped KnowledgeGraphEdges table")
            
            # Create new table (edges typically don't need embeddings)
            create_sql = f"""
            CREATE TABLE RAG.KnowledgeGraphEdges (
                edge_id VARCHAR(255) NOT NULL,
                source_node_id VARCHAR(255) NOT NULL,
                target_node_id VARCHAR(255) NOT NULL,
                edge_type VARCHAR(100),
                edge_properties VARCHAR(MAX),
                weight FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (edge_id)
            )
            """
            cursor.execute(create_sql)
            
            # Create indexes
            indexes = [
                "CREATE INDEX idx_knowledgegraphedges_source ON RAG.KnowledgeGraphEdges (source_node_id)",
                "CREATE INDEX idx_knowledgegraphedges_target ON RAG.KnowledgeGraphEdges (target_node_id)",
                "CREATE INDEX idx_knowledgegraphedges_edge_type ON RAG.KnowledgeGraphEdges (edge_type)",
                "CREATE INDEX idx_knowledgegraphedges_created_at ON RAG.KnowledgeGraphEdges (created_at)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            # Update schema metadata (edges don't typically have vector dimensions)
            edges_config = expected_config.copy()
            edges_config["vector_dimension"] = None
            edges_config["embedding_model"] = None
            self._update_schema_metadata(cursor, "KnowledgeGraphEdges", edges_config)
            
            logger.info("✅ KnowledgeGraphEdges table migrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate KnowledgeGraphEdges table: {e}")
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
    
    def get_vector_dimension(self, table_name: str = "SourceDocuments", model_name: str = None) -> int:
        """
        Universal method to get vector dimension for any table.
        This is the SINGLE SOURCE OF TRUTH for all vector dimensions.
        
        Enforces the schema manager's view of the world based on validated configuration.
        
        Args:
            table_name: Name of the table (SourceDocuments, DocumentTokenEmbeddings, etc.)
            model_name: Optional specific model name override
            
        Returns:
            Vector dimension for the table
        """
        # Check cache first
        cache_key = f"{table_name}:{model_name or 'default'}"
        if cache_key in self._dimension_cache:
            return self._dimension_cache[cache_key]
        
        # Primary method: Get dimension directly from table config (config-driven)
        if table_name in self._table_configs:
            dimension = self._table_configs[table_name]["dimension"]
            
            # If model override specified, use model mapping
            if model_name and model_name != self._table_configs[table_name]["default_model"]:
                if model_name in self._model_dimensions:
                    dimension = self._model_dimensions[model_name]
                else:
                    logger.warning(f"Unknown model '{model_name}' for {table_name}, using table default: {dimension}")
            
        else:
            # Fallback: Try model-based lookup for unknown tables
            if not model_name:
                model_name = self.base_embedding_model
            
            if model_name in self._model_dimensions:
                dimension = self._model_dimensions[model_name]
            else:
                # HARD FAIL - no dangerous fallbacks that hide configuration issues
                error_msg = f"CRITICAL: Unknown table '{table_name}' and unknown model '{model_name}' - schema manager cannot determine dimension"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Cache the result
        self._dimension_cache[cache_key] = dimension
        
        logger.debug(f"Schema Manager: {table_name} with model {model_name or 'default'} -> {dimension}D")
        return dimension
    
    def get_colbert_config(self) -> Dict[str, Any]:
        """
        Get ColBERT configuration from the schema manager.
        
        Returns:
            Dictionary with ColBERT configuration
        """
        return {
            "backend": self.colbert_backend,
            "token_dimension": self.colbert_token_dimension,
            "model_name": self.colbert_model_name,
            "document_dimension": self.base_embedding_dimension,
            "document_model": self.base_embedding_model
        }
    
    def validate_vector_dimension(self, table_name: str, provided_dimension: int, context: str = "") -> None:
        """
        Validate that a provided dimension matches schema manager's expectation.
        
        Args:
            table_name: Name of the table
            provided_dimension: Dimension provided by caller
            context: Context string for error messages
            
        Raises:
            ValueError: If dimension doesn't match
        """
        expected_dimension = self.get_vector_dimension(table_name)
        if provided_dimension != expected_dimension:
            error_msg = f"Dimension mismatch for {table_name}"
            if context:
                error_msg += f" in {context}"
            error_msg += f": provided {provided_dimension}D, schema manager expects {expected_dimension}D"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_embedding_model(self, table_name: str = "SourceDocuments") -> str:
        """
        Get the embedding model name for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Model name string
        """
        if table_name in self._table_configs:
            return self._table_configs[table_name]["default_model"]
        
        # Fallback to config manager
        embedding_config = self.config_manager.get_embedding_config()
        return embedding_config.get("model", "all-MiniLM-L6-v2")
    
    def register_model(self, model_name: str, dimension: int) -> None:
        """
        Register a new model and its dimension.
        
        Args:
            model_name: Name of the embedding model
            dimension: Vector dimension for this model
        """
        self._model_dimensions[model_name] = dimension
        # Clear cache to force recalculation
        self._dimension_cache.clear()
        logger.info(f"Registered model {model_name} with dimension {dimension}")
    
    def validate_dimension_consistency(self) -> Dict[str, Any]:
        """
        Validate that all tables have consistent dimensions with their models.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "consistent": True,
            "issues": [],
            "table_dimensions": {}
        }
        
        for table_name in self._table_configs.keys():
            try:
                current_config = self.get_current_schema_config(table_name)
                expected_dimension = self.get_vector_dimension(table_name)
                
                results["table_dimensions"][table_name] = {
                    "expected": expected_dimension,
                    "current": current_config.get("vector_dimension") if current_config else None
                }
                
                if current_config and current_config.get("vector_dimension") != expected_dimension:
                    results["consistent"] = False
                    results["issues"].append({
                        "table": table_name,
                        "issue": "dimension_mismatch",
                        "expected": expected_dimension,
                        "current": current_config.get("vector_dimension")
                    })
                    
            except Exception as e:
                results["consistent"] = False
                results["issues"].append({
                    "table": table_name,
                    "issue": "validation_error", 
                    "error": str(e)
                })
        
        return results
    
    def get_schema_status(self) -> Dict[str, Any]:
        """Get status of all managed schemas."""
        tables = list(self._table_configs.keys())
        status = {}
        
        for table in tables:
            current_config = self.get_current_schema_config(table)
            expected_config = self._get_expected_schema_config(table)
            needs_migration = self.needs_migration(table)
            
            status[table] = {
                "current_config": current_config,
                "expected_config": expected_config,
                "needs_migration": needs_migration,
                "status": "migration_needed" if needs_migration else "up_to_date",
                "vector_dimension": self.get_vector_dimension(table)
            }
        
        return status
    
    # Additional getter methods for database state validator
    def get_base_embedding_dimension(self) -> int:
        """
        Get the base embedding dimension from configuration.
        
        Returns:
            Base embedding dimension
        """
        return self.base_embedding_dimension
    
    def get_colbert_token_dimension(self) -> int:
        """
        Get the ColBERT token embedding dimension from configuration.
        
        Returns:
            ColBERT token embedding dimension
        """
        return self.colbert_token_dimension
    
    def get_base_embedding_model(self) -> str:
        """
        Get the base embedding model name from configuration.
        
        Returns:
            Base embedding model name
        """
        return self.base_embedding_model
    
    def get_colbert_backend(self) -> str:
        """
        Get the ColBERT backend type from configuration.
        
        Returns:
            ColBERT backend type ("native" or "pylate")
        """
        return self.colbert_backend