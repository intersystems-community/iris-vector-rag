#!/usr/bin/env python3
"""
Schema management system for IRIS RAG tables with automatic migration support.

This module provides robust schema versioning, configuration tracking, and
automatic migration capabilities for vector dimensions and other schema changes.
"""

import json
import logging
from typing import Any, Dict, Optional

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

    # CLASS-LEVEL CACHING (shared across all instances for performance)
    _schema_validation_cache = {}  # Cache for needs_migration() results
    _config_loaded = False  # Flag to prevent redundant config loading
    _tables_validated = set()  # Cache for ensure_table_exists() to prevent spam

    def __init__(self, connection_manager, config_manager):
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.schema_version = "1.0.0"

        # Cache for dimension lookups
        self._dimension_cache = {}

        # Load and validate configuration on initialization (only if not already loaded)
        if not SchemaManager._config_loaded:
            self._load_and_validate_config()
            SchemaManager._config_loaded = True
        else:
            # Config already loaded by another instance - set instance attributes from config
            # Use CloudConfiguration API for environment variable support (Feature 058)
            cloud_config = self.config_manager.get_cloud_config()

            self.base_embedding_model = self.config_manager.get(
                "embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            # Use cloud_config for vector dimension (supports VECTOR_DIMENSION env var)
            self.base_embedding_dimension = cloud_config.vector.vector_dimension

            # Now build mappings (these methods reference the attributes we just set)
            self._build_model_dimension_mapping()
            self._build_table_configurations()
            logger.debug("Schema Manager: Using cached configuration from previous instance")

        # Ensure schema metadata table exists
        self.ensure_schema_metadata_table()

    def _load_and_validate_config(self):
        """Load configuration from config manager and validate it makes sense."""
        logger.info("Schema Manager: Loading and validating configuration...")

        # Use CloudConfiguration API for environment variable support (Feature 058)
        cloud_config = self.config_manager.get_cloud_config()

        # Load base embedding model configuration
        self.base_embedding_model = self.config_manager.get(
            "embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Use cloud_config for vector dimension (supports VECTOR_DIMENSION env var)
        self.base_embedding_dimension = cloud_config.vector.vector_dimension

        # Validate configuration consistency
        self._validate_configuration()

        # Build unified model-to-dimension mapping from config
        self._build_model_dimension_mapping()

        # Build table-specific configurations from config
        self._build_table_configurations()

        logger.info(f"✅ Schema Manager: Configuration validated and loaded")
        logger.info(
            f"   Base embedding: {self.base_embedding_model} ({self.base_embedding_dimension}D)"
        )

    def _validate_configuration(self):
        """Validate that configuration values make sense."""
        errors = []

        # Validate base embedding dimension
        if (
            not isinstance(self.base_embedding_dimension, int)
            or self.base_embedding_dimension <= 0
        ):
            errors.append(
                f"Invalid base embedding dimension: {self.base_embedding_dimension}"
            )

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
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
        self._model_dimensions[self.base_embedding_model] = (
            self.base_embedding_dimension
        )

        logger.debug(
            f"Model-dimension mapping: {len(self._model_dimensions)} models configured"
        )

    def _build_table_configurations(self):
        """Build table-specific configurations from config."""
        # Table-specific configurations based on config
        self._table_configs = {
            "SourceDocuments": {
                "embedding_column": "embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
            },
            "DocumentChunks": {
                "embedding_column": "chunk_embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
            },
            "Entities": {
                "embedding_column": "embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
                "supports_vector_search": True,
                "supports_graph_traversal": True,
            },
            "EntityRelationships": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,  # Required by get_embedding_model
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
            },
            # IRIS Graph Core Tables for Hybrid Search
            "rdf_labels": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
                "table_type": "graph_metadata",
                "created_by": "iris_vector_graph",
            },
            "rdf_props": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
                "table_type": "graph_properties",
                "created_by": "iris_vector_graph",
            },
            "rdf_edges": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
                "table_type": "graph_relationships",
                "created_by": "iris_vector_graph",
            },
            "kg_NodeEmbeddings_optimized": {
                "embedding_column": "emb",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
                "supports_vector_search": True,
                "supports_graph_traversal": False,
                "table_type": "optimized_vectors",
                "created_by": "iris_vector_graph",
                "requires_hnsw_index": True,
            },
        }

        logger.debug(
            f"Table configurations: {len(self._table_configs)} tables configured"
        )

    def ensure_schema_metadata_table(self):
        """Create schema metadata table if it doesn't exist."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Try different schema approaches in order of preference
            schema_attempts = [
                ("RAG", "RAG.SchemaMetadata"),
                (
                    "current user",
                    "SchemaMetadata",
                ),  # No schema prefix = current user's schema
            ]

            for schema_name, table_name in schema_attempts:
                try:
                    create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
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
                    logger.info(
                        f"✅ Schema metadata table ensured in {schema_name} schema"
                    )
                    break
                except Exception as schema_error:
                    logger.warning(
                        f"Failed to create schema metadata table in {schema_name} schema: {schema_error}"
                    )
                    if (schema_name, table_name) == schema_attempts[
                        -1
                    ]:  # Last schema attempt
                        # Instead of raising, log warning and continue without metadata table
                        logger.warning(
                            "Schema metadata table creation failed in all schemas. Continuing without metadata table."
                        )
                        logger.warning(
                            "This may affect schema versioning but basic functionality will work."
                        )
                        return  # Exit gracefully
                    continue

        except Exception as e:
            logger.error(f"Failed to create schema metadata table: {e}")
            logger.warning(
                "Continuing without schema metadata table. Basic functionality will work."
            )
            # Don't raise - allow the system to continue without metadata table
        finally:
            cursor.close()

    def get_current_schema_config(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get current schema configuration for a table."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                SELECT schema_version, vector_dimension, embedding_model, configuration
                FROM RAG.SchemaMetadata 
                WHERE table_name = ?
            """,
                [table_name],
            )

            result = cursor.fetchone()
            if result:
                # Handle different result formats gracefully
                if len(result) == 4:
                    # Expected format: (schema_version, vector_dim, embedding_model, config_json)
                    schema_version, vector_dim, embedding_model, config_json = result
                    config = json.loads(config_json) if config_json else {}

                    # This won't actually exist in config yet
                    vector_data_type = config.get(
                        "vector_data_type", "FLOAT"
                    )  # Default to FLOAT

                    return {
                        "schema_version": schema_version,
                        "vector_dimension": vector_dim,
                        "embedding_model": embedding_model,
                        "vector_data_type": vector_data_type,
                        "configuration": config,
                    }
                elif len(result) == 1:
                    # Legacy or corrupted format: only one value returned
                    logger.warning(
                        f"Schema metadata for {table_name} has unexpected format (1 value instead of 4). This may indicate corrupted metadata."
                    )
                    return None
                else:
                    # Other unexpected formats
                    logger.warning(
                        f"Schema metadata for {table_name} has unexpected format ({len(result)} values instead of 4). This may indicate corrupted metadata."
                    )
                    return None
            return None

        except Exception as e:
            logger.error(f"Failed to get schema config for {table_name}: {e}")
            return None
        finally:
            cursor.close()

    def _get_expected_schema_config(
        self, table_name: str, pipeline_type: str = None
    ) -> Dict[str, Any]:
        """Get expected schema configuration based on current system config and pipeline requirements."""
        # Get model and dimension from centralized methods
        model_name = self.get_embedding_model(table_name)
        expected_dim = self.get_vector_dimension(table_name, model_name)

        # Get vector data type from configuration, default to FLOAT
        vector_data_type = self.config_manager.get(
            "storage:iris:vector_data_type", "FLOAT"
        )

        # Base configuration
        config = {
            "schema_version": self.schema_version,
            "vector_dimension": expected_dim,
            "embedding_model": model_name,
            "vector_data_type": vector_data_type,
            "configuration": {
                "managed_by_schema_manager": True,
                "supports_vector_search": True,
                "auto_migration": True,
            },
        }

        # Enhanced: Get table requirements from pipeline if specified
        if pipeline_type:
            config.update(
                self._get_table_requirements_config(table_name, pipeline_type)
            )

        # Table-specific configurations
        if table_name == "SourceDocuments":
            config["configuration"].update(
                {
                    "table_type": "document_storage",
                    "created_by": "BasicRAG",
                    "expected_columns": [
                        "id",
                        "doc_id",
                        "title",
                        "abstract",
                        "text_content",
                        "authors",
                        "keywords",
                        "embedding",
                        "metadata",
                        "created_at",
                    ],
                }
            )
        elif table_name == "DocumentChunks":
            config["configuration"].update(
                {
                    "table_type": "chunk_storage",
                    "created_by": "BasicRAG",
                    "expected_columns": [
                        "id",
                        "chunk_id",
                        "source_document_id",
                        "chunk_text",
                        "chunk_embedding",
                        "chunk_index",
                        "chunk_type",
                        "metadata",
                        "created_at",
                    ],
                }
            )
        elif table_name == "Entities":
            config["configuration"].update(
                {
                    "table_type": "entity_storage",
                    "created_by": "GraphRAG",
                    "expected_columns": [
                        "entity_id",
                        "entity_name",
                        "entity_type",
                        "source_doc_id",
                        "description",
                        "embedding",
                        "created_at",
                    ],
                    "foreign_keys": [
                        {
                            "column": "source_doc_id",
                            "references": "RAG.SourceDocuments(id)",
                            "on_delete": "CASCADE",
                        }
                    ],
                    "indexes": [
                        "idx_entities_name_lower",
                        "idx_entities_type",
                        "idx_entities_source_doc",
                    ],
                }
            )
        elif table_name == "EntityRelationships":
            config["configuration"].update(
                {
                    "table_type": "relationship_storage",
                    "created_by": "GraphRAG",
                    "expected_columns": [
                        "relationship_id",
                        "source_entity_id",
                        "target_entity_id",
                        "relationship_type",
                        "metadata",
                        "created_at",
                    ],
                    "foreign_keys": [
                        {
                            "column": "source_entity_id",
                            "references": "RAG.Entities(entity_id)",
                            "on_delete": "CASCADE",
                        },
                        {
                            "column": "target_entity_id",
                            "references": "RAG.Entities(entity_id)",
                            "on_delete": "CASCADE",
                        },
                    ],
                    "indexes": ["idx_rel_source", "idx_rel_target"],
                }
            )
        # IRIS Graph Core Tables
        elif table_name == "rdf_labels":
            config["configuration"].update(
                {
                    "table_type": "graph_metadata",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "s",  # subject
                        "label",  # entity type/label
                    ],
                    "indexes": [
                        "idx_labels_label_s",
                        "idx_labels_s_label",
                    ],
                }
            )
        elif table_name == "rdf_props":
            config["configuration"].update(
                {
                    "table_type": "graph_properties",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "s",  # subject
                        "key",  # property key
                        "val",  # property value
                    ],
                    "indexes": [
                        "idx_props_s_key",
                        "idx_props_key_val",
                    ],
                }
            )
        elif table_name == "rdf_edges":
            config["configuration"].update(
                {
                    "table_type": "graph_relationships",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "edge_id",  # primary key
                        "s",  # subject
                        "p",  # predicate
                        "o_id",  # object
                        "qualifiers",  # JSON metadata with confidence
                    ],
                    "indexes": [
                        "idx_edges_s_p",
                        "idx_edges_p_oid",
                        "idx_edges_s",
                    ],
                }
            )
        elif table_name == "kg_NodeEmbeddings_optimized":
            config["configuration"].update(
                {
                    "table_type": "optimized_vectors",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "id",  # entity identifier
                        "emb",  # vector embedding (optimized VECTOR type)
                    ],
                    "indexes": [
                        "HNSW_NodeEmb_Optimized",  # HNSW index for fast similarity
                    ],
                    "requires_hnsw_optimization": True,
                }
            )

        return config

    def _get_table_requirements_config(
        self, table_name: str, pipeline_type: str
    ) -> Dict[str, Any]:
        """Extract table configuration from pipeline requirements."""
        try:
            from ..validation.requirements import get_pipeline_requirements

            requirements = get_pipeline_requirements(pipeline_type)

            # Find the table requirement for this table
            for table_req in requirements.required_tables:
                if table_req.name == table_name:
                    return {
                        "text_content_type": table_req.text_content_type,
                        "supports_vector_search": table_req.supports_vector_search,
                    }

            # Check optional tables too
            for table_req in requirements.optional_tables:
                if table_req.name == table_name:
                    return {
                        "text_content_type": table_req.text_content_type,
                        "supports_vector_search": table_req.supports_vector_search,
                    }

        except Exception as e:
            logger.warning(f"Could not get table requirements for {pipeline_type}: {e}")

        # Default configuration
        return {"text_content_type": "LONGVARCHAR", "supports_vector_search": True}

    def needs_migration(self, table_name: str, pipeline_type: str = None) -> bool:
        """Check if table needs migration based on configuration and physical structure.

        Uses class-level caching to avoid repeated expensive validation checks.
        """
        # Check class-level cache first (shared across all instances)
        cache_key = f"{table_name}:{pipeline_type or 'default'}"
        if cache_key in SchemaManager._schema_validation_cache:
            cached_result = SchemaManager._schema_validation_cache[cache_key]
            logger.debug(f"Schema validation cache HIT for {table_name} (cached: {cached_result})")
            return cached_result

        # Cache MISS - perform full validation
        logger.debug(f"Schema validation cache MISS for {table_name} - performing full validation")

        current_config = self.get_current_schema_config(table_name)
        expected_config = self._get_expected_schema_config(table_name, pipeline_type)

        if not current_config:
            logger.info(f"Table {table_name} has no schema metadata - migration needed")
            SchemaManager._schema_validation_cache[cache_key] = True
            return True

        # Compare top-level schema config fields
        keys_to_check = [
            "schema_version",
            "vector_dimension",
            "embedding_model",
            "vector_data_type",
        ]
        for key in keys_to_check:
            if current_config.get(key) != expected_config.get(key):
                logger.info(
                    f"Migration required for {table_name} due to {key} mismatch: "
                    f"expected={expected_config.get(key)}, actual={current_config.get(key)}"
                )
                SchemaManager._schema_validation_cache[cache_key] = True
                return True

        # Compare nested configuration keys
        nested_keys = [
            "managed_by_schema_manager",
            "supports_vector_search",
            "auto_migration",
        ]
        for key in nested_keys:
            cur_val = current_config.get("configuration", {}).get(key)
            exp_val = expected_config.get("configuration", {}).get(key)
            if cur_val != exp_val:
                logger.info(
                    f"Migration required for {table_name} due to config.{key} mismatch: "
                    f"expected={exp_val}, actual={cur_val}"
                )
                SchemaManager._schema_validation_cache[cache_key] = True
                return True

        # ✅ NEW: Check that all expected columns exist in physical table
        expected_columns = expected_config.get("configuration", {}).get(
            "expected_columns", []
        )
        if expected_columns:
            try:
                physical_columns = self.verify_table_structure(table_name)
                physical_col_names = {col.lower() for col in physical_columns.keys()}

                for col in expected_columns:
                    if col.lower() not in physical_col_names:
                        logger.info(
                            f"Migration required for {table_name}: missing column '{col}' in physical schema."
                        )
                        SchemaManager._schema_validation_cache[cache_key] = True
                        return True
            except Exception as e:
                logger.warning(
                    f"Could not verify physical structure of {table_name}: {e}"
                )
                SchemaManager._schema_validation_cache[cache_key] = True
                return True  # Be conservative and migrate if in doubt

        # Schema is valid - cache the result
        logger.info(f"✅ Schema validation PASSED for {table_name} - caching result")
        SchemaManager._schema_validation_cache[cache_key] = False
        return False

    def migrate_table(
        self, table_name: str, preserve_data: bool = False, pipeline_type: str = None
    ) -> bool:
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
            expected_config = self._get_expected_schema_config(
                table_name, pipeline_type
            )

            if table_name == "SourceDocuments":
                success = self._migrate_source_documents_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "DocumentChunks":
                success = self._migrate_document_chunks_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "Entities":
                success = self._migrate_entities_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "EntityRelationships":
                success = self._migrate_entity_relationships_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            # IRIS Graph Core table migrations (case-insensitive)
            elif table_name.lower() == "rdf_labels":
                success = self._migrate_rdf_labels_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name.lower() == "rdf_props":
                success = self._migrate_rdf_props_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name.lower() == "rdf_edges":
                success = self._migrate_rdf_edges_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name.lower() == "kg_nodeembeddings_optimized":
                success = self._migrate_kg_node_embeddings_optimized_table(
                    cursor, expected_config, preserve_data
                )
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

    def _migrate_source_documents_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate SourceDocuments table with requirements-driven DDL generation."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")
            text_content_type = expected_config.get("text_content_type", "LONGVARCHAR")
            supports_ifind = expected_config.get("supports_ifind", False)

            # Try multiple table name approaches to work around IRIS schema issues
            table_attempts = [
                "RAG.SourceDocuments",  # Preferred
                "SourceDocuments",  # Fallback
            ]

            for table_name in table_attempts:
                try:
                    logger.info(
                        f"🔧 Attempting to create SourceDocuments table as {table_name}"
                    )
                    logger.info(
                        f"   Text content type: {text_content_type}, iFind support: {supports_ifind}"
                    )

                    # Handle foreign key constraints from referencing tables
                    logger.info(
                        "Checking for foreign key constraints on SourceDocuments..."
                    )
                    known_constraints = [
                        ("DOCUMENTCHUNKSFKEY2", "DocumentChunks"),
                    ]

                    for constraint_name, referencing_table in known_constraints:
                        # Check if referencing table exists first
                        cursor.execute(
                            """
                            SELECT COUNT(*) 
                            FROM INFORMATION_SCHEMA.TABLES 
                            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ?
                        """,
                            [referencing_table.upper()],
                        )
                        table_exists = cursor.fetchone()[0] > 0

                        if not table_exists:
                            logger.info(
                                f"Referencing table RAG.{referencing_table} does not exist — skipping constraint drop"
                            )
                            continue

                        try:
                            logger.info(
                                f"Dropping foreign key constraint {constraint_name} from RAG.{referencing_table}"
                            )
                            cursor.execute(
                                f"ALTER TABLE RAG.{referencing_table} DROP CONSTRAINT {constraint_name}"
                            )
                            logger.info(
                                f"✓ Successfully dropped constraint {constraint_name}"
                            )
                        except Exception as fk_error:
                            logger.warning(
                                f"Could not drop foreign key {constraint_name}: {fk_error}"
                            )
                            try:
                                logger.info(
                                    f"Attempting to drop referencing table RAG.{referencing_table}"
                                )
                                cursor.execute(
                                    f"DROP TABLE IF EXISTS RAG.{referencing_table}"
                                )
                                logger.info(
                                    f"✓ Dropped referencing table RAG.{referencing_table}"
                                )
                            except Exception as table_error:
                                logger.warning(
                                    f"Could not drop referencing table {referencing_table}: {table_error}"
                                )

                    # Drop existing table
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped existing {table_name} table")

                    # Recreate with correct schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        doc_id VARCHAR(255) PRIMARY KEY,
                        title VARCHAR(1000),
                        abstract VARCHAR(MAX),
                        text_content {text_content_type},
                        authors VARCHAR(MAX),
                        keywords VARCHAR(MAX),
                        embedding VECTOR({vector_data_type}, {vector_dim}),
                        metadata VARCHAR(MAX),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                    cursor.execute(create_sql)
                    logger.info(f"✅ Successfully created {table_name} table")

                    # Indexes (doc_id is UNIQUE via table constraint; add created_at index)
                    indexes = [
                        f"CREATE INDEX idx_source_docs_created ON {table_name} (created_at)",
                    ]
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                        except Exception as e:
                            logger.warning(f"Failed to create index: {e}")

                    # Schema metadata
                    try:
                        self._update_schema_metadata(
                            cursor, "SourceDocuments", expected_config
                        )
                    except Exception as meta_error:
                        logger.debug(f"Schema metadata update failed: {meta_error}")

                    logger.info(
                        f"✅ SourceDocuments table created successfully as {table_name}"
                    )
                    return True

                except Exception as table_error:
                    logger.warning(
                        f"Failed to create table as {table_name}: {table_error}"
                    )
                    if table_name == table_attempts[-1]:
                        logger.error("All table creation attempts failed")
                        return False
                    continue

            return False

        except Exception as e:
            logger.error(f"Failed to migrate SourceDocuments table: {e}")
            return False

    def _update_schema_metadata(self, cursor, table_name: str, config: Dict[str, Any]):
        """Update schema metadata for a table."""
        try:
            # Use MERGE or INSERT/UPDATE pattern
            cursor.execute(
                "DELETE FROM RAG.SchemaMetadata WHERE table_name = ?", [table_name]
            )

            # Handle configuration serialization safely
            configuration_json = None
            if "configuration" in config:
                try:
                    configuration_json = json.dumps(config["configuration"])
                except (TypeError, ValueError) as json_error:
                    logger.warning(
                        f"Could not serialize configuration for {table_name}: {json_error}"
                    )
                    configuration_json = json.dumps({"error": "serialization_failed"})

            cursor.execute(
                """
                INSERT INTO RAG.SchemaMetadata
                (table_name, schema_version, vector_dimension, embedding_model, configuration, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                [
                    table_name,
                    config.get("schema_version"),
                    config.get("vector_dimension"),
                    config.get("embedding_model"),
                    configuration_json,
                ],
            )

            logger.info(f"✅ Updated schema metadata for {table_name}")

        except Exception as e:
            logger.error(f"Failed to update schema metadata for {table_name}: {e}")
            raise

    def _migrate_documentchunks_add_chunk_id(self, cursor):
        """Add chunk_id column and index if missing (idempotent migration).

        This migration adds the chunk_id field that was added to the schema
        definition but not present in existing databases.
        """
        try:
            # Check if chunk_id column exists
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG'
                  AND TABLE_NAME = 'DOCUMENTCHUNKS'
                  AND COLUMN_NAME = 'chunk_id'
            """)
            if cursor.fetchone():
                logger.info("chunk_id column already exists, skipping migration")
                return

            # Add chunk_id column
            logger.info("Adding chunk_id column to RAG.DocumentChunks...")
            cursor.execute("ALTER TABLE RAG.DocumentChunks ADD chunk_id VARCHAR(255)")

            # Create index on chunk_id
            logger.info("Creating index on chunk_id...")
            cursor.execute("CREATE INDEX idx_chunks_chunk_id ON RAG.DocumentChunks (chunk_id)")

            logger.info("✓ Migration complete: chunk_id column and index added")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    def _migrate_document_chunks_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate DocumentChunks table."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")

            logger.info(
                f"🔧 Migrating DocumentChunks table to {vector_dim}D vectors with {vector_data_type} type"
            )

            # Check if table exists
            cursor.execute("""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DOCUMENTCHUNKS'
            """)
            table_exists = cursor.fetchone()[0] > 0

            if table_exists and preserve_data:
                # Just add the missing chunk_id column if needed
                logger.info("Table exists and preserving data - checking for chunk_id column")
                self._migrate_documentchunks_add_chunk_id(cursor)
                self._update_schema_metadata(cursor, "DocumentChunks", expected_config)
                logger.info("✅ DocumentChunks table migrated successfully (preserved data)")
                return True

            # If no preservation or table doesn't exist, create it fresh
            if table_exists:
                cursor.execute("DROP TABLE IF EXISTS RAG.DocumentChunks")

            create_sql = f"""
            CREATE TABLE RAG.DocumentChunks (
                id VARCHAR(255) PRIMARY KEY,
                chunk_id VARCHAR(255),
                source_document_id VARCHAR(255),
                chunk_text TEXT,
                chunk_embedding VECTOR(FLOAT, 384),
                chunk_index INTEGER,
                chunk_type VARCHAR(100),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_document_id) REFERENCES RAG.SourceDocuments(id)
            )
            """
            cursor.execute(create_sql)

            for index_sql in [
                "CREATE INDEX idx_chunks_source_doc_id ON RAG.DocumentChunks (source_document_id)",
                "CREATE INDEX idx_chunks_chunk_id ON RAG.DocumentChunks (chunk_id)",
                "CREATE INDEX idx_chunks_type ON RAG.DocumentChunks (chunk_type)",
            ]:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index on DocumentChunks: {e}")

            self._update_schema_metadata(cursor, "DocumentChunks", expected_config)
            logger.info("✅ DocumentChunks table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate DocumentChunks table: {e}")
            return False

    def ensure_table_schema(self, table_name: str, pipeline_type: str = None) -> bool:
        """
        Ensure table schema matches current configuration.
        Performs migration if needed and ensures vector indexes exist.

        Args:
            table_name: Name of the table to ensure
            pipeline_type: Optional pipeline type for requirements-driven DDL

        Returns:
            True if schema is correct or migration successful, False otherwise
        """
        try:
            # Ensure metadata table exists
            self.ensure_schema_metadata_table()

            # Check if migration is needed
            if self.needs_migration(table_name, pipeline_type):
                logger.info(f"Schema migration needed for {table_name}")
                success = self.migrate_table(table_name, pipeline_type=pipeline_type)
                if not success:
                    return False
            else:
                logger.info(f"Schema for {table_name} is up to date")

            # Ensure vector indexes after schema is correct
            if table_name in ["SourceDocuments", "Entities"]:
                logger.info(f"🔍 Ensuring vector indexes for {table_name}")
                self.ensure_all_vector_indexes()

            return True

        except Exception as e:
            logger.error(f"Failed to ensure schema for {table_name}: {e}")
            return False

    def get_vector_dimension(
        self, table_name: str = "SourceDocuments", model_name: str = None
    ) -> int:
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
            if (
                model_name
                and model_name != self._table_configs[table_name]["default_model"]
            ):
                if model_name in self._model_dimensions:
                    dimension = self._model_dimensions[model_name]
                else:
                    logger.warning(
                        f"Unknown model '{model_name}' for {table_name}, using table default: {dimension}"
                    )

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

        logger.debug(
            f"Schema Manager: {table_name} with model {model_name or 'default'} -> {dimension}D"
        )
        return dimension

    def validate_vector_dimension(
        self, table_name: str, provided_dimension: int, context: str = ""
    ) -> None:
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
        results = {"consistent": True, "issues": [], "table_dimensions": {}}

        for table_name in self._table_configs.keys():
            try:
                current_config = self.get_current_schema_config(table_name)
                expected_dimension = self.get_vector_dimension(table_name)

                results["table_dimensions"][table_name] = {
                    "expected": expected_dimension,
                    "current": (
                        current_config.get("vector_dimension")
                        if current_config
                        else None
                    ),
                }

                if (
                    current_config
                    and current_config.get("vector_dimension") != expected_dimension
                ):
                    results["consistent"] = False
                    results["issues"].append(
                        {
                            "table": table_name,
                            "issue": "dimension_mismatch",
                            "expected": expected_dimension,
                            "current": current_config.get("vector_dimension"),
                        }
                    )

            except Exception as e:
                results["consistent"] = False
                results["issues"].append(
                    {"table": table_name, "issue": "validation_error", "error": str(e)}
                )

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
                "vector_dimension": self.get_vector_dimension(table),
            }

        return status

    def _migrate_entities_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate RAG.Entities table with proper schema and foreign keys."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")

            # Try multiple table name approaches
            table_attempts = [
                "RAG.Entities",  # Preferred
                "Entities",  # Fallback
            ]

            # Ensure SourceDocuments.doc_id is unique so Entities FK can reference it
            try:
                cursor.execute(
                    "CREATE UNIQUE INDEX idx_source_docs_doc_id_unique ON RAG.SourceDocuments (doc_id)"
                )
                logger.info("✅ Ensured unique index on RAG.SourceDocuments(doc_id)")
            except Exception as e:
                # Index may already exist or another benign condition; proceed
                logger.debug(f"Unique index ensure for SourceDocuments.doc_id: {e}")

            for table_name in table_attempts:
                try:
                    logger.info(
                        f"🔧 Attempting to create Entities table as {table_name}"
                    )

                    # Drop existing table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped existing {table_name} table")

                    # Create with correct schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        entity_id VARCHAR(255) PRIMARY KEY,
                        entity_name VARCHAR(1000) NOT NULL,
                        entity_type VARCHAR(255) NOT NULL,
                        source_doc_id VARCHAR(255) NOT NULL,
                        description TEXT NULL,
                        embedding VECTOR({vector_data_type}, {vector_dim}) NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(doc_id) ON DELETE CASCADE
                    )
                    """
                    cursor.execute(create_sql)
                    logger.info(f"✅ Successfully created {table_name} table")

                    # Create indexes for GraphRAG query performance
                    indexes = [
                        f"CREATE INDEX idx_entities_name ON {table_name} (entity_name)",
                        f"CREATE INDEX idx_entities_type ON {table_name} (entity_type)",
                        f"CREATE INDEX idx_entities_source_doc ON {table_name} (source_doc_id)",
                    ]
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                            logger.debug(f"Created index: {index_sql}")
                        except Exception as e:
                            logger.warning(f"Failed to create index: {e}")

                    # Update schema metadata
                    try:
                        self._update_schema_metadata(
                            cursor, "Entities", expected_config
                        )
                    except Exception as meta_error:
                        logger.debug(f"Schema metadata update failed: {meta_error}")

                    logger.info(
                        f"✅ Entities table created successfully as {table_name}"
                    )
                    return True

                except Exception as table_error:
                    logger.warning(
                        f"Failed to create table as {table_name}: {table_error}"
                    )
                    if table_name == table_attempts[-1]:
                        logger.error("All table creation attempts failed")
                        return False
                    continue

            return False

        except Exception as e:
            logger.error(f"Failed to migrate Entities table: {e}")
            return False

    def _migrate_entity_relationships_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate RAG.EntityRelationships table with proper schema and foreign keys."""
        try:
            # Try multiple table name approaches
            table_attempts = [
                "RAG.EntityRelationships",  # Preferred
                "EntityRelationships",  # Fallback
            ]

            for table_name in table_attempts:
                try:
                    logger.info(
                        f"🔧 Attempting to create EntityRelationships table as {table_name}"
                    )

                    # Drop existing table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped existing {table_name} table")

                    # Create with correct schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        relationship_id VARCHAR(255) PRIMARY KEY,
                        source_entity_id VARCHAR(255) NOT NULL,
                        target_entity_id VARCHAR(255) NOT NULL,
                        relationship_type VARCHAR(255) NOT NULL,
                        metadata TEXT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_entity_id) REFERENCES RAG.Entities(entity_id) ON DELETE CASCADE,
                        FOREIGN KEY (target_entity_id) REFERENCES RAG.Entities(entity_id) ON DELETE CASCADE
                    )
                    """
                    cursor.execute(create_sql)
                    logger.info(f"✅ Successfully created {table_name} table")

                    # Create indexes for GraphRAG traversal performance
                    indexes = [
                        f"CREATE INDEX idx_rel_source ON {table_name} (source_entity_id)",
                        f"CREATE INDEX idx_rel_target ON {table_name} (target_entity_id)",
                    ]
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                            logger.debug(f"Created index: {index_sql}")
                        except Exception as e:
                            logger.warning(f"Failed to create index: {e}")

                    # Update schema metadata
                    try:
                        self._update_schema_metadata(
                            cursor, "EntityRelationships", expected_config
                        )
                    except Exception as meta_error:
                        logger.debug(f"Schema metadata update failed: {meta_error}")

                    logger.info(
                        f"✅ EntityRelationships table created successfully as {table_name}"
                    )
                    return True

                except Exception as table_error:
                    logger.warning(
                        f"Failed to create table as {table_name}: {table_error}"
                    )
                    if table_name == table_attempts[-1]:
                        logger.error("All table creation attempts failed")
                        return False
                    continue

            return False

        except Exception as e:
            logger.error(f"Failed to migrate EntityRelationships table: {e}")
            return False

    def ensure_vector_hnsw_index(
        self, cursor, table: str, column: str, index_name: str, try_acorn: bool = True
    ) -> None:
        """
        Ensure HNSW vector index exists on the specified table and column.

        Attempts to create index with ACORN=1 optimization first, falls back to standard HNSW.
        This function is idempotent - no-op if index already exists.

        Args:
            cursor: Database cursor
            table: Table name (e.g., "RAG.SourceDocuments")
            column: Vector column name (e.g., "embedding")
            index_name: Index name (e.g., "idx_SourceDocuments_embedding")
            try_acorn: Whether to attempt ACORN=1 optimization (default: True)
        """
        try:
            # Check if index already exists
            check_sql = """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES
                WHERE TABLE_NAME = ? AND INDEX_NAME = ?
            """
            cursor.execute(check_sql, [table.split(".")[-1], index_name])
            index_exists = cursor.fetchone()[0] > 0

            if index_exists:
                logger.debug(
                    f"HNSW index {index_name} already exists on {table}.{column}"
                )
                return

            # Try to create index with ACORN=1 first
            if try_acorn:
                try:
                    acorn_sql = f"CREATE INDEX {index_name} ON {table}({column}) AS HNSW WITH (ACORN=1)"
                    logger.info(
                        f"Creating HNSW index with ACORN=1: {index_name} on {table}.{column}"
                    )
                    cursor.execute(acorn_sql)
                    logger.info(
                        f"✅ Successfully created HNSW index with ACORN=1: {index_name}"
                    )
                    return
                except Exception as acorn_error:
                    logger.warning(
                        f"ACORN=1 syntax not supported, falling back to standard HNSW: {acorn_error}"
                    )

            # Fallback to standard HNSW
            standard_sql = f"CREATE INDEX {index_name} ON {table}({column}) AS HNSW"
            logger.info(
                f"Creating standard HNSW index: {index_name} on {table}.{column}"
            )
            cursor.execute(standard_sql)
            logger.info(f"✅ Successfully created standard HNSW index: {index_name}")

        except Exception as e:
            logger.error(
                f"Failed to create HNSW index {index_name} on {table}.{column}: {e}"
            )
            logger.warning(
                "Continuing without vector index - this may impact performance"
            )
            # Re-raise to comply with Constitutional Principle VI: Explicit Error Handling
            # Caller can decide whether to continue or abort based on business logic
            raise

    def ensure_all_vector_indexes(self) -> None:
        """
        Ensure all required vector indexes exist at startup.

        Creates HNSW indexes (with ACORN=1 when available) for all vector columns
        across the RAG system.
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Define vector indexes to ensure
            vector_indexes = [
                {
                    "table": "RAG.SourceDocuments",
                    "column": "embedding",
                    "index_name": "idx_SourceDocuments_embedding",
                },
                {
                    "table": "RAG.Entities",
                    "column": "embedding",
                    "index_name": "idx_Entities_embedding",
                },
            ]

            logger.info("Ensuring HNSW vector indexes...")
            errors = []

            for index_def in vector_indexes:
                try:
                    self.ensure_vector_hnsw_index(
                        cursor,
                        index_def["table"],
                        index_def["column"],
                        index_def["index_name"],
                        try_acorn=True,
                    )
                except Exception as index_error:
                    # Collect error but continue with other indexes
                    error_msg = f"Failed to create index {index_def['index_name']}: {index_error}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            if errors:
                # If any errors occurred, raise a comprehensive exception
                all_errors = "; ".join(errors)
                raise Exception(f"Vector index creation errors: {all_errors}")

            connection.commit()
            logger.info("✅ Vector index ensure process completed")

        except Exception as e:
            logger.error(f"Error during vector index ensure process: {e}")
            connection.rollback()
        finally:
            cursor.close()

    # ========== AUDIT TESTING METHODS ==========
    # These methods replace direct SQL anti-patterns in integration tests

    def verify_table_structure(self, table_name: str) -> Dict[str, Any]:
        """
        Verify table structure using proper abstractions (replaces direct SQL in tests).

        Args:
            table_name: Table name without schema (e.g., 'SourceDocuments')

        Returns:
            Dictionary mapping column names to data types
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """,
                [table_name.upper()],
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to verify table structure for {table_name}: {e}")
            return {}
        finally:
            cursor.close()

    # ========== IRIS GRAPH CORE TABLE MIGRATION METHODS ==========

    def _migrate_rdf_labels_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate rdf_labels table for entity type/label mapping."""
        try:
            logger.info("🔧 Migrating rdf_labels table for IRIS Graph Core")

            # Drop existing table if it exists (for now - data preservation not implemented)
            cursor.execute("DROP TABLE IF EXISTS rdf_labels")
            logger.info("Dropped existing rdf_labels table")

            # Create table with proper schema
            create_sql = """
            CREATE TABLE rdf_labels (
                s      VARCHAR(256) NOT NULL,
                label  VARCHAR(128) NOT NULL
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created rdf_labels table")

            # Create indexes for GraphRAG performance
            indexes = [
                "CREATE INDEX idx_labels_label_s ON rdf_labels(label, s)",
                "CREATE INDEX idx_labels_s_label ON rdf_labels(s, label)",
            ]
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.debug(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            # Update schema metadata
            try:
                self._update_schema_metadata(cursor, "rdf_labels", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ rdf_labels table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate rdf_labels table: {e}")
            return False

    def _migrate_rdf_props_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate rdf_props table for entity properties."""
        try:
            logger.info("🔧 Migrating rdf_props table for IRIS Graph Core")

            # Drop existing table if it exists
            cursor.execute("DROP TABLE IF EXISTS rdf_props")
            logger.info("Dropped existing rdf_props table")

            # Create table with proper schema
            create_sql = """
            CREATE TABLE rdf_props (
                s      VARCHAR(256) NOT NULL,
                key    VARCHAR(128) NOT NULL,
                val    VARCHAR(4000)
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created rdf_props table")

            # Create indexes for property lookup performance
            indexes = [
                "CREATE INDEX idx_props_s_key ON rdf_props(s, key)",
                "CREATE INDEX idx_props_key_val ON rdf_props(key, val)",
            ]
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.debug(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            # Update schema metadata
            try:
                self._update_schema_metadata(cursor, "rdf_props", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ rdf_props table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate rdf_props table: {e}")
            return False

    def _migrate_rdf_edges_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate rdf_edges table for graph relationships with JSON confidence."""
        try:
            logger.info("🔧 Migrating rdf_edges table for IRIS Graph Core")

            # Drop existing table if it exists
            cursor.execute("DROP TABLE IF EXISTS rdf_edges")
            logger.info("Dropped existing rdf_edges table")

            # Create table with proper schema including JSON qualifiers
            # IRIS uses IDENTITY instead of GENERATED BY DEFAULT AS IDENTITY
            create_sql = """
            CREATE TABLE rdf_edges (
                edge_id    BIGINT IDENTITY PRIMARY KEY,
                s          VARCHAR(256) NOT NULL,
                p          VARCHAR(128) NOT NULL,
                o_id       VARCHAR(256) NOT NULL,
                qualifiers VARCHAR(4000)
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created rdf_edges table")

            # Create indexes for graph traversal performance
            indexes = [
                "CREATE INDEX idx_edges_s_p ON rdf_edges(s, p)",
                "CREATE INDEX idx_edges_p_oid ON rdf_edges(p, o_id)",
                "CREATE INDEX idx_edges_s ON rdf_edges(s)",
            ]
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.debug(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            # Update schema metadata
            try:
                self._update_schema_metadata(cursor, "rdf_edges", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ rdf_edges table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate rdf_edges table: {e}")
            return False

    def _migrate_kg_node_embeddings_optimized_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate kg_NodeEmbeddings_optimized table for HNSW vector search."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")

            logger.info(
                f"🔧 Migrating kg_NodeEmbeddings_optimized table to {vector_dim}D vectors with {vector_data_type} type"
            )

            # Drop existing table if it exists
            cursor.execute("DROP TABLE IF EXISTS kg_NodeEmbeddings_optimized")
            logger.info("Dropped existing kg_NodeEmbeddings_optimized table")

            # Create table with optimized VECTOR column
            create_sql = f"""
            CREATE TABLE kg_NodeEmbeddings_optimized (
                id   VARCHAR(256) PRIMARY KEY,
                emb  VECTOR({vector_data_type}, {vector_dim}) NOT NULL
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created kg_NodeEmbeddings_optimized table")

            # Create HNSW index for high-performance vector search
            try:
                self.ensure_vector_hnsw_index(
                    cursor,
                    "kg_NodeEmbeddings_optimized",
                    "emb",
                    "HNSW_NodeEmb_Optimized",
                    try_acorn=True,
                )
                logger.info("✅ HNSW index created for optimized vector table")
            except Exception as index_error:
                logger.warning(f"HNSW index creation failed: {index_error}")

            # Update schema metadata
            try:
                self._update_schema_metadata(
                    cursor, "kg_NodeEmbeddings_optimized", expected_config
                )
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ kg_NodeEmbeddings_optimized table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate kg_NodeEmbeddings_optimized table: {e}")
            return False

    @classmethod
    def create_schema_manager(
        cls, pipeline_type: str, connection_manager, config_manager
    ):
        """
        Factory method to create appropriate schema manager for pipeline type.

        Args:
            pipeline_type: Type of pipeline (e.g., 'hybrid_graphrag', 'graphrag')
            connection_manager: Database connection manager
            config_manager: Configuration manager

        Returns:
            Appropriate schema manager instance
        """
        # Get pipeline requirements
        requirements = config_manager.get_pipeline_requirements(pipeline_type)
        schema_manager_type = requirements.get("schema_manager", "SchemaManager")

        if schema_manager_type == "HybridGraphRAGSchemaManager":
            from .hybrid_schema_manager import HybridGraphRAGSchemaManager

            logger.info(f"Creating HybridGraphRAGSchemaManager for {pipeline_type}")
            return HybridGraphRAGSchemaManager(connection_manager, config_manager)
        elif pipeline_type in ["graphrag", "hybrid_graphrag"]:
            # Enhanced schema manager for graph-based pipelines
            logger.info(f"Creating enhanced SchemaManager for {pipeline_type}")
            manager = cls(connection_manager, config_manager)
            manager.pipeline_type = pipeline_type
            return manager
        else:
            logger.info(f"Creating standard SchemaManager for {pipeline_type}")
            return cls(connection_manager, config_manager)

    def ensure_pipeline_schema(self, pipeline_type: str) -> bool:
        """
        Ensure schema for specific pipeline type based on requirements.

        Args:
            pipeline_type: The pipeline type to ensure schema for

        Returns:
            True if all schema requirements were met successfully
        """
        try:
            logger.info(f"Ensuring schema for pipeline type: {pipeline_type}")
            requirements = self.config_manager.get_pipeline_requirements(pipeline_type)
            schema_requirements = requirements.get("schema_requirements", {})

            # Ensure required tables
            required_tables = schema_requirements.get("tables", [])
            for table_name in required_tables:
                if not self.ensure_table_schema(table_name, pipeline_type):
                    logger.error(f"Failed to ensure required table: {table_name}")
                    return False
                else:
                    logger.info(f"✅ Table {table_name} ensured for {pipeline_type}")

            # Create required indexes
            required_indexes = schema_requirements.get("indexes", [])
            for index_config in required_indexes:
                if not self._ensure_index(index_config):
                    if index_config.get("required", True):
                        logger.error(
                            f"Failed to create required index: {index_config['name']}"
                        )
                        return False
                    else:
                        logger.info(
                            f"Optional index not created: {index_config['name']}"
                        )

            logger.info(f"✅ Schema ensured successfully for {pipeline_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure schema for {pipeline_type}: {e}")
            return False

    def ensure_table_schema(self, table_name: str, pipeline_type: str = None) -> bool:
        """
        Ensure a specific table exists with proper schema.

        Args:
            table_name: Name of the table to ensure
            pipeline_type: Optional pipeline type for context

        Returns:
            True if table exists or was created successfully
        """
        try:
            # Check if this is a standard RAG table
            standard_tables = ["SourceDocuments", "DocumentChunks", "Entities", "EntityRelationships", "Communities"]
            if table_name in standard_tables:
                return self._ensure_standard_table(table_name)

            # Check if this is an iris-vector-graph table (case-insensitive comparison)
            iris_graph_tables = ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]
            iris_graph_tables_lower = [t.lower() for t in iris_graph_tables]
            if table_name.lower() in iris_graph_tables_lower:
                # Check if migration is needed (this will create the table if it doesn't exist)
                if self.needs_migration(table_name, pipeline_type):
                    logger.info(f"Creating iris-vector-graph table: {table_name}")
                    return self.migrate_table(table_name, pipeline_type=pipeline_type)
                else:
                    logger.info(f"Table {table_name} already exists and is up to date")
                    return True
            else:
                # For other tables, delegate to specialized managers
                logger.warning(
                    f"Table {table_name} not recognized as standard RAG or iris-vector-graph table"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to ensure table {table_name}: {e}")
            return False

    def _ensure_standard_table(self, table_name: str) -> bool:
        """Ensure standard RAG tables exist."""
        try:
            # Check class-level cache first to avoid repeated validation spam
            if table_name in SchemaManager._tables_validated:
                logger.debug(f"Table RAG.{table_name} already validated (cached) - skipping check")
                return True

            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = '{table_name}'
            """
            )

            exists = cursor.fetchone()[0] > 0

            if exists:
                logger.debug(f"Table RAG.{table_name} exists - caching validation result")
                SchemaManager._tables_validated.add(table_name)  # Cache the result
                cursor.close()
                return True

            # Create table based on type
            if table_name == "SourceDocuments":
                success = self._create_source_documents_table(cursor)
            elif table_name == "Entities":
                success = self._create_entities_table(cursor)
            elif table_name == "EntityRelationships":
                success = self._create_entity_relationships_table(cursor)
            elif table_name == "Communities":
                success = self._create_communities_table(cursor)
            elif table_name == "DocumentChunks":
                success = self._create_document_chunks_table(cursor)
            else:
                logger.error(f"Unknown standard table: {table_name}")
                cursor.close()
                return False

            if success:
                conn.commit()
                logger.info(f"✅ Successfully created table RAG.{table_name}")
            else:
                conn.rollback()
                logger.error(f"❌ Failed to create table RAG.{table_name}")

            cursor.close()
            return success

        except Exception as e:
            logger.error(f"Error ensuring standard table {table_name}: {e}")
            try:
                conn.rollback()
                cursor.close()
            except:
                pass
            return False

    def _create_source_documents_table(self, cursor) -> bool:
        """Create SourceDocuments table."""
        try:
            embedding_config = self.config_manager.get_embedding_config()
            dimension = embedding_config.get("dimension", 384)

            create_sql = f"""
            CREATE TABLE RAG.SourceDocuments (
                doc_id VARCHAR(255) NOT NULL,
                text_content LONGVARCHAR,
                metadata VARCHAR(2000),
                embedding VECTOR(FLOAT, {dimension}),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (doc_id)
            )
            """
            cursor.execute(create_sql)
            return True
        except Exception as e:
            logger.error(f"Failed to create SourceDocuments table: {e}")
            return False

    def _create_entities_table(self, cursor) -> bool:
        """Create Entities table."""
        try:
            create_sql = """
            CREATE TABLE RAG.Entities (
                entity_id VARCHAR(255) NOT NULL,
                entity_name VARCHAR(500) NOT NULL,
                entity_type VARCHAR(100),
                description VARCHAR(2000),
                confidence DOUBLE DEFAULT 1.0,
                source_doc_id VARCHAR(255),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (entity_id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                "CREATE INDEX idx_entities_name ON RAG.Entities (entity_name)",
                "CREATE INDEX idx_entities_type ON RAG.Entities (entity_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True
        except Exception as e:
            logger.error(f"Failed to create Entities table: {e}")
            return False

    def _create_entity_relationships_table(self, cursor) -> bool:
        """Create EntityRelationships table."""
        try:
            create_sql = """
            CREATE TABLE RAG.EntityRelationships (
                relationship_id VARCHAR(255) NOT NULL,
                source_entity_id VARCHAR(255) NOT NULL,
                target_entity_id VARCHAR(255) NOT NULL,
                relationship_type VARCHAR(255),
                weight DOUBLE DEFAULT 1.0,
                confidence DOUBLE DEFAULT 1.0,
                source_document VARCHAR(255),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (relationship_id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                "CREATE INDEX idx_relationships_source ON RAG.EntityRelationships (source_entity_id)",
                "CREATE INDEX idx_relationships_target ON RAG.EntityRelationships (target_entity_id)",
                "CREATE INDEX idx_relationships_type ON RAG.EntityRelationships (relationship_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True
        except Exception as e:
            logger.error(f"Failed to create EntityRelationships table: {e}")
            return False

    def _create_communities_table(self, cursor) -> bool:
        """Create Communities table for GraphRAG community detection."""
        try:
            create_sql = """
            CREATE TABLE RAG.Communities (
                community_id VARCHAR(255) NOT NULL,
                name VARCHAR(500),
                description VARCHAR(5000),
                entity_ids VARCHAR(10000),
                entity_count INT DEFAULT 0,
                hierarchy_level INT DEFAULT 0,
                parent_community_id VARCHAR(255),
                summary VARCHAR(5000),
                confidence DOUBLE DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (community_id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                "CREATE INDEX idx_communities_level ON RAG.Communities (hierarchy_level)",
                "CREATE INDEX idx_communities_parent ON RAG.Communities (parent_community_id)",
            ]

            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as idx_error:
                    logger.warning(f"Index creation warning: {idx_error}")

            logger.info("✅ Communities table created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Communities table: {e}")
            return False

    def _create_document_chunks_table(self, cursor) -> bool:
        """Create DocumentChunks table."""
        try:
            dimension = self.config_manager.get("embedding:dimension", 384)

            create_sql = f"""
            CREATE TABLE RAG.DocumentChunks (
                id VARCHAR(255) PRIMARY KEY,
                chunk_id VARCHAR(255),
                source_document_id VARCHAR(255),
                chunk_text TEXT,
                chunk_embedding VECTOR(FLOAT, {dimension}),
                chunk_index INTEGER,
                chunk_type VARCHAR(100),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_document_id) REFERENCES RAG.SourceDocuments(id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                "CREATE INDEX idx_chunks_source_doc_id ON RAG.DocumentChunks (source_document_id)",
                "CREATE INDEX idx_chunks_chunk_id ON RAG.DocumentChunks (chunk_id)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True
        except Exception as e:
            logger.error(f"Failed to create DocumentChunks table: {e}")
            return False

    def _ensure_index(self, index_config: Dict[str, Any]) -> bool:
        """Create an index based on configuration."""
        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            index_name = index_config.get("name")
            table_name = index_config.get("table")
            index_type = index_config.get("type", "BTREE")

            # Basic index creation - could be enhanced for specific types
            if index_type == "HNSW":
                # HNSW indexes require special handling
                logger.info(f"HNSW index {index_name} requires specialized creation")
                cursor.close()
                return (
                    index_config.get("required", True) == False
                )  # Optional indexes return True

            cursor.close()
            return True

        except Exception as e:
            logger.error(
                f"Failed to create index {index_config.get('name', 'unknown')}: {e}"
            )
            return False
