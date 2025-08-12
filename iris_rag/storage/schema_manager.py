#!/usr/bin/env python3
"""
Schema management system for IRIS RAG tables with automatic migration support.

This module provides robust schema versioning, configuration tracking, and
automatic migration capabilities for vector dimensions and other schema changes.
"""

import logging
import json
from typing import Dict, Any, Optional, List

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
        self.base_embedding_model = self.config_manager.get(
            "embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2"
        )
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
            logger.warning(
                f"Unusual: Document embeddings ({self.base_embedding_dimension}D) >= token embeddings ({self.colbert_token_dimension}D)"
            )

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
                "dimension": self.base_embedding_dimension,
            },
            "DocumentChunks": {
                "embedding_column": "chunk_embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
            },
        }

        logger.debug(f"Table configurations: {len(self._table_configs)} tables configured")

    def ensure_schema_metadata_table(self):
        """Create schema metadata table if it doesn't exist."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Try different schema approaches in order of preference
            schema_attempts = [
                ("RAG", "RAG.SchemaMetadata"),
                ("current user", "SchemaMetadata"),  # No schema prefix = current user's schema
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
                    logger.info(f"✅ Schema metadata table ensured in {schema_name} schema")
                    break
                except Exception as schema_error:
                    logger.warning(f"Failed to create schema metadata table in {schema_name} schema: {schema_error}")
                    if (schema_name, table_name) == schema_attempts[-1]:  # Last schema attempt
                        # Instead of raising, log warning and continue without metadata table
                        logger.warning(
                            "Schema metadata table creation failed in all schemas. Continuing without metadata table."
                        )
                        logger.warning("This may affect schema versioning but basic functionality will work.")
                        return  # Exit gracefully
                    continue

        except Exception as e:
            logger.error(f"Failed to create schema metadata table: {e}")
            logger.warning("Continuing without schema metadata table. Basic functionality will work.")
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
                    vector_data_type = config.get("vector_data_type", "FLOAT")  # Default to FLOAT

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

    def _get_expected_schema_config(self, table_name: str, pipeline_type: str = None) -> Dict[str, Any]:
        """Get expected schema configuration based on current system config and pipeline requirements."""
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
                "auto_migration": True,
            },
        }

        # Enhanced: Get table requirements from pipeline if specified
        if pipeline_type:
            config.update(self._get_table_requirements_config(table_name, pipeline_type))

        # Table-specific configurations
        if table_name == "SourceDocuments":
            config["configuration"].update({
                "table_type": "document_storage",
                "created_by": "BasicRAG",
                "expected_columns": [
                    "id", "doc_id", "title", "abstract", "text_content", "authors",
                    "keywords", "embedding", "metadata", "created_at"
                ]
            })
        elif table_name == "DocumentChunks":
            config["configuration"].update({
                "table_type": "chunk_storage",
                "created_by": "HybridIFind",
                "expected_columns": [
                    "id", "chunk_id", "doc_id", "chunk_text", "chunk_embedding",
                    "chunk_index", "chunk_type", "metadata", "created_at"
                ]
            })

        return config

    def _get_table_requirements_config(self, table_name: str, pipeline_type: str) -> Dict[str, Any]:
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
        """Check if table needs migration based on configuration and physical structure."""
        current_config = self.get_current_schema_config(table_name)
        expected_config = self._get_expected_schema_config(table_name, pipeline_type)

        if not current_config:
            logger.info(f"Table {table_name} has no schema metadata - migration needed")
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
                return True

        # Compare nested configuration keys
        nested_keys = ["managed_by_schema_manager", "supports_vector_search", "auto_migration"]
        for key in nested_keys:
            cur_val = current_config.get("configuration", {}).get(key)
            exp_val = expected_config.get("configuration", {}).get(key)
            if cur_val != exp_val:
                logger.info(
                    f"Migration required for {table_name} due to config.{key} mismatch: "
                    f"expected={exp_val}, actual={cur_val}"
                )
                return True

        # ✅ NEW: Check that all expected columns exist in physical table
        expected_columns = expected_config.get("configuration", {}).get("expected_columns", [])
        if expected_columns:
            try:
                physical_columns = self.verify_table_structure(table_name)
                physical_col_names = {col.lower() for col in physical_columns.keys()}

                for col in expected_columns:
                    if col.lower() not in physical_col_names:
                        logger.info(f"Migration required for {table_name}: missing column '{col}' in physical schema.")
                        return True
            except Exception as e:
                logger.warning(f"Could not verify physical structure of {table_name}: {e}")
                return True  # Be conservative and migrate if in doubt

        return False

    def migrate_table(self, table_name: str, preserve_data: bool = False, pipeline_type: str = None) -> bool:
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
            expected_config = self._get_expected_schema_config(table_name, pipeline_type)

            if table_name == "SourceDocuments":
                success = self._migrate_source_documents_table(cursor, expected_config, preserve_data)
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "DocumentChunks":
                success = self._migrate_document_chunks_table(cursor, expected_config, preserve_data)
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
        """Migrate SourceDocuments table with requirements-driven DDL generation."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")
            text_content_type = expected_config.get("text_content_type", "LONGVARCHAR")
            supports_ifind = expected_config.get("supports_ifind", False)

            # Try multiple table name approaches to work around IRIS schema issues
            table_attempts = [
                "RAG.SourceDocuments",  # Preferred
                "SourceDocuments",      # Fallback
            ]

            for table_name in table_attempts:
                try:
                    logger.info(f"🔧 Attempting to create SourceDocuments table as {table_name}")
                    logger.info(f"   Text content type: {text_content_type}, iFind support: {supports_ifind}")

                    # Handle foreign key constraints from referencing tables
                    logger.info("Checking for foreign key constraints on SourceDocuments...")
                    known_constraints = [
                        ("DOCUMENTCHUNKSFKEY2", "DocumentChunks"),
                    ]
                    
                    for constraint_name, referencing_table in known_constraints:
                        # Check if referencing table exists first
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM INFORMATION_SCHEMA.TABLES 
                            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ?
                        """, [referencing_table.upper()])
                        table_exists = cursor.fetchone()[0] > 0

                        if not table_exists:
                            logger.info(f"Referencing table RAG.{referencing_table} does not exist — skipping constraint drop")
                            continue

                        try:
                            logger.info(f"Dropping foreign key constraint {constraint_name} from RAG.{referencing_table}")
                            cursor.execute(f"ALTER TABLE RAG.{referencing_table} DROP CONSTRAINT {constraint_name}")
                            logger.info(f"✓ Successfully dropped constraint {constraint_name}")
                        except Exception as fk_error:
                            logger.warning(f"Could not drop foreign key {constraint_name}: {fk_error}")
                            try:
                                logger.info(f"Attempting to drop referencing table RAG.{referencing_table}")
                                cursor.execute(f"DROP TABLE IF EXISTS RAG.{referencing_table}")
                                logger.info(f"✓ Dropped referencing table RAG.{referencing_table}")
                            except Exception as table_error:
                                logger.warning(f"Could not drop referencing table {referencing_table}: {table_error}")


                    # Drop existing table
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped existing {table_name} table")

                    # Recreate with correct schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        doc_id VARCHAR(255),
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

                    # Indexes
                    indexes = [
                        f"CREATE INDEX idx_source_docs_id ON {table_name} (doc_id)",
                        f"CREATE INDEX idx_source_docs_created ON {table_name} (created_at)",
                    ]
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                        except Exception as e:
                            logger.warning(f"Failed to create index: {e}")

                    # Schema metadata
                    try:
                        self._update_schema_metadata(cursor, "SourceDocuments", expected_config)
                    except Exception as meta_error:
                        logger.debug(f"Schema metadata update failed: {meta_error}")

                    logger.info(f"✅ SourceDocuments table created successfully as {table_name}")
                    return True

                except Exception as table_error:
                    logger.warning(f"Failed to create table as {table_name}: {table_error}")
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
            cursor.execute("DELETE FROM RAG.SchemaMetadata WHERE table_name = ?", [table_name])

            # Handle configuration serialization safely
            configuration_json = None
            if "configuration" in config:
                try:
                    configuration_json = json.dumps(config["configuration"])
                except (TypeError, ValueError) as json_error:
                    logger.warning(f"Could not serialize configuration for {table_name}: {json_error}")
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

    def _migrate_document_chunks_table(self, cursor, expected_config: Dict[str, Any], preserve_data: bool) -> bool:
        """Migrate DocumentChunks table."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")

            logger.info(f"🔧 Migrating DocumentChunks table to {vector_dim}D vectors with {vector_data_type} type")

            if preserve_data:
                logger.warning("Data preservation not implemented — table will be dropped")

            cursor.execute("DROP TABLE IF EXISTS RAG.DocumentChunks")

            create_sql = f"""
            CREATE TABLE RAG.DocumentChunks (
                id VARCHAR(255) PRIMARY KEY,
                chunk_id VARCHAR(255),
                doc_id VARCHAR(255),
                chunk_text TEXT,
                chunk_embedding VECTOR(FLOAT, 384),
                chunk_index INTEGER,
                chunk_type VARCHAR(100),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(id)
            )
            """
            cursor.execute(create_sql)

            for index_sql in [
                "CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks (doc_id)",
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
        Performs migration if needed.

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
                return self.migrate_table(table_name, pipeline_type=pipeline_type)
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
        results = {"consistent": True, "issues": [], "table_dimensions": {}}

        for table_name in self._table_configs.keys():
            try:
                current_config = self.get_current_schema_config(table_name)
                expected_dimension = self.get_vector_dimension(table_name)

                results["table_dimensions"][table_name] = {
                    "expected": expected_dimension,
                    "current": current_config.get("vector_dimension") if current_config else None,
                }

                if current_config and current_config.get("vector_dimension") != expected_dimension:
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
                results["issues"].append({"table": table_name, "issue": "validation_error", "error": str(e)})

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
