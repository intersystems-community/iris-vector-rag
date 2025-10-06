"""
HybridGraphRAG Schema Manager - iris_graph_core Table Management

This module manages the database schema for HybridGraphRAG pipelines,
including the creation and maintenance of iris_graph_core tables required
for advanced hybrid search capabilities.
"""

import logging
from typing import Dict, List, Optional

from .schema_manager import SchemaManager

logger = logging.getLogger(__name__)


class HybridGraphRAGSchemaManager(SchemaManager):
    """
    Schema manager for HybridGraphRAG pipelines with iris_graph_core support.

    Extends the base SchemaManager to include creation and management of
    specialized tables required for hybrid search, graph traversal, and
    multi-modal retrieval capabilities.
    """

    def __init__(self, connection_manager, config_manager):
        """Initialize HybridGraphRAG schema manager."""
        super().__init__(connection_manager, config_manager)
        self.iris_graph_core_tables = {
            "KG_NODEEMBEDDINGS_OPTIMIZED": self._create_kg_nodeembeddings_optimized_table,
            "RDF_EDGES": self._create_rdf_edges_table,
            "RDF_LABELS": self._create_rdf_labels_table,
            "RDF_PROPS": self._create_rdf_props_table,
        }

    def ensure_hybrid_graphrag_schema(self) -> bool:
        """
        Ensure all required tables for HybridGraphRAG exist.

        Returns:
            True if all tables were created/validated successfully
        """
        try:
            logger.info("🔧 Ensuring HybridGraphRAG schema (iris_graph_core tables)...")

            # First ensure base RAG tables (entities, etc)
            base_tables = ["SourceDocuments", "Entities", "EntityRelationships"]
            for table in base_tables:
                if not self.ensure_table_schema(table):
                    logger.error(f"Failed to ensure base table: {table}")
                    return False

            # Create iris_graph_core tables
            success_count = 0
            for table_name, create_func in self.iris_graph_core_tables.items():
                try:
                    if self._ensure_iris_graph_core_table(table_name, create_func):
                        success_count += 1
                        logger.info(f"✅ Table {table_name} ensured successfully")
                    else:
                        logger.error(f"❌ Failed to ensure table {table_name}")
                except Exception as e:
                    logger.error(f"❌ Error ensuring table {table_name}: {e}")

            total_tables = len(self.iris_graph_core_tables)
            if success_count == total_tables:
                logger.info(
                    f"🎉 All {total_tables} iris_graph_core tables ensured successfully"
                )
                return True
            else:
                logger.warning(
                    f"⚠️ Only {success_count}/{total_tables} tables ensured successfully"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to ensure HybridGraphRAG schema: {e}")
            return False

    def _ensure_iris_graph_core_table(self, table_name: str, create_func) -> bool:
        """
        Ensure a specific iris_graph_core table exists.

        Args:
            table_name: Name of the table to ensure
            create_func: Function to create the table if it doesn't exist

        Returns:
            True if table exists or was created successfully
        """
        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            # Check if table exists in RAG schema
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = '{table_name}'
            """
            )

            exists = cursor.fetchone()[0] > 0

            if exists:
                logger.debug(f"Table RAG.{table_name} already exists")
                cursor.close()
                return True

            # Create the table
            logger.info(f"Creating iris_graph_core table: RAG.{table_name}")
            create_success = create_func(cursor)

            if create_success:
                conn.commit()
                logger.info(f"✅ Successfully created table RAG.{table_name}")
            else:
                conn.rollback()
                logger.error(f"❌ Failed to create table RAG.{table_name}")

            cursor.close()
            return create_success

        except Exception as e:
            logger.error(f"Error ensuring table {table_name}: {e}")
            try:
                conn.rollback()
                cursor.close()
            except:
                pass
            return False

    def _create_kg_nodeembeddings_optimized_table(self, cursor) -> bool:
        """Create KG_NODEEMBEDDINGS_OPTIMIZED table for optimized vector operations."""
        try:
            # Get embedding dimension from configuration
            embedding_config = self.config_manager.get_embedding_config()
            dimension = embedding_config.get("dimension", 384)

            create_sql = f"""
            CREATE TABLE RAG.KG_NODEEMBEDDINGS_OPTIMIZED (
                node_id VARCHAR(255) NOT NULL,
                embedding VECTOR(DOUBLE, {dimension}) NOT NULL,
                node_type VARCHAR(100),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence DOUBLE DEFAULT 1.0,
                metadata VARCHAR(2000),

                PRIMARY KEY (node_id)
            )
            """

            cursor.execute(create_sql)

            # Create vector index for optimized similarity search
            vector_config = self.config_manager.get_vector_index_config()
            index_sql = f"""
            CREATE INDEX idx_kg_nodeembeddings_vector
            ON RAG.KG_NODEEMBEDDINGS_OPTIMIZED (embedding)
            USING {vector_config.get('type', 'HNSW')}
            WITH (
                M = {vector_config.get('M', 16)},
                efConstruction = {vector_config.get('efConstruction', 200)},
                Distance = '{vector_config.get('Distance', 'COSINE')}'
            )
            """

            try:
                cursor.execute(index_sql)
                logger.info("✅ HNSW index created for KG_NODEEMBEDDINGS_OPTIMIZED")
            except Exception as idx_e:
                logger.warning(
                    f"Could not create HNSW index (may require licensed IRIS): {idx_e}"
                )
                # Create basic index as fallback
                cursor.execute(
                    """
                    CREATE INDEX idx_kg_nodeembeddings_basic
                    ON RAG.KG_NODEEMBEDDINGS_OPTIMIZED (node_id, node_type)
                """
                )

            return True

        except Exception as e:
            logger.error(f"Failed to create KG_NODEEMBEDDINGS_OPTIMIZED table: {e}")
            return False

    def _create_rdf_edges_table(self, cursor) -> bool:
        """Create RDF_EDGES table for graph relationship storage."""
        try:
            create_sql = """
            CREATE TABLE RAG.RDF_EDGES (
                edge_id VARCHAR(255) NOT NULL,
                subject_id VARCHAR(255) NOT NULL,
                predicate VARCHAR(255) NOT NULL,
                object_id VARCHAR(255) NOT NULL,
                edge_type VARCHAR(100),
                weight DOUBLE DEFAULT 1.0,
                confidence DOUBLE DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_document VARCHAR(500),
                metadata VARCHAR(2000),

                PRIMARY KEY (edge_id)
            )
            """

            cursor.execute(create_sql)

            # Create indexes for efficient graph traversal
            indexes = [
                "CREATE INDEX idx_rdf_edges_subject ON RAG.RDF_EDGES (subject_id)",
                "CREATE INDEX idx_rdf_edges_object ON RAG.RDF_EDGES (object_id)",
                "CREATE INDEX idx_rdf_edges_predicate ON RAG.RDF_EDGES (predicate)",
                "CREATE INDEX idx_rdf_edges_type ON RAG.RDF_EDGES (edge_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True

        except Exception as e:
            logger.error(f"Failed to create RDF_EDGES table: {e}")
            return False

    def _create_rdf_labels_table(self, cursor) -> bool:
        """Create RDF_LABELS table for entity labeling and classification."""
        try:
            create_sql = """
            CREATE TABLE RAG.RDF_LABELS (
                label_id VARCHAR(255) NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                label_text VARCHAR(1000) NOT NULL,
                label_type VARCHAR(100),
                lang VARCHAR(10) DEFAULT 'en',
                confidence DOUBLE DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(255),

                PRIMARY KEY (label_id)
            )
            """

            cursor.execute(create_sql)

            # Create indexes for label lookup
            indexes = [
                "CREATE INDEX idx_rdf_labels_entity ON RAG.RDF_LABELS (entity_id)",
                "CREATE INDEX idx_rdf_labels_text ON RAG.RDF_LABELS (label_text)",
                "CREATE INDEX idx_rdf_labels_type ON RAG.RDF_LABELS (label_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True

        except Exception as e:
            logger.error(f"Failed to create RDF_LABELS table: {e}")
            return False

    def _create_rdf_props_table(self, cursor) -> bool:
        """Create RDF_PROPS table for entity properties and attributes."""
        try:
            create_sql = """
            CREATE TABLE RAG.RDF_PROPS (
                prop_id VARCHAR(255) NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                property_name VARCHAR(255) NOT NULL,
                property_value VARCHAR(2000),
                property_type VARCHAR(100),
                data_type VARCHAR(50) DEFAULT 'STRING',
                confidence DOUBLE DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(255),

                PRIMARY KEY (prop_id)
            )
            """

            cursor.execute(create_sql)

            # Create indexes for property lookup
            indexes = [
                "CREATE INDEX idx_rdf_props_entity ON RAG.RDF_PROPS (entity_id)",
                "CREATE INDEX idx_rdf_props_name ON RAG.RDF_PROPS (property_name)",
                "CREATE INDEX idx_rdf_props_type ON RAG.RDF_PROPS (property_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True

        except Exception as e:
            logger.error(f"Failed to create RDF_PROPS table: {e}")
            return False

    def validate_hybrid_schema(self) -> Dict[str, bool]:
        """
        Validate that all HybridGraphRAG tables exist and are properly configured.

        Returns:
            Dictionary mapping table names to validation status
        """
        validation_results = {}

        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            for table_name in self.iris_graph_core_tables.keys():
                try:
                    # Check table exists
                    cursor.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = '{table_name}'
                    """
                    )

                    exists = cursor.fetchone()[0] > 0
                    validation_results[table_name] = exists

                    if exists:
                        logger.debug(f"✅ Table RAG.{table_name} validated")
                    else:
                        logger.warning(f"❌ Table RAG.{table_name} missing")

                except Exception as e:
                    logger.error(f"Error validating table {table_name}: {e}")
                    validation_results[table_name] = False

            cursor.close()

        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            for table_name in self.iris_graph_core_tables.keys():
                validation_results[table_name] = False

        return validation_results

    def get_schema_info(self) -> Dict[str, any]:
        """Get comprehensive schema information for HybridGraphRAG."""
        info = {}

        # Add iris_graph_core specific information
        validation = self.validate_hybrid_schema()
        info["iris_graph_core_tables"] = validation
        info["hybrid_ready"] = all(validation.values())

        return info
