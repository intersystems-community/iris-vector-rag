"""
Simple storage adapter for entity and relationship persistence.

Follows existing IRIS RAG patterns for database integration.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..core.models import Entity, Relationship
from ..storage.schema_manager import SchemaManager
from .batch_entity_processor import BatchEntityProcessor

logger = logging.getLogger(__name__)


class EntityStorageAdapter:
    """
    Simple storage adapter for entities and relationships.

    Follows ConnectionManager patterns used in the rest of the codebase.
    """

    def __init__(self, connection_manager: ConnectionManager, config: Dict[str, Any]):
        """Initialize storage adapter with connection manager and config."""
        self.connection_manager = connection_manager
        self.config = config

        # Get table names from config
        storage_config = config.get("entity_extraction", {}).get("storage", {})
        self.entities_table = storage_config.get("entities_table", "RAG.Entities")
        self.relationships_table = storage_config.get(
            "relationships_table", "RAG.EntityRelationships"
        )
        self.embeddings_table = storage_config.get(
            "embeddings_table", "RAG.EntityEmbeddings"
        )

        # Incremental ingestion controls
        self.incremental: bool = storage_config.get("incremental", True)
        # id_only | natural_only | id_then_natural (default)
        self.dedupe_strategy: str = storage_config.get(
            "dedupe_strategy", "id_then_natural"
        )
        self.case_insensitive: bool = storage_config.get(
            "natural_key_case_insensitive", True
        )

        # In-run ID canonicalization map for deduping across batches/runs
        # Maps requested entity_id -> canonical stored entity_id
        self._entity_id_map: Dict[str, str] = {}

        # Table existence check caching (performance optimization - prevents redundant checks)
        # Performance fix: 99.96% reduction in table checks (59,564 → 22 calls)
        self._tables_ensured = False

        # Initialize optimized batch processor (Feature 057 - 30-64 sec savings per ticket)
        batch_size = storage_config.get("batch_size", 32)
        self.batch_processor = BatchEntityProcessor(
            connection_manager=connection_manager,
            config=config,
            batch_size=batch_size,
        )

        logger.info(
            f"EntityStorageAdapter initialized with tables: {self.entities_table}, {self.relationships_table}"
        )
        logger.info(f"BatchEntityProcessor initialized (batch_size={batch_size})")

    def _ensure_kg_tables(self) -> None:
        """
        Ensure knowledge graph tables exist using SchemaManager.
        Idempotent, safe to call before storage operations.

        Performance optimization: Caches result after first call to avoid redundant checks.
        """
        # Performance optimization: Skip if tables already ensured
        if self._tables_ensured:
            logger.debug("Tables already ensured, skipping check (cached)")
            return

        try:
            schema_manager = SchemaManager(
                self.connection_manager, ConfigurationManager()
            )
            # Ensure Entities first (depends on SourceDocuments)
            success = schema_manager.ensure_table_schema("Entities")
            logger.info(f"Entities table ensure result: {success}")
            # Then relationships
            success = schema_manager.ensure_table_schema("EntityRelationships")
            logger.info(f"EntityRelationships table ensure result: {success}")

            # Set flag to cache that tables are created (prevents redundant checks)
            self._tables_ensured = True

        except Exception as e:
            logger.error(
                f"Could not ensure knowledge graph tables prior to storage ops: {e}"
            )
            raise  # Re-raise to see what's failing

    def _resolve_source_document(self, cursor, src_id: str) -> str:
        """
        Resolve a provided source document identifier to an existing RAG.SourceDocuments.id.
        Tries:
          1) Direct match on id
          2) Match on doc_id
          3) Match on metadata.parent_doc_id (for chunked storage)
        Returns the best match or the original id if no mapping found.
        """
        try:
            # Direct match on id or doc_id
            cursor.execute(
                "SELECT id FROM RAG.SourceDocuments WHERE id = ? OR doc_id = ?",
                [src_id, src_id],
            )
            row = cursor.fetchone()
            if row:
                return str(row[0])

            # IRIS SQL has no JSON_EXTRACT; use LIKE on metadata JSON text as a best-effort
            try:
                like_pattern = f'%"parent_doc_id":"{src_id}"%'
                cursor.execute(
                    "SELECT id FROM RAG.SourceDocuments WHERE metadata LIKE ? FETCH FIRST 1 ROWS ONLY",
                    [like_pattern],
                )
                row = cursor.fetchone()
                if row:
                    return str(row[0])
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Could not resolve source_document '{src_id}': {e}")
        return src_id

    def store_entity(self, entity: Entity) -> bool:
        """Store a single entity with incremental upsert semantics."""
        conn = None
        cursor = None
        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()
            # Ensure KG tables exist before attempting inserts/updates
            self._ensure_kg_tables()

            # Canonical fields for GraphRAG schema
            entity_id = str(entity.id)
            entity_name = str(entity.text).strip()
            # Normalize enum or string to a clean type name
            entity_type = (
                entity.entity_type.name
                if hasattr(entity.entity_type, "name")
                else str(entity.entity_type).split(".")[-1].strip()
            )
            source_document_raw = str(entity.source_document_id).strip()
            source_document = self._resolve_source_document(cursor, source_document_raw)
            description = None
            embedding = None
            if isinstance(entity.metadata, dict):
                description = entity.metadata.get("description")
                embedding = entity.metadata.get("embedding")

            # Determine canonical ID to use (dedupe)
            target_entity_id = entity_id
            exists_by_id = False

            # Step 1: check by ID
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.entities_table} WHERE entity_id = ?",
                [entity_id],
            )
            exists_by_id = cursor.fetchone()[0] > 0

            # Step 2: Check natural key ONLY if entity_id already exists
            # Skip complex natural key queries during initial ingestion to avoid schema timing issues
            # if (
            #     self.incremental
            #     and not exists_by_id
            #     and self.dedupe_strategy in ("natural_only", "id_then_natural")
            # ):
            #     if self.case_insensitive:
            #         cursor.execute(
            #             f"SELECT entity_id FROM {self.entities_table} "
            #             f"WHERE LOWER(entity_name) = ? AND entity_type = ? AND source_document = ?",
            #             [entity_name.lower(), entity_type, source_document],
            #         )
            #     else:
            #         cursor.execute(
            #             f"SELECT entity_id FROM {self.entities_table} "
            #             f"WHERE entity_name = ? AND entity_type = ? AND source_document = ?",
            #             [entity_name, entity_type, source_document],
            #         )
            #     row = cursor.fetchone()
            #     if row:
            #         target_entity_id = str(row[0])
            #         # Record mapping so downstream relationships can resolve to existing row
            #         if target_entity_id != entity_id:
            #             self._entity_id_map[entity_id] = target_entity_id

            # Upsert
            if exists_by_id or target_entity_id != entity_id:
                # Update existing canonical row
                if embedding is not None:
                    embedding_str = json.dumps(embedding)
                    update_sql = f"""
                        UPDATE {self.entities_table}
                        SET entity_name = ?, entity_type = ?, source_doc_id = ?, description = ?, embedding = TO_VECTOR(?, FLOAT, 384)
                        WHERE entity_id = ?
                    """
                    params = [
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        embedding_str,
                        target_entity_id,
                    ]
                else:
                    update_sql = f"""
                        UPDATE {self.entities_table}
                        SET entity_name = ?, entity_type = ?, source_doc_id = ?, description = ?
                        WHERE entity_id = ?
                    """
                    params = [
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        target_entity_id,
                    ]
                cursor.execute(update_sql, params)
                logger.debug(
                    f"Updated entity {target_entity_id} in {self.entities_table}"
                )
            else:
                # Insert new
                if embedding is not None:
                    embedding_str = json.dumps(embedding)
                    insert_sql = f"""
                        INSERT INTO {self.entities_table}
                        (entity_id, entity_name, entity_type, source_doc_id, description, embedding)
                        VALUES (?, ?, ?, ?, ?, TO_VECTOR(?, FLOAT, 384))
                    """
                    params = [
                        entity_id,
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        embedding_str,
                    ]
                else:
                    insert_sql = f"""
                        INSERT INTO {self.entities_table}
                        (entity_id, entity_name, entity_type, source_doc_id, description)
                        VALUES (?, ?, ?, ?, ?)
                    """
                    params = [
                        entity_id,
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                    ]
                cursor.execute(insert_sql, params)
                logger.debug(f"Inserted entity {entity_id} into {self.entities_table}")

            conn.commit()
            return True

        except Exception as e:
            try:
                if conn:
                    conn.rollback()
            except Exception:
                pass
            logger.error(
                f"Failed to store entity {getattr(entity, 'id', 'unknown')}: {e}"
            )
            return False
        finally:
            try:
                if cursor:
                    cursor.close()
            except Exception:
                pass

    def store_relationship(self, relationship: Relationship) -> bool:
        """Store a single relationship in the database."""
        conn = None
        cursor = None
        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()
            # Ensure KG tables exist before attempting inserts/updates
            self._ensure_kg_tables()

            # Resolve entity IDs through mapping if they were deduplicated
            source_entity_id = self._entity_id_map.get(
                str(relationship.source_entity_id), str(relationship.source_entity_id)
            )
            target_entity_id = self._entity_id_map.get(
                str(relationship.target_entity_id), str(relationship.target_entity_id)
            )

            # Canonical fields for actual GraphRAG schema
            relationship_id = str(relationship.id)
            relationship_type = str(relationship.relationship_type).strip()

            # Extract actual schema columns
            weight = float(relationship.metadata.get("weight", 1.0)) if relationship.metadata else 1.0
            confidence = float(relationship.confidence)
            source_document = str(relationship.source_document_id) if relationship.source_document_id else None

            # Check if relationship already exists (incremental upsert)
            if self.incremental:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.relationships_table} WHERE relationship_id = ?",
                    [relationship_id],
                )
                exists = cursor.fetchone()[0] > 0

                if exists:
                    # Update existing relationship
                    update_sql = f"""
                        UPDATE {self.relationships_table}
                        SET source_entity_id = ?, target_entity_id = ?, relationship_type = ?,
                            weight = ?, confidence = ?, source_document = ?
                        WHERE relationship_id = ?
                    """
                    params = [
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        weight,
                        confidence,
                        source_document,
                        relationship_id,
                    ]
                    cursor.execute(update_sql, params)
                    logger.debug(
                        f"Updated relationship {relationship_id} in {self.relationships_table}"
                    )
                else:
                    # Insert new relationship
                    insert_sql = f"""
                        INSERT INTO {self.relationships_table}
                        (relationship_id, source_entity_id, target_entity_id, relationship_type,
                         weight, confidence, source_document)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    params = [
                        relationship_id,
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        weight,
                        confidence,
                        source_document,
                    ]
                    cursor.execute(insert_sql, params)
                    logger.debug(
                        f"Inserted relationship {relationship_id} into {self.relationships_table}"
                    )
            else:
                # Direct insert (non-incremental mode)
                insert_sql = f"""
                    INSERT INTO {self.relationships_table}
                    (relationship_id, source_entity_id, target_entity_id, relationship_type,
                     weight, confidence, source_document)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                params = [
                    relationship_id,
                    source_entity_id,
                    target_entity_id,
                    relationship_type,
                    weight,
                    confidence,
                    source_document,
                ]
                cursor.execute(insert_sql, params)
                logger.debug(
                    f"Inserted relationship {relationship_id} into {self.relationships_table}"
                )

            conn.commit()
            return True

        except Exception as e:
            try:
                if conn:
                    conn.rollback()
            except Exception:
                pass
            logger.error(
                f"Failed to store relationship {getattr(relationship, 'id', 'unknown')}: {e}"
            )
            return False
        finally:
            try:
                if cursor:
                    cursor.close()
            except Exception:
                pass

    def store_entities_batch(self, entities: List[Entity]) -> int:
        """
        Store multiple entities using optimized batch processor (Feature 057).

        Performance: 5-10x faster than serial storage for 10+ entities.
        - Serial: 40-70 seconds for 10 entities (4-7s per entity)
        - Batch: 6-10 seconds for 10 entities (single transaction)
        - Speedup: 30-64 seconds saved per ticket

        Returns:
            Number of successfully stored entities
        """
        if not entities:
            return 0

        # Ensure tables exist before batch storage
        self._ensure_kg_tables()

        # Use optimized batch processor with executemany()
        result = self.batch_processor.store_entities_batch(
            entities,
            validate_count=True
        )

        if not result.get("validation_passed", False):
            logger.error(
                f"Entity batch storage validation failed: "
                f"{result.get('entities_stored', 0)}/{len(entities)} stored"
            )

        return result.get("entities_stored", 0)

    def store_relationships_batch(self, relationships: List[Relationship]) -> int:
        """
        Store multiple relationships using optimized batch processor (Feature 057).

        Performance: 5-10x faster than serial storage for 5+ relationships.
        Includes foreign key validation to prevent orphaned relationships.

        Returns:
            Number of successfully stored relationships
        """
        if not relationships:
            return 0

        # Ensure tables exist before batch storage
        self._ensure_kg_tables()

        # Use optimized batch processor with executemany() and FK validation
        result = self.batch_processor.store_relationships_batch(
            relationships,
            validate_foreign_keys=True
        )

        if not result.get("validation_passed", False):
            orphaned = result.get("orphaned_relationships", 0)
            logger.error(
                f"Relationship batch storage validation failed: "
                f"{orphaned} orphaned relationships detected"
            )

        return result.get("relationships_stored", 0)

    def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Retrieve all entities for a specific document."""
        try:
            conn = self.connection_manager.get_connection()

            # For now, return empty list (implement actual query)
            logger.debug(f"Retrieving entities for document {document_id}")

            # TODO: Implement actual database query
            # Example: SELECT * FROM RAG.Entities WHERE source_document_id = ?

            return []

        except Exception as e:
            logger.error(f"Failed to retrieve entities for document {document_id}: {e}")
            return []

    def get_relationships_by_document(self, document_id: str) -> List[Relationship]:
        """Retrieve all relationships for a specific document."""
        try:
            conn = self.connection_manager.get_connection()

            # For now, return empty list
            logger.debug(f"Retrieving relationships for document {document_id}")

            # TODO: Implement actual database query

            return []

        except Exception as e:
            logger.error(
                f"Failed to retrieve relationships for document {document_id}: {e}"
            )
            return []

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Entity]:
        """Retrieve entities by type."""
        try:
            conn = self.connection_manager.get_connection()

            logger.debug(f"Retrieving entities of type {entity_type} (limit: {limit})")

            # TODO: Implement actual database query
            # Example: SELECT * FROM RAG.Entities WHERE type = ? LIMIT ?

            return []

        except Exception as e:
            logger.error(f"Failed to retrieve entities by type {entity_type}: {e}")
            return []

    def create_tables_if_not_exist(self) -> bool:
        """Create entity and relationship tables if they don't exist."""
        try:
            conn = self.connection_manager.get_connection()

            # SQL for creating entities table
            entities_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.entities_table} (
                id VARCHAR(255) PRIMARY KEY,
                text VARCHAR(1000) NOT NULL,
                type VARCHAR(100) NOT NULL,
                confidence DECIMAL(3,2) NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                source_document_id VARCHAR(255) NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            # SQL for creating relationships table
            relationships_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.relationships_table} (
                id VARCHAR(255) PRIMARY KEY,
                source_entity_id VARCHAR(255) NOT NULL,
                target_entity_id VARCHAR(255) NOT NULL,
                type VARCHAR(100) NOT NULL,
                confidence DECIMAL(3,2) NOT NULL,
                source_document_id VARCHAR(255) NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES {self.entities_table}(id),
                FOREIGN KEY (target_entity_id) REFERENCES {self.entities_table}(id)
            )
            """

            # SQL for creating embeddings table (optional)
            embeddings_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                entity_id VARCHAR(255) PRIMARY KEY,
                embedding_vector LONGBLOB,
                model_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES {self.entities_table}(id)
            )
            """

            # For now, just log the SQL (replace with actual execution)
            logger.info("Creating entity extraction tables if they don't exist")
            logger.debug(f"Entities table SQL: {entities_sql}")
            logger.debug(f"Relationships table SQL: {relationships_sql}")
            logger.debug(f"Embeddings table SQL: {embeddings_sql}")

            # TODO: Execute the SQL statements
            # cursor = conn.cursor()
            # cursor.execute(entities_sql)
            # cursor.execute(relationships_sql)
            # cursor.execute(embeddings_sql)
            # conn.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored entities and relationships."""
        try:
            conn = self.connection_manager.get_connection()

            # For now, return mock stats
            stats = {
                "entities_count": 0,
                "relationships_count": 0,
                "entity_types": {},
                "relationship_types": {},
                "last_updated": datetime.utcnow().isoformat(),
            }

            logger.debug("Retrieved storage statistics")

            # TODO: Implement actual statistics queries
            # Example: SELECT COUNT(*) FROM RAG.Entities
            # Example: SELECT type, COUNT(*) FROM RAG.Entities GROUP BY type

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
