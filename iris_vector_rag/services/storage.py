"""
Simple storage adapter for entity and relationship persistence.

Follows existing IRIS RAG patterns for database integration.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..core.models import Entity, Relationship
from ..exceptions import RAGException
from ..storage.catalog import IRISCatalogReader
from ..storage.schema_manager import SchemaManager
from ..storage.shape_resolver import TableShapeResolver
from ..storage.table_shape import (
    CatalogUnavailable,
    ColumnRole,
    TableNotFound,
    TableShape,
)
from .batch_entity_processor import BatchEntityProcessor, to_vector_expression

logger = logging.getLogger(__name__)

#: How much of a stream metadata column a `LIKE` can be applied to. Same projection
#: the vector store uses: `LIKE` against a LONGVARCHAR column matches nothing at all
#: in IRIS, silently, so the text is narrowed to a searchable width first.
_STREAM_METADATA_PROJECTION = 32000


class EntityStorageError(RAGException):
    """Entities or relationships could not be stored.

    `stored` is how many of the batch were written before the failure; `failed` lists
    `{"entity_id", "reason"}` for those that were not.
    """

    def __init__(self, message: str, stored: int = 0, failed: Optional[list] = None):
        super().__init__(message)
        self.stored = stored
        self.failed = failed or []


def _split_table(qualified: str) -> tuple[str, str]:
    """`"RAG.Entities"` -> `("RAG", "Entities")`; an unqualified name gets `RAG`."""
    if "." in qualified:
        schema, _, table = qualified.partition(".")
        return schema, table
    return "RAG", qualified


def _schema_of(qualified: str) -> str:
    return _split_table(qualified)[0]


class EntityStorageAdapter:
    """
    Simple storage adapter for entities and relationships.

    Follows ConnectionManager patterns used in the rest of the codebase.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config: Dict[str, Any],
        shapes: Optional[TableShapeResolver] = None,
    ):
        """Initialize storage adapter with connection manager and config.

        Args:
            connection_manager: Manager for database connections.
            config: The `entity_extraction.storage` block is read from here.
            shapes: Resolver for this session's table shapes (optional). One is built
                from this adapter's own connection on first use when not supplied;
                pass a resolver over a `RecordedCatalog` to describe a shape in a test.
        """
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
        # The document table an entity's source_doc_id points into. Derived from the
        # entities table's schema so a caller who moved the graph to another schema
        # does not have to name the document table twice.
        self.documents_table = storage_config.get(
            "documents_table",
            f"{_schema_of(self.entities_table)}.SourceDocuments",
        )
        # Feature 082: one answer to "which column identifies a document", read from
        # the database once. Resolved lazily so the constructor stays pure.
        self.shapes = shapes

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

        # In-memory cache of entities stored in this adapter instance
        self._entity_cache: Dict[str, Entity] = {}

        # Table existence check caching (performance optimization - prevents redundant checks)
        # Performance fix: 99.96% reduction in table checks (59,564 → 22 calls)
        self._tables_ensured = False
        self._configuration: Optional[ConfigurationManager] = None

        # Initialize optimized batch processor (Feature 057 - 30-64 sec savings per ticket)
        batch_size = storage_config.get("batch_size", 32)
        self.batch_processor = BatchEntityProcessor(
            connection_manager=connection_manager,
            config=config,
            batch_size=batch_size,
            source_document_resolver=self._resolve_source_document,
        )

        logger.info(
            f"EntityStorageAdapter initialized with tables: {self.entities_table}, {self.relationships_table}"
        )
        logger.info(f"BatchEntityProcessor initialized (batch_size={batch_size})")

    def _config_manager(self) -> ConfigurationManager:
        """One `ConfigurationManager` for this adapter, built on first use."""
        if self._configuration is None:
            self._configuration = ConfigurationManager()
        return self._configuration

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
            # The tables live in the entities table's own schema, not whatever
            # schema a fresh ConfigurationManager would default to.
            schema_manager = SchemaManager(
                self.connection_manager,
                self._config_manager(),
                schema_prefix=_schema_of(self.entities_table),
            )
            # Ensure Entities first (depends on SourceDocuments)
            success = schema_manager.ensure_table_schema("Entities")
            logger.info(f"Entities table ensure result: {success}")
            # Then relationships
            success = schema_manager.ensure_table_schema("EntityRelationships")
            logger.info(f"EntityRelationships table ensure result: {success}")

            # Ensure Entities columns exist for legacy schemas
            self._ensure_entity_columns()

            # Set flag to cache that tables are created (prevents redundant checks)
            self._tables_ensured = True

        except Exception as e:
            logger.error(
                f"Could not ensure knowledge graph tables prior to storage ops: {e}"
            )
            raise  # Re-raise to see what's failing

    def _ensure_entity_columns(self) -> None:
        """Ensure legacy Entities schema has required columns."""
        conn = None
        cursor = None

        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            schema, table = _split_table(self.entities_table)

            try:
                # The catalog is read through the one module that owns that SQL, so
                # this file has no second account of what INFORMATION_SCHEMA means.
                snapshot = IRISCatalogReader(conn).snapshot(schema, table)
                columns = {
                    str(column["COLUMN_NAME"]).lower()
                    for column in (snapshot.columns if snapshot else ())
                }
            except Exception as exc:
                logger.warning(
                    f"Failed to inspect {schema}.{table} columns, attempting fallbacks: {exc}"
                )
                columns = set()

            if "entity_id" not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {schema}.{table} ADD entity_id VARCHAR(255)"
                    )
                    if "id" in columns:
                        cursor.execute(
                            f"UPDATE {schema}.{table} SET entity_id = id WHERE entity_id IS NULL"
                        )
                    conn.commit()
                    columns.add("entity_id")
                except Exception as exc:
                    logger.warning(f"Failed to add entity_id column: {exc}")

            if "entity_name" not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {schema}.{table} ADD entity_name VARCHAR(255)"
                    )
                    if "name" in columns:
                        cursor.execute(
                            f"UPDATE {schema}.{table} SET entity_name = name WHERE entity_name IS NULL"
                        )
                    conn.commit()
                    columns.add("entity_name")
                except Exception as exc:
                    logger.warning(f"Failed to add entity_name column: {exc}")

            if "entity_type" not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {schema}.{table} ADD entity_type VARCHAR(100)"
                    )
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Failed to add entity_type column: {exc}")

            if "source_doc_id" not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {schema}.{table} ADD source_doc_id VARCHAR(255)"
                    )
                    if "doc_id" in columns:
                        cursor.execute(
                            f"UPDATE {schema}.{table} SET source_doc_id = doc_id WHERE source_doc_id IS NULL"
                        )
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Failed to add source_doc_id column: {exc}")

            if "doc_id" not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {schema}.{table} ADD doc_id VARCHAR(255)"
                    )
                    if "source_doc_id" in columns:
                        cursor.execute(
                            f"UPDATE {schema}.{table} SET doc_id = source_doc_id WHERE doc_id IS NULL"
                        )
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Failed to add doc_id column: {exc}")

            if "description" not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {schema}.{table} ADD description VARCHAR(4000)"
                    )
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Failed to add description column: {exc}")
        except Exception as exc:
            logger.warning(f"Failed to ensure Entities columns: {exc}")
        finally:
            if cursor:
                cursor.close()

    def _shape_resolver(self) -> TableShapeResolver:
        if self.shapes is None:
            self.shapes = TableShapeResolver(
                IRISCatalogReader(self.connection_manager.get_connection())
            )
        return self.shapes

    def _documents_shape(self) -> Optional[TableShape]:
        """The document table's shape, or None if it cannot be described.

        Resolution feeds a best-effort lookup, so an absent table or an unreachable
        catalog leaves the caller's identifier alone rather than failing a write.
        """
        schema, table = _split_table(self.documents_table)
        try:
            return self._shape_resolver().resolve(schema, table)
        except (TableNotFound, CatalogUnavailable) as exc:
            logger.debug(f"Could not describe {self.documents_table}: {exc}")
            return None

    def _resolve_source_document(self, cursor, src_id: str) -> str:
        """Map a supplied identifier onto a row of the document table.

        Two lookups at most, both against the one column the shape names as the
        table's identifier:

        1. The identifier itself.
        2. `parent_doc_id` inside the metadata JSON, for an id that names a chunk.

        What this replaces: `WHERE id = ? OR doc_id = ?`, which guessed at both
        spellings and so scanned two columns on every call, matched the wrong one on
        a table carrying both, and named a column that does not exist on a canonical
        table at all. IRIS has no JSON_EXTRACT, so step 2 is still a text match --
        against a projection of the column when the column is a stream, because
        `LIKE` on a LONGVARCHAR matches nothing and reports no error.

        Returns the resolved identifier, or `src_id` unchanged when nothing matches.
        """
        shape = self._documents_shape()
        if shape is None:
            return src_id

        identifier = shape.column_for(ColumnRole.IDENTIFIER).name
        table = shape.qualified_name
        try:
            cursor.execute(
                f"SELECT {identifier} FROM {table} WHERE {identifier} = ?",
                [src_id],
            )
            row = cursor.fetchone()
            if row:
                return str(row[0])

            metadata = shape.optional_column_for(ColumnRole.METADATA)
            if metadata is None:
                return src_id

            expression = (
                metadata.name
                if metadata.filterable_with_like
                else f"SUBSTRING({metadata.name},1,{_STREAM_METADATA_PROJECTION})"
            )
            try:
                cursor.execute(
                    f"SELECT TOP 1 {identifier} FROM {table} WHERE {expression} LIKE ?",
                    [f'%"parent_doc_id":"{src_id}"%'],
                )
                row = cursor.fetchone()
                if row:
                    return str(row[0])
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Could not resolve source_document '{src_id}': {e}")
        return src_id

    def store_entity(self, entity: Optional[Entity] = None, **kwargs) -> bool:
        """Store a single entity with incremental upsert semantics."""
        if entity is None:
            entity_id = kwargs.get("entity_id") or kwargs.get("id")
            entity_name = kwargs.get("entity_name") or kwargs.get("text")
            entity_type = kwargs.get("entity_type")

            if not entity_id or not entity_name or not entity_type:
                raise ValueError(
                    "entity_id, entity_name (or text), and entity_type are required"
                )

            confidence = kwargs.get("confidence", 0.95)
            start_offset = kwargs.get("start_offset", 0)
            end_offset = kwargs.get("end_offset", start_offset + len(entity_name))
            source_document_id = kwargs.get("source_document_id", "TEST-DOC-CONTRACT")
            metadata = kwargs.get("metadata", {})

            entity = Entity(
                id=str(entity_id),
                text=str(entity_name),
                entity_type=str(entity_type),
                confidence=float(confidence),
                start_offset=int(start_offset),
                end_offset=int(end_offset),
                source_document_id=str(source_document_id),
                metadata=metadata,
            )
        elif kwargs:
            raise ValueError("Provide either an Entity or keyword fields, not both")

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
            entity_type = str(entity.entity_type).split(".")[-1].strip()
            source_document_raw = str(entity.source_document_id).strip()
            source_document = self._resolve_source_document(cursor, source_document_raw)
            description = None
            embedding = None
            if isinstance(entity.metadata, dict):
                description = entity.metadata.get("description")
                embedding = entity.metadata.get("embedding")
            if embedding is not None and len(embedding) == 0:
                embedding = None  # an empty vector has no width to store

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
                        SET entity_name = ?, entity_type = ?, source_doc_id = ?, description = ?, confidence = ?, embedding = {to_vector_expression(len(embedding))}
                        WHERE entity_id = ?
                    """
                    params = [
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        float(entity.confidence),
                        embedding_str,
                        target_entity_id,
                    ]
                else:
                    update_sql = f"""
                        UPDATE {self.entities_table}
                        SET entity_name = ?, entity_type = ?, source_doc_id = ?, description = ?, confidence = ?
                        WHERE entity_id = ?
                    """
                    params = [
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        float(entity.confidence),
                        target_entity_id,
                    ]
                cursor.execute(update_sql, params)

                conn.commit()
                logger.debug(
                    f"Updated entity {target_entity_id} in {self.entities_table}"
                )
            else:
                # Insert new
                if embedding is not None:
                    embedding_str = json.dumps(embedding)
                    insert_sql = f"""
                        INSERT INTO {self.entities_table}
                        (entity_id, entity_name, entity_type, source_doc_id, description, confidence, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, {to_vector_expression(len(embedding))})
                    """
                    params = [
                        entity_id,
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        float(entity.confidence),
                        embedding_str,
                    ]
                else:
                    insert_sql = f"""
                        INSERT INTO {self.entities_table}
                        (entity_id, entity_name, entity_type, source_doc_id, description, confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """
                    params = [
                        entity_id,
                        entity_name,
                        entity_type,
                        source_document,
                        description,
                        float(entity.confidence),
                    ]
                cursor.execute(insert_sql, params)

                conn.commit()
                logger.debug(f"Inserted entity {entity_id} into {self.entities_table}")

            conn.commit()
            self._entity_cache[str(entity.id)] = entity
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
            weight = (
                float(relationship.metadata.get("weight", 1.0))
                if relationship.metadata
                else 1.0
            )
            confidence = float(relationship.confidence)
            source_document = (
                str(relationship.source_document_id)
                if relationship.source_document_id
                else None
            )

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
        Store multiple entities using the batch processor (Feature 057).

        Chunked by ``storage.batch_size``; existing entity ids are updated, new ones
        inserted; ``confidence`` is written.

        Returns:
            Number of entities stored (all of them, or this raises).

        Raises:
            EntityStorageError: some entities were not stored. ``stored`` and
                ``failed`` on the exception say which.
        """
        if not entities:
            return 0

        # Ensure tables exist before batch storage
        self._ensure_kg_tables()

        result = self.batch_processor.store_entities_batch(
            entities, validate_count=True
        )

        if not result.get("validation_passed", False):
            raise EntityStorageError(
                result.get("error")
                or f"Entity batch storage failed: {result.get('entities_stored', 0)}"
                f"/{len(entities)} stored",
                stored=result.get("entities_stored", 0),
                failed=result.get("failed", []),
            )

        return result.get("entities_stored", 0)

    def store_relationships_batch(self, relationships: List[Relationship]) -> int:
        """
        Store multiple relationships using the batch processor (Feature 057).

        Includes foreign key validation to prevent orphaned relationships.

        Returns:
            Number of relationships stored.

        Raises:
            EntityStorageError: a relationship referenced an entity that is not
                stored, or the write failed.
        """
        if not relationships:
            return 0

        # Ensure tables exist before batch storage
        self._ensure_kg_tables()

        result = self.batch_processor.store_relationships_batch(
            relationships, validate_foreign_keys=True
        )

        if not result.get("validation_passed", False):
            orphaned = result.get("orphaned_relationships", 0)
            raise EntityStorageError(
                result.get("error")
                or f"Relationship batch storage failed: {orphaned} orphaned "
                "relationships detected",
                stored=result.get("relationships_stored", 0),
            )

        return result.get("relationships_stored", 0)

    def _read(self, sql: str, params: Optional[list] = None) -> list:
        """Run a read and return its rows. Errors propagate: an empty list means
        "no rows", never "the query failed"."""
        self._ensure_kg_tables()
        conn = self.connection_manager.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params or [])
            return [tuple(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    _ENTITY_COLUMNS = (
        "entity_id, entity_name, entity_type, source_doc_id, description, confidence"
    )

    @staticmethod
    def _entity_from_row(row: tuple) -> Entity:
        entity_id, name, entity_type, source_doc, description, confidence = row
        return Entity(
            id=str(entity_id),
            text=str(name),
            entity_type=str(entity_type),
            confidence=float(confidence) if confidence is not None else 0.0,
            start_offset=0,
            end_offset=0,
            source_document_id=str(source_doc) if source_doc is not None else "",
            metadata={"description": description} if description else {},
        )

    def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Entities stored with ``source_doc_id = document_id``.

        Offsets are not stored and come back as 0.
        """
        rows = self._read(
            f"SELECT {self._ENTITY_COLUMNS} FROM {self.entities_table} "
            "WHERE source_doc_id = ?",
            [document_id],
        )
        return [self._entity_from_row(r) for r in rows]

    def get_relationships_by_document(self, document_id: str) -> List[Relationship]:
        """Relationships stored with ``source_document = document_id``."""
        rows = self._read(
            "SELECT relationship_id, source_entity_id, target_entity_id, "
            "relationship_type, confidence, source_document "
            f"FROM {self.relationships_table} WHERE source_document = ?",
            [document_id],
        )
        return [
            Relationship(
                id=str(rid),
                source_entity_id=str(src),
                target_entity_id=str(dst),
                relationship_type=str(rtype),
                confidence=float(confidence) if confidence is not None else 0.0,
                source_document_id=str(doc) if doc is not None else "",
            )
            for rid, src, dst, rtype, confidence, doc in rows
        ]

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Entity]:
        """Up to ``limit`` entities whose type is ``entity_type``."""
        top = int(limit)
        if top < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        rows = self._read(
            f"SELECT TOP {top} {self._ENTITY_COLUMNS} FROM {self.entities_table} "
            "WHERE entity_type = ?",
            [entity_type],
        )
        return [self._entity_from_row(r) for r in rows]

    def search_entities(
        self,
        query: str,
        fuzzy: bool = True,
        edit_distance_threshold: int = 2,
        max_results: int = 10,
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name with fuzzy matching.

        The entity-type and confidence filters run in SQL. The name match does not:
        every remaining row is read and compared in Python (Levenshtein distance,
        token subset, substring), so cost grows with the size of the entities table.
        Matches are case-insensitive (e.g. "Scott Derrickson" matches
        "Scott Derrickson director").

        Args:
            query: Search query string
            fuzzy: Enable fuzzy matching (always True, parameter for API compatibility)
            edit_distance_threshold: Maximum edit distance for fuzzy matching (not implemented, for API compatibility)
            max_results: Maximum number of results to return
            entity_types: Optional list of entity types to filter by
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of entity dictionaries with fields:
                - entity_id: Entity identifier
                - entity_name: Entity name text
                - entity_type: Entity type
                - source_doc_id: Source document ID
                - description: Entity description (may be None)
                - confidence: Confidence score (0.0-1.0)
                - similarity_score: Fuzzy match similarity (0.0-1.0)
                - edit_distance: Levenshtein edit distance

        Example:
            >>> results = adapter.search_entities("Scott Derrickson", fuzzy=True, max_results=5)
            >>> print(results[0]["entity_name"])  # "Scott Derrickson director"
            >>> print(results[0]["similarity_score"])  # 1.0

        Note:
            With ``fuzzy=False`` only an exact (case-insensitive) name matches.
            ``edit_distance_threshold`` is the largest Levenshtein distance accepted.
        """
        if not query or not str(query).strip():
            return []

        if max_results == 0:
            return []

        # Ensure tables exist and schema is up to date before searching
        try:
            self._ensure_kg_tables()
        except Exception as exc:
            logger.warning(f"Failed to ensure KG tables before search: {exc}")

        normalized_query = str(query).strip()
        lower_query = normalized_query.lower()
        short_query = len(lower_query) <= 2

        def levenshtein_distance(a: str, b: str) -> int:
            if a == b:
                return 0
            if not a:
                return len(b)
            if not b:
                return len(a)
            previous = list(range(len(b) + 1))
            for i, char_a in enumerate(a, 1):
                current = [i]
                for j, char_b in enumerate(b, 1):
                    cost = 0 if char_a == char_b else 1
                    current.append(
                        min(
                            previous[j] + 1,
                            current[j - 1] + 1,
                            previous[j - 1] + cost,
                        )
                    )
                previous = current
            return previous[-1]

        def similarity_score_for(distance: int, a: str, b: str) -> float:
            max_len = max(len(a), len(b))
            if max_len == 0:
                return 1.0
            return max(0.0, 1.0 - (distance / max_len))

        conn = None
        cursor = None
        try:
            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            # Build query with optional filters
            sql = f"""
                SELECT entity_id, entity_name, entity_type, source_doc_id, description, confidence
                FROM {self.entities_table}
                WHERE 1=1
            """
            params = []

            entity_type_filter = (
                {str(t).upper() for t in entity_types} if entity_types else None
            )

            # Add entity type filter if provided
            if entity_type_filter:
                placeholders = ", ".join("?" * len(entity_type_filter))
                sql += f" AND UPPER(entity_type) IN ({placeholders})"
                params.extend(sorted(entity_type_filter))

            # Add confidence filter
            if min_confidence > 0.0:
                sql += " AND confidence >= ?"
                params.append(min_confidence)

            logger.debug(
                f"Searching entities with query: {normalized_query} (fuzzy={fuzzy}, edit_distance={edit_distance_threshold}, similarity_threshold={similarity_threshold}, types: {entity_types}, min_confidence: {min_confidence}, max_results: {max_results})"
            )
            try:
                cursor.execute(sql, params)
                rows = list(cursor.fetchall())
            except Exception as exc:
                logger.warning(
                    f"Search query failed, retrying after schema ensure: {exc}"
                )
                self._ensure_entity_columns()
                try:
                    cursor.execute(sql, params)
                    rows = list(cursor.fetchall())
                except Exception as retry_exc:
                    logger.error(
                        f"Search query failed after retry, using cache only: {retry_exc}"
                    )
                    rows = []

            if self._entity_cache:
                existing_ids = {row[0] for row in rows}
                for cached in self._entity_cache.values():
                    cached_id = str(cached.id)
                    if cached_id in existing_ids:
                        continue
                    if (
                        entity_type_filter
                        and str(cached.entity_type).upper() not in entity_type_filter
                    ):
                        continue
                    if float(cached.confidence) < min_confidence:
                        continue
                    rows.append(
                        (
                            cached_id,
                            cached.text,
                            cached.entity_type,
                            cached.source_document_id,
                            (
                                cached.metadata.get("description")
                                if cached.metadata
                                else None
                            ),
                            cached.confidence,
                        )
                    )

            # Convert rows to dictionaries (for hipporag2 compatibility)
            entities = []
            for row in rows:
                (
                    entity_id,
                    entity_name,
                    entity_type,
                    source_doc_id,
                    description,
                    confidence,
                ) = row
                name = str(entity_name)
                name_lower = name.lower()
                distance = levenshtein_distance(lower_query, name_lower)
                similarity = similarity_score_for(distance, lower_query, name_lower)

                def tokenize(text: str) -> List[str]:
                    return re.findall(r"\w+", text.lower())

                query_words = tokenize(lower_query)
                name_words = tokenize(name_lower)
                token_subset_match = bool(query_words) and all(
                    token in name_words for token in query_words
                )

                if fuzzy:
                    if similarity_threshold == 1.0:
                        matches = distance == 0
                    elif edit_distance_threshold is not None:
                        if edit_distance_threshold == 0:
                            matches = distance == 0
                        else:
                            matches = (
                                distance <= edit_distance_threshold
                                or token_subset_match
                            )
                    else:
                        if short_query:
                            matches = name_lower.startswith(lower_query)
                        else:
                            matches = lower_query in name_lower

                    if matches and similarity >= similarity_threshold:
                        entities.append(
                            {
                                "entity_id": entity_id,
                                "entity_name": entity_name,
                                "entity_type": entity_type,
                                "source_doc_id": source_doc_id,
                                "description": description,
                                "confidence": float(confidence) if confidence else 0.0,
                                "similarity_score": similarity,
                                "edit_distance": distance,
                            }
                        )
                else:
                    if name_lower == lower_query:
                        entities.append(
                            {
                                "entity_id": entity_id,
                                "entity_name": entity_name,
                                "entity_type": entity_type,
                                "source_doc_id": source_doc_id,
                                "description": description,
                                "confidence": float(confidence) if confidence else 0.0,
                                "similarity_score": 1.0,
                                "edit_distance": 0,
                            }
                        )

            if fuzzy:
                entities.sort(
                    key=lambda item: (
                        -(1 if item["edit_distance"] == 0 else 0),
                        item["edit_distance"],
                        len(str(item["entity_name"])),
                    )
                )

            if max_results > 0:
                entities = entities[:max_results]

            logger.debug(
                f"Found {len(entities)} entities matching query: {normalized_query}"
            )
            return entities

        except Exception as e:
            logger.error(f"Failed to search entities with query '{query}': {e}")
            return []

        finally:
            try:
                if cursor:
                    cursor.close()
            except Exception:
                pass

    def create_tables_if_not_exist(self) -> bool:
        """Create the entity and relationship tables (in the entities table's schema).

        Returns True when they exist. Raises when they cannot be created.
        """
        self._ensure_kg_tables()
        return True

    def get_storage_stats(self) -> Dict[str, Any]:
        """Counts of stored entities and relationships, in total and by type."""
        entities = self._read(f"SELECT COUNT(*) FROM {self.entities_table}")[0][0]
        relationships = self._read(f"SELECT COUNT(*) FROM {self.relationships_table}")[
            0
        ][0]
        entity_types = self._read(
            f"SELECT entity_type, COUNT(*) FROM {self.entities_table} "
            "GROUP BY entity_type"
        )
        relationship_types = self._read(
            f"SELECT relationship_type, COUNT(*) FROM {self.relationships_table} "
            "GROUP BY relationship_type"
        )
        return {
            "entities_count": int(entities),
            "relationships_count": int(relationships),
            "entity_types": {str(t): int(n) for t, n in entity_types},
            "relationship_types": {str(t): int(n) for t, n in relationship_types},
        }
