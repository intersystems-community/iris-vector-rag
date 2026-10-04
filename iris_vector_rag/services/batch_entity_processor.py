"""
Batch Entity Storage Processor for GraphRAG Performance Optimization (Feature 057).

Replaces serial entity storage with batched operations using IRIS DBAPI executemany()
to achieve 30-64 second performance improvement per ticket (80-92% reduction in storage time).

Performance Impact:
- Serial: 40-70 seconds for 10 entities (4-7s per entity × 10)
- Batch: 6-10 seconds for 10 entities (single transaction with executemany)
- Speedup: 5-10x faster storage operations

Key Optimizations:
1. Single transaction boundary per ticket (commit once, not per entity)
2. IRIS DBAPI cursor.executemany() for batch INSERT operations
3. Foreign key validation before storage (prevents orphaned relationships)
4. A chunk that fails is rolled back and retried in smaller halves, down to
   single rows, so one bad row costs one row and is reported in ``failed``
5. ``batch_size`` chunks the work; each chunk is its own transaction

Entity storage is therefore not all-or-nothing across a call: chunks that
succeeded stay committed. The result says exactly which entities were not stored.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from ..core.connection import ConnectionManager
from ..core.models import Entity, Relationship
from ..storage.legacy_columns import ensure_confidence_column
from ..storage.table_shape import CANONICAL_VECTOR_TYPE

logger = logging.getLogger(__name__)


def to_vector_expression(dimension: int) -> str:
    """`TO_VECTOR(?, <type>, <dimension>)` for an embedding of this width.

    The width is the embedding's own: IRIS rejects a value of another width than the
    column, loudly, which a constant here would not survive.
    """
    return f"TO_VECTOR(?, {CANONICAL_VECTOR_TYPE}, {int(dimension)})"


class BatchEntityProcessor:
    """
    Batch processor for entity and relationship storage with performance optimization.

    Uses IRIS DBAPI executemany() to reduce transaction overhead from O(n) to O(1),
    achieving 5-10x speedup for typical GraphRAG workloads.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config: Dict[str, Any],
        batch_size: int = 32,
        source_document_resolver: Optional[Callable[[Any, str], str]] = None,
    ):
        """
        Initialize batch entity processor.

        Args:
            connection_manager: IRIS database connection manager
            config: Configuration dict with table names
            batch_size: Maximum entities per transaction (default: 32); a larger
                call is split into chunks of this size
            source_document_resolver: Optional ``(cursor, raw_id) -> stored_id``
                mapping an entity's source document onto a document-table row, as
                ``EntityStorageAdapter.store_entity`` does
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.connection_manager = connection_manager
        self.source_document_resolver = source_document_resolver
        self.config = config
        self.batch_size = batch_size
        self._confidence_ensured: set = set()

        # Get table names from config
        storage_config = config.get("entity_extraction", {}).get("storage", {})
        self.entities_table = storage_config.get("entities_table", "RAG.Entities")
        self.relationships_table = storage_config.get(
            "relationships_table", "RAG.EntityRelationships"
        )

        logger.info(
            f"BatchEntityProcessor initialized (batch_size={batch_size}, "
            f"tables: {self.entities_table}, {self.relationships_table})"
        )

    def _ensure_confidence(self, conn: Any, table: str) -> None:
        """Add `confidence` to a table from an older release, once per table."""
        if table not in self._confidence_ensured:
            ensure_confidence_column(conn, table)
            self._confidence_ensured.add(table)

    def store_entities_batch(
        self, entities: List[Entity], validate_count: bool = True
    ) -> Dict[str, Any]:
        """
        Store entities in chunks of ``batch_size``, inserting new ids and updating
        existing ones (the same upsert as ``EntityStorageAdapter.store_entity``).

        Every row carries ``confidence``. An entity id repeated in the call keeps its
        last occurrence. A chunk that fails is rolled back and retried in halves;
        rows that still fail alone are reported, not raised.

        Args:
            entities: List of entities to store
            validate_count: Re-count the stored ids in the table afterwards and
                compare with the number of distinct ids in the call

        Returns:
            Dict with:
                - entities_stored: Number of distinct entities written
                - validation_passed: True when every entity was stored (and, if
                  requested, the table holds the expected count)
                - storage_time_ms: Time taken for storage operation
                - failed: ``[{"entity_id", "reason"}]`` for entities not stored
                - error: Message when anything was not stored

        Example:
            >>> processor = BatchEntityProcessor(conn_mgr, config)
            >>> result = processor.store_entities_batch(entities)
            >>> assert result['entities_stored'] == len(entities)
            >>> assert result['validation_passed'] is True
        """
        if not entities:
            return {
                "entities_stored": 0,
                "validation_passed": True,
                "storage_time_ms": 0,
                "failed": [],
            }

        import time

        start_time = time.time()

        conn = None
        cursor = None
        stored_ids: List[str] = []
        failed: List[Dict[str, str]] = []

        try:
            conn = self.connection_manager.get_connection()
            self._ensure_confidence(conn, self.entities_table)
            cursor = conn.cursor()

            distinct: Dict[str, Entity] = {str(e.id): e for e in entities}
            records = [self._entity_record(cursor, e) for e in distinct.values()]

            for offset in range(0, len(records), self.batch_size):
                self._store_chunk(
                    conn,
                    cursor,
                    records[offset : offset + self.batch_size],
                    stored_ids,
                    failed,
                )

            errors: List[str] = []
            if failed:
                errors.append(
                    f"{len(failed)} of {len(records)} entities not stored: "
                    + ", ".join(
                        f"{f['entity_id']} ({f['reason']})" for f in failed[:10]
                    )
                )
            if validate_count:
                in_table = self._count_in_table(cursor, [r["id"] for r in records])
                if in_table != len(records):
                    errors.append(
                        f"Entity count mismatch: expected {len(records)} in "
                        f"{self.entities_table}, found {in_table}"
                    )

            storage_time_ms = (time.time() - start_time) * 1000
            result: Dict[str, Any] = {
                "entities_stored": len(stored_ids),
                "validation_passed": not errors,
                "storage_time_ms": storage_time_ms,
                "failed": failed,
            }
            if errors:
                result["error"] = "; ".join(errors)
                logger.error(f"Batch entity storage incomplete: {result['error']}")
            else:
                logger.info(
                    f"Batch stored {len(stored_ids)} entities in {storage_time_ms:.1f}ms"
                )
            return result

        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            storage_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Batch entity storage failed: {e}")
            return {
                "entities_stored": len(stored_ids),
                "validation_passed": False,
                "storage_time_ms": storage_time_ms,
                "failed": failed,
                "error": str(e),
            }

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass

    def _entity_record(self, cursor, entity: Entity) -> Dict[str, Any]:
        """The fields of one entity row, with its optional embedding normalised."""
        entity_type = (
            entity.entity_type.name
            if hasattr(entity.entity_type, "name")
            else str(entity.entity_type).split(".")[-1].strip()
        )
        source_document = str(entity.source_document_id).strip()
        if self.source_document_resolver is not None:
            source_document = self.source_document_resolver(cursor, source_document)
        description = None
        embedding = None
        if isinstance(entity.metadata, dict):
            description = entity.metadata.get("description")
            embedding = entity.metadata.get("embedding")
        # An empty embedding is no embedding; its width would be 0.
        has_embedding = embedding is not None and len(embedding) > 0
        return {
            "id": str(entity.id),
            "name": str(entity.text).strip(),
            "type": entity_type,
            "doc": source_document,
            "description": description,
            "confidence": float(entity.confidence),
            "embedding": json.dumps(list(embedding)) if has_embedding else None,
            "dimension": len(embedding) if has_embedding else 0,
        }

    def _store_chunk(
        self,
        conn,
        cursor,
        chunk: List[Dict[str, Any]],
        stored_ids: List[str],
        failed: List[Dict[str, str]],
    ) -> None:
        """Write one chunk in one transaction; on failure retry it in halves."""
        try:
            existing = self._existing_ids(cursor, [r["id"] for r in chunk])
            for is_update in (False, True):
                part = [r for r in chunk if (r["id"] in existing) == is_update]
                for dimension in sorted({r["dimension"] for r in part}):
                    rows = [r for r in part if r["dimension"] == dimension]
                    sql, params = self._statement(is_update, dimension, rows)
                    cursor.executemany(sql, params)
            conn.commit()
            stored_ids.extend(r["id"] for r in chunk)
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            if len(chunk) == 1:
                reason = str(exc)
                if chunk[0]["dimension"]:
                    reason += (
                        f" (the row carries a {chunk[0]['dimension']}-wide embedding; "
                        f"{self.entities_table} needs a matching embedding column)"
                    )
                failed.append({"entity_id": chunk[0]["id"], "reason": reason})
                return
            logger.warning(
                f"Chunk of {len(chunk)} entities failed ({exc}); retrying in halves"
            )
            middle = len(chunk) // 2
            self._store_chunk(conn, cursor, chunk[:middle], stored_ids, failed)
            self._store_chunk(conn, cursor, chunk[middle:], stored_ids, failed)

    def _statement(self, is_update: bool, dimension: int, rows: List[Dict[str, Any]]):
        """One INSERT or UPDATE and its parameter rows, all of one width."""
        with_vector = dimension > 0
        vector = to_vector_expression(dimension)
        if is_update:
            sets = (
                "entity_name = ?, entity_type = ?, source_doc_id = ?, "
                "description = ?, confidence = ?"
            )
            if with_vector:
                sets += f", embedding = {vector}"
            sql = f"UPDATE {self.entities_table} SET {sets} WHERE entity_id = ?"
            params = [
                (r["name"], r["type"], r["doc"], r["description"], r["confidence"])
                + ((r["embedding"],) if with_vector else ())
                + (r["id"],)
                for r in rows
            ]
        else:
            columns = (
                "entity_id, entity_name, entity_type, source_doc_id, "
                "description, confidence"
            )
            values = "?, ?, ?, ?, ?, ?"
            if with_vector:
                columns += ", embedding"
                values += f", {vector}"
            sql = f"INSERT INTO {self.entities_table} ({columns}) VALUES ({values})"
            params = [
                (
                    r["id"],
                    r["name"],
                    r["type"],
                    r["doc"],
                    r["description"],
                    r["confidence"],
                )
                + ((r["embedding"],) if with_vector else ())
                for r in rows
            ]
        return sql, params

    def _existing_ids(self, cursor, ids: List[str]) -> set:
        marks = ",".join("?" * len(ids))
        cursor.execute(
            f"SELECT entity_id FROM {self.entities_table} WHERE entity_id IN ({marks})",
            ids,
        )
        return {str(row[0]) for row in cursor.fetchall()}

    def _count_in_table(self, cursor, ids: List[str]) -> int:
        total = 0
        for offset in range(0, len(ids), self.batch_size):
            chunk = ids[offset : offset + self.batch_size]
            marks = ",".join("?" * len(chunk))
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.entities_table} "
                f"WHERE entity_id IN ({marks})",
                chunk,
            )
            total += int(cursor.fetchone()[0])
        return total

    def store_relationships_batch(
        self,
        relationships: List[Relationship],
        validate_foreign_keys: bool = True,
    ) -> Dict[str, Any]:
        """
        Store multiple relationships in a single batch operation using executemany().

        Performance: 5-10x faster than serial storage for 5+ relationships.

        Args:
            relationships: List of relationships to store
            validate_foreign_keys: Whether to check entity IDs exist before storage

        Returns:
            Dict with:
                - relationships_stored: Number successfully stored
                - orphaned_relationships: Count of invalid foreign keys (0 if validation passed)
                - validation_passed: Whether foreign key validation succeeded
                - storage_time_ms: Time taken for storage operation
                - error: Error message if failed

        Example:
            >>> result = processor.store_relationships_batch(relationships)
            >>> assert result['relationships_stored'] == len(relationships)
            >>> assert result['orphaned_relationships'] == 0
        """
        if not relationships:
            return {
                "relationships_stored": 0,
                "orphaned_relationships": 0,
                "validation_passed": True,
                "storage_time_ms": 0,
            }

        import time

        start_time = time.time()

        conn = None
        cursor = None
        relationships_stored = 0

        try:
            conn = self.connection_manager.get_connection()
            self._ensure_confidence(conn, self.relationships_table)
            cursor = conn.cursor()

            # Validate foreign keys if requested (prevents orphaned relationships)
            orphaned_count = 0
            if validate_foreign_keys:
                entity_ids = set()
                for rel in relationships:
                    entity_ids.add(str(rel.source_entity_id))
                    entity_ids.add(str(rel.target_entity_id))

                # Check which entity IDs exist in database
                placeholders = ",".join(["?"] * len(entity_ids))
                check_sql = f"SELECT entity_id FROM {self.entities_table} WHERE entity_id IN ({placeholders})"
                cursor.execute(check_sql, list(entity_ids))

                existing_ids = {str(row[0]) for row in cursor.fetchall()}
                missing_ids = entity_ids - existing_ids

                if missing_ids:
                    orphaned_count = len(
                        [
                            r
                            for r in relationships
                            if str(r.source_entity_id) in missing_ids
                            or str(r.target_entity_id) in missing_ids
                        ]
                    )

                    # Enhanced logging to diagnose missing entities
                    missing_ids_list = sorted(list(missing_ids))[:10]  # Show first 10
                    logger.error(
                        f"Foreign key validation failed: {len(missing_ids)} missing entity IDs, "
                        f"{orphaned_count} orphaned relationships"
                    )
                    logger.error(
                        f"Sample missing entity IDs (first 10): {missing_ids_list}"
                    )
                    logger.info(
                        f"Total entity IDs referenced: {len(entity_ids)}, "
                        f"Found in database: {len(existing_ids)}, "
                        f"Missing: {len(missing_ids)}"
                    )

                    # Filter out relationships with missing entity IDs
                    relationships = [
                        r
                        for r in relationships
                        if str(r.source_entity_id) not in missing_ids
                        and str(r.target_entity_id) not in missing_ids
                    ]

            # Prepare batch data for executemany()
            batch_data = []

            for rel in relationships:
                relationship_id = str(rel.id)
                source_entity_id = str(rel.source_entity_id)
                target_entity_id = str(rel.target_entity_id)
                relationship_type = str(rel.relationship_type).strip()
                weight = float(rel.metadata.get("weight", 1.0)) if rel.metadata else 1.0
                confidence = float(rel.confidence)
                source_document = (
                    str(rel.source_document_id) if rel.source_document_id else None
                )

                batch_data.append(
                    (
                        relationship_id,
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        weight,
                        confidence,
                        source_document,
                    )
                )

            # Execute batch INSERT using executemany()
            if batch_data:
                insert_sql = f"""
                    INSERT INTO {self.relationships_table}
                    (relationship_id, source_entity_id, target_entity_id, relationship_type,
                     weight, confidence, source_document)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """

                # Execute batch operation (single database round-trip)
                cursor.executemany(insert_sql, batch_data)
                relationships_stored = len(batch_data)

                logger.debug(
                    f"Batch INSERT: {relationships_stored} relationships in single operation"
                )

            # Commit transaction (single commit for all relationships)
            conn.commit()

            storage_time_ms = (time.time() - start_time) * 1000

            validation_passed = orphaned_count == 0

            logger.info(
                f"Batch stored {relationships_stored} relationships in {storage_time_ms:.1f}ms "
                f"(orphaned: {orphaned_count})"
            )

            return {
                "relationships_stored": relationships_stored,
                "orphaned_relationships": orphaned_count,
                "validation_passed": validation_passed,
                "storage_time_ms": storage_time_ms,
            }

        except Exception as e:
            # Rollback transaction on error (ensures atomicity)
            if conn:
                try:
                    conn.rollback()
                    logger.warning("Transaction rolled back due to error")
                except Exception:
                    pass

            storage_time_ms = (time.time() - start_time) * 1000

            logger.error(f"Batch relationship storage failed: {e}")
            return {
                "relationships_stored": 0,
                "orphaned_relationships": 0,
                "validation_passed": False,
                "storage_time_ms": storage_time_ms,
                "error": str(e),
            }

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass

    def store_entities_and_relationships(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Store entities, then relationships.

        This is not atomic: entities commit (chunk by chunk) before relationships
        are attempted, and a relationship failure does not undo the entities. The
        result reports what was stored. If any entity was not stored, relationships
        are not attempted and the entity error is returned.

        Args:
            entities: List of entities to store
            relationships: List of relationships to store
            validate: Whether to validate entity count and foreign keys

        Returns:
            Dict with combined storage results and validation status

        Example:
            >>> result = processor.store_entities_and_relationships(
            ...     entities, relationships, validate=True
            ... )
            >>> assert result['entities_stored'] == len(entities)
            >>> assert result['relationships_stored'] == len(relationships)
            >>> assert result['validation_passed'] is True
        """
        import time

        start_time = time.time()

        # Store entities first (relationships depend on entities)
        entity_result = self.store_entities_batch(entities, validate_count=validate)

        if not entity_result.get("validation_passed", False):
            return {
                "entities_stored": entity_result.get("entities_stored", 0),
                "relationships_stored": 0,
                "validation_passed": False,
                "total_time_ms": (time.time() - start_time) * 1000,
                "failed": entity_result.get("failed", []),
                "error": entity_result.get("error", "Entity storage failed"),
            }

        # Store relationships (with foreign key validation)
        relationship_result = self.store_relationships_batch(
            relationships, validate_foreign_keys=validate
        )

        total_time_ms = (time.time() - start_time) * 1000

        validation_passed = entity_result.get(
            "validation_passed", False
        ) and relationship_result.get("validation_passed", False)

        return {
            "entities_stored": entity_result.get("entities_stored", 0),
            "relationships_stored": relationship_result.get("relationships_stored", 0),
            "orphaned_relationships": relationship_result.get(
                "orphaned_relationships", 0
            ),
            "validation_passed": validation_passed,
            "total_time_ms": total_time_ms,
            "entity_storage_time_ms": entity_result.get("storage_time_ms", 0),
            "relationship_storage_time_ms": relationship_result.get(
                "storage_time_ms", 0
            ),
        }
