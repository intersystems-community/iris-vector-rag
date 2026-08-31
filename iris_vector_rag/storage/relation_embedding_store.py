"""Relation embedding store for KG edge embeddings (Feature 081)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from iris_vector_rag.common.db_vector_utils import insert_vector  # noqa: E402

logger = logging.getLogger(__name__)

_ALTER_SQL = (
    "ALTER TABLE RAG.EntityRelationships ADD relation_embedding VECTOR(FLOAT, 384) NULL"
)
_INDEX_SQL = (
    "CREATE INDEX idx_hnsw_rel_embedding ON RAG.EntityRelationships (relation_embedding)"
    " AS HNSW(M=16, efConstruction=200, Distance='COSINE')"
)
_COUNT_SQL = (
    "SELECT COUNT(*) FROM RAG.EntityRelationships WHERE relation_embedding IS NOT NULL"
)
_UPDATE_EMBEDDING_SQL = (
    "UPDATE RAG.EntityRelationships "
    "SET relation_embedding = TO_VECTOR(?, FLOAT, 384) "
    "WHERE relationship_id = ?"
)

# IRIS SQLCODE values for "already exists" conditions
_ALREADY_EXISTS_CODES = {-306, -201, -324}  # -324 = index already defined


class RelationEmbeddingStore:
    """Manages relation embeddings in RAG.EntityRelationships.

    Extends the existing KG table with a VECTOR(FLOAT, 384) column and HNSW
    index so that relationship descriptions can be retrieved by semantic
    similarity at query time (Feature 081 global/mix modes).
    """

    def __init__(self, connection_manager: Any, config_manager: Any) -> None:
        self._conn_mgr = connection_manager
        self._cfg_mgr = config_manager
        self._schema_ensured = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Idempotently add relation_embedding column and HNSW index."""
        conn = self._conn_mgr.get_connection("iris")
        cursor = conn.cursor()
        try:
            try:
                cursor.execute(_ALTER_SQL)
                conn.commit()
                logger.debug(
                    "Added relation_embedding column to RAG.EntityRelationships"
                )
            except Exception as exc:
                code = getattr(exc, "errorCode", None) or getattr(exc, "sqlcode", None)
                if (
                    code in _ALREADY_EXISTS_CODES
                    or "already exists" in str(exc).lower()
                ):
                    logger.debug(
                        "relation_embedding column already exists — skipping ALTER"
                    )
                    conn.rollback()
                else:
                    conn.rollback()
                    raise

            try:
                cursor.execute(_INDEX_SQL)
                conn.commit()
                logger.debug("Created HNSW index idx_hnsw_rel_embedding")
            except Exception as exc:
                code = getattr(exc, "errorCode", None) or getattr(exc, "sqlcode", None)
                msg = str(exc).lower()
                if (
                    code in _ALREADY_EXISTS_CODES
                    or "already defined" in msg
                    or "already exists" in msg
                    or "duplicate" in msg
                ):
                    logger.debug(
                        "HNSW index idx_hnsw_rel_embedding already exists — skipping"
                    )
                    conn.rollback()
                else:
                    conn.rollback()
                    raise
        finally:
            cursor.close()
        self._schema_ensured = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cursor(self):
        conn = self._conn_mgr.get_connection("iris")
        return conn.cursor()

    def _get_embedding_manager(self):
        from iris_vector_rag.embeddings.manager import EmbeddingManager

        return EmbeddingManager(self._cfg_mgr)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def embed_and_store(
        self,
        relationship_id: str,
        relationship_type: str,
        source_entity: str,
        target_entity: str,
        description: str = "",
    ) -> None:
        """Embed a relationship description and upsert it into IRIS.

        The embedding text follows the LightRAG format:
            "{type}: {source} → {target}. {description}"
        """
        parts = [f"{relationship_type}: {source_entity} → {target_entity}"]
        if description and description.strip():
            parts.append(description.strip())
        text = ". ".join(parts)

        emb_mgr = self._get_embedding_manager()
        vec: List[float] = emb_mgr.embed_text(text)
        embedding_str = "[" + ",".join(map(str, vec)) + "]"

        conn = self._conn_mgr.get_connection("iris")
        cursor = conn.cursor()
        try:
            # UPDATE is preferred: relationship rows are always written before embedding.
            # insert_vector builds a bare INSERT that fails on NOT NULL columns (SQLCODE -108)
            # when the row already exists and IRIS doesn't raise a UNIQUE error.
            cursor.execute(_UPDATE_EMBEDDING_SQL, [embedding_str, relationship_id])
            conn.commit()
            logger.debug("Embedded relationship_id=%s", relationship_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbour search over relation embeddings.

        Returns list of dicts with keys: relationship_id, source_entity_id,
        target_entity_id, relationship_type, score.
        """
        from iris_vector_rag.common.db_vector_utils import vector_similarity_search

        conn = self._conn_mgr.get_connection("iris")
        cursor = conn.cursor()
        try:
            # vector_similarity_search does not accept a where_clause param; NULL rows are
            # naturally excluded because VECTOR_COSINE returns NULL for NULL embeddings
            # and the function filters by score ranking.
            results = vector_similarity_search(
                cursor=cursor,
                table_name="RAG.EntityRelationships",
                vector_column="relation_embedding",
                query_vector=query_embedding,
                top_k=top_k,
                id_column="relationship_id",
                return_columns=[
                    "source_entity_id",
                    "target_entity_id",
                    "relationship_type",
                ],
                metric="COSINE",
                dtype="FLOAT",
            )
        finally:
            cursor.close()

        if not results:
            return []
        # IRIS returns scores as strings or None; normalize to float
        for row in results:
            if "score" in row:
                raw = row["score"]
                if raw is None:
                    row["score"] = 0.0
                elif not isinstance(raw, float):
                    try:
                        row["score"] = float(raw)
                    except (TypeError, ValueError):
                        row["score"] = 0.0
        return results

    def count_embedded(self) -> int:
        """Return count of rows with a non-NULL relation_embedding."""
        cursor = self._get_cursor()
        try:
            cursor.execute(_COUNT_SQL)
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            cursor.close()
