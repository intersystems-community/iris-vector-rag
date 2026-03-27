"""
ColBERT in-database schema for IRIS.

Tables:
  RAG.ColBERTDocuments        — document chunks to return
  RAG.DocumentTokenEmbeddings — one row per (doc, token_position)
  RAG.ColBERTCentroids        — PLAID-phase centroids (optional)
  RAG.ColBERTDocCentroids     — doc → centroid mapping (optional)

Index strategy:
  HNSW on tok_vec with DotProduct distance (ColBERT vectors are L2-normalised
  so dot product == cosine similarity, and IRIS HNSW supports DotProduct natively).

  Only created after bulk-ingest to avoid per-row maintenance overhead.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

TOKEN_DIM = 128


class ColBERTSchema:
    def __init__(self, conn):
        self._conn = conn

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def create_tables(self, token_dim: int = TOKEN_DIM) -> None:
        cur = self._conn.cursor()
        try:
            for ddl in [
                f"""
                CREATE TABLE RAG.ColBERTDocuments (
                    doc_id      VARCHAR(64)        NOT NULL,
                    parent_id   VARCHAR(64),
                    chunk_index INTEGER,
                    text_content LONGVARCHAR,
                    metadata    LONGVARCHAR,
                    PRIMARY KEY (doc_id)
                )
                """,
                f"""
                CREATE TABLE RAG.DocumentTokenEmbeddings (
                    doc_id   VARCHAR(64)                  NOT NULL,
                    tok_pos  INTEGER                      NOT NULL,
                    tok_text VARCHAR(64),
                    tok_vec  VECTOR(FLOAT, {token_dim})   NOT NULL,
                    centroid_id INTEGER,
                    PRIMARY KEY (doc_id, tok_pos)
                )
                """,
            ]:
                try:
                    cur.execute(ddl)
                except Exception as e:
                    if "already exists" in str(e).lower() or "-201" in str(e):
                        pass
                    else:
                        raise
            self._conn.commit()
            logger.info("ColBERT tables created (or already exist)")
        finally:
            cur.close()

    def create_hnsw_index(
        self,
        m: int = 16,
        ef_construction: int = 200,
        token_dim: int = TOKEN_DIM,
    ) -> None:
        """Create HNSW index on token vectors.

        Must be called AFTER bulk-ingest (not before) to avoid per-row
        write amplification during load.
        """
        cur = self._conn.cursor()
        try:
            cur.execute("DROP INDEX IF EXISTS idx_tok_vec_hnsw")
        except Exception:
            pass
        try:
            cur.execute(
                f"""
                CREATE INDEX idx_tok_vec_hnsw
                ON TABLE RAG.DocumentTokenEmbeddings (tok_vec)
                AS HNSW(Distance='DotProduct', M={m}, efConstruction={ef_construction})
                """
            )
            self._conn.commit()
            logger.info(f"HNSW index created (M={m}, efConstruction={ef_construction})")
        except Exception as e:
            logger.warning(f"HNSW index creation failed (may need licensed IRIS): {e}")
        finally:
            cur.close()

    def create_doc_index(self) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tok_doc_id "
                "ON RAG.DocumentTokenEmbeddings (doc_id)"
            )
            self._conn.commit()
        except Exception as e:
            logger.debug(f"Doc-id index: {e}")
        finally:
            cur.close()

    def drop_tables(self) -> None:
        cur = self._conn.cursor()
        for tbl in [
            "RAG.DocumentTokenEmbeddings",
            "RAG.ColBERTDocCentroids",
            "RAG.ColBERTCentroids",
            "RAG.ColBERTDocuments",
        ]:
            try:
                cur.execute(f"DROP TABLE {tbl}")
            except Exception:
                pass
        try:
            self._conn.commit()
        except Exception:
            pass
        cur.close()

    # ------------------------------------------------------------------
    # Inspection helpers used by tests / benchmarks
    # ------------------------------------------------------------------

    def table_exists(self, table_name: str) -> bool:
        schema, table = table_name.split(".")
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
                [schema.upper(), table.upper()],
            )
            return cur.fetchone()[0] > 0
        finally:
            cur.close()

    def index_exists(self, index_name: str) -> bool:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES "
                "WHERE INDEX_NAME = ?",
                [index_name.upper()],
            )
            return cur.fetchone()[0] > 0
        except Exception:
            return False
        finally:
            cur.close()

    def row_count(self, table_name: str) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cur.fetchone()[0]
        finally:
            cur.close()

    def explain(self, sql: str, params: Optional[list] = None) -> str:
        cur = self._conn.cursor()
        try:
            explain_sql = f"EXPLAIN {sql}"
            if params:
                cur.execute(explain_sql, params)
            else:
                cur.execute(explain_sql)
            rows = cur.fetchall()
            return "\n".join(str(r) for r in rows)
        finally:
            cur.close()
