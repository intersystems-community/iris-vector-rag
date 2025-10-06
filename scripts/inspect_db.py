#!/usr/bin/env python3
"""
IRIS RAG DB inspector: prints counts and sample rows for SourceDocuments, Entities, EntityRelationships.
"""
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_connection():
    try:
        from common.iris_dbapi_connector import get_iris_dbapi_connection

        conn = get_iris_dbapi_connection()
        if conn is None:
            raise RuntimeError("get_iris_dbapi_connection returned None")
        return conn
    except Exception as e:
        logger.error(f"Failed to get IRIS connection: {e}")
        return None


def fetch_int(cursor, sql):
    try:
        cursor.execute(sql)
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception as e:
        logger.warning(f"Query failed: {sql} -> {e}")
        return None


def table_exists(cursor, table_name):
    try:
        cursor.execute(f"SELECT 1 FROM {table_name} WHERE 1=0")
        return True
    except Exception:
        return False


def get_samples(cursor, table_name, cols, limit=5):
    try:
        col_list = ", ".join(cols)
        cursor.execute(f"SELECT TOP {limit} {col_list} FROM {table_name}")
        rows = cursor.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning(f"Sample fetch failed for {table_name}: {e}")
        return []


def main():
    conn = get_connection()
    if not conn:
        print(json.dumps({"success": False, "error": "connection_failed"}))
        return 1
    cur = conn.cursor()
    report = {"timestamp": datetime.utcnow().isoformat() + "Z"}

    # SourceDocuments
    sd_table = "RAG.SourceDocuments"
    sd_exists = table_exists(cur, sd_table)
    report["SourceDocuments"] = {"exists": sd_exists}
    if sd_exists:
        report["SourceDocuments"]["count_total"] = fetch_int(
            cur, f"SELECT COUNT(*) FROM {sd_table}"
        )
        report["SourceDocuments"]["count_with_embedding"] = fetch_int(
            cur, f"SELECT COUNT(*) FROM {sd_table} WHERE embedding IS NOT NULL"
        )
        report["SourceDocuments"]["count_chunks"] = fetch_int(
            cur, f"SELECT COUNT(*) FROM {sd_table} WHERE doc_id LIKE '%_chunk_%'"
        )
        report["SourceDocuments"]["samples"] = get_samples(
            cur, sd_table, ["doc_id", "title"], limit=5
        )

    # Entities
    ent_table = "RAG.Entities"
    ent_exists = table_exists(cur, ent_table)
    report["Entities"] = {"exists": ent_exists}
    if ent_exists:
        report["Entities"]["count_total"] = fetch_int(
            cur, f"SELECT COUNT(*) FROM {ent_table}"
        )
        report["Entities"]["samples"] = get_samples(
            cur, ent_table, ["entity_id", "entity_name", "entity_type"], limit=5
        )

    # EntityRelationships
    rel_table = "RAG.EntityRelationships"
    rel_exists = table_exists(cur, rel_table)
    report["EntityRelationships"] = {"exists": rel_exists}
    if rel_exists:
        report["EntityRelationships"]["count_total"] = fetch_int(
            cur, f"SELECT COUNT(*) FROM {rel_table}"
        )
        report["EntityRelationships"]["samples"] = get_samples(
            cur, rel_table, ["relationship_id", "relationship_type"], limit=5
        )

    try:
        cur.close()
        conn.close()
    except Exception:
        pass

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
