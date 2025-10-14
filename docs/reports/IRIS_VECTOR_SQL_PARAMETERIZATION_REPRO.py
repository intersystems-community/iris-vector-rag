# docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.py
import os
import sys

sys.path.insert(0, ".")
import random
from typing import List

from common.db_vector_utils import insert_vector
from common.iris_dbapi_connector import get_iris_dbapi_connection
from common.utils import get_embedding_func
from common.vector_sql_utils import (
    execute_vector_search_with_params,
    format_vector_search_sql,
    format_vector_search_sql_with_params,
)

TABLE = "USER.ReproVectors"  # isolated table for reproducibility
VECTOR_COL = "embedding"
DIM = 384


def ensure_table(cursor):
    cursor.execute(
        """
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'USER' AND TABLE_NAME = 'ReproVectors'
    """
    )
    exists = cursor.fetchone()[0] > 0
    if not exists:
        # Create minimal repro table
        cursor.execute(
            f"""
            CREATE TABLE {TABLE} (
              doc_id VARCHAR(64) PRIMARY KEY,
              title VARCHAR(200),
              {VECTOR_COL} VECTOR(FLOAT, {DIM})
            )
        """
        )
        cursor.connection.commit()


def upsert_two_rows(cursor, embed_func):
    texts = ["Repro doc A about medical trials", "Repro doc B about clinical research"]
    for i, text in enumerate(texts, start=1):
        emb: List[float] = embed_func(text)  # returns 384-dim
        ok = insert_vector(
            cursor=cursor,
            table_name=TABLE,
            vector_column_name=VECTOR_COL,
            vector_data=emb,
            target_dimension=DIM,
            key_columns={"doc_id": f"repro_{i}"},
            additional_data={"title": f"Repro Title {i}"},
        )
        if not ok:
            raise RuntimeError("insert_vector failed")
    cursor.connection.commit()


def print_rows(cursor, note):
    cursor.execute(f"SELECT doc_id, title FROM {TABLE}")
    rows = cursor.fetchall()
    print(f"{note} — rows in {TABLE}: {len(rows)}")
    for r in rows:
        print("  ", r)


def main():
    conn = get_iris_dbapi_connection()
    cur = conn.cursor()

    # 1) Ensure table + sample rows
    ensure_table(cur)
    embed = get_embedding_func()
    upsert_two_rows(cur, embed)
    print_rows(cur, "After upsert")

    # 2) Working pattern — single bound parameter (no explicit type/dimension binds)
    print(
        "\n=== WORKING: Single bound vector parameter (VECTOR_DOT_PRODUCT with TO_VECTOR(?)) ==="
    )
    # Build a query embedding and convert to comma-separated string (no brackets)
    qvec = embed("medical research document")
    qvec_str = ",".join(map(str, qvec))

    sql_working = f"""
        SELECT TOP 2 doc_id, title, VECTOR_DOT_PRODUCT({VECTOR_COL}, TO_VECTOR(?)) AS similarity
        FROM {TABLE}
        WHERE {VECTOR_COL} IS NOT NULL
        ORDER BY similarity DESC
    """
    try:
        cur.execute(sql_working, (qvec_str,))
        rows = cur.fetchall()
        print("WORKING results:", rows)
    except Exception as e:
        print("UNEXPECTED failure in working pattern:", e)

    # 3) Failing pattern A — parameterized vector + explicit type/dimension
    print(
        "\n=== FAILING A: format_vector_search_sql_with_params (IRIS rewrites TOP/type/dim into :%qpar) ==="
    )
    try:
        sql_params = format_vector_search_sql_with_params(
            table_name=TABLE,
            vector_column=VECTOR_COL,
            embedding_dim=DIM,
            top_k=2,
            id_column="doc_id",
            content_column="title",
        )
        # vector_string with brackets for TO_VECTOR text literal
        qvec_bracketed = "[" + ",".join(map(str, qvec)) + "]"
        # execute with parameter (as designed by utility)
        cur.execute(sql_params, [qvec_bracketed])
        rows = cur.fetchall()
        print("SHOULD HAVE FAILED but returned:", rows)
    except Exception as e:
        print("EXPECTED failure A:", e)

    # 4) Failing pattern B — fully literalized vector string embedded in SQL text
    print(
        "\n=== FAILING B: format_vector_search_sql (no DBAPI params, still IRIS injects :%qpar in cached class) ==="
    )
    try:
        qvec_bracketed = "[" + ",".join(map(str, qvec)) + "]"
        sql_literal = format_vector_search_sql(
            table_name=TABLE,
            vector_column=VECTOR_COL,
            vector_string=qvec_bracketed,
            embedding_dim=DIM,
            top_k=2,
            id_column="doc_id",
            content_column="title",
        )
        # Execute with NO DBAPI params
        cur.execute(sql_literal)
        rows = cur.fetchall()
        print("SHOULD HAVE FAILED but returned:", rows)
    except Exception as e:
        print("EXPECTED failure B:", e)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
