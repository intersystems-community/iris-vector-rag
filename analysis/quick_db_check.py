#!/usr/bin/env python3
"""Quick database schema and evaluation results check."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.iris_dbapi_connector import get_iris_dbapi_connection


def check_database_schema():
    """Check what tables exist in RAG schema."""
    conn = get_iris_dbapi_connection()
    cursor = conn.cursor()

    print("=== RAG DATABASE SCHEMA ===")
    cursor.execute(
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'RAG'"
    )
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
            count = cursor.fetchone()[0]
            print(f"✅ RAG.{table_name}: {count} rows")
        except Exception as e:
            print(f"❌ RAG.{table_name}: Error - {e}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    check_database_schema()
