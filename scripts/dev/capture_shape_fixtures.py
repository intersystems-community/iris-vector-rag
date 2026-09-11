#!/usr/bin/env python3
"""Capture recorded catalog snapshots for every declared table shape.

Feature 082. Creates each shape in `tests/fixtures/shapes/ddl.py` for real against
`iris-vector-rag-iris`, dumps what the catalog says about it, and drops the tables.
The fixtures are recorded rather than hand-written on purpose: hand-written rows
drift from what IRIS actually reports, and would "correct" the two things that make
this feature necessary (VECTOR reported as varchar, property names with the
underscores stripped).

Usage:
    python scripts/dev/capture_shape_fixtures.py [--schema ShapeTest] [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.shapes.ddl import (  # noqa: E402
    SHAPE_DDL,
    SHAPE_TABLES,
    create_statement,
    drop_statement,
)

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "shapes"

COLUMN_SQL = """
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE,
           ORDINAL_POSITION, IS_IDENTITY, AUTO_INCREMENT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
    ORDER BY ORDINAL_POSITION
"""

CLASS_SQL = """
    SELECT ID FROM %Dictionary.CompiledClass
    WHERE SqlSchemaName = ? AND SqlTableName = ?
"""

PROPERTY_SQL = """
    SELECT SqlFieldName, Type, Parameters
    FROM %Dictionary.CompiledProperty
    WHERE parent = ?
    ORDER BY SqlFieldName
"""

KEY_SQL = """
    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
"""


def rows_as_dicts(cursor, sql: str, params: list) -> list[dict]:
    cursor.execute(sql, params)
    names = [d[0] for d in cursor.description]
    # Values are copied out of the DataRow before the cursor moves on: a DataRow is
    # invalidated by the cursor closing.
    return [
        dict(zip(names, [row[i] for i in range(len(names))]))
        for row in cursor.fetchall()
    ]


def capture(cursor, schema: str, table: str) -> dict:
    columns = rows_as_dicts(cursor, COLUMN_SQL, [schema, table])
    cursor.execute(CLASS_SQL, [schema, table])
    class_rows = cursor.fetchall()
    class_name = str(class_rows[0][0]) if class_rows else None
    properties = rows_as_dicts(cursor, PROPERTY_SQL, [class_name]) if class_name else []
    keys = [
        str(r["COLUMN_NAME"]) for r in rows_as_dicts(cursor, KEY_SQL, [schema, table])
    ]
    return {
        "schema": schema,
        "table": table,
        "class_name": class_name,
        "columns": columns,
        "properties": properties,
        "key_columns": keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="ShapeTest", help="scratch SQL schema")
    parser.add_argument("--keep", action="store_true", help="do not drop the tables")
    args = parser.parse_args()

    os.environ.setdefault("IRIS_HOST", "localhost")
    os.environ.setdefault("IRIS_PORT", "51972")

    from iris_vector_rag.common.iris_connection import get_iris_connection

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_iris_connection()
    written = []
    try:
        cursor = conn.cursor()
        for shape in SHAPE_DDL:
            table = SHAPE_TABLES[shape]
            cursor.execute(drop_statement(shape, args.schema))
            cursor.execute(create_statement(shape, args.schema))
            conn.commit()
            snapshot = capture(cursor, args.schema, table)
            path = FIXTURE_DIR / f"{shape}.json"
            path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
            written.append(path)
            print(
                f"{shape:<14} {len(snapshot['columns'])} columns, "
                f"{len(snapshot['properties'])} properties, "
                f"class={snapshot['class_name']}"
            )
    finally:
        if not args.keep:
            cursor = conn.cursor()
            for shape in SHAPE_DDL:
                try:
                    cursor.execute(drop_statement(shape, args.schema))
                except Exception as exc:  # noqa: BLE001 - cleanup is advisory
                    print(f"warning: could not drop {shape}: {exc}", file=sys.stderr)
            conn.commit()
        conn.close()

    print(f"\nwrote {len(written)} fixtures to {FIXTURE_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
