"""A real IRIS instance still finds a table named in another case (feature 082).

The unit suite asserts the SQL folds case; this asserts IRIS agrees, because the
whole point of feature 074's `UPPER()` comparison was what a live instance does with
a name that differs from the DDL's spelling. Without this, the unit test could keep
passing against a driver or a view that behaves differently.

Container: `iris-vector-rag-iris` on 51972, per CLAUDE.md. Never another one.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.schema_manager import SchemaManager

pytestmark = pytest.mark.integration

SCHEMA = "RAG"
TABLE = "CaseProbeDocuments"


@pytest.fixture(scope="module")
def probe_table():
    """One table, declared in mixed case, dropped again."""
    from iris_vector_rag.common.iris_connection import get_iris_connection

    conn = get_iris_connection()
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE} CASCADE")
    cursor.execute(
        f"CREATE TABLE {SCHEMA}.{TABLE} "
        "(doc_id VARCHAR(255) PRIMARY KEY, text_content VARCHAR(2000))"
    )
    conn.commit()
    cursor.close()
    try:
        yield conn
    finally:
        cursor = conn.cursor()
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE} CASCADE")
            conn.commit()
        except Exception:  # noqa: BLE001 - teardown is advisory
            pass
        cursor.close()
        conn.close()


@pytest.mark.parametrize(
    "schema,table",
    [
        (SCHEMA, TABLE),
        (SCHEMA.lower(), TABLE.lower()),
        (SCHEMA.upper(), TABLE.upper()),
    ],
)
def test_the_catalog_reader_finds_the_table_however_it_is_spelled(
    probe_table, schema, table
):
    snapshot = IRISCatalogReader(probe_table).snapshot(schema, table)

    assert snapshot is not None, f"{schema}.{table} should resolve to {SCHEMA}.{TABLE}"
    assert {str(row["COLUMN_NAME"]).lower() for row in snapshot.columns} == {
        "doc_id",
        "text_content",
    }


def test_the_snapshot_reports_the_declared_spelling(probe_table):
    """What IRIS calls it, not what the caller asked for."""
    snapshot = IRISCatalogReader(probe_table).snapshot(SCHEMA.lower(), TABLE.lower())

    assert snapshot is not None
    assert (snapshot.schema.upper(), snapshot.table.upper()) == (
        SCHEMA.upper(),
        TABLE.upper(),
    )


def test_a_missing_table_is_still_missing(probe_table):
    assert IRISCatalogReader(probe_table).snapshot(SCHEMA, "NoSuchTableAtAll") is None


@pytest.mark.parametrize(
    "name", [f"{SCHEMA}.{TABLE}", f"{SCHEMA.lower()}.{TABLE.lower()}"]
)
def test_schema_manager_table_exists_ignores_case(probe_table, name):
    config = ConfigurationManager()
    manager = SchemaManager(ConnectionManager(config), config)

    assert manager.table_exists(name) is True
