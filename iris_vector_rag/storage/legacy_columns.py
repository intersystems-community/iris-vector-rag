"""Bring an older entity table up to the columns the writers need.

The entity writers always INSERT a `confidence` column. Tables created by earlier
releases do not have one, and every write then failed with SQLCODE -29. The repair is
additive: ask the catalog which columns exist, and `ALTER TABLE ... ADD` the one that is
missing. Nothing is dropped, rewritten or recreated, so existing rows survive; their
`confidence` stays NULL (IRIS applies a column DEFAULT to new rows only).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .catalog import CatalogReader, IRISCatalogReader

logger = logging.getLogger(__name__)

CONFIDENCE_DDL = "ADD confidence DOUBLE DEFAULT 1.0"


def _split(qualified: str) -> tuple[str, str]:
    schema, _, table = qualified.rpartition(".")
    return (schema or "RAG"), table


def ensure_confidence_column(
    connection: Any,
    qualified_table: str,
    reader: Optional[CatalogReader] = None,
) -> bool:
    """Add `confidence DOUBLE DEFAULT 1.0` to ``qualified_table`` if it has none.

    Returns True when the column was added. A table that does not exist, or already has
    the column, is left alone and gives False. A failed ALTER rolls back and raises:
    writing would only fail again, one row at a time.
    """
    schema, table = _split(qualified_table)
    snapshot = (reader or IRISCatalogReader(connection)).snapshot(schema, table)
    if snapshot is None:
        return False
    present = {str(c["COLUMN_NAME"]).lower() for c in snapshot.columns}
    if "confidence" in present:
        return False

    cursor = connection.cursor()
    try:
        cursor.execute(f"ALTER TABLE {schema}.{table} {CONFIDENCE_DDL}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
    logger.warning(
        "%s.%s had no confidence column (created by an older release); added "
        "confidence DOUBLE DEFAULT 1.0. Existing rows are untouched (confidence NULL).",
        schema,
        table,
    )
    return True
