"""Shared iris-devtester cleanup helpers for tests."""

from __future__ import annotations

import importlib
import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

RAG_DELETE_ORDER = [
    "DocumentChunks",
    "EntityRelationships",
    "Entities",
    "DocumentTokenEmbeddings",
    "SourceDocuments",
]


def reset_rag_schema(connection, schema: str = "RAG", strict: bool = False) -> None:
    """Reset a RAG schema using iris-devtester helpers.

    Prefers the dedicated rag_reset helper if available, otherwise falls back
    to truncate_schema with the recommended ordering.
    """
    try:
        module = importlib.import_module("iris_devtester.testing.rag_reset")
        reset_func = getattr(module, "reset_rag_schema")
        reset_func(connection)
        return
    except Exception as exc:
        logger.debug("idt reset_rag_schema unavailable: %s", exc)

    try:
        module = importlib.import_module("iris_devtester.testing.schema_reset")
        resetter_cls = getattr(module, "SchemaResetter")
        resetter = resetter_cls(connection)
        resetter.reset_rag_schema(
            schema=schema,
            order=RAG_DELETE_ORDER,
            include_system=False,
            strict=strict,
        )
        return
    except Exception as exc:
        if strict:
            raise
        logger.warning("idt schema reset failed: %s", exc)


def truncate_tables_schema(
    connection,
    tables: Iterable[str],
    strict: bool = False,
) -> Optional[str]:
    """Infer schema from table list and truncate via iris-devtester.

    Returns the schema name when truncated, otherwise None.
    """
    table_list = [t for t in tables if t]
    if not table_list:
        return None

    schemas = {t.split(".", 1)[0] for t in table_list if "." in t}
    if not schemas:
        return None

    schema = sorted(schemas)[0]
    reset_rag_schema(connection, schema=schema, strict=strict)
    return schema
