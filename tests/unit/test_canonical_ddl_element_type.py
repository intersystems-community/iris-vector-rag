"""The canonical table's vector element type comes from the shape (082, FR-006/FR-010).

`is_canonical` is true only of a `VECTOR(FLOAT, n)` embedding, so a creation path that
takes the element type from the environment can emit a table the library then refuses
to call canonical. That is two accounts of one fact -- `CANONICAL_DOCUMENT_SHAPE` says
FLOAT, `IRIS_VECTOR_DATA_TYPE` says whatever the last process to set it said -- and it
was reachable: one E2E fixture exports `IRIS_VECTOR_DATA_TYPE=DOUBLE` and never unsets
it, after which every table the library created for itself was non-canonical.

The dimension is a real configuration choice (it follows the embedding model), so it
still comes from the caller. The element type is what canonical means.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.table_shape import (
    CANONICAL_CHUNK_SHAPE,
    CANONICAL_DOCUMENT_SHAPE,
    CANONICAL_VECTOR_TYPE,
)


@pytest.fixture
def manager() -> SchemaManager:
    """A schema manager with nothing but what the DDL rendering needs."""
    instance = SchemaManager.__new__(SchemaManager)
    instance.connection_manager = MagicMock()
    instance.config_manager = MagicMock(spec=ConfigurationManager)
    instance.schema_prefix = "RAG"
    return instance


@pytest.mark.parametrize(
    "shape,table",
    [
        (CANONICAL_DOCUMENT_SHAPE, "SourceDocuments"),
        (CANONICAL_CHUNK_SHAPE, "DocumentChunks"),
    ],
)
def test_the_element_type_is_the_shape_s_own(manager, shape, table, monkeypatch):
    monkeypatch.setenv("IRIS_VECTOR_DATA_TYPE", "DOUBLE")

    sql = manager._canonical_create_sql(shape, table, 384)

    assert f"VECTOR({CANONICAL_VECTOR_TYPE}, 384)" in sql, sql
    assert "DOUBLE" not in sql, sql


def test_the_dimension_still_comes_from_the_caller(manager):
    """A 768-dimension model is a configuration choice, not a second shape."""
    sql = manager._canonical_create_sql(
        CANONICAL_DOCUMENT_SHAPE, "SourceDocuments", 768
    )

    assert f"VECTOR({CANONICAL_VECTOR_TYPE}, 768)" in sql, sql
