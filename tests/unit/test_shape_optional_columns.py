"""Two recorded shapes that leave an optional role empty (feature 082).

The six shapes recorded for US1 all carry a `metadata` column and all carry
something that fills the TITLE role (`title`, `filename` or `source`). So the two
cases the code claims to handle -- a table with no metadata column and a table with
no title -- could only be described by taking a recorded snapshot and deleting a
column from it, which asserts against the test's own idea of a table rather than
against one IRIS ever described. `no_metadata` and `no_title` are captured from real
DDL like every other shape, so what the catalog reports for a genuinely absent
column is what these tests see.

Both are the `fixture` shape minus one optional column, which is why they classify
as FIXTURE: `ShapeKind` describes the identifier and text columns, and an absent
optional column does not make a new kind.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import (
    ColumnRole,
    RoleNotPresent,
    ShapeKind,
    TableShape,
)
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES

NO_METADATA = SHAPE_TABLES["no_metadata"]
NO_TITLE = SHAPE_TABLES["no_title"]


@pytest.fixture
def no_metadata(recorded_catalog: RecordedCatalog) -> TableShape:
    return TableShape.from_snapshot(
        recorded_catalog.snapshot(SHAPE_SCHEMA, NO_METADATA)
    )


@pytest.fixture
def no_title(recorded_catalog: RecordedCatalog) -> TableShape:
    return TableShape.from_snapshot(recorded_catalog.snapshot(SHAPE_SCHEMA, NO_TITLE))


# --- the fixtures describe what they claim to describe -----------------------


def test_the_recorded_table_really_has_no_metadata_column(no_metadata):
    """Guards the fixture itself: a recapture must not quietly add the column back."""
    assert "metadata" not in no_metadata.columns
    assert sorted(no_metadata.columns) == ["content", "doc_id", "embedding", "title"]


def test_the_recorded_table_really_has_no_title_column(no_title):
    assert sorted(no_title.columns) == ["content", "doc_id", "embedding", "metadata"]


# --- an absent optional role ------------------------------------------------


def test_a_missing_metadata_column_is_absent_not_guessed(no_metadata):
    assert no_metadata.has(ColumnRole.METADATA) is False
    assert no_metadata.optional_column_for(ColumnRole.METADATA) is None


def test_asking_for_metadata_that_is_not_there_names_the_table(no_metadata):
    with pytest.raises(RoleNotPresent) as raised:
        no_metadata.column_for(ColumnRole.METADATA)

    message = str(raised.value)
    assert f"{SHAPE_SCHEMA}.{NO_METADATA}" in message, message
    assert "METADATA" in message, message


def test_a_missing_title_column_is_absent_not_guessed(no_title):
    """`content` is a TEXT candidate and must not be pressed into the TITLE role."""
    assert no_title.has(ColumnRole.TITLE) is False
    assert no_title.optional_column_for(ColumnRole.TITLE) is None


def test_asking_for_a_title_that_is_not_there_lists_the_candidates(no_title):
    with pytest.raises(RoleNotPresent) as raised:
        no_title.column_for(ColumnRole.TITLE)

    message = str(raised.value)
    assert "title, source, filename" in message, message


# --- the roles that are there are unaffected --------------------------------


@pytest.mark.parametrize("table", [NO_METADATA, NO_TITLE])
def test_the_required_roles_still_resolve(recorded_catalog, table):
    shape = TableShape.from_snapshot(recorded_catalog.snapshot(SHAPE_SCHEMA, table))

    assert shape.column_for(ColumnRole.IDENTIFIER).name == "doc_id"
    assert shape.column_for(ColumnRole.TEXT).name == "content"
    assert shape.writable_identifier().name == "doc_id"


@pytest.mark.parametrize("table", [NO_METADATA, NO_TITLE])
def test_the_table_is_still_vector_searchable(recorded_catalog, table):
    """The absent column is not the embedding, so the vector spec is intact."""
    shape = TableShape.from_snapshot(recorded_catalog.snapshot(SHAPE_SCHEMA, table))

    spec = shape.vector_spec()
    assert (spec.element_type, spec.dimension) == ("FLOAT", 384)


@pytest.mark.parametrize("table", [NO_METADATA, NO_TITLE])
def test_a_shape_missing_an_optional_column_is_not_canonical(recorded_catalog, table):
    shape = TableShape.from_snapshot(recorded_catalog.snapshot(SHAPE_SCHEMA, table))

    assert shape.is_canonical is False
    assert shape.kind is ShapeKind.FIXTURE


@pytest.mark.parametrize("table", [NO_METADATA, NO_TITLE])
def test_no_legacy_column_is_invented_for_the_absent_role(recorded_catalog, table):
    """A role with no bound column has no legacy spelling either."""
    shape = TableShape.from_snapshot(recorded_catalog.snapshot(SHAPE_SCHEMA, table))

    for role in ColumnRole:
        if not shape.has(role):
            assert shape.legacy_column_for(role) is None, role


# --- through the resolver, the way a call site gets there -------------------


@pytest.mark.parametrize(
    ("table", "absent"),
    [(NO_METADATA, ColumnRole.METADATA), (NO_TITLE, ColumnRole.TITLE)],
)
def test_the_resolver_reports_the_absent_role(recorded_catalog, table, absent):
    resolver = TableShapeResolver(recorded_catalog)

    shape = resolver.resolve(SHAPE_SCHEMA, table)

    assert shape.has(absent) is False
    assert shape.qualified_name == f"{SHAPE_SCHEMA}.{table}"
    # Cached like any other shape: an absent role is a fact about the table, not a
    # reason to go back to the catalog.
    assert resolver.resolve(SHAPE_SCHEMA, table) is shape
