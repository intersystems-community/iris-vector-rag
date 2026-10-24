"""Declared columns in a table shape (feature 092, FR-001 to FR-003).

The recorded catalogs are MedTeam's and Štafeta's tables as IRIS reported them
(`tests/fixtures/byot/catalog/`), so a declared role is checked against the columns a
real team table has, not against a hand-written wish.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import (
    ColumnRole,
    ShapeUnresolvable,
    TableShape,
)

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "byot" / "catalog"


@pytest.fixture
def teams() -> RecordedCatalog:
    return RecordedCatalog.from_fixture_dir(CATALOG)


MEDTEAM = {
    ColumnRole.IDENTIFIER: "EvidenceId",
    ColumnRole.TEXT: "Text",
    ColumnRole.EMBEDDING: "Embedding",
    ColumnRole.TITLE: "Title",
    ColumnRole.PARENT: "ResourceId",
}
STAFETA = {
    ColumnRole.IDENTIFIER: ("DocRef", "Seq"),
    ColumnRole.TEXT: "Text",
    ColumnRole.EMBEDDING: "Embedding",
    ColumnRole.PARENT: "DocRef",
}


def test_without_a_declaration_a_team_table_does_not_resolve(teams):
    # The gap spike S3 measured: neither `EvidenceId` nor `Text` is in the name lists.
    with pytest.raises(ShapeUnresolvable, match="IDENTIFIER"):
        TableShape.from_snapshot(teams.snapshot("BYOTMed", "EvidenceChunk"))


def test_declared_columns_fill_their_roles(teams):
    shape = TableShape.from_snapshot(
        teams.snapshot("BYOTMed", "EvidenceChunk"), declared=MEDTEAM
    )
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "EvidenceId"
    assert shape.column_for(ColumnRole.TEXT).name == "Text"
    assert shape.column_for(ColumnRole.EMBEDDING).name == "Embedding"
    assert shape.column_for(ColumnRole.PARENT).name == "ResourceId"
    assert shape.vector_spec().dimension == 384
    assert shape.vector_spec().element_type == "DOUBLE"
    assert shape.attached is True


def test_a_declared_column_is_matched_without_regard_to_case(teams):
    shape = TableShape.from_snapshot(
        teams.snapshot("BYOTMed", "EvidenceChunk"),
        declared={ColumnRole.IDENTIFIER: "evidenceid", ColumnRole.TEXT: "TEXT"},
    )
    # The catalog's spelling is what goes into SQL.
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "EvidenceId"
    assert shape.column_for(ColumnRole.TEXT).name == "Text"


def test_an_undeclared_role_still_resolves_by_name(teams):
    # Štafeta's vector column is called `Embedding`, which the name list knows.
    shape = TableShape.from_snapshot(
        teams.snapshot("BYOTStf", "DocChunk"),
        declared={ColumnRole.IDENTIFIER: ("DocRef", "Seq"), ColumnRole.TEXT: "Text"},
    )
    assert shape.column_for(ColumnRole.EMBEDDING).name == "Embedding"


def test_a_declared_column_the_table_lacks_names_the_table_and_the_column(teams):
    with pytest.raises(ShapeUnresolvable) as caught:
        TableShape.from_snapshot(
            teams.snapshot("BYOTMed", "EvidenceChunk"),
            declared={**MEDTEAM, ColumnRole.TEXT: "Body"},
        )
    message = str(caught.value)
    assert "Body" in message and "BYOTMed" in message and "EvidenceChunk" in message


def test_a_composite_identifier_is_joined_with_a_hash(teams):
    shape = TableShape.from_snapshot(
        teams.snapshot("BYOTStf", "DocChunk"), declared=STAFETA
    )
    assert shape.identifier_columns == ("DocRef", "Seq")
    assert shape.identifier_sql == '"DocRef" || \'#\' || "Seq"'
    # The first column stands for the role where one column is needed.
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "DocRef"


def test_a_single_declared_identifier_is_quoted(teams):
    shape = TableShape.from_snapshot(
        teams.snapshot("BYOTMed", "EvidenceChunk"), declared=MEDTEAM
    )
    assert shape.identifier_columns == ()
    assert shape.identifier_sql == '"EvidenceId"'
    assert shape.sql_name(ColumnRole.TEXT) == '"Text"'


def test_parent_is_never_resolved_by_name(recorded_catalog):
    shape = TableShape.from_snapshot(
        recorded_catalog.snapshot("ShapeTest", "ShapeCanonical")
    )
    assert not shape.has(ColumnRole.PARENT)
    assert shape.optional_column_for(ColumnRole.PARENT) is None


@pytest.mark.parametrize(
    "table",
    [
        "ShapeCanonical",
        "ShapeLegacyId",
        "ShapeLegacySerial",
        "ShapeFixture",
        "ShapeMixed",
        "ShapeDoubleVector",
        "ShapeNoMetadata",
        "ShapeNoTitle",
    ],
)
def test_the_recorded_shapes_are_unchanged_without_a_declaration(
    recorded_catalog, table
):
    shape = TableShape.from_snapshot(recorded_catalog.snapshot("ShapeTest", table))
    assert shape.attached is False
    assert shape.identifier_columns == ()
    identifier = shape.column_for(ColumnRole.IDENTIFIER).name
    # An own table's SQL keeps the bare names it has always used.
    assert shape.identifier_sql == identifier
    assert shape.sql_name(ColumnRole.TEXT) == shape.column_for(ColumnRole.TEXT).name


def test_the_resolver_keys_its_cache_on_the_declaration(teams):
    resolver = TableShapeResolver(teams)
    med = resolver.resolve("BYOTMed", "EvidenceChunk", declared=MEDTEAM)
    again = resolver.resolve("BYOTMed", "EvidenceChunk", declared=dict(MEDTEAM))
    other = resolver.resolve(
        "BYOTMed",
        "EvidenceChunk",
        declared={**MEDTEAM, ColumnRole.TEXT: "Department"},
    )
    assert again is med
    assert other.column_for(ColumnRole.TEXT).name == "Department"
    assert resolver.reads == 2
