"""Unit tests for the table-shape value objects (feature 082, T004-T008).

These tests need no database. `TableShape.from_snapshot` is a pure function of
recorded catalog rows, which is the whole point of the seam: a test states the shape
it wants and asserts on behaviour, instead of scripting cursor responses in the
order the implementation happens to issue its queries.

The recorded snapshots under `tests/fixtures/shapes/` were captured from real DDL
against IRIS 2026.1 (`scripts/dev/capture_shape_fixtures.py`), so the awkward facts
are present rather than idealised away: a VECTOR column reports `varchar`, a
`VARCHAR(MAX)` column reports `longvarchar`, and property names have their
underscores stripped.
"""

from __future__ import annotations

import json

import pytest

from iris_vector_rag.storage.catalog import CatalogSnapshot, RecordedCatalog
from iris_vector_rag.storage.table_shape import (
    CANONICAL_CHUNK_SHAPE,
    CANONICAL_DOCUMENT_SHAPE,
    ColumnFact,
    ColumnRole,
    IdentifierNotWritable,
    NonCanonicalCreation,
    NotVectorSearchable,
    RoleNotPresent,
    ShapeKind,
    ShapeUnresolvable,
    TableShape,
    VectorSpec,
    create_table_statement,
)
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES, create_statement

# The Parameters string exactly as %Dictionary.CompiledProperty reports it for a
# VECTOR(FLOAT, 384) column.
REAL_PARAMETERS = (
    "CALCSELECTIVITY,1,CAPTION,,DATATYPE,FLOAT,EXTERNALSQLNAME,,EXTERNALSQLTYPE,,"
    "INDEXNULLMARKER,,JAVATYPE,,LEN,384,STORAGEDEFAULT,GLOBALNODE,STORAGEDEFAULTTARGET,"
)


@pytest.fixture
def shapes(recorded_catalog: RecordedCatalog):
    """Every recorded shape, resolved, keyed by kind name."""
    return {
        kind: TableShape.from_snapshot(recorded_catalog.snapshot(SHAPE_SCHEMA, table))
        for kind, table in SHAPE_TABLES.items()
    }


def column(name, sql_type, class_type, **kwargs) -> ColumnFact:
    defaults = dict(
        max_length=None,
        nullable=True,
        position=1,
        is_identity=False,
        is_key=False,
        vector=None,
    )
    defaults.update(kwargs)
    return ColumnFact(name=name, sql_type=sql_type, class_type=class_type, **defaults)


# --- T004: ColumnFact derived properties ------------------------------------


def test_stream_column_is_a_stream():
    col = column(
        "chunk_text", "longvarchar", "%Stream.GlobalCharacter", max_length=2147483647
    )
    assert col.is_stream is True
    assert col.is_long_text is True
    assert col.filterable_with_like is False


def test_long_string_is_not_a_stream():
    """`content` and `chunk_text` are both longvarchar; only one is a stream.

    Length alone cannot tell them apart, which is why the class type decides.
    """
    col = column("content", "longvarchar", "%Library.String", max_length=2147483647)
    assert col.is_stream is False
    assert col.is_long_text is True
    assert col.filterable_with_like is False


def test_short_varchar_is_filterable_with_like():
    col = column("metadata", "varchar", "%Library.String", max_length=2000)
    assert col.is_stream is False
    assert col.is_long_text is False
    assert col.filterable_with_like is True


def test_binary_column_is_not_a_stream_and_not_a_vector():
    col = column("embedding", "varbinary", "%Library.Binary")
    assert col.is_stream is False
    assert col.vector is None


# --- T005: VectorSpec parsing -----------------------------------------------


def test_vector_spec_parses_real_parameters():
    spec = VectorSpec.from_parameters(REAL_PARAMETERS)
    assert spec == VectorSpec(element_type="FLOAT", dimension=384)


def test_vector_spec_parses_double_precision():
    params = REAL_PARAMETERS.replace("DATATYPE,FLOAT", "DATATYPE,DOUBLE")
    assert VectorSpec.from_parameters(params).element_type == "DOUBLE"


def test_vector_spec_preserves_empty_values():
    """Empty values are real: `CAPTION,,` is CAPTION set to the empty string.

    Dropping them would shift every subsequent key-value pair by one.
    """
    params = "CAPTION,,DATATYPE,FLOAT,JAVATYPE,,LEN,768"
    assert VectorSpec.from_parameters(params) == VectorSpec("FLOAT", 768)


@pytest.mark.parametrize(
    "params",
    [
        "CALCSELECTIVITY,1,LEN,384",  # no DATATYPE
        "CALCSELECTIVITY,1,DATATYPE,FLOAT",  # no LEN
        "",
    ],
)
def test_vector_spec_refuses_to_guess(params):
    with pytest.raises(ShapeUnresolvable):
        VectorSpec.from_parameters(params)


def test_vector_spec_never_reads_the_environment(monkeypatch):
    """FR-001: the removed IRIS_VECTOR_DATA_TYPE fallback stays removed."""
    monkeypatch.setenv("IRIS_VECTOR_DATA_TYPE", "DECIMAL")
    with pytest.raises(ShapeUnresolvable):
        VectorSpec.from_parameters("LEN,384")


# --- T006: role resolution --------------------------------------------------


def test_canonical_shape_binds_every_role(shapes):
    shape = shapes["canonical"]
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "doc_id"
    assert shape.column_for(ColumnRole.TEXT).name == "text_content"
    assert shape.column_for(ColumnRole.TITLE).name == "title"
    assert shape.column_for(ColumnRole.METADATA).name == "metadata"
    assert shape.column_for(ColumnRole.EMBEDDING).name == "embedding"
    assert shape.column_for(ColumnRole.CREATED_AT).name == "created_at"
    assert shape.is_canonical is True


def test_canonical_text_column_wins_on_a_mixed_table(shapes):
    """FR-003. This is the defect: the writer used text_content, the reader content."""
    shape = shapes["mixed"]
    assert shape.column_for(ColumnRole.TEXT).name == "text_content"
    assert shape.legacy_column_for(ColumnRole.TEXT).name == "content"


def test_no_legacy_column_when_the_table_holds_only_one_family(shapes):
    assert shapes["canonical"].legacy_column_for(ColumnRole.TEXT) is None
    assert shapes["legacy_id"].legacy_column_for(ColumnRole.TEXT) is None


def test_legacy_shapes_bind_their_own_columns(shapes):
    assert shapes["legacy_id"].column_for(ColumnRole.IDENTIFIER).name == "id"
    assert shapes["legacy_id"].column_for(ColumnRole.TEXT).name == "content"
    assert shapes["legacy_id"].column_for(ColumnRole.TITLE).name == "source"
    assert shapes["legacy_serial"].column_for(ColumnRole.TITLE).name == "filename"
    assert shapes["fixture"].column_for(ColumnRole.IDENTIFIER).name == "doc_id"
    assert shapes["fixture"].column_for(ColumnRole.TEXT).name == "content"


def test_declared_key_outranks_name_precedence(shapes):
    """The mixed table has both doc_id and id; doc_id is the declared key."""
    assert shapes["mixed"].column_for(ColumnRole.IDENTIFIER).name == "doc_id"
    assert shapes["mixed"].column_for(ColumnRole.IDENTIFIER).is_key is True


def test_key_outranks_name_even_when_the_name_is_not_canonical():
    """A table whose key is `record_ref` uses it, regardless of an `id` column."""
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="Odd",
        class_name="ShapeTest.Odd",
        columns=[
            _col_row("record_ref", "varchar", 1, length=255, nullable="NO"),
            _col_row("id", "varchar", 2, length=255),
            _col_row("text_content", "longvarchar", 3, length=2147483647),
        ],
        properties=[
            _prop_row("record_ref", "%Library.String"),
            _prop_row("id", "%Library.String"),
            _prop_row("text_content", "%Library.String"),
        ],
        key_columns=["record_ref"],
    )
    shape = TableShape.from_snapshot(snapshot)
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "record_ref"


def test_name_precedence_applies_without_a_declared_key():
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="NoKey",
        class_name="ShapeTest.NoKey",
        columns=[
            _col_row("id", "varchar", 1, length=255),
            _col_row("doc_id", "varchar", 2, length=255),
            _col_row("content", "longvarchar", 3, length=2147483647),
        ],
        properties=[
            _prop_row("id", "%Library.String"),
            _prop_row("doc_id", "%Library.String"),
            _prop_row("content", "%Library.String"),
        ],
        key_columns=[],
    )
    shape = TableShape.from_snapshot(snapshot)
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "doc_id"


def test_lookup_is_case_insensitive_but_spelling_is_preserved():
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="Shouty",
        class_name="ShapeTest.Shouty",
        columns=[
            _col_row("DOC_ID", "varchar", 1, length=255, nullable="NO"),
            _col_row("Text_Content", "longvarchar", 2, length=2147483647),
        ],
        properties=[
            _prop_row("DOC_ID", "%Library.String"),
            _prop_row("Text_Content", "%Library.String"),
        ],
        key_columns=["DOC_ID"],
    )
    shape = TableShape.from_snapshot(snapshot)
    assert shape.column_for(ColumnRole.IDENTIFIER).name == "DOC_ID"
    assert shape.column_for(ColumnRole.TEXT).name == "Text_Content"
    assert "doc_id" in shape.columns


def test_vector_spec_comes_from_the_table(shapes):
    assert shapes["canonical"].vector_spec() == VectorSpec("FLOAT", 384)
    assert shapes["double_vector"].vector_spec() == VectorSpec("DOUBLE", 384)


def test_information_schema_reporting_varchar_does_not_hide_a_vector(shapes):
    """The standing gotcha, asserted: the SQL view says varchar, the shape says vector."""
    col = shapes["canonical"].column_for(ColumnRole.EMBEDDING)
    assert col.sql_type == "varchar"
    assert col.class_type == "%Library.Vector"
    assert col.vector == VectorSpec("FLOAT", 384)


# --- T007: errors -----------------------------------------------------------


def test_absent_embedding_is_not_searchable(shapes):
    shape = shapes["legacy_id"]
    with pytest.raises(NotVectorSearchable) as exc:
        shape.vector_spec()
    message = str(exc.value)
    assert "ShapeTest" in message and "ShapeLegacyId" in message


def test_non_vector_embedding_names_the_type_found():
    """The live RAG.DocumentChunks has a %Library.Binary embedding column."""
    snapshot = CatalogSnapshot(
        schema="RAG",
        table="DocumentChunks",
        class_name="RAG.DocumentChunks",
        columns=[
            _col_row("id", "varchar", 1, length=255, nullable="NO"),
            _col_row("chunk_text", "longvarchar", 2, length=2147483647),
            _col_row("embedding", "varbinary", 3),
        ],
        properties=[
            _prop_row("id", "%Library.String"),
            _prop_row("chunk_text", "%Stream.GlobalCharacter"),
            _prop_row("embedding", "%Library.Binary"),
        ],
        key_columns=["id"],
    )
    shape = TableShape.from_snapshot(snapshot)
    with pytest.raises(NotVectorSearchable) as exc:
        shape.vector_spec()
    assert "%Library.Binary" in str(exc.value)


def test_identity_identifier_refuses_a_caller_supplied_value(shapes):
    shape = shapes["legacy_serial"]
    assert shape.column_for(ColumnRole.IDENTIFIER).is_identity is True
    with pytest.raises(IdentifierNotWritable) as exc:
        shape.writable_identifier()
    message = str(exc.value)
    assert "ShapeLegacySerial" in message and "id" in message


def test_writable_identifier_is_returned_when_it_is_writable(shapes):
    assert shapes["canonical"].writable_identifier().name == "doc_id"


def test_missing_role_names_the_role(shapes):
    with pytest.raises(RoleNotPresent) as exc:
        shapes["fixture"].column_for(ColumnRole.CREATED_AT)
    assert "CREATED_AT" in str(exc.value)
    assert shapes["fixture"].optional_column_for(ColumnRole.CREATED_AT) is None
    assert shapes["fixture"].has(ColumnRole.CREATED_AT) is False


def test_every_error_names_schema_and_table(shapes):
    errors = []
    for shape, call in (
        (shapes["legacy_id"], lambda s: s.vector_spec()),
        (shapes["legacy_serial"], lambda s: s.writable_identifier()),
        (shapes["fixture"], lambda s: s.column_for(ColumnRole.CREATED_AT)),
    ):
        with pytest.raises(Exception) as exc:  # noqa: PT011 - family is the point
            call(shape)
        errors.append((shape, str(exc.value)))
    for shape, message in errors:
        assert shape.schema in message, message
        assert shape.table in message, message


def test_a_table_with_no_text_column_is_unresolvable():
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="Bare",
        class_name="ShapeTest.Bare",
        columns=[_col_row("doc_id", "varchar", 1, length=255, nullable="NO")],
        properties=[_prop_row("doc_id", "%Library.String")],
        key_columns=["doc_id"],
    )
    with pytest.raises(ShapeUnresolvable) as exc:
        TableShape.from_snapshot(snapshot)
    assert "TEXT" in str(exc.value)


def test_a_table_with_no_columns_is_unresolvable():
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="Empty",
        class_name="ShapeTest.Empty",
        columns=[],
        properties=[],
        key_columns=[],
    )
    with pytest.raises(ShapeUnresolvable):
        TableShape.from_snapshot(snapshot)


def test_a_role_column_with_no_class_view_is_unresolvable():
    """A column the class view does not describe cannot be typed, so it is an error."""
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="Half",
        class_name="ShapeTest.Half",
        columns=[
            _col_row("doc_id", "varchar", 1, length=255, nullable="NO"),
            _col_row("text_content", "longvarchar", 2, length=2147483647),
        ],
        properties=[_prop_row("doc_id", "%Library.String")],
        key_columns=["doc_id"],
    )
    with pytest.raises(ShapeUnresolvable) as exc:
        TableShape.from_snapshot(snapshot)
    assert "text_content" in str(exc.value)


# --- T008: purity and classification ----------------------------------------


def test_from_snapshot_is_pure(recorded_catalog):
    snapshot = recorded_catalog.snapshot(SHAPE_SCHEMA, SHAPE_TABLES["canonical"])
    assert TableShape.from_snapshot(snapshot) == TableShape.from_snapshot(snapshot)


def test_shapes_are_frozen(shapes):
    with pytest.raises(
        Exception
    ):  # noqa: PT011 - dataclasses raise FrozenInstanceError
        shapes["canonical"].table = "other"


@pytest.mark.parametrize(
    "kind_name,expected",
    [
        ("canonical", ShapeKind.CANONICAL),
        ("legacy_serial", ShapeKind.LEGACY_SERIAL),
        ("legacy_id", ShapeKind.LEGACY_ID),
        ("fixture", ShapeKind.FIXTURE),
        ("mixed", ShapeKind.MIXED),
        ("double_vector", ShapeKind.DOUBLE_VECTOR),
    ],
)
def test_recorded_shapes_classify(shapes, kind_name, expected):
    assert shapes[kind_name].kind is expected


def test_only_the_canonical_shape_is_canonical(shapes):
    canonical = [name for name, shape in shapes.items() if shape.is_canonical]
    assert canonical == ["canonical"]


def test_an_unrecognised_but_resolvable_shape_is_unknown_not_an_error():
    """FR-002 names six shapes; a seventh must still work."""
    snapshot = CatalogSnapshot(
        schema="ShapeTest",
        table="Seventh",
        class_name="ShapeTest.Seventh",
        columns=[
            _col_row("doc_id", "varchar", 1, length=255, nullable="NO"),
            _col_row("abstract", "longvarchar", 2, length=2147483647),
        ],
        properties=[
            _prop_row("doc_id", "%Library.String"),
            _prop_row("abstract", "%Library.String"),
        ],
        key_columns=["doc_id"],
    )
    shape = TableShape.from_snapshot(snapshot)
    assert shape.kind is ShapeKind.UNKNOWN
    assert shape.column_for(ColumnRole.TEXT).name == "abstract"


def test_columns_are_ordered_by_position(shapes):
    positions = [col.position for col in shapes["mixed"].columns.values()]
    assert positions == sorted(positions)


def test_recorded_fixtures_are_loadable_json(shape_fixtures):
    """Guards against a hand-edited fixture: every file parses and has all sections."""
    for path in sorted(shape_fixtures.glob("*.json")):
        data = json.loads(path.read_text())
        assert set(data) >= {
            "schema",
            "table",
            "class_name",
            "columns",
            "properties",
            "key_columns",
        }
        assert data["columns"], path


# --- row builders (shaped exactly like the catalog's own rows) ---------------


def _col_row(name, data_type, position, length=None, nullable="YES", identity="NO"):
    return {
        "COLUMN_NAME": name,
        "DATA_TYPE": data_type,
        "CHARACTER_MAXIMUM_LENGTH": length,
        "IS_NULLABLE": nullable,
        "ORDINAL_POSITION": position,
        "IS_IDENTITY": identity,
        "AUTO_INCREMENT": identity,
    }


def _prop_row(sql_field_name, class_type, parameters=""):
    return {
        "SqlFieldName": sql_field_name,
        "Type": class_type,
        "Parameters": parameters,
    }


# --- US3: the DDL a creation path emits is rendered from the canonical shape ---
#
# There is one statement per canonical table and it is derived, not typed: a literal
# in the schema manager is how the two shapes of RAG.SourceDocuments came to exist in
# the first place. These tests pin the rendered text exactly, because that text is
# what a fresh instance ends up with.

CANONICAL_DOCUMENT_DDL = """CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(1000),
    abstract VARCHAR(MAX),
    text_content VARCHAR(MAX),
    authors VARCHAR(MAX),
    keywords VARCHAR(MAX),
    embedding VECTOR(FLOAT, 384),
    metadata VARCHAR(2000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""

CANONICAL_CHUNK_DDL = """CREATE TABLE RAG.DocumentChunks (
    doc_id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255),
    chunk_text VARCHAR(MAX),
    chunk_embedding VECTOR(FLOAT, 384),
    metadata VARCHAR(2000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


def test_canonical_document_ddl_is_rendered_from_the_shape():
    """The document table's statement comes from CANONICAL_DOCUMENT_SHAPE (FR-006)."""
    assert create_table_statement(CANONICAL_DOCUMENT_SHAPE) == CANONICAL_DOCUMENT_DDL


def test_canonical_chunk_ddl_is_rendered_from_the_shape():
    """The chunk table's statement comes from CANONICAL_CHUNK_SHAPE (FR-006)."""
    assert create_table_statement(CANONICAL_CHUNK_SHAPE) == CANONICAL_CHUNK_DDL


def test_canonical_ddl_is_rendered_for_the_schema_and_table_asked_for():
    """A caller with `IRIS_SCHEMA_PREFIX` set gets its own schema, same shape."""
    statement = create_table_statement(
        CANONICAL_DOCUMENT_SHAPE, schema="Other", table="Documents"
    )
    assert statement.startswith("CREATE TABLE Other.Documents (")
    assert statement.splitlines()[1:] == CANONICAL_DOCUMENT_DDL.splitlines()[1:]


def test_canonical_ddl_carries_the_vector_type_and_dimension_it_is_given():
    """A configured 768-dimension DOUBLE table is still the canonical shape.

    The element type and dimension are the two facts a deployment legitimately
    varies; every column name and every other type stays derived.
    """
    statement = create_table_statement(
        CANONICAL_DOCUMENT_SHAPE, vector=VectorSpec("DOUBLE", 768)
    )
    assert "    embedding VECTOR(DOUBLE, 768),\n" in statement
    assert statement.count("VECTOR(") == 1


def test_canonical_ddl_matches_the_recorded_canonical_fixture():
    """The renderer and the fixture family describe the same table.

    `tests/fixtures/shapes/ddl.py` is what the recorded snapshots were captured from,
    so if these two ever disagree the fixtures stop describing what the library
    creates -- which is the drift the recorded fixtures exist to catch.
    """
    rendered = create_table_statement(
        CANONICAL_DOCUMENT_SHAPE, schema=SHAPE_SCHEMA, table=SHAPE_TABLES["canonical"]
    )
    fixture = create_statement("canonical", SHAPE_SCHEMA)
    assert _normalised(rendered) == _normalised(fixture)


def test_a_non_canonical_shape_is_refused_for_creation():
    """Nothing may create a second shape of the document table (FR-006, FR-007)."""
    snapshot = CatalogSnapshot(
        schema="RAG",
        table="Legacy",
        class_name="RAG.Legacy",
        columns=[
            _col_row("id", "varchar", 1, length=255, nullable="NO"),
            _col_row("content", "varchar", 2, length=2000),
        ],
        properties=[
            _prop_row("id", "%Library.String"),
            _prop_row("content", "%Library.String"),
        ],
        key_columns=["id"],
    )
    legacy = TableShape.from_snapshot(snapshot)
    assert legacy.kind is not ShapeKind.CANONICAL

    with pytest.raises(NonCanonicalCreation) as raised:
        create_table_statement(legacy)

    assert "RAG.Legacy" in str(raised.value)


def _normalised(statement: str) -> str:
    return " ".join(statement.split())
