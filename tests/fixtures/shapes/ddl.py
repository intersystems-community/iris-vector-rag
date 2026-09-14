"""DDL for the recorded document-table shapes (feature 082).

One definition, two consumers: `scripts/dev/capture_shape_fixtures.py` records the
catalog's description of each of these tables, and
`tests/integration/test_table_shape_six_shapes.py` creates them for real. Keeping
one source means a recorded fixture cannot drift from the shape it claims to
describe.

The first six are the shapes observed in production and test environments, named in
`specs/082-table-shape/spec.md` US1:

canonical       what the library creates today
legacy_serial   database-generated integer identifier, filename as the title
legacy_id       id / source / content, no embedding column at all
fixture         doc_id with the legacy `content` text column
mixed           both column families after an in-place upgrade
double_vector   canonical, but the vectors are double precision

The last two exist because all six of those carry a `metadata` column and all six
carry a column that fills the TITLE role, so the two absent-optional-column cases
the code handles could only be described by deleting a column from a recorded
snapshot -- a test asserting against its own idea of a table. These are captured
from real DDL like the rest:

no_metadata     the `fixture` shape with no metadata column
no_title        the `fixture` shape with nothing that fills the TITLE role
"""

from __future__ import annotations

VECTOR_DIMENSION = 384

# Scratch SQL schema the fixtures were captured under, and the one the integration
# suite creates its tables in. No underscore, so the class-package mapping stays
# readable.
SHAPE_SCHEMA = "ShapeTest"

# Table name is substituted, so the same shape can be created under several names
# (the parity suite needs every table alive at once).
SHAPE_DDL: dict[str, str] = {
    "canonical": """
        CREATE TABLE {table} (
            doc_id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(1000),
            abstract VARCHAR(MAX),
            text_content VARCHAR(MAX),
            authors VARCHAR(MAX),
            keywords VARCHAR(MAX),
            embedding VECTOR(FLOAT, {dim}),
            metadata VARCHAR(2000),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "legacy_serial": """
        CREATE TABLE {table} (
            id INTEGER IDENTITY PRIMARY KEY,
            filename VARCHAR(500),
            content VARCHAR(MAX),
            embedding VECTOR(FLOAT, {dim}),
            metadata VARCHAR(2000),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "legacy_id": """
        CREATE TABLE {table} (
            id VARCHAR(255) PRIMARY KEY,
            source VARCHAR(500),
            content VARCHAR(MAX),
            metadata VARCHAR(MAX),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            test_run_id VARCHAR(255)
        )
    """,
    "fixture": """
        CREATE TABLE {table} (
            doc_id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(1000),
            content VARCHAR(MAX),
            embedding VECTOR(FLOAT, {dim}),
            metadata VARCHAR(2000)
        )
    """,
    "mixed": """
        CREATE TABLE {table} (
            doc_id VARCHAR(255) PRIMARY KEY,
            id VARCHAR(255),
            title VARCHAR(1000),
            source VARCHAR(500),
            text_content VARCHAR(MAX),
            content VARCHAR(MAX),
            embedding VECTOR(FLOAT, {dim}),
            metadata VARCHAR(2000),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "double_vector": """
        CREATE TABLE {table} (
            doc_id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(1000),
            text_content VARCHAR(MAX),
            embedding VECTOR(DOUBLE, {dim}),
            metadata VARCHAR(2000),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "no_metadata": """
        CREATE TABLE {table} (
            doc_id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(1000),
            content VARCHAR(MAX),
            embedding VECTOR(FLOAT, {dim})
        )
    """,
    "no_title": """
        CREATE TABLE {table} (
            doc_id VARCHAR(255) PRIMARY KEY,
            content VARCHAR(MAX),
            embedding VECTOR(FLOAT, {dim}),
            metadata VARCHAR(2000)
        )
    """,
}

# Default table name per shape, used by the fixture capture and by tests that only
# need one table of each shape alive at a time.
SHAPE_TABLES: dict[str, str] = {
    "canonical": "ShapeCanonical",
    "legacy_serial": "ShapeLegacySerial",
    "legacy_id": "ShapeLegacyId",
    "fixture": "ShapeFixture",
    "mixed": "ShapeMixed",
    "double_vector": "ShapeDoubleVector",
    "no_metadata": "ShapeNoMetadata",
    "no_title": "ShapeNoTitle",
}


def create_statement(shape: str, schema: str, table: str | None = None) -> str:
    """Render one shape's CREATE TABLE for `schema`.`table`."""
    if shape not in SHAPE_DDL:
        raise KeyError(f"unknown shape {shape!r}; known: {sorted(SHAPE_DDL)}")
    qualified = f"{schema}.{table or SHAPE_TABLES[shape]}"
    return SHAPE_DDL[shape].format(table=qualified, dim=VECTOR_DIMENSION).strip()


def drop_statement(shape: str, schema: str, table: str | None = None) -> str:
    """Render the matching DROP TABLE."""
    qualified = f"{schema}.{table or SHAPE_TABLES[shape]}"
    return f"DROP TABLE IF EXISTS {qualified}"
