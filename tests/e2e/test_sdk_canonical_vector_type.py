"""Every SDK creation path declares the canonical vector element type (feature 086).

IRIS refuses to compare vectors whose element types differ (SQLCODE -259), so the type a
column is created with and the type a writer names in `TO_VECTOR` have to agree. Two claims
are checked here:

* what `RAG.SDK.Schema.Initialize` creates is the type the Python side calls canonical, and
  a vector converted with that type is comparable against the column it just created;
* what `RAG.SDK.Pipeline.AddDocument` writes is the type the column it writes into actually
  has, whether that is the canonical `FLOAT` or the `DOUBLE` an older table was created with.

The declaration is read out of the class dictionary rather than `INFORMATION_SCHEMA`, which
reports every VECTOR column as varchar (a driver limitation, per CLAUDE.md).

Everything runs in a scratch schema of this feature's own, so no test here can reach
`RAG.SourceDocuments`. ObjectScript is called in `iris-vector-rag-iris` -- the only
container this project uses -- because a `.cls` runs on the server, not on disk.
"""

from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

import pytest

from iris_vector_rag.common.iris_connection import get_iris_connection
from iris_vector_rag.storage.table_shape import CANONICAL_VECTOR_TYPE

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_docker,
    pytest.mark.requires_database,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_SCHEMA = "RAG086Canonical"
LEGACY_TABLE = f"{SCRATCH_SCHEMA}.LegacyDoubleDocuments"
# The dimension the repo ships with, for the creation path, and a small one for the write
# paths: a writer that hardcodes 384 cannot serve an 8-wide column.
DIM = 384
SMALL_DIM = 8
SCRATCH_TABLES = (
    f"{SCRATCH_SCHEMA}.SourceDocuments",
    f"{SCRATCH_SCHEMA}.DocumentChunks",
    f"{SCRATCH_SCHEMA}.Entities",
    LEGACY_TABLE,
)


def _container_name() -> str:
    """The container CLAUDE.md declares for this project, and no other."""
    for line in (REPO_ROOT / "CLAUDE.md").read_text().splitlines():
        if line.startswith("IRIS_CONTAINER:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("CLAUDE.md declares no IRIS_CONTAINER line")


def _session(script: str) -> str:
    """Run ObjectScript on the server and return everything the session printed."""
    completed = subprocess.run(
        ["docker", "exec", "-i", _container_name(), "iris", "session", "IRIS"],
        input=script if script.endswith("\n") else script + "\n",
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    return output


def _marker(output: str, marker: str) -> str:
    """The rest of the line a marker appears on, so prompt noise is not parsed as data."""
    for line in output.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    raise AssertionError(f"{marker!r} is not in the session output:\n{output}")


def _declared_vector(table: str, column: str) -> tuple[str, int]:
    """The element type and dimension IRIS recorded for a VECTOR column."""
    schema, _, name = table.rpartition(".")
    script = (
        'zn "USER"\n'
        "set rs=##class(%SQL.Statement).%ExecDirect(,"
        '"SELECT CLASSNAME FROM INFORMATION_SCHEMA.TABLES '
        'WHERE TABLE_SCHEMA=? AND TABLE_NAME=?",'
        f'"{schema}","{name}")\n'
        'if \'rs.%Next() { write "SPEC:no-such-table",! halt }\n'
        "set cls=rs.%GetData(1)\n"
        # A DDL-created property is named after its column with the punctuation removed, so
        # chunk_embedding is the property chunkembedding: look it up by SQL field name.
        "set rs2=##class(%SQL.Statement).%ExecDirect(,"
        '"SELECT Name FROM %Dictionary.CompiledProperty '
        'WHERE parent = ? AND (SqlFieldName = ? OR Name = ?)",'
        f'cls,"{column}","{column}")\n'
        'if \'rs2.%Next() { write "SPEC:no-such-property",! halt }\n'
        'set prop=##class(%Dictionary.CompiledProperty).%OpenId(cls_"||"_rs2.%GetData(1))\n'
        'if \'$IsObject(prop) { write "SPEC:unopenable-property",! halt }\n'
        'write "SPEC:",prop.Parameters.GetAt("DATATYPE"),"|",'
        'prop.Parameters.GetAt("LEN"),!\n'
        "halt\n"
    )
    spec = _marker(_session(script), "SPEC:")
    element_type, _, length = spec.partition("|")
    assert length.isdigit(), f"{table}.{column} reported {spec!r}"
    return element_type.upper(), int(length)


def _vector_text(seed: int, dim: int) -> str:
    """A unit-length vector as the comma-separated text both TO_VECTOR and the SDK take."""
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in values))
    return ",".join(f"{value / norm:.6f}" for value in values)


def _drop_scratch_tables(connection) -> None:
    cursor = connection.cursor()
    try:
        for table in SCRATCH_TABLES:
            try:
                cursor.execute(f"DROP TABLE {table}")
            except (
                Exception
            ):  # noqa: BLE001 - a table that is not there needs no dropping
                pass
        connection.commit()
    finally:
        cursor.close()


@pytest.fixture
def scratch_schema():
    """A schema of this feature's own, empty before the test and gone after it."""
    connection = get_iris_connection()
    _drop_scratch_tables(connection)
    try:
        yield SCRATCH_SCHEMA
    finally:
        _drop_scratch_tables(connection)
        # Do not close: get_iris_connection hands out a cached connection the rest of the
        # session reuses.


def _search_score(
    table: str, doc_id: str, element_type: str, vector: str, dim: int
) -> float:
    """The cosine of a stored row against the vector that was stored in it."""
    connection = get_iris_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"SELECT VECTOR_COSINE(embedding, TO_VECTOR(?, {element_type}, {dim})) "
            f"FROM {table} WHERE doc_id = ?",
            [vector, doc_id],
        )
        row = cursor.fetchone()
        assert row is not None, f"{doc_id} is not in {table}"
        return float(row[0])
    finally:
        cursor.close()


def test_initialize_declares_the_canonical_element_type(scratch_schema):
    """FR-001, FR-002: the SDK's DDL creates FLOAT columns Python can search as they are."""
    output = _session(
        'zn "USER"\n'
        f'write "INIT:",##class(RAG.SDK.Schema).Initialize({DIM}, "{scratch_schema}"),!\n'
        "halt\n"
    )
    assert '"status":"ok"' in _marker(output, "INIT:"), output

    assert _declared_vector(f"{scratch_schema}.SourceDocuments", "embedding") == (
        CANONICAL_VECTOR_TYPE,
        DIM,
    )
    assert _declared_vector(f"{scratch_schema}.DocumentChunks", "chunk_embedding") == (
        CANONICAL_VECTOR_TYPE,
        DIM,
    )

    # A vector converted with the type the Python side calls canonical has to compare
    # against the column the SDK just created. A cross-element-type comparison comes back
    # as SQLCODE -259, and nothing here retries with a second type.
    vector = _vector_text(1, DIM)
    connection = get_iris_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"INSERT INTO {scratch_schema}.SourceDocuments "
            f"(doc_id, text_content, embedding) "
            f"VALUES (?, ?, TO_VECTOR(?, {CANONICAL_VECTOR_TYPE}, {DIM}))",
            ["t086_canonical", "created by the SDK, searched from Python", vector],
        )
        connection.commit()
    finally:
        cursor.close()

    score = _search_score(
        f"{scratch_schema}.SourceDocuments",
        "t086_canonical",
        CANONICAL_VECTOR_TYPE,
        vector,
        DIM,
    )
    assert score > 0.99, score


def test_add_document_writes_the_type_a_legacy_double_column_has(scratch_schema):
    """FR-003, FR-004: a table created before the canonical type is still writable."""
    connection = get_iris_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"CREATE TABLE {LEGACY_TABLE} (doc_id VARCHAR(64) NOT NULL, "
            f"text_content LONGVARCHAR, metadata LONGVARCHAR, "
            f"embedding VECTOR(DOUBLE, {SMALL_DIM}), PRIMARY KEY (doc_id))"
        )
        connection.commit()
    finally:
        cursor.close()
    assert _declared_vector(LEGACY_TABLE, "embedding") == ("DOUBLE", SMALL_DIM)

    vector = _vector_text(2, SMALL_DIM)
    output = _session(
        'zn "USER"\n'
        'write "ADD:",##class(RAG.SDK.Pipeline).AddDocument('
        f'"t086_double","a row in a legacy double table","{{}}","{vector}",'
        f'"{LEGACY_TABLE}"),!\n'
        "halt\n"
    )
    assert '"status":"ok"' in _marker(output, "ADD:"), output

    score = _search_score(LEGACY_TABLE, "t086_double", "DOUBLE", vector, SMALL_DIM)
    assert score > 0.99, score


def test_add_document_writes_the_canonical_type_into_an_initialized_table(
    scratch_schema,
):
    """FR-004: the same call against a FLOAT table Initialize created."""
    table = f"{scratch_schema}.SourceDocuments"
    output = _session(
        'zn "USER"\n'
        f'write "INIT:",##class(RAG.SDK.Schema).Initialize({SMALL_DIM}, '
        f'"{scratch_schema}"),!\n'
        "halt\n"
    )
    assert '"status":"ok"' in _marker(output, "INIT:"), output
    assert _declared_vector(table, "embedding") == (CANONICAL_VECTOR_TYPE, SMALL_DIM)

    vector = _vector_text(3, SMALL_DIM)
    output = _session(
        'zn "USER"\n'
        'write "ADD:",##class(RAG.SDK.Pipeline).AddDocument('
        f'"t086_float","a row in a canonical table","{{}}","{vector}","{table}"),!\n'
        "halt\n"
    )
    assert '"status":"ok"' in _marker(output, "ADD:"), output

    score = _search_score(table, "t086_float", CANONICAL_VECTOR_TYPE, vector, SMALL_DIM)
    assert score > 0.99, score
