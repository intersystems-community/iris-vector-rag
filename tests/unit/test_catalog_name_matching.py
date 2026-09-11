"""A table is found whatever case its name is spelled in (feature 082).

Feature 074 taught `table_exists` to compare `UPPER(TABLE_SCHEMA)` and
`UPPER(TABLE_NAME)`, because callers spell table names the way their configuration
spells them and IRIS stores them the way the DDL declared them. Routing the check
through the catalog reader dropped that, and nothing noticed: `RecordedCatalog`
lower-cases its keys, so every unit test using a recorded fixture kept passing while
the live reader started missing tables that differ only in case.

Case folding belongs in one place -- resolving the table's identity -- and every
later query uses the spelling the catalog itself reported, so the snapshot describes
the table as IRIS names it, not as the caller asked for it.
"""

from __future__ import annotations

from unittest.mock import Mock

from iris_vector_rag.storage.catalog import IRISCatalogReader, RecordedCatalog


class ScriptedCursor:
    """Answers each statement in turn from a script of (description, rows)."""

    def __init__(self, script):
        self._script = list(script)
        self.statements: list[tuple[str, object]] = []
        self.description: list[tuple[str]] = []
        self._rows: list[tuple] = []
        self.closed = False

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))
        description, rows = self._script.pop(0) if self._script else ([], [])
        self.description = [(name,) for name in description]
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)

    def close(self):
        self.closed = True


IDENTITY_ROW = (["TABLE_SCHEMA", "TABLE_NAME"], [("RAG", "SourceDocuments")])
COLUMN_ROWS = (
    ["COLUMN_NAME", "DATA_TYPE"],
    [("doc_id", "varchar"), ("embedding", "varchar")],
)
CLASS_ROW = (["ID"], [("RAG.SourceDocuments",)])
PROPERTY_ROWS = (["SqlFieldName", "Type"], [("doc_id", "%Library.String")])
KEY_ROWS = (["COLUMN_NAME"], [("doc_id",)])

FOUND = [IDENTITY_ROW, COLUMN_ROWS, CLASS_ROW, PROPERTY_ROWS, KEY_ROWS]
MISSING = [(["TABLE_SCHEMA", "TABLE_NAME"], [])]


def reader_for(script) -> tuple[IRISCatalogReader, ScriptedCursor]:
    cursor = ScriptedCursor(script)
    connection = Mock()
    connection.cursor.return_value = cursor
    return IRISCatalogReader(connection), cursor


def test_a_table_asked_for_in_the_wrong_case_is_still_found():
    reader, _ = reader_for(FOUND)

    snapshot = reader.snapshot("rag", "sourcedocuments")

    assert snapshot is not None
    assert [str(row["COLUMN_NAME"]) for row in snapshot.columns] == [
        "doc_id",
        "embedding",
    ]


def test_the_snapshot_reports_the_spelling_the_catalog_uses():
    """Not the caller's spelling: the DDL paths need the real name."""
    reader, _ = reader_for(FOUND)

    snapshot = reader.snapshot("rag", "sourcedocuments")

    assert (snapshot.schema, snapshot.table) == ("RAG", "SourceDocuments")


def test_identity_is_the_one_place_that_folds_case():
    reader, cursor = reader_for(FOUND)

    reader.snapshot("rag", "sourcedocuments")

    identity_sql, identity_params = cursor.statements[0]
    assert "UPPER(TABLE_SCHEMA) = UPPER(?)" in identity_sql
    assert "UPPER(TABLE_NAME) = UPPER(?)" in identity_sql
    assert identity_params == ["rag", "sourcedocuments"]

    later = cursor.statements[1:]
    assert later, "the remaining catalog queries should still run"
    assert not any("UPPER(" in sql for sql, _ in later)


def test_every_later_query_uses_the_resolved_name():
    reader, cursor = reader_for(FOUND)

    reader.snapshot("rag", "sourcedocuments")

    for sql, params in cursor.statements[1:]:
        if params and "RAG.SourceDocuments" not in list(params):
            assert list(params)[:2] == ["RAG", "SourceDocuments"], sql


def test_a_table_that_does_not_exist_costs_one_query():
    """The common `table_exists` answer is no; it should not pay for four reads."""
    reader, cursor = reader_for(MISSING)

    assert reader.snapshot("RAG", "NoSuchTable") is None
    assert len(cursor.statements) == 1
    assert cursor.closed


def test_the_two_readers_agree_about_case():
    """`RecordedCatalog` has always folded case; the live reader now matches it."""
    recorded = RecordedCatalog(
        {("RAG", "SourceDocuments"): object()}  # type: ignore[dict-item]
    )
    live, _ = reader_for(FOUND)

    assert recorded.snapshot("rag", "sourcedocuments") is not None
    assert live.snapshot("rag", "sourcedocuments") is not None
