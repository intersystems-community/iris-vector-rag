"""Reading the IRIS catalog, and replaying a recording of it (feature 082).

This is the only module in the package that talks to `INFORMATION_SCHEMA` and
`%Dictionary.*`. `TableShape` consumes what comes out of it, so the shape logic can
be tested without a database and the SQL can be tested against a real one.

Three queries per table, plus a fourth only when the first returns nothing:

1. `INFORMATION_SCHEMA.COLUMNS` -- names, SQL types, lengths, nullability, ordinal
   position, and the identity flags.
2. `%Dictionary.CompiledClass` -- the class backing the table. This lookup is not
   optional and cannot be replaced by string concatenation: the live namespace has
   `RAG.llmcache` for SQL table `RAG.llm_cache`, so `"RAG." + table` would be right
   eleven times out of twelve and silently wrong the twelfth.
3. `%Dictionary.CompiledProperty` -- the class-side type of each column, joined on
   `SqlFieldName`. Not on `Name`: `Name` strips underscores, so `chunk_text` is
   stored as `chunktext` and a `Name = 'chunk_embedding'` predicate matches nothing.
4. `INFORMATION_SCHEMA.KEY_COLUMN_USAGE` -- the declared key. It can report the same
   column twice (an `INTEGER IDENTITY PRIMARY KEY` yields `['id', 'id']`), so callers
   must treat the result as a set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from iris_vector_rag.storage.table_shape import CatalogUnavailable

__all__ = [
    "CatalogReader",
    "CatalogSnapshot",
    "IRISCatalogReader",
    "RecordedCatalog",
]

COLUMN_SQL = """
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE,
           ORDINAL_POSITION, IS_IDENTITY, AUTO_INCREMENT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
    ORDER BY ORDINAL_POSITION
"""

CLASS_SQL = """
    SELECT ID FROM %Dictionary.CompiledClass
    WHERE SqlSchemaName = ? AND SqlTableName = ?
"""

PROPERTY_SQL = """
    SELECT SqlFieldName, Type, Parameters
    FROM %Dictionary.CompiledProperty
    WHERE parent = ?
    ORDER BY SqlFieldName
"""

KEY_SQL = """
    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
"""

TABLE_SQL = """
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
"""


@dataclass(frozen=True)
class CatalogSnapshot:
    """Catalog rows for one table, as read or as recorded.

    Rows are kept in the catalog's own vocabulary (`COLUMN_NAME`, `SqlFieldName`)
    rather than translated on the way in, so a recorded fixture is readable as what
    IRIS actually said and cannot quietly disagree with the live reader.
    """

    schema: str
    table: str
    class_name: Optional[str] = None
    columns: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    properties: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    key_columns: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CatalogSnapshot":
        return cls(
            schema=data["schema"],
            table=data["table"],
            class_name=data.get("class_name"),
            columns=list(data.get("columns") or []),
            properties=list(data.get("properties") or []),
            key_columns=list(data.get("key_columns") or []),
        )


class CatalogReader(Protocol):
    """The seam. One method, so a test double is three lines."""

    def snapshot(self, schema: str, table: str) -> Optional[CatalogSnapshot]:
        """Return the table's catalog rows, or None if the table does not exist."""
        ...


class IRISCatalogReader:
    """Reads the live catalog over a DBAPI connection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def snapshot(self, schema: str, table: str) -> Optional[CatalogSnapshot]:
        try:
            cursor = self._connection.cursor()
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise CatalogUnavailable(
                f"could not open a cursor to read the catalog: {exc}", schema, table
            ) from exc
        try:
            columns = self._rows(cursor, COLUMN_SQL, [schema, table])
            if not columns and not self._rows(cursor, TABLE_SQL, [schema, table]):
                return None

            cursor.execute(CLASS_SQL, [schema, table])
            class_rows = cursor.fetchall()
            class_name = str(class_rows[0][0]) if class_rows else None

            properties = (
                self._rows(cursor, PROPERTY_SQL, [class_name]) if class_name else []
            )
            keys = _dedupe(
                str(row["COLUMN_NAME"])
                for row in self._rows(cursor, KEY_SQL, [schema, table])
            )
        except CatalogUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - the driver error becomes the cause
            raise CatalogUnavailable(
                f"catalog query failed: {exc}", schema, table
            ) from exc
        finally:
            _close(cursor)

        return CatalogSnapshot(
            schema=schema,
            table=table,
            class_name=class_name,
            columns=columns,
            properties=properties,
            key_columns=keys,
        )

    @staticmethod
    def _rows(cursor: Any, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        cursor.execute(sql, list(params))
        names = [d[0] for d in cursor.description]
        # Values are copied out of each DataRow here: a DataRow is invalidated when
        # the cursor moves on or closes, so holding one and reading it later yields
        # nothing.
        return [
            {name: row[i] for i, name in enumerate(names)} for row in cursor.fetchall()
        ]


class RecordedCatalog:
    """Replays snapshots captured from a real instance.

    The fixtures under `tests/fixtures/shapes/` were produced by
    `scripts/dev/capture_shape_fixtures.py` against real DDL, so a unit test using
    this reader asserts against what IRIS reports, not against what a hand-written
    fixture wishes it reported.
    """

    def __init__(self, snapshots: Mapping[tuple[str, str], CatalogSnapshot]) -> None:
        self._snapshots = {
            (schema.lower(), table.lower()): snapshot
            for (schema, table), snapshot in snapshots.items()
        }

    @classmethod
    def from_fixture_dir(cls, path: Path) -> "RecordedCatalog":
        snapshots: dict[tuple[str, str], CatalogSnapshot] = {}
        for fixture in sorted(Path(path).glob("*.json")):
            snapshot = CatalogSnapshot.from_dict(json.loads(fixture.read_text()))
            snapshots[(snapshot.schema, snapshot.table)] = snapshot
        if not snapshots:
            raise FileNotFoundError(f"no recorded shape fixtures in {path}")
        return cls(snapshots)

    def snapshot(self, schema: str, table: str) -> Optional[CatalogSnapshot]:
        return self._snapshots.get((schema.lower(), table.lower()))

    @property
    def tables(self) -> list[tuple[str, str]]:
        return [(s.schema, s.table) for s in self._snapshots.values()]


def _dedupe(values: Iterable[str]) -> list[str]:
    """Order-preserving dedupe: KEY_COLUMN_USAGE can report a column twice."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _close(cursor: Any) -> None:
    try:
        cursor.close()
    except Exception:  # noqa: BLE001 - closing is advisory
        pass
