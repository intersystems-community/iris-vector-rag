"""Rows copied out of the cursor, so they outlive it.

This driver's row objects stop being readable once their cursor closes, which has
cost this project silent empty results and truncated fields. A ``Rows`` holds plain
tuples, materialised while the cursor was still open.

Feature: 083-iris-session (FR-011)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


class Rows:
    """The values a statement returned, and the columns they came under."""

    __slots__ = ("columns", "rows", "_rowcount")

    def __init__(
        self,
        columns: Sequence[str] = (),
        rows: Sequence[Sequence[Any]] = (),
        rowcount: Optional[int] = None,
    ) -> None:
        self.columns: Tuple[str, ...] = tuple(columns)
        self.rows: Tuple[Tuple[Any, ...], ...] = tuple(tuple(row) for row in rows)
        self._rowcount = rowcount

    @property
    def rowcount(self) -> int:
        """What the driver reported, or how many rows came back."""
        return len(self.rows) if self._rowcount is None else self._rowcount

    def one(self) -> Optional[Tuple[Any, ...]]:
        """The single row, or ``None``.

        No rows is an answer, not an error: a caller asking for a count of an
        empty table wants zero, not an exception about row objects.
        """
        if not self.rows:
            return None
        if len(self.rows) > 1:
            raise ValueError(f"expected at most one row, got {len(self.rows)} rows")
        return self.rows[0]

    def dicts(self) -> List[Dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __bool__(self) -> bool:
        return bool(self.rows)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rows):
            return self.columns == other.columns and self.rows == other.rows
        return NotImplemented

    def __repr__(self) -> str:
        return f"Rows(columns={self.columns!r}, rows={len(self.rows)})"


def read_cursor(cursor: Any) -> Rows:
    """Copy everything readable out of a cursor that is still open."""
    description = getattr(cursor, "description", None)
    columns = tuple(str(column[0]) for column in description) if description else ()
    rows: Tuple[Tuple[Any, ...], ...] = ()
    if description:
        fetched = cursor.fetchall()
        rows = tuple(tuple(row) for row in (fetched or ()))
    reported = getattr(cursor, "rowcount", None)
    rowcount = None
    if isinstance(reported, int) and reported >= 0 and not description:
        rowcount = reported
    return Rows(columns=columns, rows=rows, rowcount=rowcount)
