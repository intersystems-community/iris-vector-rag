"""Keep the case of entity and relationship types.

IRIS gives a string column the SQLUPPER collation unless the DDL says otherwise. On
such a column `GROUP BY`, `DISTINCT` and `=` fold case, so

    SELECT entity_type, COUNT(*) FROM RAG.Entities GROUP BY entity_type

returns `DISEASE` for a stored `Disease`, and `WHERE entity_type = 'disease'` matches
it. Type labels are compared and reported as the extractor wrote them, so the two type
columns are declared `COLLATE EXACT`. Nothing else here is: identifiers, names and
free text keep the default, where folding is what a lookup wants.

Tables made by an earlier release have the default collation. `ensure_exact_collation`
reads each column's collation from `%Dictionary.CompiledProperty` (the `COLLATION`
entry of its `Parameters`; empty means the default) and alters the ones that are not
EXACT with `ALTER TABLE ... ALTER COLUMN c VARCHAR(n) COLLATE EXACT`. Measured on IRIS
2026.1: the ALTER keeps every row and the column's nullability, and the indexes on the
column are rebuilt by IRIS itself, so no index is dropped or recreated here.

The one hazard is a lock. The ALTER waits for a connection that holds a table lock (or
has an unreleased read on the table), and a caller that hangs there is worse than one
that leaves the column as it was. So the ALTER runs on a worker thread (which has its
own connection: the session keeps one per thread) with the server's `LOCK_TIMEOUT` set
to the caller's timeout, which makes the server abandon the wait (SQLCODE -110) instead
of altering the table at some later moment under queries that do not expect it to
change (measured: a query racing a collation ALTER can die with <INVALID OREF>). A
blocked ALTER therefore leaves the column as it was, logs why, and the next ensure
retries. The caller's own connection is never committed or rolled back by this step.
"""

from __future__ import annotations

import contextlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .catalog import CatalogSnapshot, IRISCatalogReader

logger = logging.getLogger(__name__)

__all__ = [
    "ExactCollationError",
    "EXACT_CASE_COLUMNS",
    "ensure_exact_collation",
    "collation_of",
    "columns_needing_exact",
]


class ExactCollationError(RuntimeError):
    """A column could not be made COLLATE EXACT; nothing was changed."""


#: Table (unqualified) -> the columns whose stored case is part of their meaning.
EXACT_CASE_COLUMNS: Mapping[str, Sequence[str]] = {
    "Entities": ("entity_type",),
    "EntityRelationships": ("relationship_type",),
}

DEFAULT_COLLATION = "SQLUPPER"
DEFAULT_TIMEOUT_SECONDS = 30.0
GRACE_SECONDS = 5.0

_COLLATION_PARAMETER = re.compile(r"(?:^|,)COLLATION,([^,]*)")


def collation_of(snapshot: CatalogSnapshot, column: str) -> Optional[str]:
    """The collation a column was declared with, upper-case, or None if unknown.

    An empty `COLLATION` parameter is the default, reported as `SQLUPPER`. None means
    the catalog has no property for the column (not a string column, or no such
    column).
    """
    for prop in snapshot.properties:
        if str(prop.get("SqlFieldName", "")).lower() != column.lower():
            continue
        match = _COLLATION_PARAMETER.search(str(prop.get("Parameters") or ""))
        if match is None:
            return None
        return (match.group(1) or DEFAULT_COLLATION).upper()
    return None


def columns_needing_exact(
    snapshot: CatalogSnapshot, columns: Iterable[str]
) -> Dict[str, int]:
    """The VARCHAR columns among `columns` that are not EXACT, with their lengths."""
    by_name = {str(c["COLUMN_NAME"]).lower(): c for c in snapshot.columns}
    needed: Dict[str, int] = {}
    for column in columns:
        fact = by_name.get(column.lower())
        if fact is None:
            continue
        if str(fact.get("DATA_TYPE", "")).lower() not in (
            "varchar",
            "character varying",
        ):
            logger.info(
                "%s.%s.%s is %s, not VARCHAR; collation left as it is",
                snapshot.schema,
                snapshot.table,
                column,
                fact.get("DATA_TYPE"),
            )
            continue
        current = collation_of(snapshot, column)
        length = fact.get("CHARACTER_MAXIMUM_LENGTH")
        if current is None or current == "EXACT" or not length:
            continue
        needed[str(fact["COLUMN_NAME"])] = int(length)
    return needed


def ensure_exact_collation(
    connection_manager: Any,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    raise_on_failure: bool = True,
) -> Dict[str, int]:
    """Make the case-bearing columns of `schema.table` EXACT; return those altered.

    `columns` defaults to `EXACT_CASE_COLUMNS` for the table. A missing table, a
    missing column or a column already EXACT is left alone. Data is never touched. If
    the ALTER cannot finish within `timeout` seconds (another connection holds an
    lock or unreleased read), the column is left as it is and a warning says why; the next
    call retries. The call returns within about `timeout` plus a few seconds. A failure
    raises `ExactCollationError`; `raise_on_failure=False` logs it and returns `{}`,
    for callers (the schema manager) for whom the repair is best effort.
    """
    if columns is None:
        columns = EXACT_CASE_COLUMNS.get(table, ())
    if not columns:
        return {}

    connection = connection_manager.get_connection()
    snapshot = IRISCatalogReader(connection).snapshot(schema, table)
    if snapshot is None:
        return {}
    needed = columns_needing_exact(snapshot, columns)
    if not needed:
        return {}
    qualified = f"{snapshot.schema}.{snapshot.table}"

    def alter() -> None:
        worker_connection = connection_manager.get_connection()
        cursor = worker_connection.cursor()
        try:
            worker_connection.commit()
            # The server abandons a lock wait after this long (SQLCODE -110), so an
            # ALTER that is told to give up does give up, instead of completing at
            # some later moment under queries that do not expect the table to change.
            cursor.execute(f"SET OPTION LOCK_TIMEOUT={max(1, int(timeout))}")
            for column, length in needed.items():
                cursor.execute(
                    f"ALTER TABLE {qualified} ALTER COLUMN {column} "
                    f"VARCHAR({length}) COLLATE EXACT"
                )
            worker_connection.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                worker_connection.rollback()
            raise
        finally:
            with contextlib.suppress(Exception):
                cursor.close()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="exact-collation")
    try:
        executor.submit(alter).result(timeout + GRACE_SECONDS)
    except (FutureTimeout, Exception) as exc:
        reason = (
            f"did not finish in {timeout:.0f}s (a lock held by another connection?)"
            if isinstance(exc, FutureTimeout)
            else f"failed ({exc})"
        )
        message = (
            f"{qualified} {sorted(needed)} left at the default (case-folding) "
            f"collation: ALTER ... COLLATE EXACT {reason}. No data was changed; the "
            "next ensure retries."
        )
        if raise_on_failure:
            raise ExactCollationError(message) from exc
        logger.warning(message)
        return {}
    finally:
        executor.shutdown(wait=False)

    logger.warning(
        "%s %s had the default collation (GROUP BY/DISTINCT/= fold case); altered to "
        "COLLATE EXACT. Rows and indexes are kept.",
        qualified,
        sorted(needed),
    )
    return needed
