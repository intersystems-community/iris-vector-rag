"""Tables the library reads but does not own (feature 092, "bring your own table").

A team's chunk table has its own column names, its own embedding model and columns
every query must filter on. This module turns the settings that describe such a
table into values the store can use: which column fills which role
(`ColumnMapping`, fed to `TableShape` as declared roles) and which columns a search
may and must filter on (`FilterSpec`).

Everything here is pure: settings in, values out. The names follow the vocabulary
that `HybridGraphRAGPipeline.attach_existing_corpus` (069) and `RAG.SDK.Bridge`
(070, 085) already use: `id_col`, `text_col`, `embedding_col`, `metadata_col`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from iris_vector_rag.exceptions import (
    FilterError,
    RequiredFilterMissing,
    VectorStoreConfigurationError,
)
from iris_vector_rag.storage.table_shape import (
    _IDENTIFIER,
    ColumnRole,
    Declared,
    TableShape,
    quote_identifier,
)

__all__ = ["ColumnMapping", "FilterSpec", "compile_where", "quote_identifier"]

#: The comparisons a filter may make; a list is ``IN``.
_OPERATORS = frozenset({"=", "<", "<=", ">", ">="})
_SCALARS = (str, int, float, dt.date, dt.datetime)

#: SQL types (as INFORMATION_SCHEMA reports them) a value is cast to before it is
#: compared, so "2026-10-05" compares as a date and not as text.
_CASTS = {"date": "DATE", "timestamp": "TIMESTAMP", "datetime": "TIMESTAMP"}

#: Setting key -> the role it declares.
_ROLE_KEYS: Mapping[str, ColumnRole] = {
    "id_col": ColumnRole.IDENTIFIER,
    "text_col": ColumnRole.TEXT,
    "embedding_col": ColumnRole.EMBEDDING,
    "metadata_col": ColumnRole.METADATA,
    "title_col": ColumnRole.TITLE,
    "parent_col": ColumnRole.PARENT,
}


def _setting(settings: Mapping[str, Any], overrides: Mapping[str, Any], key: str):
    """The override when one was given, else the configured value."""
    value = overrides.get(key)
    return settings.get(key) if value is None else value


def _names(value: Any, key: str) -> Tuple[str, ...]:
    """A list setting: a list, a tuple, or a comma-separated string (environment)."""
    if value is None or value == "":
        return ()
    items = value.split(",") if isinstance(value, str) else list(value)
    names = tuple(str(item).strip() for item in items if str(item).strip())
    for name in names:
        if not _IDENTIFIER.match(name):
            raise VectorStoreConfigurationError(
                f"storage:iris:{key} must name plain SQL columns, got {name!r}"
            )
    return names


@dataclass(frozen=True)
class ColumnMapping:
    """The declared column for each role of an attached table."""

    declared: Mapping[ColumnRole, Declared]

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> "ColumnMapping":
        """From ``storage:iris:`` settings, with constructor ``overrides`` winning."""
        declared: Dict[ColumnRole, Declared] = {}
        for key, role in _ROLE_KEYS.items():
            names = _names(_setting(settings, overrides, key), key)
            if not names:
                continue
            if role is not ColumnRole.IDENTIFIER and len(names) > 1:
                raise VectorStoreConfigurationError(
                    f"storage:iris:{key} names one column; only id_col may list several "
                    f"(got {', '.join(names)})"
                )
            declared[role] = names if len(names) > 1 else names[0]
        return cls(declared=declared)

    @property
    def attached(self) -> bool:
        return bool(self.declared)


@dataclass(frozen=True)
class FilterSpec:
    """The real columns a search may filter on, and the ones it must."""

    filter_columns: Tuple[str, ...] = ()
    required_filters: Tuple[str, ...] = ()

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> "FilterSpec":
        filter_columns = _names(
            _setting(settings, overrides, "filter_columns"), "filter_columns"
        )
        required = _names(
            _setting(settings, overrides, "required_filters"), "required_filters"
        )
        allowed = {name.lower() for name in filter_columns}
        stray = [name for name in required if name.lower() not in allowed]
        if stray:
            raise FilterError(
                f"required_filters must also be filter_columns: {', '.join(stray)} "
                f"is not in filter_columns ({', '.join(filter_columns) or 'none'})",
                rejected_keys=stray,
                allowed_keys=list(filter_columns),
            )
        return cls(filter_columns=filter_columns, required_filters=required)

    def column(self, key: str) -> Optional[str]:
        """The declared spelling of ``key``, matched without regard to case."""
        lowered = key.lower()
        return next((c for c in self.filter_columns if c.lower() == lowered), None)


def _bindable(column: str, value: Any) -> Any:
    """A filter value as bound: dates as ISO text, other scalars as they are."""
    if isinstance(value, bool):
        # IRIS has no boolean type; a BIT column holds 0 and 1.
        raise FilterError(f"filter on {column}: {value!r} is a bool; use 0 or 1")
    if not isinstance(value, _SCALARS):
        raise FilterError(
            f"filter on {column}: {type(value).__name__} is not a filter value "
            "(use str, int, float, date or datetime)"
        )
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


def compile_where(
    where: Optional[Mapping[str, Any]], shape: TableShape, spec: FilterSpec
) -> Tuple[str, List[Any]]:
    """``(predicate, params)`` for a search's ``where``; ``("", [])`` for none.

    ``where`` maps a column to a value (``=``), an ``(op, value)`` tuple, or a
    non-empty list (``IN``). Every key must be one of the spec's filter columns and
    a column of the table; every required filter must be present. The predicate is
    terms joined by ``AND`` with no outer parentheses (see `build_bound_vector_dot_sql`
    on the DB-API's doubled-parenthesis defect), and it holds only quoted names and
    ``?`` markers: every value is a parameter.

    Raises:
        RequiredFilterMissing: a required filter is absent.
        FilterError: anything else that cannot be applied as written.
    """
    absent = [c for c in spec.filter_columns if c.lower() not in shape.columns]
    if absent:
        raise FilterError(
            f"filter_columns names {', '.join(absent)}, which "
            f"{shape.qualified_name} does not have",
            rejected_keys=absent,
        )
    where = dict(where or {})
    present = {key.lower() for key in where}
    missing = [c for c in spec.required_filters if c.lower() not in present]
    if missing:
        raise RequiredFilterMissing(
            f"{shape.qualified_name} requires a filter on {', '.join(missing)}; "
            "a search without it would read every row",
            rejected_keys=missing,
            allowed_keys=list(spec.filter_columns),
        )

    terms: List[str] = []
    params: List[Any] = []
    for key, condition in where.items():
        column = spec.column(key)
        if column is None:
            raise FilterError(
                f"cannot filter on {key}: it is not in filter_columns "
                f"({', '.join(spec.filter_columns) or 'none declared'})",
                rejected_keys=[key],
                allowed_keys=list(spec.filter_columns),
            )
        fact = shape.columns[column.lower()]
        name = quote_identifier(fact.name)
        cast = _CASTS.get(fact.sql_type)
        marker = f"CAST(? AS {cast})" if cast else "?"

        if isinstance(condition, (list, set, frozenset)):
            values = list(condition)
            if not values:
                raise FilterError(
                    f"filter on {fact.name}: an empty list matches nothing"
                )
            terms.append(f"{name} IN ({', '.join(marker for _ in values)})")
            params.extend(_bindable(fact.name, v) for v in values)
            continue

        if isinstance(condition, tuple):
            if len(condition) != 2 or condition[0] not in _OPERATORS:
                raise FilterError(
                    f"filter on {fact.name}: {condition!r} is not an (operator, value) "
                    f"pair with an operator from {', '.join(sorted(_OPERATORS))} "
                    "(use a list for IN)"
                )
            op, value = condition
        else:
            op, value = "=", condition
        terms.append(f"{name} {op} {marker}")
        params.append(_bindable(fact.name, value))

    return " AND ".join(terms), params
