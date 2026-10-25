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

__all__ = [
    "ColumnMapping",
    "FilterSpec",
    "compile_where",
    "quote_identifier",
    "where_kwargs",
]

#: The comparisons a filter may make; a list is ``IN``.
_OPERATORS = frozenset({"=", "<", "<=", ">", ">="})

#: LangChain's operator names (review, 2026-10-24), so a LangChain filter works as is.
_LANGCHAIN_OPERATORS = {
    "$eq": "=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$in": "IN",
}
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

    ``where`` maps a column to a value (``=``), an ``(op, value)`` tuple, a
    non-empty list (``IN``), or LangChain's ``{"$gte": a, "$lt": b}`` (see
    `_conditions`). Every key must be one of the spec's filter columns and
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
    for key in where:
        if key.startswith("$"):
            raise FilterError(
                f"{key} is not supported in where: conditions on several columns are "
                "always ANDed, so list them as keys",
                rejected_keys=[key],
            )
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
        for op, value in _conditions(fact.name, condition):
            term, values = _term(fact, op, value)
            terms.append(term)
            params.extend(values)

    return " AND ".join(terms), params


def _conditions(column: str, condition: Any) -> List[Tuple[str, Any]]:
    """``(op, value)`` pairs for one column's condition, in any accepted spelling.

    A value means ``=``, a list means ``IN``, a tuple is ``(op, value)``, and a dict
    is LangChain's form, ``{"$gte": a, "$lt": b}``. LangChain's ``$eq``, ``$gt``,
    ``$gte``, ``$lt``, ``$lte`` and ``$in`` are accepted everywhere an operator is.
    """
    if isinstance(condition, dict):
        if not condition:
            raise FilterError(f"filter on {column}: an empty condition matches nothing")
        return [(_operator(column, op), value) for op, value in condition.items()]
    if isinstance(condition, tuple):
        if len(condition) != 2:
            raise FilterError(
                f"filter on {column}: {condition!r} is not an (operator, value) pair"
            )
        return [(_operator(column, condition[0]), condition[1])]
    if isinstance(condition, (list, set, frozenset)):
        return [("IN", condition)]
    return [("=", condition)]


def _operator(column: str, op: Any) -> str:
    """The SQL operator for ``op``, given as SQL (``>=``) or as LangChain (``$gte``)."""
    sql = _LANGCHAIN_OPERATORS.get(op, op) if isinstance(op, str) else op
    if sql not in _OPERATORS and sql != "IN":
        raise FilterError(
            f"filter on {column}: operator {op!r} is not supported; use one of "
            f"{', '.join(sorted(_OPERATORS))} (or {', '.join(sorted(_LANGCHAIN_OPERATORS))})"
        )
    return sql


def _term(fact, op: str, value: Any) -> Tuple[str, List[Any]]:
    """One predicate term and its parameters."""
    name = quote_identifier(fact.name)
    cast = _CASTS.get(fact.sql_type)
    marker = f"CAST(? AS {cast})" if cast else "?"
    if op == "IN":
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise FilterError(f"filter on {fact.name}: IN takes a list, got {value!r}")
        values = list(value)
        if not values:
            raise FilterError(f"filter on {fact.name}: an empty list matches nothing")
        return (
            f"{name} IN ({', '.join(marker for _ in values)})",
            [_bindable(fact.name, v) for v in values],
        )
    return f"{name} {op} {marker}", [_bindable(fact.name, value)]


def where_kwargs(where: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``{"where": where}`` when there are real-column filters, else ``{}``.

    The one place that decides how ``where`` is handed on. It is passed only when
    set, so a store or override written before feature 092, whose searches take no
    ``where``, keeps working for every query that does not filter on real columns.
    """
    return {"where": where} if where else {}
