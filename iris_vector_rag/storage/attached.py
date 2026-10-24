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

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from iris_vector_rag.exceptions import FilterError, VectorStoreConfigurationError
from iris_vector_rag.storage.table_shape import (
    _IDENTIFIER,
    ColumnRole,
    Declared,
    quote_identifier,
)

__all__ = ["ColumnMapping", "FilterSpec", "quote_identifier"]

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
