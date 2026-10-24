"""Resolving a table's shape once per session (feature 082).

The resolver owns the cache; `TableShape` owns the facts. That split is the point:
the shape stays a pure value, and the "how often do we ask the database" question
lives in one small object whose whole state is a dict.

The cache is per resolver, and a resolver belongs to a session. There is no
module-level cache, because two sessions may point at different instances or
different namespaces, and a shared cache would let one session's schema answer the
other's question.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

from iris_vector_rag.storage.catalog import CatalogReader
from iris_vector_rag.storage.table_shape import (
    CatalogUnavailable,
    ColumnRole,
    Declared,
    TableNotFound,
    TableShape,
)

__all__ = ["TableShapeResolver"]

CacheKey = Tuple[str, str, Tuple]


class TableShapeResolver:
    """Resolves and caches one `TableShape` per table."""

    def __init__(self, reader: CatalogReader) -> None:
        self._reader = reader
        self._shapes: Dict[CacheKey, TableShape] = {}
        self._reads = 0

    @property
    def reads(self) -> int:
        """Catalog snapshots actually taken. SC-002 is measured on this."""
        return self._reads

    def resolve(
        self,
        schema: str,
        table: str,
        declared: Optional[Mapping[ColumnRole, Declared]] = None,
    ) -> TableShape:
        """The table's shape, with ``declared`` columns bound first (feature 092).

        The declaration is part of the cache key, so two stores that read one table
        through different mappings never share a shape.
        """
        key = _key(schema, table, declared)
        cached = self._shapes.get(key)
        if cached is not None:
            return cached

        try:
            snapshot = self._reader.snapshot(schema, table)
        except CatalogUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - the driver error becomes the cause
            # A failed read is never a shape. Substituting a default here is exactly
            # how a mistyped table name used to surface as a vector dimension error
            # several layers away.
            raise CatalogUnavailable(
                f"could not read the catalog: {exc}", schema, table
            ) from exc

        if snapshot is None:
            # Not cached: a table created after this call must be visible next time,
            # which is what a test that creates its tables in a fixture depends on.
            raise TableNotFound("no such table in this namespace", schema, table)

        self._reads += 1
        shape = TableShape.from_snapshot(snapshot, declared=declared)
        self._shapes[key] = shape
        return shape

    def invalidate(self, schema: str, table: str) -> None:
        """Forget one table, under every declaration. A no-op if never resolved."""
        prefix = _key(schema, table)[:2]
        for key in [k for k in self._shapes if k[:2] == prefix]:
            del self._shapes[key]

    def invalidate_all(self) -> None:
        self._shapes.clear()

    def cached(
        self,
        schema: str,
        table: str,
        declared: Optional[Mapping[ColumnRole, Declared]] = None,
    ) -> Optional[TableShape]:
        """The cached shape, without resolving. For diagnostics and tests."""
        return self._shapes.get(_key(schema, table, declared))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"TableShapeResolver(cached={sorted('.'.join(k[:2]) for k in self._shapes)}, "
            f"reads={self._reads})"
        )


def _key(
    schema: str, table: str, declared: Optional[Mapping[ColumnRole, Declared]] = None
) -> CacheKey:
    """IRIS identifiers are case-insensitive, so the cache key must be too."""
    frozen = tuple(
        sorted(
            (
                role.value,
                (
                    names.lower()
                    if isinstance(names, str)
                    else tuple(n.lower() for n in names)
                ),
            )
            for role, names in (declared or {}).items()
        )
    )
    return (schema.lower(), table.lower(), frozen)
