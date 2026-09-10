"""Unit tests for `TableShapeResolver` (feature 082, T009).

The resolver owns the cache, not the shape. Its contract is short: at most one
catalog snapshot per table per resolver, explicit invalidation, no shared state, and
distinct errors for "no such table" and "the catalog query itself failed".
"""

from __future__ import annotations

import pytest

from iris_vector_rag.storage.catalog import CatalogSnapshot, RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import (
    CatalogUnavailable,
    ColumnRole,
    TableNotFound,
)
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES

CANONICAL = SHAPE_TABLES["canonical"]
MIXED = SHAPE_TABLES["mixed"]


class CountingCatalog:
    """Wraps a reader and counts the snapshots actually taken."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[str, str]] = []

    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None:
        self.calls.append((schema, table))
        return self.inner.snapshot(schema, table)


class MissingCatalog:
    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None:
        return None


class BrokenCatalog:
    def __init__(self, error: Exception):
        self.error = error

    def snapshot(self, schema: str, table: str) -> CatalogSnapshot | None:
        raise self.error


@pytest.fixture
def counting(recorded_catalog: RecordedCatalog) -> CountingCatalog:
    return CountingCatalog(recorded_catalog)


def test_first_resolve_takes_one_snapshot(counting):
    resolver = TableShapeResolver(counting)
    shape = resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    assert shape.column_for(ColumnRole.TEXT).name == "text_content"
    assert counting.calls == [(SHAPE_SCHEMA, CANONICAL)]
    assert resolver.reads == 1


def test_repeated_resolves_take_no_further_snapshots(counting):
    """FR-005 / SC-002: N operations, one resolution."""
    resolver = TableShapeResolver(counting)
    first = resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    for _ in range(100):
        assert resolver.resolve(SHAPE_SCHEMA, CANONICAL) is first
    assert resolver.reads == 1


def test_different_tables_are_cached_separately(counting):
    resolver = TableShapeResolver(counting)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, MIXED)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    assert resolver.reads == 2


def test_lookup_is_case_insensitive_in_the_cache(counting):
    resolver = TableShapeResolver(counting)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA.upper(), CANONICAL.upper())
    assert resolver.reads == 1


def test_invalidate_forces_exactly_one_more_snapshot(counting):
    resolver = TableShapeResolver(counting)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.invalidate(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    assert resolver.reads == 2


def test_invalidate_is_a_no_op_for_an_unresolved_table(counting):
    resolver = TableShapeResolver(counting)
    resolver.invalidate(SHAPE_SCHEMA, CANONICAL)
    assert resolver.reads == 0


def test_invalidate_all_clears_every_table(counting):
    resolver = TableShapeResolver(counting)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, MIXED)
    resolver.invalidate_all()
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, MIXED)
    assert resolver.reads == 4


def test_invalidating_one_table_leaves_the_other_cached(counting):
    resolver = TableShapeResolver(counting)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, MIXED)
    resolver.invalidate(SHAPE_SCHEMA, MIXED)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    resolver.resolve(SHAPE_SCHEMA, MIXED)
    assert resolver.reads == 3


def test_two_resolvers_share_no_state(recorded_catalog):
    """Two sessions may point at different instances; one's shape is not the other's."""
    left = CountingCatalog(recorded_catalog)
    right = CountingCatalog(recorded_catalog)
    TableShapeResolver(left).resolve(SHAPE_SCHEMA, CANONICAL)
    resolver = TableShapeResolver(right)
    resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    assert right.calls == [(SHAPE_SCHEMA, CANONICAL)]
    assert resolver.reads == 1


def test_absent_table_raises_table_not_found():
    resolver = TableShapeResolver(MissingCatalog())
    with pytest.raises(TableNotFound) as exc:
        resolver.resolve("RAG", "NoSuchTable")
    message = str(exc.value)
    assert "RAG" in message and "NoSuchTable" in message


def test_absent_table_is_not_cached_as_a_failure():
    """A table created after a failed lookup must be visible on the next attempt."""

    class AppearingCatalog:
        def __init__(self, inner):
            self.inner = inner
            self.exists = False

        def snapshot(self, schema, table):
            return self.inner.snapshot(SHAPE_SCHEMA, CANONICAL) if self.exists else None

    catalog = AppearingCatalog(RecordedCatalog.from_fixture_dir(_fixture_dir()))
    resolver = TableShapeResolver(catalog)
    with pytest.raises(TableNotFound):
        resolver.resolve(SHAPE_SCHEMA, CANONICAL)
    catalog.exists = True
    assert resolver.resolve(SHAPE_SCHEMA, CANONICAL).is_canonical


def test_driver_failure_surfaces_as_catalog_unavailable():
    boom = RuntimeError("SQLCODE -461")
    resolver = TableShapeResolver(BrokenCatalog(boom))
    with pytest.raises(CatalogUnavailable) as exc:
        resolver.resolve("RAG", "SourceDocuments")
    assert exc.value.__cause__ is boom
    assert "RAG" in str(exc.value) and "SourceDocuments" in str(exc.value)


def test_a_driver_failure_is_never_turned_into_a_default_shape():
    resolver = TableShapeResolver(BrokenCatalog(RuntimeError("connection reset")))
    with pytest.raises(CatalogUnavailable):
        resolver.resolve("RAG", "SourceDocuments")
    assert resolver.reads == 0


def _fixture_dir():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "fixtures" / "shapes"
