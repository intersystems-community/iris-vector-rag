"""The similarity measure and the table's HNSW index (feature 092, US5: FR-013).

IRIS requires an HNSW index's ``Distance`` to match the query function, so an
attached table searched with a different measure is searched without its index. The
store says so, once per table.
"""

from __future__ import annotations

import logging
from pathlib import Path

from iris_vector_rag.storage.catalog import IRISCatalogReader, RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from tests.unit.test_byot_store_vector import QUERY, STAFETA, TeamCursor, team_store

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "byot" / "catalog"

#: %Dictionary.CompiledIndex rows as IRIS 2026.3 returned them for Štafeta's table.
STF_INDEX_ROWS = [
    ("DocChunkDocRef", "DocRef", None, None),
    (
        "DocChunkHNSW",
        "Embedding",
        "%SQL.Index.HNSW",
        "Distance,Cosine,M,16,ProjectMap,1,SUPPORTSPARTITIONING,0,"
        "SUPPORTSSHARDING,0,efConstruction,64",
    ),
    ("DocChunkText", "Text", "%iFind.Index.Basic", "FILER,%iFind.Filer.Basic"),
]


class IndexCursor:
    def __init__(self, rows):
        self.rows, self.executed = rows, []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class Connection:
    def __init__(self, rows):
        self.cursor_ = IndexCursor(rows)

    def cursor(self):
        return self.cursor_


def test_the_reader_finds_the_hnsw_distance_of_a_column():
    reader = IRISCatalogReader(Connection(STF_INDEX_ROWS))
    assert reader.hnsw_index("BYOTStf.DocChunk", "Embedding") == (
        "DocChunkHNSW",
        "Cosine",
    )
    assert reader.hnsw_index("BYOTStf.DocChunk", "Text") is None


def test_a_column_with_underscores_matches_its_property_name():
    rows = [("EmbIdx", "chunkembedding", "%SQL.Index.HNSW", "Distance,DotProduct")]
    reader = IRISCatalogReader(Connection(rows))
    assert reader.hnsw_index("RAG.Chunks", "chunk_embedding") == (
        "EmbIdx",
        "DotProduct",
    )


class HnswCatalog(RecordedCatalog):
    """The recorded team catalogs, plus Štafeta's HNSW index."""

    def __init__(self, distance="Cosine"):
        base = RecordedCatalog.from_fixture_dir(CATALOG)
        super().__init__({(s, t): base.snapshot(s, t) for s, t in base.tables})
        self.distance = distance
        self.asked = 0

    def hnsw_index(self, class_name, column):
        self.asked += 1
        if class_name == "BYOTStf.DocChunk" and column == "Embedding":
            return ("DocChunkHNSW", self.distance) if self.distance else None
        return None


def store_over(catalog, cursor, **kwargs):
    store = team_store(STAFETA, cursor, **kwargs)
    store.shapes = TableShapeResolver(catalog)
    return store


def test_a_mismatch_is_warned_about_once(caplog):
    catalog = HnswCatalog("Cosine")
    store = store_over(catalog, TeamCursor(), metric="dot")
    with caplog.at_level(logging.WARNING):
        store.similarity_search_by_embedding(QUERY, 3)
        store.similarity_search_by_embedding(QUERY, 3)
    warnings = [r.getMessage() for r in caplog.records if "HNSW" in r.getMessage()]
    assert len(warnings) == 1
    assert "DocChunkHNSW" in warnings[0] and "Cosine" in warnings[0]
    assert "dot" in warnings[0]
    assert catalog.asked == 1


def test_a_match_or_no_index_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        store_over(HnswCatalog("Cosine"), TeamCursor()).similarity_search_by_embedding(
            QUERY, 3
        )
        store_over(
            HnswCatalog(None), TeamCursor(), metric="dot"
        ).similarity_search_by_embedding(QUERY, 3)
    assert not [r for r in caplog.records if "HNSW" in r.getMessage()]


def test_a_dot_product_index_matches_metric_dot(caplog):
    with caplog.at_level(logging.WARNING):
        store_over(
            HnswCatalog("DotProduct"), TeamCursor(), metric="dot"
        ).similarity_search_by_embedding(QUERY, 3)
    assert not [r for r in caplog.records if "HNSW" in r.getMessage()]


def test_a_reader_without_the_question_is_not_asked():
    # The plain recorded catalog has no hnsw_index: no check, no error.
    store = team_store(STAFETA, TeamCursor(), metric="dot")
    store.similarity_search_by_embedding(QUERY, 3)


def test_the_library_s_own_tables_are_checked_too(recorded_catalog, caplog):
    """Review decision (2026-10-24): an unused index is a performance bug on any table."""
    from tests.unit.test_vector_store_shape import FakeCursor, make_store

    class OwnHnsw(RecordedCatalog):
        def __init__(self, base):
            super().__init__({(s, t): base.snapshot(s, t) for s, t in base.tables})

        def hnsw_index(self, class_name, column):
            return ("OwnHNSW", "Cosine")

    store = make_store("canonical", FakeCursor(), recorded_catalog)
    store.shapes = TableShapeResolver(OwnHnsw(recorded_catalog))
    with caplog.at_level(logging.WARNING):
        store.similarity_search_by_embedding([0.1] * 384, 3)
    warnings = [r.getMessage() for r in caplog.records if "HNSW" in r.getMessage()]
    assert len(warnings) == 1 and "OwnHNSW" in warnings[0] and "'dot'" in warnings[0]
