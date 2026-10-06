"""Attaching a corpus describes it once, from the catalog (feature 082, T054).

Two things `attach_existing_corpus` used to do for itself, and no longer does:

* it required the caller to name the identifier, text and embedding columns, so a
  caller who got the spelling wrong attached a corpus whose vector search silently
  returned nothing;
* it asked `INFORMATION_SCHEMA.INDEXES` whether the embedding column is indexed,
  which is the twelfth copy of a catalog query this feature reduced to one
  (FR-013, FR-015, SC-003).

The columns now default to the shape of the table being attached, and index
presence comes from `IRISCatalogReader`. Explicit column arguments still win: a
corpus in someone else's schema can carry any spelling at all, which is the whole
point of feature 069.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shapes"


class RecordingCursor:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(str(sql).split()))

    def fetchone(self):
        return None

    def close(self):
        pass


class RecordingEngine:
    """The iris_vector_graph engine, reduced to what attaching a corpus asks of it."""

    def __init__(self, dimension=384, row_count=100):
        self.cursor_obj = RecordingCursor()
        self.conn = Mock()
        self.conn.cursor.return_value = self.cursor_obj
        self._validation = {"dimension": dimension, "row_count": row_count}
        self.validated: list[tuple[str, str]] = []
        self.mapped: list[dict] = []

    def validate_vector_table(self, table, embedding_col):
        self.validated.append((table, embedding_col))
        return dict(self._validation)

    def map_sql_table(self, table, id_col, label, property_columns=None):
        self.mapped.append(
            {
                "table": table,
                "id_col": id_col,
                "label": label,
                "property_columns": property_columns,
            }
        )


class RecordedIndexes:
    """`IRISCatalogReader`'s answer about which columns are indexed.

    An instance is both the reader and its factory: the pipeline constructs
    `IRISCatalogReader(connection)`, so calling the instance returns itself.
    """

    def __init__(self, *columns: str) -> None:
        self.columns = {column.upper() for column in columns}
        self.asked: list[tuple[str, str]] = []

    def __call__(self, connection):
        self.connection = connection
        return self

    def indexed_columns(self, schema: str, table: str) -> set[str]:
        self.asked.append((schema, table))
        return set(self.columns)


def pipeline_with(engine: RecordingEngine) -> HybridGraphRAGPipeline:
    """A pipeline with the graph engine and the shapes, and nothing else.

    `__new__` is how `tests/test_attach_corpus.py` builds one too: the constructor
    connects to IRIS and imports iris_vector_graph, neither of which this asks about.
    """
    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    pipeline.iris_engine = engine
    pipeline._attached_corpora = {}
    pipeline.connection_manager = Mock()
    pipeline.config_manager = Mock()
    pipeline.shapes = TableShapeResolver(RecordedCatalog.from_fixture_dir(FIXTURE_DIR))
    return pipeline


def table(shape: str) -> str:
    return f"{SHAPE_SCHEMA}.{SHAPE_TABLES[shape]}"


def test_the_columns_default_to_the_shape(monkeypatch):
    """No column named by the caller: the table says which ones it has."""
    catalog = RecordedIndexes("EMBEDDING")
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader", catalog
    )
    engine = RecordingEngine()
    pipeline = pipeline_with(engine)

    result = pipeline.attach_existing_corpus(table("fixture"), graph_label="Doc")

    assert result["id_col"] == "doc_id"
    assert result["embedding_col"] == "embedding"
    # `content` on this table, which is also the column the store reads.
    assert engine.mapped[0]["property_columns"] == ["content"]
    assert engine.validated == [(table("fixture"), "embedding")]


def test_a_mixed_table_attaches_its_canonical_columns(monkeypatch):
    """`content` and `text_content` both present: the canonical one is attached."""
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader",
        RecordedIndexes("EMBEDDING"),
    )
    engine = RecordingEngine()

    pipeline_with(engine).attach_existing_corpus(table("mixed"), graph_label="Doc")

    assert engine.mapped[0]["id_col"] == "doc_id"
    assert engine.mapped[0]["property_columns"] == ["text_content"]


def test_explicit_columns_win(monkeypatch):
    """Feature 069's contract: any table, any spelling, named by the caller."""
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader",
        RecordedIndexes(),
    )
    engine = RecordingEngine()

    result = pipeline_with(engine).attach_existing_corpus(
        table("mixed"),
        id_col="id",
        text_col="content",
        embedding_col="embedding",
        graph_label="Doc",
    )

    assert result["id_col"] == "id"
    assert engine.mapped[0]["property_columns"] == ["content"]


def test_index_presence_comes_from_the_catalog_module(monkeypatch):
    """The pipeline asks; it does not write the query."""
    catalog = RecordedIndexes("EMBEDDING")
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader", catalog
    )
    engine = RecordingEngine()

    result = pipeline_with(engine).attach_existing_corpus(
        table("canonical"), graph_label="Doc"
    )

    assert result["has_hnsw_index"] is True
    assert catalog.asked == [(SHAPE_SCHEMA, SHAPE_TABLES["canonical"])]
    assert not any(
        "INFORMATION_SCHEMA" in sql for sql in engine.cursor_obj.statements
    ), engine.cursor_obj.statements


def test_an_unindexed_embedding_column_is_reported(monkeypatch):
    """Brute-force search is a warning, not a failure -- but it is reported."""
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader",
        RecordedIndexes("DOC_ID"),
    )
    engine = RecordingEngine()

    result = pipeline_with(engine).attach_existing_corpus(
        table("canonical"), graph_label="Doc"
    )

    assert result["has_hnsw_index"] is False


def test_a_catalog_failure_leaves_index_presence_unknown(monkeypatch):
    """`None` means "not established", which is what the AttachResult promises."""
    broken = Mock(side_effect=RuntimeError("catalog unavailable"))
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader", broken
    )
    engine = RecordingEngine()

    result = pipeline_with(engine).attach_existing_corpus(
        table("canonical"), graph_label="Doc"
    )

    assert result["has_hnsw_index"] is None


def test_a_table_without_an_embedding_column_is_refused(monkeypatch):
    """Nothing to attach a vector search to, said before any mapping happens."""
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.hybrid_graphrag.IRISCatalogReader",
        RecordedIndexes(),
    )
    engine = RecordingEngine()

    with pytest.raises(Exception, match="EMBEDDING"):
        pipeline_with(engine).attach_existing_corpus(
            table("legacy_id"), graph_label="Doc"
        )

    assert engine.mapped == []


def test_a_query_of_the_wrong_width_raises_the_dimension_error_with_both_widths():
    """`DimensionMismatchError` takes (got, expected); the attach check once passed a message."""
    from iris_vector_rag.exceptions import DimensionMismatchError

    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    pipeline._attached_corpora = {"Docs": {"dimension": 384}}

    with pytest.raises(DimensionMismatchError) as caught:
        pipeline._validate_query_dimension("Docs", [0.0] * 768)

    assert (caught.value.got, caught.value.expected) == (768, 384)
    pipeline._validate_query_dimension("Docs", [0.0] * 384)
    pipeline._validate_query_dimension("NotAttached", [0.0] * 5)
