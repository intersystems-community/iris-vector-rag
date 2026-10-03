"""HNSW vector index ensure, with the catalog answering as a catalog (feature 082).

What these tests used to do: hand `ensure_vector_hnsw_index` a `Mock` cursor whose
`fetchone()` returned `[1]`, then assert the SQL text of the existence check --
`SELECT COUNT(*)`, `INFORMATION_SCHEMA.INDEXES`. That pinned a query that now
lives in `catalog.py`, and it described the answer in the shape of a row rather
than as the thing the caller wants to know: which indexes exist.

So the catalog is replaced at the seam the index path actually uses
(`IRISCatalogReader.index_names`) and the answer is a set of names (FR-015). The
cursor is still real work here -- it is the DDL channel, and the exact
`CREATE INDEX` text is the behaviour under test.
"""

from unittest.mock import Mock, patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.exceptions import SchemaIndexError
from iris_vector_rag.storage.catalog import CatalogSnapshot
from iris_vector_rag.storage.schema_manager import SchemaManager


class RecordedIndexes:
    """The catalog's answer about a table's indexes, as `IRISCatalogReader` gives it.

    An instance is both the reader and its factory: the index path constructs
    `IRISCatalogReader(connection)`, so calling the instance returns itself and
    records the connection it was handed.
    """

    def __init__(self, *names: str) -> None:
        self.names = {name.upper() for name in names}
        self.asked: list[tuple[str, object]] = []

    def __call__(self, connection):
        self.connection = connection
        return self

    def index_names(self, schema: str, table: str, cursor=None) -> set[str]:
        self.asked.append((schema, table, cursor))
        return set(self.names)


class RecordingCursor:
    """Records the statements it is given, and can fail one of them."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.statements: list[str] = []
        self.closed = False
        self._fail_on = fail_on

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if self._fail_on and self._fail_on in sql:
            raise RuntimeError(f"{self._fail_on} not supported by this instance")

    def close(self):
        self.closed = True

    @property
    def created(self) -> list[str]:
        return [sql for sql in self.statements if "CREATE INDEX" in sql]


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def manager():
    """A schema manager whose configuration is real and whose database is not.

    The dimension is passed explicitly rather than mocked: a `Mock` config manager
    yields a `Mock` dimension, which the constructor now rejects outright.
    """
    with patch.object(SchemaManager, "ensure_schema_metadata_table"):
        schema_manager = SchemaManager(
            Mock(), ConfigurationManager(), base_embedding_dimension=384
        )
    # Both vector tables exist and carry their vector column; the catalog is not under test.
    schema_manager._snapshot = lambda table, **kwargs: CatalogSnapshot(
        schema="RAG", table=table, columns=[{"COLUMN_NAME": "embedding"}]
    )
    return schema_manager


class TestEnsureHNSWIndex:
    def test_existing_index_is_left_alone(self, manager):
        """Idempotent: an index the catalog already reports is not re-created."""
        catalog = RecordedIndexes("IDX_SOURCEDOCUMENTS_EMBEDDING")
        cursor = RecordingCursor()

        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", catalog):
            manager.ensure_vector_hnsw_index(
                cursor,
                "RAG.SourceDocuments",
                "embedding",
                "idx_SourceDocuments_embedding",
            )

        assert cursor.statements == []
        # The table's own schema, and the caller's own cursor: the answer has to come
        # from inside the transaction the DDL will run in (ST-05).
        assert catalog.asked == [("RAG", "SourceDocuments", cursor)]

    def test_acorn_is_tried_first(self, manager):
        """A missing index is created with the ACORN=1 optimisation."""
        cursor = RecordingCursor()

        with patch(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            RecordedIndexes(),
        ):
            manager.ensure_vector_hnsw_index(
                cursor, "RAG.Entities", "embedding", "idx_Entities_embedding"
            )

        assert cursor.statements == [
            "CREATE INDEX idx_Entities_embedding ON RAG.Entities(embedding) "
            "AS HNSW WITH (ACORN=1)"
        ]

    def test_acorn_failure_falls_back_to_standard_hnsw(self, manager):
        """An instance without ACORN still gets an index, on the second statement."""
        cursor = RecordingCursor(fail_on="ACORN=1")

        with patch(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            RecordedIndexes(),
        ):
            manager.ensure_vector_hnsw_index(
                cursor,
                "RAG.SourceDocuments",
                "embedding",
                "idx_SourceDocuments_embedding",
            )

        assert cursor.statements == [
            "CREATE INDEX idx_SourceDocuments_embedding ON RAG.SourceDocuments(embedding) "
            "AS HNSW WITH (ACORN=1)",
            "CREATE INDEX idx_SourceDocuments_embedding ON RAG.SourceDocuments(embedding) "
            "AS HNSW",
        ]

    def test_try_acorn_false_skips_the_optimisation(self, manager):
        cursor = RecordingCursor()

        with patch(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            RecordedIndexes(),
        ):
            manager.ensure_vector_hnsw_index(
                cursor,
                "RAG.Entities",
                "embedding",
                "idx_Entities_embedding",
                try_acorn=False,
            )

        assert cursor.statements == [
            "CREATE INDEX idx_Entities_embedding ON RAG.Entities(embedding) AS HNSW"
        ]

    def test_a_catalog_failure_reaches_the_caller(self, manager):
        """The index check is not allowed to fail quietly into "create it anyway"."""
        broken = Mock(side_effect=RuntimeError("catalog unavailable"))
        cursor = RecordingCursor()

        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", broken):
            with pytest.raises(RuntimeError, match="catalog unavailable"):
                manager.ensure_vector_hnsw_index(
                    cursor, "RAG.Entities", "embedding", "idx_Entities_embedding"
                )

        assert cursor.statements == []


class TestEnsureAllVectorIndexes:
    def test_both_vector_tables_get_an_index(self, manager):
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        manager.connection_manager.get_connection.return_value = connection

        with patch(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            RecordedIndexes(),
        ):
            manager.ensure_all_vector_indexes()

        assert cursor.created == [
            "CREATE INDEX idx_SourceDocuments_embedding ON RAG.SourceDocuments(embedding) "
            "AS HNSW WITH (ACORN=1)",
            "CREATE INDEX idx_Entities_embedding ON RAG.Entities(embedding) "
            "AS HNSW WITH (ACORN=1)",
        ]
        assert connection.commits == 1
        assert cursor.closed

    def test_existing_indexes_produce_no_ddl(self, manager):
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        manager.connection_manager.get_connection.return_value = connection
        catalog = RecordedIndexes(
            "IDX_SOURCEDOCUMENTS_EMBEDDING", "IDX_ENTITIES_EMBEDDING"
        )

        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", catalog):
            manager.ensure_all_vector_indexes()

        assert cursor.statements == []
        assert connection.commits == 1

    def test_a_failure_rolls_back_closes_and_reaches_the_caller(self, manager):
        """ST-06: it used to log, roll back and return None, so nobody learned."""
        cursor = RecordingCursor(fail_on="CREATE INDEX")
        connection = RecordingConnection(cursor)
        manager.connection_manager.get_connection.return_value = connection

        with patch(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            RecordedIndexes(),
        ):
            with pytest.raises(SchemaIndexError):
                manager.ensure_all_vector_indexes()

        assert connection.rollbacks == 1
        assert connection.commits == 0
        assert cursor.closed
