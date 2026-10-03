"""Storage correctness (ST-04 to ST-07, ST-10 to ST-12, ST-16 to ST-18 of the 2026-10 review).

Pure unit tests: a recording cursor stands in for the database only where the
statement text or the call order is the behaviour under test. The IRIS gate is in
tests/integration/test_storage_correctness_iris.py.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.exceptions import (
    DestructiveMigrationError,
    SchemaIndexError,
    VectorStoreConfigurationError,
)
from iris_vector_rag.storage.catalog import INDEX_SQL, CatalogSnapshot
from iris_vector_rag.storage.clob_handler import (
    ensure_string_content,
    process_document_row,
)
from iris_vector_rag.storage.metadata_filter_manager import MetadataFilterManager
from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore


class Cursor:
    """Records statements; `rows` maps a statement fragment to the row it fetches."""

    def __init__(self, rows=None, fail_on=None, rowcount=1):
        self.statements: list[str] = []
        self.rows = rows or {}
        self.fail_on = fail_on
        self.rowcount = rowcount
        self._last = None
        self.closed = False

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError(f"{self.fail_on} failed")
        self._last = sql

    def fetchone(self):
        for fragment, row in self.rows.items():
            if self._last and fragment in self._last:
                return row
        return None

    def close(self):
        self.closed = True

    @property
    def drops(self):
        return [s for s in self.statements if s.lstrip().upper().startswith("DROP")]


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _manager(prefix="RAG"):
    with patch.object(SchemaManager, "ensure_schema_metadata_table"):
        manager = SchemaManager(
            Mock(),
            ConfigurationManager(),
            schema_prefix=prefix,
            base_embedding_dimension=8,
        )
    # Every table "exists" unless a test says otherwise; the catalog is not under test.
    manager._snapshot = lambda table, **kwargs: CatalogSnapshot(
        schema=prefix, table=table, columns=[{"COLUMN_NAME": "embedding"}]
    )
    return manager


# --- ST-04: migrate_table refuses to drop a populated table ---------------------------


class TestMigrateTableRefusesDestruction:
    @pytest.mark.parametrize(
        "table",
        [
            "SourceDocuments",
            "DocumentChunks",
            "Entities",
            "EntityRelationships",
            "rdf_labels",
            "rdf_props",
            "rdf_edges",
            "kg_NodeEmbeddings_optimized",
        ],
    )
    def test_populated_table_is_refused_and_nothing_is_dropped(self, table):
        manager = _manager()
        cursor = Cursor(rows={"SELECT TOP 1": (1,)})
        connection = Connection(cursor)
        manager.connection_manager.get_connection.return_value = connection

        with pytest.raises(DestructiveMigrationError, match="allow_destructive"):
            manager.migrate_table(table)

        assert cursor.drops == []
        assert connection.commits == 0

    def test_preserve_data_does_not_unlock_it(self):
        """`preserve_data` was never implemented; it must not read as a promise."""
        manager = _manager()
        cursor = Cursor(rows={"SELECT TOP 1": (1,)})
        manager.connection_manager.get_connection.return_value = Connection(cursor)

        with pytest.raises(DestructiveMigrationError):
            manager.migrate_table("Entities", preserve_data=True)
        assert cursor.drops == []

    def test_dropping_source_documents_also_checks_the_chunks_it_takes_with_it(self):
        manager = _manager()
        # SourceDocuments is empty, DocumentChunks is not
        cursor = Cursor(rows={"DocumentChunks": (1,)})
        manager.connection_manager.get_connection.return_value = Connection(cursor)

        with pytest.raises(DestructiveMigrationError, match="DocumentChunks"):
            manager.migrate_table("SourceDocuments")
        assert cursor.drops == []

    def test_a_table_that_cannot_be_counted_is_refused_not_assumed_empty(self):
        manager = _manager()
        cursor = Cursor(fail_on="SELECT TOP 1")
        manager.connection_manager.get_connection.return_value = Connection(cursor)
        # a missing table is "empty"; any other failure must be loud
        with patch.object(manager, "_snapshot", return_value=object()):
            with pytest.raises(RuntimeError):
                manager.migrate_table("Entities")
        assert cursor.drops == []

    def test_allow_destructive_lets_a_populated_table_through(self):
        manager = _manager()
        cursor = Cursor(rows={"SELECT TOP 1": (1,)})
        manager.connection_manager.get_connection.return_value = Connection(cursor)

        with patch.object(manager, "_migrate_entities_table", return_value=True) as m:
            assert manager.migrate_table("Entities", allow_destructive=True) is True
        m.assert_called_once()

    def test_an_absent_or_empty_table_needs_no_flag(self):
        manager = _manager()
        cursor = Cursor()  # SELECT TOP 1 fetches nothing
        manager.connection_manager.get_connection.return_value = Connection(cursor)
        with patch.object(manager, "_snapshot", return_value=None):
            with patch.object(
                manager, "_migrate_entities_table", return_value=True
            ) as m:
                assert manager.migrate_table("Entities") is True
        m.assert_called_once()

    def test_a_failed_migration_reports_false_and_rolls_back(self):
        manager = _manager()
        cursor = Cursor()
        connection = Connection(cursor)
        manager.connection_manager.get_connection.return_value = connection
        with patch.object(manager, "_snapshot", return_value=None):
            with patch.object(manager, "_migrate_entities_table", return_value=False):
                assert manager.migrate_table("Entities") is False
        assert connection.rollbacks == 1


class TestMigrationNamesAreQualified:
    """ST-04: no migration creates or drops a table in the connecting user's schema."""

    @pytest.mark.parametrize(
        "method,table",
        [
            ("_migrate_rdf_labels_table", "rdf_labels"),
            ("_migrate_rdf_props_table", "rdf_props"),
            ("_migrate_rdf_edges_table", "rdf_edges"),
            (
                "_migrate_kg_node_embeddings_optimized_table",
                "kg_NodeEmbeddings_optimized",
            ),
            ("_migrate_entities_table", "Entities"),
            ("_migrate_entity_relationships_table", "EntityRelationships"),
        ],
    )
    def test_every_ddl_statement_names_the_schema(self, method, table):
        manager = _manager("MYAPP")
        cursor = Cursor()
        config = {"vector_dimension": 8, "vector_data_type": "FLOAT"}
        with patch.object(manager, "ensure_vector_hnsw_index"):
            getattr(manager, method)(cursor, config, False)

        ddl = [
            s
            for s in cursor.statements
            if s.lstrip().upper().startswith(("DROP TABLE", "CREATE TABLE"))
        ]
        assert ddl, "the migration ran no DDL"
        for statement in ddl:
            assert "MYAPP." in statement, statement

    def test_a_failed_create_does_not_retry_on_an_unqualified_name(self):
        manager = _manager("MYAPP")
        cursor = Cursor(fail_on="CREATE TABLE MYAPP.Entities")
        config = {"vector_dimension": 8, "vector_data_type": "FLOAT"}
        assert manager._migrate_entities_table(cursor, config, False) is False
        assert not [s for s in cursor.statements if "TABLE Entities" in s]
        assert not [s for s in cursor.statements if "TABLE IF EXISTS Entities" in s]


# --- ST-17: SchemaMetadata and table_exists honour the prefix ---------------------------


class TestSchemaPrefixHonoured:
    def test_table_exists_uses_the_managers_schema_for_a_bare_name(self):
        manager = _manager("MYAPP")
        with patch.object(manager, "_snapshot", return_value=object()) as snap:
            assert manager.table_exists("Entities") is True
        snap.assert_called_once_with("Entities", schema="MYAPP")

    def test_table_exists_still_honours_a_dotted_name(self):
        manager = _manager("MYAPP")
        with patch.object(manager, "_snapshot", return_value=None) as snap:
            assert manager.table_exists("Other.Entities") is False
        snap.assert_called_once_with("Entities", schema="Other")

    def test_schema_metadata_is_never_created_in_the_current_users_schema(self):
        manager = _manager("MYAPP")
        manager._initialized = False
        cursor = Cursor(fail_on="MYAPP.SchemaMetadata")
        manager.connection_manager.get_connection.return_value = Connection(cursor)
        manager.ensure_schema_metadata_table()
        creates = [s for s in cursor.statements if "CREATE TABLE" in s]
        assert len(creates) == 1 and "MYAPP.SchemaMetadata" in creates[0]


# --- ST-05 to ST-07, ST-18 ---------------------------------------------------------------


class TestIndexEnsuringReportsTruth:
    def test_index_sql_is_scoped_by_schema(self):
        assert "TABLE_SCHEMA" in INDEX_SQL

    def test_ensure_vector_hnsw_index_asks_for_the_tables_own_schema(self):
        manager = _manager("MYAPP")
        asked = []

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                asked.append((schema, table))
                return set()

        cursor = Cursor()
        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            manager.ensure_vector_hnsw_index(
                cursor, "MYAPP.Entities", "embedding", "idx_Entities_embedding"
            )
        assert asked == [("MYAPP", "Entities")]

    def test_ensure_all_vector_indexes_raises_when_an_index_cannot_be_made(self):
        manager = _manager()
        cursor = Cursor(fail_on="CREATE INDEX")
        connection = Connection(cursor)
        manager.connection_manager.get_connection.return_value = connection

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                return set()

        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            with pytest.raises(SchemaIndexError, match="idx_SourceDocuments_embedding"):
                manager.ensure_all_vector_indexes()
        assert connection.rollbacks == 1
        assert cursor.closed

    def test_ensure_all_vector_indexes_returns_the_names_it_ensured(self):
        manager = _manager()
        cursor = Cursor()
        manager.connection_manager.get_connection.return_value = Connection(cursor)

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                return set()

        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            ensured = manager.ensure_all_vector_indexes()
        assert ensured == ["idx_SourceDocuments_embedding", "idx_Entities_embedding"]

    def test_a_table_without_the_vector_column_is_left_out_not_reported(self):
        manager = _manager()
        cursor = Cursor()
        manager.connection_manager.get_connection.return_value = Connection(cursor)
        manager._snapshot = lambda table, **kw: CatalogSnapshot(
            schema="RAG",
            table=table,
            columns=[
                {"COLUMN_NAME": "embedding" if table == "SourceDocuments" else "x"}
            ],
        )

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                return set()

        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            assert manager.ensure_all_vector_indexes() == [
                "idx_SourceDocuments_embedding"
            ]

    def test_a_missing_table_is_an_error(self):
        manager = _manager()
        manager.connection_manager.get_connection.return_value = Connection(Cursor())
        manager._snapshot = lambda table, **kw: None
        with pytest.raises(SchemaIndexError, match="does not exist"):
            manager.ensure_all_vector_indexes()

    def test_ensure_index_creates_a_btree_index_on_the_named_table(self):
        manager = _manager("MYAPP")
        cursor = Cursor()
        connection = Connection(cursor)
        manager.connection_manager.get_connection.return_value = connection

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                return set()

        config = {
            "name": "idx_rel_source",
            "table": "EntityRelationships",
            "columns": ["source_entity_id"],
        }
        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            assert manager._ensure_index(config) is True
        assert cursor.statements == [
            "CREATE INDEX idx_rel_source ON MYAPP.EntityRelationships (source_entity_id)"
        ]
        assert connection.commits == 1
        assert cursor.closed

    def test_ensure_index_leaves_an_existing_index_alone(self):
        manager = _manager("MYAPP")
        cursor = Cursor()
        manager.connection_manager.get_connection.return_value = Connection(cursor)

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                return {"IDX_REL_SOURCE"}

        config = {
            "name": "idx_rel_source",
            "table": "EntityRelationships",
            "columns": ["source_entity_id"],
        }
        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            assert manager._ensure_index(config) is True
        assert cursor.statements == []

    def test_ensure_index_without_a_table_raises(self):
        manager = _manager()
        with pytest.raises(SchemaIndexError, match="table"):
            manager._ensure_index({"name": "idx_x", "columns": ["a"]})

    def test_ensure_index_reports_false_when_the_create_fails_and_closes_the_cursor(
        self,
    ):
        manager = _manager()
        cursor = Cursor(fail_on="CREATE INDEX")
        connection = Connection(cursor)
        manager.connection_manager.get_connection.return_value = connection

        class Reader:
            def __init__(self, connection):
                pass

            def index_names(self, schema, table, cursor=None):
                return set()

        config = {"name": "idx_x", "table": "Entities", "columns": ["a"]}
        with patch("iris_vector_rag.storage.schema_manager.IRISCatalogReader", Reader):
            assert manager._ensure_index(config) is False
        assert cursor.closed
        assert connection.rollbacks == 1

    def test_ensure_index_rejects_an_unsafe_identifier(self):
        manager = _manager()
        with pytest.raises(SchemaIndexError):
            manager._ensure_index(
                {"name": "x; DROP TABLE y", "table": "Entities", "columns": ["a"]}
            )

    def test_ensure_index_hnsw_goes_through_the_hnsw_path(self):
        manager = _manager("MYAPP")
        cursor = Cursor()
        manager.connection_manager.get_connection.return_value = Connection(cursor)
        config = {
            "name": "idx_e_emb",
            "table": "Entities",
            "columns": ["embedding"],
            "type": "HNSW",
        }
        with patch.object(manager, "ensure_vector_hnsw_index") as hnsw:
            assert manager._ensure_index(config) is True
        hnsw.assert_called_once_with(
            cursor, "MYAPP.Entities", "embedding", "idx_e_emb", try_acorn=True
        )

    def test_pipeline_schema_fails_when_a_required_index_is_not_made(self):
        manager = _manager()
        manager.config_manager = MagicMock()
        manager.config_manager.get_pipeline_requirements.return_value = {
            "schema_requirements": {
                "tables": [],
                "indexes": [{"name": "idx_x", "table": "Entities", "columns": ["a"]}],
            }
        }
        with patch.object(manager, "_ensure_index", return_value=False):
            assert manager.ensure_pipeline_schema("basic") is False


class TestDdlConforms:
    def test_no_text_columns_in_the_canonical_migrations(self):
        import inspect

        source = inspect.getsource(SchemaManager)
        import re

        assert not re.search(
            r"\bTEXT\s+(NULL|NOT NULL|,)", source
        ), "project rule: varchar(N), not TEXT"


# --- ST-10 to ST-12 ------------------------------------------------------------------------


class TestClobDictsKeepTheirTypes:
    def test_process_document_row_keeps_scalars(self):
        row = {"chunk_index": 3, "ratio": 1.5, "flag": True, "missing": None}
        assert process_document_row(row) == row

    def test_process_document_row_keeps_nested_types(self):
        row = {"inner": {"n": 2}, "many": [1, "a", {"k": 4}]}
        assert process_document_row(row) == row

    def test_clob_values_still_become_strings(self):
        row = {"text": io.BytesIO(b"hello"), "raw": b"bytes", "n": 7}
        out = process_document_row(row)
        assert out == {"text": "hello", "raw": "bytes", "n": 7}

    def test_ensure_string_content_keeps_metadata_types(self):
        out = ensure_string_content(
            {"page_content": io.BytesIO(b"body"), "metadata": {"chunk_index": 3}}
        )
        assert out["page_content"] == "body"
        assert out["metadata"] == {"chunk_index": 3}


class TestFilterKeys:
    @pytest.mark.parametrize(
        "key", ["category", "priority", "organization", "brand", "order_id"]
    )
    def test_legitimate_keys_containing_or_and_are_accepted(self, key):
        MetadataFilterManager({"storage": {"iris": {"custom_filter_keys": [key]}}})

    @pytest.mark.parametrize(
        "key", ["select", "DROP", "or", "And", "union", "1abc", "a-b", "a;b", "a b"]
    )
    def test_keywords_and_malformed_keys_are_still_rejected(self, key):
        with pytest.raises(VectorStoreConfigurationError):
            MetadataFilterManager({"storage": {"iris": {"custom_filter_keys": [key]}}})


class TestTableNames:
    def _validate(self, name):
        IRISVectorStore.__new__(IRISVectorStore)._validate_table_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "RAG.Descriptions",
            "RAG.Forecasts",
            "MYAPP.Updates_2024",
            "RAG.Scripts",
            "RAG.Castings",
            "RAG.SourceDocuments",
        ],
    )
    def test_valid_names_are_accepted(self, name):
        self._validate(name)

    @pytest.mark.parametrize(
        "name",
        [
            "RAG.x; DROP TABLE y",
            "RAG.docs--",
            "RAG.a b",
            "RAG.docs/*x*/",
            "RAG.docs) UNION SELECT 1",
            "RAG.drop",
            "drop.docs",
            "RAG",
            "RAG.1docs",
            "RAG.a.b",
            "",
        ],
    )
    def test_injection_attempts_and_malformed_names_are_rejected(self, name):
        with pytest.raises(VectorStoreConfigurationError):
            self._validate(name)


# --- ST-16 ------------------------------------------------------------------------------------


class TestRelationEmbeddingStore:
    def _store(self, cursor, prefix=None):
        conn = Connection(cursor)
        connections = MagicMock()
        connections.get_connection.return_value = conn
        config = MagicMock()
        config.get_schema_prefix.return_value = prefix or "RAG"
        store = RelationEmbeddingStore(connections, config)
        store._dimension = 3
        emb = MagicMock()
        emb.embed_text.return_value = [0.1, 0.2, 0.3]
        store._get_embedding_manager = lambda: emb
        return store, conn

    def test_a_zero_row_update_raises_and_rolls_back(self):
        cursor = Cursor(rowcount=0)
        store, conn = self._store(cursor)
        with pytest.raises(LookupError, match="missing-id"):
            store.embed_and_store("missing-id", "KNOWS", "a", "b")
        assert conn.commits == 0
        assert conn.rollbacks == 1
        assert cursor.closed

    def test_a_one_row_update_commits(self):
        cursor = Cursor(rowcount=1)
        store, conn = self._store(cursor)
        store.embed_and_store("r1", "KNOWS", "a", "b")
        assert conn.commits == 1

    def test_sql_uses_the_configured_schema_not_rag(self):
        cursor = Cursor(rowcount=1)
        store, _ = self._store(cursor, prefix="MYAPP")
        store.embed_and_store("r1", "KNOWS", "a", "b")
        assert "MYAPP.EntityRelationships" in cursor.statements[0]
        assert "RAG." not in cursor.statements[0]

    def test_an_explicit_prefix_wins(self):
        connections, config = MagicMock(), MagicMock()
        config.get_schema_prefix.return_value = "FROMCONFIG"
        store = RelationEmbeddingStore(connections, config, schema_prefix="EXPLICIT")
        assert store.table == "EXPLICIT.EntityRelationships"

    def test_an_invalid_prefix_is_refused(self):
        with pytest.raises(ValueError):
            RelationEmbeddingStore(MagicMock(), MagicMock(), schema_prefix="x; drop")
