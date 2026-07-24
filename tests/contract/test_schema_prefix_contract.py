"""
Contract test for configurable schema prefix with a real IRIS instance.

Feature: 079-schema-prefix
Requires: IRIS running at localhost:51972 (iris-vector-rag-iris container).

Tests that SchemaManager uses IRIS_SCHEMA_PREFIX env var and schema_prefix
constructor arg to create tables under the correct SQL schema, not 'RAG'.
"""

import os
import uuid

import pytest

pytestmark = [pytest.mark.requires_database, pytest.mark.contract]


@pytest.fixture
def unique_prefix():
    """Generate a unique schema prefix for this test run to avoid collisions."""
    suffix = uuid.uuid4().hex[:6].upper()
    return f"TIVR{suffix}"


@pytest.fixture
def schema_manager_factory():
    """Factory that constructs a SchemaManager with a given prefix using real IRIS."""
    from iris_vector_rag.storage.schema_manager import SchemaManager
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.config.manager import ConfigurationManager

    managers = []

    def _make(prefix):
        cfg = ConfigurationManager()
        conn_mgr = ConnectionManager(cfg)
        sm = SchemaManager(conn_mgr, cfg, schema_prefix=prefix)
        managers.append((sm, conn_mgr))
        return sm

    yield _make

    # Cleanup: close connections
    for sm, conn_mgr in managers:
        try:
            conn_mgr.close_connection()
        except Exception:
            pass


class TestSchemaPrefixContract:
    """Contract tests for schema prefix with real IRIS."""

    def test_schema_prefix_attribute_set_correctly(
        self, unique_prefix, schema_manager_factory
    ):
        """SchemaManager.schema_prefix is set to the supplied value."""
        sm = schema_manager_factory(unique_prefix)
        assert sm.schema_prefix == unique_prefix

    def test_qn_generates_correct_qualified_name(
        self, unique_prefix, schema_manager_factory
    ):
        """_qn() generates prefix.TableName with real prefix."""
        sm = schema_manager_factory(unique_prefix)
        assert sm._qn("SourceDocuments") == f"{unique_prefix}.SourceDocuments"
        assert sm._qn("SchemaMetadata") == f"{unique_prefix}.SchemaMetadata"
        assert sm._qn("Entities") == f"{unique_prefix}.Entities"

    def test_env_var_prefix_picked_up_by_config_manager(self, unique_prefix):
        """ConfigurationManager.get_schema_prefix() reads IRIS_SCHEMA_PREFIX."""
        from iris_vector_rag.config.manager import ConfigurationManager

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("IRIS_SCHEMA_PREFIX", unique_prefix)
            cfg = ConfigurationManager()
            assert cfg.get_schema_prefix() == unique_prefix

    def test_env_var_prefix_used_by_schema_manager(self, unique_prefix):
        """SchemaManager reads IRIS_SCHEMA_PREFIX from environment."""
        from iris_vector_rag.storage.schema_manager import SchemaManager
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag.config.manager import ConfigurationManager

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("IRIS_SCHEMA_PREFIX", unique_prefix)
            cfg = ConfigurationManager()
            conn_mgr = ConnectionManager(cfg)
            sm = SchemaManager(conn_mgr, cfg)
            assert sm.schema_prefix == unique_prefix

    def test_ensure_schema_metadata_table_uses_prefix(
        self, unique_prefix, schema_manager_factory
    ):
        """
        ensure_schema_metadata_table() creates a table under the prefix schema.

        We call it and then query INFORMATION_SCHEMA to verify the table
        was created under unique_prefix, not 'RAG'.
        """
        sm = schema_manager_factory(unique_prefix)
        sm.ensure_schema_metadata_table()

        # Query INFORMATION_SCHEMA to verify table exists under unique_prefix
        conn = sm.connection_manager.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE UPPER(TABLE_SCHEMA) = UPPER(?) AND UPPER(TABLE_NAME) = 'SCHEMAMETADATA'
                """,
                [unique_prefix],
            )
            count = cursor.fetchone()[0]
        finally:
            cursor.close()
            # Drop the test table to clean up
            try:
                cleanup_cursor = conn.cursor()
                cleanup_cursor.execute(
                    f"DROP TABLE IF EXISTS {sm._qn('SchemaMetadata')}"
                )
                conn.commit()
                cleanup_cursor.close()
            except Exception:
                pass

        assert count > 0, (
            f"Expected SchemaMetadata table under schema '{unique_prefix}' "
            f"but found {count} tables"
        )
