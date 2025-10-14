"""
Tests for HNSW vector index ensure functionality.

Tests the automatic creation and management of HNSW indexes
with ACORN-1 optimization fallback.
"""

import logging
from unittest.mock import Mock, call, patch

import pytest

from iris_rag.storage.schema_manager import SchemaManager


class TestEnsureHNSWIndex:
    """Tests for ensure_vector_hnsw_index method."""

    @pytest.fixture
    def schema_manager(self):
        """Create a schema manager with mocked dependencies."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()

        # Mock config manager to return default values
        mock_config_manager.get.side_effect = lambda key, default=None: {
            "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model.dimension": 384,
            "storage:iris:vector_data_type": "FLOAT",
        }.get(key, default)

        with patch.object(SchemaManager, "ensure_schema_metadata_table"):
            manager = SchemaManager(mock_connection_manager, mock_config_manager)

        return manager

    def test_ensure_hnsw_index_idempotent(self, schema_manager):
        """Test that ensure_vector_hnsw_index is idempotent (no-op if already exists)."""
        # Setup mock cursor
        mock_cursor = Mock()

        # Mock index already exists
        mock_cursor.fetchone.return_value = [1]  # index count > 0

        # Call ensure_vector_hnsw_index
        schema_manager.ensure_vector_hnsw_index(
            mock_cursor,
            "RAG.SourceDocuments",
            "embedding",
            "idx_SourceDocuments_embedding",
        )

        # Verify check query was executed
        expected_check_sql = """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
                WHERE TABLE_NAME = ? AND INDEX_NAME = ?
            """
        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args[0]
        assert "SELECT COUNT(*)" in args[0]
        assert "INFORMATION_SCHEMA.INDEXES" in args[0]

        # Verify no CREATE INDEX was executed (since index already exists)
        create_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "CREATE INDEX" in str(call)
        ]
        assert len(create_calls) == 0

    def test_ensure_hnsw_index_fallback_without_acorn(self, schema_manager):
        """Test fallback to standard HNSW when ACORN syntax fails."""
        # Setup mock cursor
        mock_cursor = Mock()

        # Mock index doesn't exist
        mock_cursor.fetchone.return_value = [0]  # index count = 0

        # Mock ACORN syntax failure, then success on standard HNSW
        def execute_side_effect(sql, *args):
            if "ACORN=1" in sql:
                raise Exception("ACORN syntax not supported")
            # Standard HNSW succeeds
            return None

        mock_cursor.execute.side_effect = execute_side_effect

        # Call ensure_vector_hnsw_index
        schema_manager.ensure_vector_hnsw_index(
            mock_cursor,
            "RAG.SourceDocuments",
            "embedding",
            "idx_SourceDocuments_embedding",
        )

        # Verify both ACORN and standard HNSW were attempted
        execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]

        # Should have check, ACORN attempt, and standard HNSW
        assert len(mock_cursor.execute.call_args_list) == 3

        # Find the CREATE INDEX calls
        create_calls = [call for call in execute_calls if "CREATE INDEX" in call]
        assert len(create_calls) == 2

        # First should be ACORN attempt
        assert "ACORN=1" in create_calls[0]

        # Second should be standard HNSW
        assert "AS HNSW" in create_calls[1]
        assert "ACORN=1" not in create_calls[1]

    def test_ensure_hnsw_index_acorn_success(self, schema_manager):
        """Test successful creation with ACORN=1."""
        # Setup mock cursor
        mock_cursor = Mock()

        # Mock index doesn't exist
        mock_cursor.fetchone.return_value = [0]  # index count = 0

        # Call ensure_vector_hnsw_index
        schema_manager.ensure_vector_hnsw_index(
            mock_cursor, "RAG.Entities", "embedding", "idx_Entities_embedding"
        )

        # Verify check and ACORN creation
        assert mock_cursor.execute.call_count == 2

        # Check the CREATE INDEX call
        create_call = None
        for call_args in mock_cursor.execute.call_args_list:
            if "CREATE INDEX" in str(call_args):
                create_call = call_args[0][0]
                break

        assert create_call is not None
        assert (
            "CREATE INDEX idx_Entities_embedding ON RAG.Entities(embedding) AS HNSW WITH (ACORN=1)"
            == create_call
        )

    def test_ensure_hnsw_index_entities_table(self, schema_manager):
        """Test ensure_vector_hnsw_index with Entities table."""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [0]  # index doesn't exist

        schema_manager.ensure_vector_hnsw_index(
            mock_cursor,
            "RAG.Entities",
            "embedding",
            "idx_Entities_embedding",
            try_acorn=False,  # Skip ACORN for this test
        )

        # Should have check query and standard HNSW creation
        assert mock_cursor.execute.call_count == 2

        # Find CREATE INDEX call
        create_sql = None
        for call_args in mock_cursor.execute.call_args_list:
            sql = call_args[0][0]
            if "CREATE INDEX" in sql:
                create_sql = sql
                break

        assert (
            create_sql
            == "CREATE INDEX idx_Entities_embedding ON RAG.Entities(embedding) AS HNSW"
        )

    def test_ensure_hnsw_index_with_try_acorn_false(self, schema_manager):
        """Test ensure_vector_hnsw_index with try_acorn=False."""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [0]  # index doesn't exist

        schema_manager.ensure_vector_hnsw_index(
            mock_cursor,
            "RAG.SourceDocuments",
            "embedding",
            "idx_SourceDocuments_embedding",
            try_acorn=False,
        )

        # Should only attempt standard HNSW
        execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
        create_calls = [call for call in execute_calls if "CREATE INDEX" in call]

        assert len(create_calls) == 1
        assert "ACORN=1" not in create_calls[0]
        assert "AS HNSW" in create_calls[0]


class TestEnsureAllVectorIndexes:
    """Tests for ensure_all_vector_indexes method."""

    @pytest.fixture
    def schema_manager(self):
        """Create a schema manager with mocked dependencies."""
        mock_connection_manager = Mock()
        mock_config_manager = Mock()

        # Mock config manager to return default values
        mock_config_manager.get.side_effect = lambda key, default=None: {
            "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model.dimension": 384,
            "storage:iris:vector_data_type": "FLOAT",
        }.get(key, default)

        with patch.object(SchemaManager, "ensure_schema_metadata_table"):
            manager = SchemaManager(mock_connection_manager, mock_config_manager)

        return manager

    def test_ensure_all_vector_indexes_success(self, schema_manager):
        """Test ensure_all_vector_indexes creates indexes for all tables."""
        # Setup mock connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        schema_manager.connection_manager.get_connection.return_value = mock_connection

        # Mock no indexes exist
        mock_cursor.fetchone.return_value = [0]

        # Call ensure_all_vector_indexes
        schema_manager.ensure_all_vector_indexes()

        # Verify connection management
        schema_manager.connection_manager.get_connection.assert_called_once()
        mock_connection.cursor.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

        # Should create indexes for both SourceDocuments and Entities
        execute_calls = mock_cursor.execute.call_args_list

        # Find CREATE INDEX calls
        create_calls = []
        for call_args in execute_calls:
            sql = call_args[0][0]
            if "CREATE INDEX" in sql:
                create_calls.append(sql)

        assert len(create_calls) == 2

        # Check SourceDocuments index
        source_docs_create = [
            call for call in create_calls if "SourceDocuments" in call
        ]
        assert len(source_docs_create) == 1
        assert "idx_SourceDocuments_embedding" in source_docs_create[0]

        # Check Entities index
        entities_create = [call for call in create_calls if "Entities" in call]
        assert len(entities_create) == 1
        assert "idx_Entities_embedding" in entities_create[0]

    def test_ensure_all_vector_indexes_handles_errors(self, schema_manager):
        """Test ensure_all_vector_indexes handles errors gracefully."""
        # Setup mock connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        schema_manager.connection_manager.get_connection.return_value = mock_connection

        # Mock an error during execution
        mock_cursor.execute.side_effect = Exception("Database error")

        # Call ensure_all_vector_indexes - should not raise
        schema_manager.ensure_all_vector_indexes()

        # Verify rollback was called
        mock_connection.rollback.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_ensure_all_vector_indexes_with_existing_indexes(self, schema_manager):
        """Test ensure_all_vector_indexes when indexes already exist."""
        # Setup mock connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        schema_manager.connection_manager.get_connection.return_value = mock_connection

        # Mock indexes already exist
        mock_cursor.fetchone.return_value = [1]  # index count > 0

        # Call ensure_all_vector_indexes
        schema_manager.ensure_all_vector_indexes()

        # Should still have check queries but no CREATE INDEX calls
        execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
        create_calls = [call for call in execute_calls if "CREATE INDEX" in call]

        assert len(create_calls) == 0  # No indexes created since they exist
        mock_connection.commit.assert_called_once()
