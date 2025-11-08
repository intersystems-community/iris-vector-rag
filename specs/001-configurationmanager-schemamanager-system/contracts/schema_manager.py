"""
Contract tests for SchemaManager component.

These tests define the expected behavior and interface contracts that the
SchemaManager implementation must satisfy.
"""

from typing import Any, Dict, Optional

import pytest


class TestSchemaManagerContract:
    """Contract tests for SchemaManager interface."""

    def test_vector_dimension_authority(self, schema_manager):
        """
        Contract: System MUST provide centralized vector dimension authority
        for all tables based on embedding models and configuration.
        """
        # Given: A table name and optional model
        # When: Getting vector dimension
        # Then: Correct dimension returned based on model and table config
        dimension = schema_manager.get_vector_dimension("SourceDocuments")
        assert isinstance(dimension, int)
        assert dimension > 0

    def test_schema_migration_detection(self, schema_manager):
        """
        Contract: System MUST automatically detect schema migration needs
        by comparing current vs expected table configurations.
        """
        # Given: A table with potential schema changes
        # When: Checking migration needs
        # Then: Boolean result indicates if migration required
        needs_migration = schema_manager.needs_migration("SourceDocuments")
        assert isinstance(needs_migration, bool)

    def test_safe_schema_migration(self, schema_manager):
        """
        Contract: System MUST perform safe schema migrations with
        proper transaction rollback on failure.
        """
        # Given: A table requiring migration
        # When: Performing migration
        # Then: Migration succeeds or fails safely with rollback
        success = schema_manager.migrate_table("SourceDocuments", preserve_data=False)
        assert isinstance(success, bool)

    def test_schema_metadata_tracking(self, schema_manager):
        """
        Contract: System MUST maintain schema version metadata for all managed tables.
        """
        # Given: A managed table
        # When: Getting schema configuration
        # Then: Metadata includes version, dimensions, and model info
        config = schema_manager.get_current_schema_config("SourceDocuments")
        if config:  # May be None if not yet created
            assert "schema_version" in config
            assert "vector_dimension" in config
            assert "embedding_model" in config

    def test_vector_dimension_validation(self, schema_manager):
        """
        Contract: System MUST validate vector dimension consistency
        across all pipeline components.
        """
        # Given: A table and dimension
        # When: Validating dimension
        # Then: Validation passes for correct dimensions, fails for incorrect
        expected_dim = schema_manager.get_vector_dimension("SourceDocuments")

        # Should not raise for correct dimension
        schema_manager.validate_vector_dimension("SourceDocuments", expected_dim)

        # Should raise for incorrect dimension
        with pytest.raises(ValueError):
            schema_manager.validate_vector_dimension(
                "SourceDocuments", expected_dim + 100
            )

    def test_hnsw_index_management(self, schema_manager, mock_cursor):
        """
        Contract: System MUST create and manage HNSW vector indexes
        with ACORN=1 optimization when available.
        """
        # Given: A table with vector column
        # When: Ensuring HNSW index
        # Then: Index creation attempted with ACORN=1 fallback
        schema_manager.ensure_vector_hnsw_index(
            mock_cursor, "RAG.SourceDocuments", "embedding", "test_index"
        )

    def test_table_structure_verification(self, schema_manager):
        """
        Contract: System MUST provide audit methods for integration testing
        that replace direct SQL access patterns.
        """
        # Given: A table name
        # When: Verifying table structure
        # Then: Column information returned without direct SQL
        structure = schema_manager.verify_table_structure("SourceDocuments")
        assert isinstance(structure, dict)

    def test_embedding_model_registration(self, schema_manager):
        """
        Contract: System MUST support registration of new embedding models
        with their dimensions.
        """
        # Given: A new model and dimension
        # When: Registering model
        # Then: Model is available for dimension lookups
        test_model = "test-model"
        test_dimension = 512

        schema_manager.register_model(test_model, test_dimension)
        assert (
            schema_manager.get_vector_dimension("SourceDocuments", test_model)
            == test_dimension
        )

    def test_dimension_consistency_validation(self, schema_manager):
        """
        Contract: System MUST validate that all tables have consistent
        dimensions with their models.
        """
        # Given: Managed tables
        # When: Validating dimension consistency
        # Then: Validation results include consistency status and issues
        results = schema_manager.validate_dimension_consistency()

        assert "consistent" in results
        assert "issues" in results
        assert "table_dimensions" in results
        assert isinstance(results["consistent"], bool)

    def test_schema_status_reporting(self, schema_manager):
        """
        Contract: System MUST provide comprehensive schema status
        for all managed tables.
        """
        # Given: Managed schema tables
        # When: Getting schema status
        # Then: Status includes current config, expected config, and migration needs
        status = schema_manager.get_schema_status()

        assert isinstance(status, dict)
        for table_status in status.values():
            assert "current_config" in table_status
            assert "expected_config" in table_status
            assert "needs_migration" in table_status
            assert "status" in table_status
