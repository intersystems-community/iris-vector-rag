"""
Contract tests for SchemaManager component.

These tests validate the expected behavior and interface contracts for the
SchemaManager implementation against the functional requirements.
"""

import time
from typing import Any, Dict, Optional

import pytest

# These imports will fail initially - this is expected for contract tests
try:
    from common.iris_connection_manager import IRISConnectionManager
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.storage.schema_manager import SchemaManager
except ImportError:
    # Contract tests are written before implementation
    SchemaManager = None
    ConfigurationManager = None
    IRISConnectionManager = None


@pytest.mark.requires_database
class TestSchemaManagerContract:
    """Contract tests for SchemaManager functionality requiring live IRIS database."""

    @pytest.fixture
    def connection_manager(self):
        """Provide IRIS connection manager for tests."""
        if IRISConnectionManager:
            return IRISConnectionManager()
        return None

    @pytest.fixture
    def config_manager(self):
        """Provide configuration manager for tests."""
        if ConfigurationManager:
            return ConfigurationManager()
        return None

    @pytest.fixture
    def schema_manager(self, connection_manager, config_manager):
        """Provide schema manager instance for tests."""
        if SchemaManager and connection_manager and config_manager:
            return SchemaManager(connection_manager, config_manager)
        return None

    def test_schema_manager_exists(self):
        """SchemaManager class must exist and be importable."""
        assert SchemaManager is not None, "SchemaManager class must be implemented"

    def test_vector_dimension_authority(self, schema_manager):
        """FR-003: System MUST provide centralized vector dimension authority."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test dimension lookup for standard table
        dimension = schema_manager.get_vector_dimension("SourceDocuments")
        assert isinstance(dimension, int), "Vector dimension must be integer"
        assert dimension > 0, "Vector dimension must be positive"

        # Test dimension consistency across calls
        dimension2 = schema_manager.get_vector_dimension("SourceDocuments")
        assert dimension == dimension2, "Dimension must be consistent across calls"

    def test_schema_migration_detection(self, schema_manager):
        """FR-004: System MUST detect schema migration needs automatically."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test migration detection for a standard table
        needs_migration = schema_manager.needs_migration("SourceDocuments")
        assert isinstance(
            needs_migration, bool
        ), "Migration detection must return boolean"

        # Test status reporting
        status = schema_manager.get_schema_status()
        assert isinstance(status, dict), "Schema status must be dictionary"
        assert "SourceDocuments" in status, "Status must include managed tables"

    def test_safe_schema_migrations(self, schema_manager):
        """FR-005: System MUST perform safe schema migrations with rollback."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test migration execution
        result = schema_manager.migrate_table("SourceDocuments")
        assert isinstance(result, bool), "Migration result must be boolean"

        # If migration succeeded, verify table structure
        if result:
            structure = schema_manager.verify_table_structure("SourceDocuments")
            assert structure is not None, "Migrated table must have valid structure"

    def test_schema_metadata_tracking(self, schema_manager):
        """FR-006: System MUST maintain schema version metadata."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Ensure metadata table exists
        schema_manager.ensure_schema_metadata_table()

        # Test metadata retrieval
        metadata = schema_manager.get_table_metadata("SourceDocuments")
        if metadata:
            assert "schema_version" in metadata, "Metadata must include schema version"
            assert (
                "vector_dimension" in metadata
            ), "Metadata must include vector dimension"

    def test_table_specific_configurations(self, schema_manager):
        """FR-007: System MUST support table-specific configurations."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test table configuration retrieval
        config = schema_manager.get_table_configuration("SourceDocuments")
        assert config is not None, "Table configuration must exist"

        # Verify configuration structure
        assert "table_name" in config, "Config must include table name"
        assert "dimension" in config, "Config must include vector dimension"

    def test_vector_dimension_consistency_validation(self, schema_manager):
        """FR-008: System MUST validate vector dimension consistency."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test dimension consistency validation
        results = schema_manager.validate_dimension_consistency()
        assert isinstance(results, dict), "Validation results must be dictionary"
        assert "consistent" in results, "Results must include consistency status"

        # Test specific dimension validation
        dimension = schema_manager.get_vector_dimension("SourceDocuments")
        is_valid = schema_manager.validate_vector_dimension(
            "SourceDocuments", dimension
        )
        assert is_valid is True, "Valid dimension should pass validation"

        # Test invalid dimension detection
        with pytest.raises(ValueError):
            schema_manager.validate_vector_dimension("SourceDocuments", -1)

    def test_hnsw_index_management(self, schema_manager):
        """FR-009: System MUST create and manage HNSW vector indexes."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test HNSW index creation
        result = schema_manager.ensure_vector_hnsw_index("SourceDocuments", "embedding")
        assert isinstance(result, bool), "Index creation must return boolean"

        # Test ACORN=1 optimization when available
        if hasattr(schema_manager, "check_acorn_support"):
            acorn_supported = schema_manager.check_acorn_support()
            if acorn_supported:
                # Verify ACORN=1 is used in index creation
                index_config = schema_manager.get_index_configuration(
                    "SourceDocuments", "embedding"
                )
                if index_config:
                    assert (
                        index_config.get("acorn") == 1
                    ), "ACORN=1 must be used when supported"

    def test_audit_methods_for_testing(self, schema_manager):
        """FR-010: System MUST provide audit methods replacing direct SQL access."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test table structure verification
        structure = schema_manager.verify_table_structure("SourceDocuments")
        assert structure is not None, "Audit method must return table structure"

        # Test performance requirements
        start_time = time.time()
        for _ in range(10):  # Multiple calls to test performance
            schema_manager.verify_table_structure("SourceDocuments")
        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / 10) * 1000
        # Allow more time for audit methods as they're not in critical path
        assert (
            avg_time_ms < 100
        ), f"Audit method took {avg_time_ms:.2f}ms, should be <100ms"

    def test_schema_migration_performance(self, schema_manager):
        """Schema migrations MUST complete within 5 seconds."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test migration timing
        start_time = time.time()
        result = schema_manager.migrate_table("SourceDocuments")
        end_time = time.time()

        migration_time = end_time - start_time
        assert (
            migration_time < 5.0
        ), f"Migration took {migration_time:.2f}s, must be <5s"

    def test_transaction_safety_with_rollback(self, schema_manager):
        """Schema operations MUST be transaction-safe with rollback capability."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Get current table state
        initial_structure = schema_manager.verify_table_structure("SourceDocuments")

        # Test rollback behavior by simulating failure
        # (This would typically be done by injecting a failure condition)
        try:
            # Attempt operation that might fail
            result = schema_manager.migrate_table("SourceDocuments")

            # Verify table structure remains consistent
            final_structure = schema_manager.verify_table_structure("SourceDocuments")
            assert final_structure is not None, "Table structure must remain valid"

        except Exception as e:
            # If operation failed, ensure rollback occurred
            rollback_structure = schema_manager.verify_table_structure(
                "SourceDocuments"
            )
            assert (
                rollback_structure == initial_structure
            ), "Failed operation must rollback to original state"

    def test_explicit_error_handling(self, schema_manager):
        """FR-006: System MUST provide explicit error handling with actionable context."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Test error handling for invalid table
        with pytest.raises(Exception) as exc_info:
            schema_manager.get_vector_dimension("NonexistentTable")

        error_msg = str(exc_info.value)
        assert len(error_msg) > 0, "Error message must not be empty"
        assert "NonexistentTable" in error_msg, "Error must reference specific table"

        # Test error handling for invalid dimension
        with pytest.raises(ValueError) as exc_info:
            schema_manager.validate_vector_dimension("SourceDocuments", 0)

        error_msg = str(exc_info.value)
        assert (
            "dimension" in error_msg.lower()
        ), "Error must reference dimension validation"

    def test_iris_integration_compliance(self, schema_manager):
        """System MUST use standardized IRIS database interfaces."""
        if schema_manager is None:
            pytest.skip("SchemaManager not available")

        # Verify connection manager is used (not direct IRIS API calls)
        assert hasattr(
            schema_manager, "connection_manager"
        ), "SchemaManager must use connection manager"

        # Test that operations work with IRIS database
        connection_test = schema_manager.connection_manager.test_connection()
        assert connection_test is True, "IRIS connection must be functional"
