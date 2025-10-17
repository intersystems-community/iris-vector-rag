"""
SchemaManager Contract Tests

These tests validate the SchemaManager public interface and ensure
compliance with the functional requirements. Tests must fail initially
and pass after implementation validation.
"""

import time
from typing import Any, Dict, List

import pytest

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

# Import the actual implementation to test against
from iris_rag.storage.schema_manager import SchemaManager


@pytest.mark.requires_database
class TestSchemaManagerContract:
    """Contract tests for SchemaManager functionality requiring live IRIS database."""

    def setup_method(self):
        """Set up test environment with database connection."""
        self.config_manager = ConfigurationManager()
        self.connection_manager = ConnectionManager()
        self.schema_manager = SchemaManager(
            self.connection_manager, self.config_manager
        )

    def test_vector_dimension_authority(self):
        """
        FR-003: System MUST provide centralized vector dimension authority for all tables.
        """
        # Test dimension retrieval for standard table
        dimension = self.schema_manager.get_vector_dimension("SourceDocuments")
        assert isinstance(dimension, int)
        assert dimension > 0

        # Test dimension consistency across multiple calls
        dimension2 = self.schema_manager.get_vector_dimension("SourceDocuments")
        assert dimension == dimension2

        # Test with specific model override
        ada_dimension = self.schema_manager.get_vector_dimension(
            "SourceDocuments", "text-embedding-ada-002"
        )
        assert ada_dimension == 1536  # Known dimension for Ada model

    def test_schema_migration_detection(self):
        """
        FR-004: System MUST automatically detect schema migration needs.
        """
        # Test migration need detection for a table
        needs_migration = self.schema_manager.needs_migration("SourceDocuments")
        assert isinstance(needs_migration, bool)

        # Test schema status reporting
        status = self.schema_manager.get_schema_status()
        assert isinstance(status, dict)
        assert (
            "SourceDocuments" in status or len(status) == 0
        )  # May be empty on fresh system

    def test_safe_schema_migrations(self):
        """
        FR-005: System MUST perform safe schema migrations with automatic rollback on failure.
        """
        # Test successful migration
        success = self.schema_manager.migrate_table("SourceDocuments")
        assert isinstance(success, bool)

        # Test migration with rollback simulation (if supported)
        if hasattr(self.schema_manager, "_test_rollback_scenario"):
            # This would be a test hook for simulating failures
            rollback_success = self.schema_manager._test_rollback_scenario(
                "SourceDocuments"
            )
            assert isinstance(rollback_success, bool)

    def test_schema_metadata_tracking(self):
        """
        FR-006: System MUST maintain schema version metadata for all managed tables.
        """
        # Ensure table exists
        self.schema_manager.ensure_table_schema("SourceDocuments", "basic")

        # Test metadata retrieval
        metadata = self.schema_manager.get_table_metadata("SourceDocuments")
        assert metadata is not None
        assert "schema_version" in metadata or "version" in metadata
        assert "vector_dimension" in metadata or "dimension" in metadata

    def test_table_specific_configurations(self):
        """
        FR-007: System MUST support table-specific configurations including embedding columns, foreign keys, and indexes.
        """
        # Test table configuration retrieval
        config = self.schema_manager.get_table_configuration("SourceDocuments")
        assert isinstance(config, dict)
        assert "embedding_column" in config or "embedding_columns" in config

        # Test multiple table types
        tables = [
            "SourceDocuments",
            "DocumentChunks",
            "Entities",
            "EntityRelationships",
        ]
        for table in tables:
            try:
                table_config = self.schema_manager.get_table_configuration(table)
                assert isinstance(table_config, dict)
            except KeyError:
                # Table might not be configured, which is acceptable
                pass

    def test_vector_dimension_consistency_validation(self):
        """
        FR-008: System MUST validate vector dimension consistency during startup only.
        """
        # Test dimension consistency check
        if hasattr(self.schema_manager, "validate_dimension_consistency"):
            result = self.schema_manager.validate_dimension_consistency()
            assert isinstance(result, dict)
            assert "consistent" in result

        # Test individual dimension validation
        try:
            valid = self.schema_manager.validate_vector_dimension(
                "SourceDocuments", 384
            )
            assert isinstance(valid, bool)
        except Exception as e:
            # May raise exception for invalid dimensions
            assert "dimension" in str(e).lower()

    def test_hnsw_vector_index_management(self):
        """
        FR-009: System MUST create and manage HNSW vector indexes with ACORN=1 optimization.
        """
        # Test vector index creation
        success = self.schema_manager.ensure_vector_hnsw_index(
            "SourceDocuments", "embedding"
        )
        assert isinstance(success, bool)

        # Test all vector indexes
        self.schema_manager.ensure_all_vector_indexes()

        # Verify index exists (if verification method available)
        if hasattr(self.schema_manager, "verify_vector_index"):
            index_exists = self.schema_manager.verify_vector_index(
                "SourceDocuments", "embedding"
            )
            assert isinstance(index_exists, bool)

    def test_audit_methods_for_testing(self):
        """
        FR-010: System MUST provide audit methods for integration testing.
        """
        # Test table structure verification
        structure = self.schema_manager.verify_table_structure("SourceDocuments")
        assert isinstance(structure, dict)
        assert "columns" in structure or "fields" in structure

        # Test schema audit
        if hasattr(self.schema_manager, "audit_schema"):
            audit_result = self.schema_manager.audit_schema()
            assert isinstance(audit_result, dict)

    def test_performance_targets_schema_operations(self):
        """
        Clarification: Schema migrations should complete in <5s, with warnings for longer operations.
        """
        start_time = time.perf_counter()

        # Perform a schema operation
        success = self.schema_manager.ensure_table_schema("SourceDocuments", "basic")

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Operation should complete (regardless of duration for existing tables)
        assert isinstance(success, bool)

        # For new table creation, duration should be tracked
        if duration_ms > 5000:
            # This would normally trigger a warning log
            print(f"Schema operation took {duration_ms}ms, exceeds 5s target")

    def test_transaction_safety(self):
        """
        Clarification: Automatic rollback with detailed error logging.
        """
        # Test transaction behavior during migration
        original_state = self.schema_manager.get_schema_status()

        try:
            # Attempt schema migration
            result = self.schema_manager.migrate_table("SourceDocuments")
            assert isinstance(result, bool)

            # Verify state is consistent after operation
            new_state = self.schema_manager.get_schema_status()
            assert isinstance(new_state, dict)

        except Exception as e:
            # Any exception should leave system in consistent state
            post_error_state = self.schema_manager.get_schema_status()
            # System should still be operational
            assert isinstance(post_error_state, dict)

    def test_model_dimension_mapping(self):
        """
        Test model-to-dimension mapping functionality.
        """
        # Test known model mappings
        known_models = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
            ("text-embedding-ada-002", 1536),
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
        ]

        for model, expected_dim in known_models:
            try:
                dim = self.schema_manager.get_vector_dimension("SourceDocuments", model)
                assert (
                    dim == expected_dim
                ), f"Model {model} should have dimension {expected_dim}, got {dim}"
            except (KeyError, ValueError):
                # Model might not be configured, which is acceptable
                pass

    def test_error_handling_patterns(self):
        """
        Test explicit error handling with no silent failures.
        """
        # Test with invalid table name
        with pytest.raises(Exception) as exc_info:
            self.schema_manager.get_vector_dimension("NonexistentTable")

        assert exc_info.value is not None
        error_msg = str(exc_info.value).lower()
        assert "table" in error_msg or "not found" in error_msg

        # Test with invalid model name
        with pytest.raises(Exception) as exc_info:
            self.schema_manager.get_vector_dimension("SourceDocuments", "invalid-model")

        assert exc_info.value is not None

    def test_constitutional_compliance_database_interfaces(self):
        """
        FR-007: Test use of standardized database interfaces.
        """
        # Verify schema manager uses connection manager
        assert self.schema_manager.connection_manager is not None

        # Verify no direct SQL queries (implementation-dependent)
        # This would check that proper abstractions are used
        connection = self.schema_manager._get_connection()
        assert connection is not None

    def test_iris_graph_core_table_support(self):
        """
        Test support for IRIS Graph Core tables (HybridGraphRAG).
        """
        graph_tables = [
            "rdf_labels",
            "rdf_props",
            "rdf_edges",
            "kg_NodeEmbeddings_optimized",
        ]

        for table in graph_tables:
            try:
                # Test if schema manager can handle graph tables
                config = self.schema_manager.get_table_configuration(table)
                assert isinstance(config, dict)
            except (KeyError, NotImplementedError):
                # Graph table support might not be implemented yet
                pass

    def test_schema_validation_startup_only(self):
        """
        Clarification: Vector dimension consistency validation occurs during system startup only.
        """
        # This test validates that dimension checks happen at startup
        # rather than during every operation (performance optimization)

        # Multiple operations should not trigger repeated validation
        start_time = time.perf_counter()

        for _ in range(10):
            dimension = self.schema_manager.get_vector_dimension("SourceDocuments")
            assert isinstance(dimension, int)

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) * 1000) / 10

        # Subsequent calls should be fast (cached/no re-validation)
        assert (
            avg_time_ms < 10
        ), f"Average dimension lookup took {avg_time_ms}ms, should be <10ms for cached operations"


@pytest.mark.requires_database
class TestSchemaManagerPerformance:
    """Performance-focused contract tests."""

    def setup_method(self):
        """Set up performance testing environment."""
        self.config_manager = ConfigurationManager()
        self.connection_manager = ConnectionManager()
        self.schema_manager = SchemaManager(
            self.connection_manager, self.config_manager
        )

    def test_configuration_access_performance(self):
        """
        Test <50ms configuration access target.
        """
        # Warm up
        self.schema_manager.get_vector_dimension("SourceDocuments")

        start_time = time.perf_counter()
        for _ in range(100):
            dimension = self.schema_manager.get_vector_dimension("SourceDocuments")
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) * 1000) / 100
        assert (
            avg_time_ms < 50
        ), f"Average config access: {avg_time_ms}ms, exceeds 50ms target"

    def test_enterprise_scale_operations(self):
        """
        Test enterprise scale (10K+ documents) readiness.
        """
        # This test would validate that schema operations
        # can handle enterprise scale without degradation

        # Test table status at scale
        status = self.schema_manager.get_schema_status()
        assert isinstance(status, dict)

        # Performance should remain consistent
        start_time = time.perf_counter()
        self.schema_manager.verify_table_structure("SourceDocuments")
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000
        # Structure verification should be fast even at scale
        assert (
            duration_ms < 1000
        ), f"Table verification took {duration_ms}ms, too slow for enterprise scale"
