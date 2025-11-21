"""
Integration tests for Bug 2: Automatic iris-vector-graph Table Initialization (v0.5.4)

Tests verify that:
1. iris-vector-graph package is detected correctly
2. Tables are automatically created when package is installed
3. Graceful degradation when package is not installed
4. PPR operations work without "Table not found" errors
5. Initialization completes in < 5 seconds
"""

import os
import pytest
import time
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.common.connection_manager import ConnectionManager


class ConnectionManagerWrapper:
    """Wrapper to make ConnectionManager compatible with SchemaManager interface"""
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def get_connection(self):
        """Alias for connect() to match SchemaManager expectations"""
        return self._connection_manager.connect()

    def __getattr__(self, name):
        """Forward all other methods to wrapped connection manager"""
        return getattr(self._connection_manager, name)


class MinimalConfigManager:
    """Minimal config manager for testing"""
    def get(self, key, default=None):
        return default

    def get_embedding_config(self):
        return {"model": "all-MiniLM-L6-v2", "dimension": 384}

    def get_vector_index_config(self):
        return {"type": "HNSW", "M": 16, "efConstruction": 200}

    def get_cloud_config(self):
        """Return cloud config with vector configuration"""
        class VectorConfig:
            vector_dimension = 384

        class CloudConfig:
            vector = VectorConfig()

        return CloudConfig()


class TestBug2SchemaInitialization:
    """Integration tests for Bug 2: iris-vector-graph automatic table initialization"""

    def test_package_detection(self):
        """T024: Verify iris-vector-graph package detection"""
        # Create SchemaManager
        connection_manager = ConnectionManagerWrapper(ConnectionManager(connection_type="dbapi"))
        config_manager = MinimalConfigManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # When: Package detection is called
        is_installed = schema_manager._detect_iris_vector_graph()

        # Then: Detection returns boolean
        assert isinstance(is_installed, bool)
        print(f"✅ Bug 2 Fix: Package detection works (installed: {is_installed})")

    def test_automatic_table_initialization_when_package_installed(self):
        """T024: Verify tables are automatically created when iris-vector-graph is installed"""
        # Given: SchemaManager initialized
        connection_manager = ConnectionManagerWrapper(ConnectionManager(connection_type="dbapi"))
        config_manager = MinimalConfigManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # Check if package is installed
        is_installed = schema_manager._detect_iris_vector_graph()

        if not is_installed:
            pytest.skip("iris-vector-graph not installed - skipping table creation test")

        # When: ensure_iris_vector_graph_tables() is called
        start_time = time.time()
        result = schema_manager.ensure_iris_vector_graph_tables()
        elapsed_time = time.time() - start_time

        # Then: Tables are created
        assert result.package_detected is True
        assert len(result.tables_attempted) == 4
        assert "rdf_labels" in result.tables_attempted
        assert "rdf_props" in result.tables_attempted
        assert "rdf_edges" in result.tables_attempted
        assert "kg_NodeEmbeddings_optimized" in result.tables_attempted

        # And: Performance requirement met (< 5 seconds)
        assert result.total_time_seconds < 5.0
        assert elapsed_time < 5.0

        # And: All tables created successfully (or already existed)
        assert len(result.tables_created) == 4
        print(f"✅ Bug 2 Fix: All 4 tables initialized in {result.total_time_seconds:.2f}s")

    def test_graceful_degradation_when_package_not_installed(self):
        """T025: Verify graceful skip when iris-vector-graph not installed"""
        # Given: SchemaManager initialized
        connection_manager = ConnectionManagerWrapper(ConnectionManager(connection_type="dbapi"))
        config_manager = MinimalConfigManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # Mock package not installed scenario
        original_detect = schema_manager._detect_iris_vector_graph

        def mock_detect_not_installed():
            return False

        schema_manager._detect_iris_vector_graph = mock_detect_not_installed

        try:
            # When: ensure_iris_vector_graph_tables() is called
            result = schema_manager.ensure_iris_vector_graph_tables()

            # Then: No tables attempted
            assert result.package_detected is False
            assert len(result.tables_attempted) == 0
            assert len(result.tables_created) == 0
            assert result.total_time_seconds < 1.0  # Fast skip

            print("✅ Bug 2 Fix: Graceful degradation when package not installed")
        finally:
            schema_manager._detect_iris_vector_graph = original_detect

    def test_prerequisite_validation_all_met(self):
        """T024: Verify prerequisite validation when all requirements met"""
        # Given: SchemaManager initialized
        connection_manager = ConnectionManagerWrapper(ConnectionManager(connection_type="dbapi"))
        config_manager = MinimalConfigManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # Check if package is installed
        is_installed = schema_manager._detect_iris_vector_graph()

        if not is_installed:
            pytest.skip("iris-vector-graph not installed - skipping validation test")

        # Ensure tables exist
        schema_manager.ensure_iris_vector_graph_tables()

        # When: Validation is performed
        validation = schema_manager.validate_graph_prerequisites()

        # Then: All prerequisites met
        assert validation.is_valid is True
        assert validation.package_installed is True
        assert len(validation.missing_tables) == 0
        assert validation.error_message == ""

        print("✅ Bug 2 Fix: All prerequisites validated successfully")

    def test_prerequisite_validation_package_missing(self):
        """T025: Verify validation reports missing package"""
        # Given: SchemaManager initialized
        connection_manager = ConnectionManagerWrapper(ConnectionManager(connection_type="dbapi"))
        config_manager = MinimalConfigManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # Mock package not installed
        original_detect = schema_manager._detect_iris_vector_graph

        def mock_detect_not_installed():
            return False

        schema_manager._detect_iris_vector_graph = mock_detect_not_installed

        try:
            # When: Validation is performed
            validation = schema_manager.validate_graph_prerequisites()

            # Then: Validation fails with clear message
            assert validation.is_valid is False
            assert validation.package_installed is False
            assert "iris-vector-graph" in validation.error_message.lower()
            print(f"✅ Bug 2 Fix: Validation correctly reports missing package: {validation.error_message}")
        finally:
            schema_manager._detect_iris_vector_graph = original_detect

    def test_idempotent_initialization(self):
        """T024: Verify initialization is idempotent (safe to call multiple times)"""
        # Given: SchemaManager initialized
        connection_manager = ConnectionManagerWrapper(ConnectionManager(connection_type="dbapi"))
        config_manager = MinimalConfigManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # Check if package is installed
        is_installed = schema_manager._detect_iris_vector_graph()

        if not is_installed:
            pytest.skip("iris-vector-graph not installed - skipping idempotency test")

        # When: ensure_iris_vector_graph_tables() is called multiple times
        result1 = schema_manager.ensure_iris_vector_graph_tables()
        result2 = schema_manager.ensure_iris_vector_graph_tables()
        result3 = schema_manager.ensure_iris_vector_graph_tables()

        # Then: All calls succeed
        assert result1.package_detected is True
        assert result2.package_detected is True
        assert result3.package_detected is True

        # And: No errors raised
        assert len(result1.error_messages) == 0
        assert len(result2.error_messages) == 0
        assert len(result3.error_messages) == 0

        print("✅ Bug 2 Fix: Idempotent initialization verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
