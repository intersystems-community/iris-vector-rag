"""
Contract tests for iris-vector-graph table initialization.

Feature: 060-fix-users-tdyar
Status: TDD Phase - These tests MUST fail initially

These tests define the contract for ensure_iris_vector_graph_tables() functionality
BEFORE implementation. Following TDD principles (Constitution III), all tests
should fail with AttributeError when first run.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict


# Data structure contracts (these will be imported from implementation later)
@dataclass
class InitializationResult:
    """Result of iris-vector-graph table initialization."""
    package_detected: bool
    tables_attempted: List[str]
    tables_created: Dict[str, bool]
    total_time_seconds: float
    error_messages: Dict[str, str]


class TestGraphTableInitialization:
    """Contract tests for ensure_iris_vector_graph_tables() method."""

    def test_graph_tables_created_when_package_installed(self, schema_manager_with_db):
        """
        CONTRACT: All 4 graph tables created when package installed.

        GIVEN iris-vector-graph is installed
        AND database connection is valid
        WHEN ensure_iris_vector_graph_tables() is called
        THEN all 4 tables are created successfully
        AND InitializationResult.package_detected == True
        AND InitializationResult.tables_created shows all tables as True
        AND total_time_seconds < 5.0
        """
        schema_manager = schema_manager_with_db

        # Mock package detection as installed
        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            # This will fail initially with AttributeError:
            # 'SchemaManager' object has no attribute 'ensure_iris_vector_graph_tables'
            result = schema_manager.ensure_iris_vector_graph_tables(pipeline_type="graphrag")

            # Validate result structure
            assert isinstance(result, InitializationResult), \
                "Must return InitializationResult instance"

            # Validate package detection
            assert result.package_detected is True, \
                "Should detect installed package"

            # Validate tables attempted
            expected_tables = ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]
            assert result.tables_attempted == expected_tables, \
                f"Should attempt all 4 tables: {expected_tables}"

            # Validate all tables created successfully
            assert len(result.tables_created) == 4, \
                "Should have creation status for all 4 tables"
            for table in expected_tables:
                assert table in result.tables_created, \
                    f"Missing creation status for {table}"
                assert result.tables_created[table] is True, \
                    f"Table {table} should be created successfully"

            # Validate performance
            assert result.total_time_seconds < 5.0, \
                f"Initialization took {result.total_time_seconds}s, must be < 5s"

            # Validate no errors
            assert len(result.error_messages) == 0, \
                f"Should have no errors, got: {result.error_messages}"

    def test_graph_tables_skipped_when_package_not_installed(self, schema_manager):
        """
        CONTRACT: No tables created when package not installed.

        GIVEN iris-vector-graph is NOT installed
        WHEN ensure_iris_vector_graph_tables() is called
        THEN no tables are created
        AND InitializationResult.package_detected == False
        AND InitializationResult.tables_attempted is empty
        AND InitializationResult.tables_created is empty
        """
        # Mock package detection as not installed
        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=False):
            # This will fail initially with AttributeError
            result = schema_manager.ensure_iris_vector_graph_tables()

            assert isinstance(result, InitializationResult), \
                "Must return InitializationResult instance"
            assert result.package_detected is False, \
                "Should detect package not installed"
            assert result.tables_attempted == [], \
                "Should not attempt any tables when package not installed"
            assert result.tables_created == {}, \
                "Should not create any tables when package not installed"
            assert result.error_messages == {}, \
                "Should have no errors when package not installed"

    def test_idempotent_table_creation(self, schema_manager_with_db):
        """
        CONTRACT: Table creation is idempotent (safe to call multiple times).

        GIVEN iris-vector-graph is installed
        AND ensure_iris_vector_graph_tables() has been called once (tables exist)
        WHEN ensure_iris_vector_graph_tables() is called again
        THEN operation succeeds without errors
        AND InitializationResult shows all tables as successfully created
        AND no duplicate tables are created
        """
        schema_manager = schema_manager_with_db

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            # First call - create tables
            # This will fail initially with AttributeError
            result1 = schema_manager.ensure_iris_vector_graph_tables()
            assert all(result1.tables_created.values()), \
                "First call should create all tables"

            # Second call - tables already exist
            result2 = schema_manager.ensure_iris_vector_graph_tables()
            assert all(result2.tables_created.values()), \
                "Second call should succeed (idempotent)"
            assert len(result2.error_messages) == 0, \
                "Second call should have no errors"

    def test_partial_table_creation_failure(self, schema_manager_with_db):
        """
        CONTRACT: Partial failures tracked, other tables continue.

        GIVEN iris-vector-graph is installed
        AND database permissions allow creating 2 tables but not the other 2
        WHEN ensure_iris_vector_graph_tables() is called
        THEN successful tables are created
        AND InitializationResult.tables_created shows mixed success/failure
        AND InitializationResult.error_messages contains details for failed tables
        """
        schema_manager = schema_manager_with_db

        def mock_ensure_table_schema(table_name, pipeline_type=None):
            """Mock that fails for specific tables."""
            if table_name in ["rdf_edges", "kg_NodeEmbeddings_optimized"]:
                return False  # Simulate permission failure
            return True  # Success for other tables

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'ensure_table_schema', side_effect=mock_ensure_table_schema):
                # This will fail initially with AttributeError
                result = schema_manager.ensure_iris_vector_graph_tables()

                # Validate mixed results
                assert result.tables_created["rdf_labels"] is True, \
                    "rdf_labels should succeed"
                assert result.tables_created["rdf_props"] is True, \
                    "rdf_props should succeed"
                assert result.tables_created["rdf_edges"] is False, \
                    "rdf_edges should fail (simulated permission error)"
                assert result.tables_created["kg_NodeEmbeddings_optimized"] is False, \
                    "kg_NodeEmbeddings_optimized should fail"

                # Validate error tracking
                assert "rdf_edges" in result.error_messages or \
                       "kg_NodeEmbeddings_optimized" in result.error_messages, \
                    "Should track errors for failed tables"

    def test_initialization_performance(self, schema_manager_with_db):
        """
        CONTRACT: Initialization completes in < 5 seconds.

        GIVEN iris-vector-graph is installed
        WHEN ensure_iris_vector_graph_tables() is called on empty database
        THEN total_time_seconds < 5.0
        AND InitializationResult contains accurate timing
        """
        schema_manager = schema_manager_with_db

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            start_time = time.time()

            # This will fail initially with AttributeError
            result = schema_manager.ensure_iris_vector_graph_tables()

            elapsed = time.time() - start_time

            assert result.total_time_seconds < 5.0, \
                f"Initialization took {result.total_time_seconds}s, must be < 5s"
            assert abs(result.total_time_seconds - elapsed) < 0.5, \
                "Recorded time should match actual elapsed time (within 0.5s)"

    def test_tables_created_in_dependency_order(self, schema_manager_with_db):
        """
        CONTRACT: Tables created in correct dependency order.

        GIVEN iris-vector-graph is installed
        WHEN ensure_iris_vector_graph_tables() is called
        THEN tables are created in order: rdf_labels, rdf_props, rdf_edges,
             kg_NodeEmbeddings_optimized
        AND node tables created before edge tables
        """
        schema_manager = schema_manager_with_db
        creation_order = []

        def mock_ensure_table_schema(table_name, pipeline_type=None):
            """Track table creation order."""
            creation_order.append(table_name)
            return True

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'ensure_table_schema', side_effect=mock_ensure_table_schema):
                # This will fail initially with AttributeError
                result = schema_manager.ensure_iris_vector_graph_tables()

                # Validate order
                expected_order = ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]
                assert creation_order == expected_order, \
                    f"Tables should be created in order: {expected_order}, got: {creation_order}"

                # Validate nodes before edges
                edges_index = creation_order.index("rdf_edges")
                labels_index = creation_order.index("rdf_labels")
                assert labels_index < edges_index, \
                    "Node tables (rdf_labels) must be created before edge tables (rdf_edges)"


class TestInitializationResultDataStructure:
    """Contract tests for InitializationResult data structure."""

    def test_initialization_result_invariants(self):
        """
        CONTRACT: InitializationResult maintains data invariants.

        GIVEN InitializationResult instance
        WHEN package_detected == False
        THEN tables_created must be empty dict
        AND error_messages keys must be subset of tables_attempted
        AND total_time_seconds >= 0
        """
        # Package not detected case
        result = InitializationResult(
            package_detected=False,
            tables_attempted=[],
            tables_created={},
            total_time_seconds=0.0,
            error_messages={},
        )

        assert result.tables_created == {}, \
            "tables_created must be empty when package not detected"

        # Partial failure case
        result = InitializationResult(
            package_detected=True,
            tables_attempted=["rdf_labels", "rdf_props", "rdf_edges"],
            tables_created={"rdf_labels": True, "rdf_props": True, "rdf_edges": False},
            total_time_seconds=3.5,
            error_messages={"rdf_edges": "Permission denied"},
        )

        assert len(result.tables_created) == len(result.tables_attempted), \
            "tables_created must have entry for each attempted table"
        assert set(result.error_messages.keys()).issubset(set(result.tables_attempted)), \
            "error_messages keys must be subset of tables_attempted"
        assert result.total_time_seconds >= 0, \
            "total_time_seconds must be non-negative"


@pytest.fixture
def schema_manager():
    """
    Fixture providing SchemaManager instance (no database).

    NOTE: This fixture will fail initially because SchemaManager
    does not yet have the ensure_iris_vector_graph_tables() method.
    """
    from iris_vector_rag.storage.schema_manager import SchemaManager
    from iris_vector_rag.config.config_manager import ConfigurationManager

    config = ConfigurationManager()
    manager = SchemaManager(
        connection_string=config.get("database.connection_string"),
        base_embedding_dimension=config.get("embeddings.dimension"),
    )

    return manager


@pytest.fixture
def schema_manager_with_db(iris_connection):
    """
    Fixture providing SchemaManager with real database connection.

    Requires iris_connection fixture from conftest.py
    """
    from iris_vector_rag.storage.schema_manager import SchemaManager

    manager = SchemaManager(
        connection_string=iris_connection.connection_string,
        base_embedding_dimension=384,
    )

    yield manager

    # Cleanup: drop tables after test
    tables = ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]
    for table in tables:
        try:
            manager.drop_table(table)
        except Exception:
            pass  # Table may not exist


# Expected test results for TDD phase:
# test_graph_tables_created_when_package_installed: FAIL (AttributeError: 'ensure_iris_vector_graph_tables')
# test_graph_tables_skipped_when_package_not_installed: FAIL (AttributeError: 'ensure_iris_vector_graph_tables')
# test_idempotent_table_creation: FAIL (AttributeError: 'ensure_iris_vector_graph_tables')
# test_partial_table_creation_failure: FAIL (AttributeError: 'ensure_iris_vector_graph_tables')
# test_initialization_performance: FAIL (AttributeError: 'ensure_iris_vector_graph_tables')
# test_tables_created_in_dependency_order: FAIL (AttributeError: 'ensure_iris_vector_graph_tables')
# test_initialization_result_invariants: PASS (data structure test, no implementation dependency)
