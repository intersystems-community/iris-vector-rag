"""
Contract tests for iris-vector-graph prerequisite validation.

Feature: 060-fix-users-tdyar
Status: TDD Phase - These tests MUST fail initially

These tests define the contract for validate_graph_prerequisites() functionality
BEFORE implementation. Following TDD principles (Constitution III), all tests
should fail with AttributeError when first run.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List


# Data structure contracts (these will be imported from implementation later)
@dataclass
class ValidationResult:
    """Result of prerequisite validation."""
    is_valid: bool
    package_installed: bool
    missing_tables: List[str]
    error_message: str


class TestGraphPrerequisiteValidation:
    """Contract tests for validate_graph_prerequisites() method."""

    def test_prerequisite_validation_all_met(self, schema_manager_with_tables):
        """
        CONTRACT: Validation succeeds when all prerequisites met.

        GIVEN iris-vector-graph is installed
        AND all 4 graph tables exist in database
        WHEN validate_graph_prerequisites() is called
        THEN ValidationResult.is_valid == True
        AND ValidationResult.package_installed == True
        AND ValidationResult.missing_tables == []
        AND ValidationResult.error_message == ""
        """
        schema_manager = schema_manager_with_tables

        # Mock package installed and all tables exist
        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'table_exists', return_value=True):
                # This will fail initially with AttributeError:
                # 'SchemaManager' object has no attribute 'validate_graph_prerequisites'
                result = schema_manager.validate_graph_prerequisites()

                assert isinstance(result, ValidationResult), \
                    "Must return ValidationResult instance"
                assert result.is_valid is True, \
                    "Should be valid when all prerequisites met"
                assert result.package_installed is True, \
                    "Should detect package installed"
                assert result.missing_tables == [], \
                    "Should have no missing tables"
                assert result.error_message == "", \
                    "Should have empty error message when valid"

    def test_prerequisite_validation_package_missing(self, schema_manager):
        """
        CONTRACT: Validation fails when package not installed.

        GIVEN iris-vector-graph is NOT installed
        WHEN validate_graph_prerequisites() is called
        THEN ValidationResult.is_valid == False
        AND ValidationResult.package_installed == False
        AND ValidationResult.error_message contains "iris-vector-graph package not installed"
        """
        # Mock package not installed
        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=False):
            # This will fail initially with AttributeError
            result = schema_manager.validate_graph_prerequisites()

            assert isinstance(result, ValidationResult), \
                "Must return ValidationResult instance"
            assert result.is_valid is False, \
                "Should be invalid when package not installed"
            assert result.package_installed is False, \
                "Should detect package not installed"
            assert "iris-vector-graph" in result.error_message.lower(), \
                "Error message should mention iris-vector-graph"
            assert "not installed" in result.error_message.lower(), \
                "Error message should indicate package not installed"

    def test_prerequisite_validation_tables_missing(self, schema_manager_with_db):
        """
        CONTRACT: Validation fails when tables missing.

        GIVEN iris-vector-graph is installed
        AND only 2 of 4 graph tables exist (rdf_labels and rdf_props present, others missing)
        WHEN validate_graph_prerequisites() is called
        THEN ValidationResult.is_valid == False
        AND ValidationResult.package_installed == True
        AND ValidationResult.missing_tables == ["rdf_edges", "kg_NodeEmbeddings_optimized"]
        AND ValidationResult.error_message lists specific missing tables
        """
        schema_manager = schema_manager_with_db

        def mock_table_exists(table_name):
            """Mock that only some tables exist."""
            return table_name in ["rdf_labels", "rdf_props"]

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'table_exists', side_effect=mock_table_exists):
                # This will fail initially with AttributeError
                result = schema_manager.validate_graph_prerequisites()

                assert result.is_valid is False, \
                    "Should be invalid when tables missing"
                assert result.package_installed is True, \
                    "Should detect package installed"
                assert "rdf_edges" in result.missing_tables, \
                    "Should list rdf_edges as missing"
                assert "kg_NodeEmbeddings_optimized" in result.missing_tables, \
                    "Should list kg_NodeEmbeddings_optimized as missing"
                assert len(result.missing_tables) == 2, \
                    "Should have exactly 2 missing tables"
                assert "rdf_edges" in result.error_message, \
                    "Error message should mention rdf_edges"
                assert "kg_NodeEmbeddings_optimized" in result.error_message, \
                    "Error message should mention kg_NodeEmbeddings_optimized"

    def test_prerequisite_validation_before_ppr(self, schema_manager):
        """
        CONTRACT: Validation provides explicit error for PPR operations.

        GIVEN iris-vector-graph is installed
        AND NOT all tables exist
        WHEN validate_graph_prerequisites() is called before PPR operation
        THEN ValidationResult.is_valid == False
        AND calling code can raise RuntimeError with specific missing components
        """
        def mock_table_exists(table_name):
            """Mock missing tables."""
            return table_name not in ["rdf_edges", "kg_NodeEmbeddings_optimized"]

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'table_exists', side_effect=mock_table_exists):
                # This will fail initially with AttributeError
                result = schema_manager.validate_graph_prerequisites()

                assert result.is_valid is False, \
                    "Validation should fail when prerequisites not met"

                # Simulate calling code that raises RuntimeError
                if not result.is_valid:
                    error_msg = f"Cannot perform PPR: {result.error_message}"
                    assert "rdf_edges" in error_msg, \
                        "Error should list specific missing tables"
                    assert "kg_NodeEmbeddings_optimized" in error_msg, \
                        "Error should list specific missing tables"

    def test_clear_error_when_tables_missing(self, schema_manager):
        """
        CONTRACT: Error messages are clear and actionable.

        GIVEN iris-vector-graph is installed
        AND tables rdf_edges and kg_NodeEmbeddings_optimized are missing
        WHEN validate_graph_prerequisites() is called
        THEN error_message clearly indicates:
            - Which specific tables are missing
            - Suggested remediation (run ensure_iris_vector_graph_tables())
            - Distinguishes from "package not installed" error
        """
        def mock_table_exists(table_name):
            """Mock specific missing tables."""
            return table_name not in ["rdf_edges", "kg_NodeEmbeddings_optimized"]

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'table_exists', side_effect=mock_table_exists):
                # This will fail initially with AttributeError
                result = schema_manager.validate_graph_prerequisites()

                # Validate error message clarity
                error = result.error_message
                assert "rdf_edges" in error, \
                    "Error should list rdf_edges as missing"
                assert "kg_NodeEmbeddings_optimized" in error, \
                    "Error should list kg_NodeEmbeddings_optimized as missing"

                # Error should be actionable
                assert "missing" in error.lower() or "not found" in error.lower(), \
                    "Error should indicate tables are missing"

                # Should NOT say "package not installed" since package IS installed
                assert "not installed" not in error.lower(), \
                    "Error should distinguish tables missing from package not installed"

    def test_validation_performance(self, schema_manager_with_tables):
        """
        CONTRACT: Validation completes in < 1 second.

        GIVEN iris-vector-graph is installed and tables exist
        WHEN validate_graph_prerequisites() is called
        THEN execution completes in < 1.0 seconds
        """
        import time

        schema_manager = schema_manager_with_tables

        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'table_exists', return_value=True):
                start_time = time.time()

                # This will fail initially with AttributeError
                result = schema_manager.validate_graph_prerequisites()

                elapsed = time.time() - start_time

                assert elapsed < 1.0, \
                    f"Validation took {elapsed}s, must be < 1s"
                assert result.is_valid is True, \
                    "Validation should succeed"

    def test_validation_is_stateless(self, schema_manager):
        """
        CONTRACT: Validation is stateless and repeatable.

        GIVEN validate_graph_prerequisites() has been called once
        WHEN validate_graph_prerequisites() is called again
        THEN returns same result without side effects
        AND does not modify any instance state
        """
        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=True):
            with patch.object(schema_manager, 'table_exists', return_value=True):
                # This will fail initially with AttributeError
                result1 = schema_manager.validate_graph_prerequisites()
                result2 = schema_manager.validate_graph_prerequisites()

                assert result1.is_valid == result2.is_valid, \
                    "Repeated calls should return same validity"
                assert result1.package_installed == result2.package_installed, \
                    "Repeated calls should return same package status"
                assert result1.missing_tables == result2.missing_tables, \
                    "Repeated calls should return same missing tables"


class TestValidationResultDataStructure:
    """Contract tests for ValidationResult data structure."""

    def test_validation_result_invariants(self):
        """
        CONTRACT: ValidationResult maintains data invariants.

        GIVEN ValidationResult instance
        WHEN package_installed == False
        THEN is_valid must be False
        AND if is_valid == True, missing_tables must be empty
        AND if is_valid == False, error_message must be non-empty
        """
        # Package not installed case
        result = ValidationResult(
            is_valid=False,
            package_installed=False,
            missing_tables=[],
            error_message="iris-vector-graph package not installed",
        )

        assert result.is_valid is False, \
            "is_valid must be False when package not installed"
        assert result.error_message != "", \
            "error_message must be non-empty when invalid"

        # All valid case
        result = ValidationResult(
            is_valid=True,
            package_installed=True,
            missing_tables=[],
            error_message="",
        )

        assert result.missing_tables == [], \
            "missing_tables must be empty when valid"
        assert result.error_message == "", \
            "error_message must be empty when valid"

        # Tables missing case
        result = ValidationResult(
            is_valid=False,
            package_installed=True,
            missing_tables=["rdf_edges", "kg_NodeEmbeddings_optimized"],
            error_message="Missing required tables: rdf_edges, kg_NodeEmbeddings_optimized",
        )

        assert result.is_valid is False, \
            "is_valid must be False when tables missing"
        assert len(result.missing_tables) > 0, \
            "missing_tables must be non-empty when invalid"
        assert result.error_message != "", \
            "error_message must be non-empty when invalid"


class TestBackwardCompatibility:
    """Contract tests for backward compatibility."""

    def test_backward_compatibility_without_package(self, schema_manager):
        """
        CONTRACT: Existing pipelines without iris-vector-graph unaffected.

        GIVEN iris-vector-graph is NOT installed
        WHEN existing pipeline initialization runs
        THEN no errors are raised
        AND pipeline proceeds normally
        AND no graph tables are created
        """
        with patch.object(schema_manager, '_detect_iris_vector_graph', return_value=False):
            # Call validation (should succeed gracefully)
            # This will fail initially with AttributeError
            result = schema_manager.validate_graph_prerequisites()

            # Validation should indicate invalid but not crash
            assert result.is_valid is False, \
                "Validation should indicate prerequisites not met"
            assert result.package_installed is False, \
                "Should detect package not installed"

            # This is expected behavior - pipeline can proceed without graph features


@pytest.fixture
def schema_manager():
    """
    Fixture providing SchemaManager instance (no database).

    NOTE: This fixture will fail initially because SchemaManager
    does not yet have the validate_graph_prerequisites() method.
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


@pytest.fixture
def schema_manager_with_tables(schema_manager_with_db):
    """
    Fixture providing SchemaManager with all graph tables created.

    Creates all 4 iris-vector-graph tables before test.
    """
    schema_manager = schema_manager_with_db

    # Create all tables (this will be done via ensure_iris_vector_graph_tables in real implementation)
    tables = ["rdf_labels", "rdf_props", "rdf_edges", "kg_NodeEmbeddings_optimized"]
    for table in tables:
        schema_manager.ensure_table_schema(table, pipeline_type="graphrag")

    yield schema_manager

    # Cleanup
    for table in tables:
        try:
            schema_manager.drop_table(table)
        except Exception:
            pass


# Expected test results for TDD phase:
# test_prerequisite_validation_all_met: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_prerequisite_validation_package_missing: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_prerequisite_validation_tables_missing: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_prerequisite_validation_before_ppr: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_clear_error_when_tables_missing: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_validation_performance: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_validation_is_stateless: FAIL (AttributeError: 'validate_graph_prerequisites')
# test_validation_result_invariants: PASS (data structure test, no implementation dependency)
# test_backward_compatibility_without_package: FAIL (AttributeError: 'validate_graph_prerequisites')
