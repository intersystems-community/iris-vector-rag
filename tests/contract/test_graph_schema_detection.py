"""
Contract tests for iris-vector-graph package detection.

Feature: 060-fix-users-tdyar
Status: TDD Phase - These tests MUST fail initially

These tests define the contract for iris-vector-graph detection functionality
BEFORE implementation. Following TDD principles (Constitution III), all tests
should fail with AttributeError or ImportError when first run.
"""

import pytest
from unittest.mock import patch, MagicMock
import importlib.util


class TestGraphPackageDetection:
    """Contract tests for _detect_iris_vector_graph() method."""

    def test_iris_vector_graph_detection_installed(self, schema_manager):
        """
        CONTRACT: _detect_iris_vector_graph() returns True when package installed.

        GIVEN iris-vector-graph is installed in the Python environment
        WHEN _detect_iris_vector_graph() is called
        THEN returns True
        AND does not import the package (no side effects)
        AND does not raise any exceptions
        """
        # Mock iris-vector-graph as installed
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_spec = MagicMock()
            mock_find_spec.return_value = mock_spec

            # This will fail initially with AttributeError:
            # 'SchemaManager' object has no attribute '_detect_iris_vector_graph'
            result = schema_manager._detect_iris_vector_graph()

            assert result is True, "Should detect installed package"
            mock_find_spec.assert_called_once_with("iris_vector_graph")

    def test_iris_vector_graph_detection_not_installed(self, schema_manager):
        """
        CONTRACT: _detect_iris_vector_graph() returns False when package not installed.

        GIVEN iris-vector-graph is NOT installed
        WHEN _detect_iris_vector_graph() is called
        THEN returns False
        AND does not raise any exceptions
        """
        # Mock iris-vector-graph as not installed
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_find_spec.return_value = None

            # This will fail initially with AttributeError
            result = schema_manager._detect_iris_vector_graph()

            assert result is False, "Should detect missing package"
            mock_find_spec.assert_called_once_with("iris_vector_graph")

    def test_detection_is_stateless(self, schema_manager):
        """
        CONTRACT: Package detection is stateless and repeatable.

        GIVEN _detect_iris_vector_graph() has been called once
        WHEN _detect_iris_vector_graph() is called again
        THEN returns same result without side effects
        AND does not modify any instance state
        """
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_spec = MagicMock()
            mock_find_spec.return_value = mock_spec

            # This will fail initially with AttributeError
            result1 = schema_manager._detect_iris_vector_graph()
            result2 = schema_manager._detect_iris_vector_graph()

            assert result1 == result2, "Repeated calls should return same result"
            assert result1 is True, "Should consistently detect installed package"

    def test_detection_handles_errors_gracefully(self, schema_manager):
        """
        CONTRACT: Detection handles errors without raising exceptions.

        GIVEN importlib.util.find_spec() raises an exception
        WHEN _detect_iris_vector_graph() is called
        THEN returns False without raising
        AND logs appropriate error message
        """
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_find_spec.side_effect = ImportError("Simulated import error")

            # This will fail initially with AttributeError
            result = schema_manager._detect_iris_vector_graph()

            # Should handle error gracefully, return False
            assert result is False, "Should return False on error"

    def test_detection_does_not_import_package(self, schema_manager):
        """
        CONTRACT: Detection uses find_spec, not import statement.

        GIVEN iris-vector-graph package check
        WHEN _detect_iris_vector_graph() is called
        THEN uses importlib.util.find_spec() for detection
        AND does NOT execute 'import iris_vector_graph'
        AND avoids package initialization side effects
        """
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_spec = MagicMock()
            mock_find_spec.return_value = mock_spec

            # This will fail initially with AttributeError
            result = schema_manager._detect_iris_vector_graph()

            # Verify find_spec was used (not direct import)
            mock_find_spec.assert_called_once_with("iris_vector_graph")
            assert result is True


class TestPackageDetectionIntegration:
    """Integration contract tests for package detection in real environment."""

    @pytest.mark.integration
    def test_detection_in_real_environment(self, schema_manager):
        """
        CONTRACT: Detection works in real Python environment.

        GIVEN a real Python environment (no mocks)
        WHEN _detect_iris_vector_graph() is called
        THEN returns boolean based on actual package availability
        AND result matches importlib.util.find_spec() directly
        """
        # This will fail initially with AttributeError
        result = schema_manager._detect_iris_vector_graph()

        # Verify against direct check
        expected = importlib.util.find_spec("iris_vector_graph") is not None

        assert result == expected, \
            f"Detection result ({result}) should match direct check ({expected})"
        assert isinstance(result, bool), "Result must be boolean type"


@pytest.fixture
def schema_manager():
    """
    Fixture providing SchemaManager instance for testing.

    NOTE: This fixture will fail initially because SchemaManager
    does not yet have the _detect_iris_vector_graph() method.
    """
    from iris_vector_rag.storage.schema_manager import SchemaManager
    from iris_vector_rag.config.config_manager import ConfigurationManager

    config = ConfigurationManager()
    manager = SchemaManager(
        connection_string=config.get("database.connection_string"),
        base_embedding_dimension=config.get("embeddings.dimension"),
    )

    return manager


# Expected test results for TDD phase:
# test_iris_vector_graph_detection_installed: FAIL (AttributeError: '_detect_iris_vector_graph')
# test_iris_vector_graph_detection_not_installed: FAIL (AttributeError: '_detect_iris_vector_graph')
# test_detection_is_stateless: FAIL (AttributeError: '_detect_iris_vector_graph')
# test_detection_handles_errors_gracefully: FAIL (AttributeError: '_detect_iris_vector_graph')
# test_detection_does_not_import_package: FAIL (AttributeError: '_detect_iris_vector_graph')
# test_detection_in_real_environment: FAIL (AttributeError: '_detect_iris_vector_graph')
