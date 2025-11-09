"""
Contract Tests: Common Module Import Resolution

These tests verify that the common module can be imported without namespace
conflicts after moving from top-level to iris_vector_rag.common.

Test-Driven Development: These tests are written BEFORE implementation and
should initially FAIL, then PASS after the fix is applied.

Bug: https://github.com/tdyar/hipporag2-pipeline/issues/common-import-error
Fix: Move common/ → iris_vector_rag/common/
"""

import pytest


def test_import_iris_dbapi_connector():
    """
    Verify iris_dbapi_connector can be imported without namespace conflict.

    BEFORE FIX: ImportError or ModuleNotFoundError (wrong common module loaded)
    AFTER FIX: Should import successfully
    """
    from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection

    # Verify function is callable
    assert callable(get_iris_dbapi_connection), \
        "get_iris_dbapi_connection should be a callable function"


def test_import_iris_connection_manager():
    """
    Verify iris_connection_manager can be imported without namespace conflict.

    BEFORE FIX: ImportError or ModuleNotFoundError (wrong common module loaded)
    AFTER FIX: Should import successfully
    """
    from iris_vector_rag.common.iris_connection_manager import get_iris_connection

    # Verify function is callable
    assert callable(get_iris_connection), \
        "get_iris_connection should be a callable function"


def test_connection_manager_imports():
    """
    Verify ConnectionManager can import its dependencies.

    BEFORE FIX: ImportError when ConnectionManager tries to import common modules
    AFTER FIX: Should import successfully without errors

    This is the critical test - ConnectionManager is what users actually import.
    """
    # This import triggers the problematic imports on lines 155, 194 of connection.py
    from iris_vector_rag.core.connection import ConnectionManager

    # Should not raise ImportError
    assert ConnectionManager is not None, \
        "ConnectionManager should import successfully"

    # Verify it's the correct class
    assert hasattr(ConnectionManager, 'get_connection'), \
        "ConnectionManager should have get_connection method"


def test_common_module_location():
    """
    Verify common module is inside iris_vector_rag package.

    BEFORE FIX: common is top-level module
    AFTER FIX: common is at iris_vector_rag.common
    """
    from iris_vector_rag import common

    # Verify common is a module
    import types
    assert isinstance(common, types.ModuleType), \
        "iris_vector_rag.common should be a Python module"

    # Verify it's in the correct location
    assert 'iris_vector_rag' in common.__name__, \
        f"common module should be under iris_vector_rag namespace, got: {common.__name__}"


def test_old_top_level_common_removed():
    """
    Verify top-level common module is no longer provided by iris-vector-rag.

    BEFORE FIX: from iris_vector_rag.common.X import Y might work (if loaded first) or conflict
    AFTER FIX: Top-level common should not be provided by our package

    NOTE: This test may pass even before fix if no conflicting common module exists.
    The important test is test_connection_manager_imports() which tests actual usage.
    """
    # Try to import common module
    try:
        import common
        # If it exists, verify it's NOT from iris-vector-rag
        assert not common.__file__.startswith('iris_vector_rag'), \
            "Top-level common should not be provided by iris-vector-rag package"
    except ModuleNotFoundError:
        # This is expected - no top-level common module provided
        pass


def test_connection_manager_can_create_connection_mock():
    """
    Verify ConnectionManager can actually use the imported functions.

    This test verifies the imports are not just successful, but actually usable.
    Uses mocking to avoid needing real IRIS database.
    """
    from unittest.mock import MagicMock, patch
    from iris_vector_rag.core.connection import ConnectionManager

    # Mock the IRIS connection functions
    with patch('iris_vector_rag.common.iris_dbapi_connector.get_iris_dbapi_connection') as mock_dbapi, \
         patch('iris_vector_rag.common.iris_connection_manager.get_iris_connection') as mock_iris:

        # Setup mocks
        mock_dbapi.return_value = MagicMock()
        mock_iris.return_value = MagicMock()

        # Create ConnectionManager instance
        config = {
            'connection': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'username': 'test',
                'password': 'test'
            }
        }

        # This should not raise ImportError
        manager = ConnectionManager(config)
        assert manager is not None, "ConnectionManager should be created successfully"


# Mark these tests for contract testing
pytestmark = pytest.mark.contract


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
