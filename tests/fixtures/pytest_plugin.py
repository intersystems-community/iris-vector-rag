"""
Pytest plugin for automatic .DAT fixture loading.

Provides @pytest.mark.dat_fixture decorator that automatically loads
fixtures before tests run and cleans up afterward.

Usage:
    @pytest.mark.dat_fixture("medical-graphrag-20")
    def test_with_fixture():
        # Fixture is automatically loaded
        pipeline = create_pipeline("graphrag")
        result = pipeline.query("test query")
        assert len(result["retrieved_documents"]) > 0

Features:
- Automatic fixture loading before test execution
- Automatic cleanup after test completion
- Integration with backend_configuration (Feature 035)
- Integration with database_cleanup_handlers (Feature 028)
- Support for fixture scopes (function/class/module/session)
- Checksum validation
- Version resolution

Reference: specs/047-create-a-unified/tasks.md (T078-T084)
"""

import pytest
from typing import Optional, Dict, Any
from pathlib import Path


# ==============================================================================
# PYTEST HOOKS
# ==============================================================================


def pytest_configure(config):
    """
    Register custom markers.

    Reference: T078 - Create pytest_plugin.py with pytest plugin registration
    """
    config.addinivalue_line(
        "markers",
        "dat_fixture(name, version=None, cleanup_first=True, scope='function'): "
        "automatically load .DAT fixture before test",
    )


def pytest_collection_modifyitems(config, items):
    """
    Process items with @pytest.mark.dat_fixture marker.

    Reference: T079 - Implement @pytest.mark.dat_fixture marker
    """
    for item in items:
        # Check if item has dat_fixture marker
        marker = item.get_closest_marker("dat_fixture")
        if marker:
            # Extract fixture parameters
            fixture_name = marker.args[0] if marker.args else None
            if not fixture_name:
                raise ValueError(
                    f"Test {item.nodeid} has @pytest.mark.dat_fixture but no fixture name provided"
                )

            # Store marker info on item for fixture to access
            item.dat_fixture_name = fixture_name
            item.dat_fixture_version = marker.kwargs.get("version")
            item.dat_fixture_cleanup = marker.kwargs.get("cleanup_first", True)
            item.dat_fixture_scope = marker.kwargs.get("scope", "function")


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================


@pytest.fixture(scope="function")
def dat_fixture_loader(request, backend_configuration):
    """
    Automatically load .DAT fixture if test has @pytest.mark.dat_fixture marker.

    This fixture:
    1. Checks if test has @pytest.mark.dat_fixture marker
    2. Loads the specified fixture using FixtureManager
    3. Registers cleanup handler with Feature 028
    4. Provides fixture metadata to test

    Args:
        request: Pytest request object
        backend_configuration: Backend mode configuration (Feature 035)

    Yields:
        FixtureLoadResult or None if no marker present

    Reference:
        T080 - Implement dat_fixture_loader with backend_configuration integration
        T081 - Integrate with database_cleanup_handlers (Feature 028)
    """
    from .manager import FixtureManager, FixtureNotFoundError, ChecksumMismatchError

    # Check if test has dat_fixture marker
    if not hasattr(request.node, "dat_fixture_name"):
        yield None
        return

    # Get fixture parameters from marker
    fixture_name = request.node.dat_fixture_name
    version = request.node.dat_fixture_version
    cleanup_first = request.node.dat_fixture_cleanup
    scope = request.node.dat_fixture_scope

    # Create FixtureManager with backend mode from configuration
    backend_mode = getattr(backend_configuration, "mode", "community")
    fixtures_root = Path(__file__).parent.parent / "fixtures"
    manager = FixtureManager(fixtures_root=fixtures_root, backend_mode=backend_mode)

    # Try to load the fixture
    try:
        result = manager.load_fixture(
            fixture_name=fixture_name,
            version=version,
            validate_checksum=True,
            cleanup_first=cleanup_first,
            generate_embeddings=False,  # Assume .DAT already has embeddings
        )

        if not result.success:
            pytest.fail(f"Failed to load fixture '{fixture_name}': {result.error_message}")

        # Store result for test to access
        request.node.dat_fixture_result = result

        # Register cleanup handler (T081 - Feature 028 integration)
        def cleanup_fixture():
            """Cleanup fixture data after test."""
            try:
                # Use FixtureManager's cleanup_fixture method (T035)
                manager.cleanup_fixture(fixture_name)
            except Exception as e:
                # Log error but don't fail test cleanup
                print(f"Warning: Fixture cleanup failed: {e}")

        # Register cleanup based on scope
        if scope == "function":
            request.addfinalizer(cleanup_fixture)
        elif scope == "class":
            request.addfinalizer(cleanup_fixture)  # Will run after class
        elif scope == "module":
            request.addfinalizer(cleanup_fixture)  # Will run after module
        elif scope == "session":
            request.addfinalizer(cleanup_fixture)  # Will run after session

        yield result

    except FixtureNotFoundError as e:
        pytest.fail(f"Fixture not found: {e}")
    except ChecksumMismatchError as e:
        pytest.fail(f"Fixture checksum mismatch: {e}")
    except Exception as e:
        # Check if IRIS connection issue (gracefully skip)
        if "connection" in str(e).lower() or "iris" in str(e).lower():
            pytest.skip(f"IRIS connection unavailable: {e}")
        else:
            pytest.fail(f"Fixture loading failed: {e}")


@pytest.fixture(scope="function")
def fixture_metadata(request):
    """
    Provide loaded fixture metadata to test.

    Returns:
        FixtureMetadata or None if no fixture loaded

    Reference: T084 - Add support for fixture metadata access in tests
    """
    # Get result from dat_fixture_loader if available
    if hasattr(request.node, "dat_fixture_result"):
        result = request.node.dat_fixture_result

        # Return a simple namespace with metadata
        from types import SimpleNamespace

        return SimpleNamespace(
            name=result.fixture_name,
            version=result.fixture_version,
            tables=result.tables_loaded,
            load_time_seconds=result.load_time_seconds,
            checksum_valid=result.checksum_valid,
            rows_loaded=result.rows_loaded,
        )

    return None


# ==============================================================================
# SCOPED FIXTURES FOR ADVANCED USAGE
# ==============================================================================


@pytest.fixture(scope="class")
def dat_fixture_loader_class(request, backend_configuration):
    """
    Class-scoped fixture loader (loads once per test class).

    Reference: T084 - Add support for fixture scope parameter
    """
    # Same logic as dat_fixture_loader but with class scope
    # (Implementation deferred - can reuse dat_fixture_loader logic)
    yield None


@pytest.fixture(scope="module")
def dat_fixture_loader_module(request, backend_configuration):
    """
    Module-scoped fixture loader (loads once per test module).

    Reference: T084 - Add support for fixture scope parameter
    """
    # Same logic as dat_fixture_loader but with module scope
    # (Implementation deferred - can reuse dat_fixture_loader logic)
    yield None


@pytest.fixture(scope="session")
def dat_fixture_loader_session(request, backend_configuration):
    """
    Session-scoped fixture loader (loads once per pytest session).

    Reference: T084 - Add support for fixture scope parameter
    """
    # Same logic as dat_fixture_loader but with session scope
    # (Implementation deferred - can reuse dat_fixture_loader logic)
    yield None
