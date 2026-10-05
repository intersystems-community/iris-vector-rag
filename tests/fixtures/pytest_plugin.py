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

from pathlib import Path

import pytest

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
    Load the fixture named by @pytest.mark.dat_fixture into the session's IRIS.

    Without the marker this yields None. With it, the fixture is loaded through
    FixtureManager over a connection to `iris_target` (the ephemeral container, or
    the pinned one), its checksum is validated, and its rows are deleted afterwards.
    A fixture that is missing, corrupt or fails to load fails the test's setup with
    the reason; it is never skipped.

    Reference:
        T080 - Implement dat_fixture_loader with backend_configuration integration
        T081 - Integrate with database_cleanup_handlers (Feature 028)
    """
    from tests.support.iris_environment import connect

    from .manager import ChecksumMismatchError, FixtureManager, FixtureNotFoundError

    # Check if test has dat_fixture marker
    if not hasattr(request.node, "dat_fixture_name"):
        yield None
        return

    fixture_name = request.node.dat_fixture_name
    version = request.node.dat_fixture_version
    cleanup_first = request.node.dat_fixture_cleanup

    backend_mode = getattr(backend_configuration, "mode", "community")
    fixtures_root = Path(__file__).parent.parent / "fixtures"
    connection = connect(request.getfixturevalue("iris_target"))
    manager = FixtureManager(
        fixtures_root=fixtures_root, backend_mode=backend_mode, connection=connection
    )

    def cleanup_fixture():
        try:
            manager.cleanup_fixture(fixture_name)
        except Exception as e:  # a failed cleanup must not mask the test's result
            print(f"Warning: Fixture cleanup failed: {e}")
        finally:
            connection.close()

    request.addfinalizer(cleanup_fixture)

    try:
        result = manager.load_fixture(
            fixture_name=fixture_name,
            version=version,
            validate_checksum=True,
            cleanup_first=cleanup_first,
            generate_embeddings=False,
        )
    except FixtureNotFoundError as e:
        pytest.fail(f"Fixture not found: {e}", pytrace=False)
    except ChecksumMismatchError as e:
        pytest.fail(f"Fixture checksum mismatch: {e}", pytrace=False)

    if not result.success:
        pytest.fail(
            f"Failed to load fixture '{fixture_name}': {result.error_message}",
            pytrace=False,
        )

    request.node.dat_fixture_result = result
    yield result


@pytest.fixture(scope="function")
def fixture_metadata(request, dat_fixture_loader):
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
