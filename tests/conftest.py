"""Global pytest configuration and fixtures for the RAG templates framework.

This module contains shared pytest fixtures and configuration that are used
across all test modules. It provides common setup and teardown functionality,
test data, and mock objects.

Note: pytest-randomly has been disabled due to incompatibility with thinc/numpy
random seeding (ValueError: Seed must be between 0 and 2**32 - 1).
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add repository root to Python path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import framework modules
from iris_rag.config.manager import ConfigurationManager

# Coverage testing imports
import coverage
from datetime import datetime
import subprocess
import sys


# Removed custom event_loop fixture to avoid deprecation warning
# pytest-asyncio will handle event loop management automatically


# Coverage Testing Fixtures
@pytest.fixture(scope="session")
def coverage_instance():
    """Create a coverage instance for test coverage measurement."""
    cov = coverage.Coverage(
        source=["iris_rag", "common"],
        omit=[
            "*/tests/*",
            "*/test_*",
            "*/__pycache__/*",
            "*/venv/*",
            "*/.venv/*",
        ],
        config_file=True
    )
    cov.start()
    yield cov
    cov.stop()
    cov.save()


@pytest.fixture
def coverage_context():
    """Provide context for coverage measurement in individual tests."""
    return {
        "start_time": datetime.now(),
        "module_name": None,
        "test_name": None,
    }


@pytest.fixture
def iris_database_config():
    """IRIS database configuration for coverage testing per constitutional requirements."""
    # Use port discovery to find available IRIS instance
    iris_ports = [11972, 21972, 1972]  # Default, Licensed, System

    for port in iris_ports:
        try:
            # Test connection using subprocess to avoid import issues
            result = subprocess.run([
                sys.executable, "-c",
                f"""
import sqlalchemy_iris
from sqlalchemy import create_engine, text
try:
    engine = create_engine(f'iris://_SYSTEM:SYS@localhost:{port}/USER')
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('SUCCESS')
except Exception:
    print('FAILED')
"""
            ], capture_output=True, text=True, timeout=5)

            if "SUCCESS" in result.stdout:
                return {
                    "host": "localhost",
                    "port": port,
                    "username": "_SYSTEM",
                    "password": "SYS",
                    "namespace": "USER",
                    "connection_string": f"iris://_SYSTEM:SYS@localhost:{port}/USER"
                }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue

    # No IRIS instance found, return mock config for unit tests
    return {
        "host": "localhost",
        "port": 1972,
        "username": "_SYSTEM",
        "password": "SYS",
        "namespace": "USER",
        "connection_string": "iris://_SYSTEM:SYS@localhost:1972/USER",
        "mock": True
    }


@pytest.fixture
def iris_test_session(iris_database_config):
    """Create IRIS database session for testing with constitutional compliance."""
    if iris_database_config.get("mock", False):
        # Return mock session for unit tests when IRIS not available
        yield Mock()
        return

    try:
        import sqlalchemy_iris
        from sqlalchemy import create_engine

        engine = create_engine(iris_database_config["connection_string"])

        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")

        yield engine

        # Cleanup: Drop any test tables created during testing
        with engine.connect() as conn:
            try:
                conn.execute("DROP TABLE IF EXISTS test_coverage_data")
                conn.execute("DROP TABLE IF EXISTS test_module_coverage")
                conn.commit()
            except Exception:
                pass  # Ignore cleanup errors

    except Exception as e:
        # Fall back to mock for unit tests
        yield Mock()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide mock configuration data for tests."""
    return {
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False,
        },
        "redis": {
            "url": "redis://localhost:6379/0",
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-api-key",
        },
        "vector_store": {
            "provider": "iris",
            "connection_string": "localhost:1972/USER",
        },
        "memory": {
            "provider": "mem0",
            "config": {
                "vector_store": {
                    "provider": "iris",
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4",
                    },
                },
            },
        },
    }


@pytest.fixture
def config_manager(temp_dir: Path, mock_config: Dict[str, Any]) -> ConfigurationManager:
    """Create a ConfigurationManager instance for testing."""
    config_file = temp_dir / "test_config.yaml"

    # Write config to temporary file
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)

    # Set environment variable to point to test config
    os.environ["RAG_TEMPLATES_CONFIG"] = str(config_file)

    try:
        manager = ConfigurationManager()
        yield manager
    finally:
        # Clean up environment variable
        if "RAG_TEMPLATES_CONFIG" in os.environ:
            del os.environ["RAG_TEMPLATES_CONFIG"]


@pytest.fixture
def iris_config_manager(
    temp_dir: Path, mock_config: Dict[str, Any]
) -> ConfigurationManager:
    """Create an IRIS ConfigManager instance for testing."""
    config_file = temp_dir / "iris_config.yaml"

    import yaml

    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)

    manager = ConfigurationManager(config_path=str(config_file))
    return manager


@pytest.fixture
def mock_database_session():
    """Create a mock database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get = Mock(return_value=None)
    mock_redis.set = Mock(return_value=True)
    mock_redis.delete = Mock(return_value=1)
    mock_redis.exists = Mock(return_value=False)
    mock_redis.expire = Mock(return_value=True)
    return mock_redis


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Generated response")
    mock_llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock_llm


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    mock_store = Mock()
    mock_store.add_documents = Mock(return_value=["doc1", "doc2"])
    mock_store.similarity_search = Mock(return_value=[])
    mock_store.similarity_search_with_score = Mock(return_value=[])
    return mock_store


@pytest.fixture
def mock_mem0_client():
    """Create a mock Mem0 client for testing."""
    mock_mem0 = AsyncMock()
    mock_mem0.add = AsyncMock(return_value={"id": "memory-123"})
    mock_mem0.search = AsyncMock(return_value=[])
    mock_mem0.get = AsyncMock(return_value=None)
    mock_mem0.delete = AsyncMock(return_value=True)
    return mock_mem0


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "This is a sample document about artificial intelligence.",
            "metadata": {"source": "test", "type": "article"},
        },
        {
            "id": "doc2",
            "content": "This document discusses machine learning algorithms.",
            "metadata": {"source": "test", "type": "research"},
        },
        {
            "id": "doc3",
            "content": "A comprehensive guide to retrieval-augmented generation.",
            "metadata": {"source": "test", "type": "guide"},
        },
    ]


@pytest.fixture
def sample_queries():
    """Provide sample queries for testing."""
    return [
        "What is artificial intelligence?",
        "How do machine learning algorithms work?",
        "Explain retrieval-augmented generation",
        "What are the benefits of RAG systems?",
    ]


@pytest.fixture
def mock_pipeline():
    """Create a mock RAG pipeline for testing."""
    mock_pipeline = Mock()
    mock_pipeline.process = Mock(
        return_value={"query": "test query", "response": "test response", "sources": []}
    )
    return mock_pipeline


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Cleanup environment variables after each test."""
    original_env = os.environ.copy()
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def docker_services():
    """Wait for Docker services to be ready."""
    # This fixture can be used with pytest-docker to wait for services
    # Implementation would depend on the specific services needed
    pass


# Async fixtures for testing async code
@pytest_asyncio.fixture
async def async_mock_llm_client():
    """Create an async mock LLM client for testing."""
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="Async generated response")
    mock_llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock_llm


@pytest_asyncio.fixture
async def async_mock_mem0_client():
    """Create an async mock Mem0 client for testing."""
    mock_mem0 = AsyncMock()
    mock_mem0.add = AsyncMock(return_value={"id": "async-memory-123"})
    mock_mem0.search = AsyncMock(return_value=[])
    mock_mem0.get = AsyncMock(return_value=None)
    mock_mem0.delete = AsyncMock(return_value=True)
    return mock_mem0


# Markers for different test categories
pytest_plugins = ["pytest_asyncio"]


# Configure test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_docker: mark test as requiring Docker")
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet connection"
    )
    # Constitutional compliance markers for coverage testing
    config.addinivalue_line("markers", "requires_database: mark test as requiring live IRIS database per constitution")
    config.addinivalue_line("markers", "clean_iris: mark test as requiring fresh/clean IRIS instance per constitution")
    config.addinivalue_line("markers", "coverage_critical: mark test as critical for coverage measurement")
    config.addinivalue_line("markers", "performance: mark test as performance validation")


def pytest_sessionstart(session):
    """Verify IRIS database is available before running tests."""
    # Skip health check if only running unit tests
    if session.config.getoption("markexpr") == "unit":
        return

    # Skip if running specific unit test files
    test_paths = [str(item.fspath) for item in session.items]
    if all("unit" in path for path in test_paths):
        return

    # Try to find IRIS on common ports
    iris_ports = [11972, 21972, 1972]
    iris_available = False

    for port in iris_ports:
        try:
            result = subprocess.run([
                sys.executable, "-c",
                f"""
import sqlalchemy_iris
from sqlalchemy import create_engine, text
try:
    engine = create_engine(f'iris://_SYSTEM:SYS@localhost:{port}/USER')
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('SUCCESS')
except Exception:
    print('FAILED')
"""
            ], capture_output=True, text=True, timeout=5)

            if "SUCCESS" in result.stdout:
                iris_available = True
                break
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue

    if not iris_available and any("e2e" in path or "integration" in path for path in test_paths):
        pytest.exit(
            "IRIS database not running. E2E and integration tests require IRIS.\n"
            "Start IRIS with: docker-compose up -d\n"
            "Verify with: docker logs iris-pgwire-db --tail 50",
            returncode=1
        )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# Performance monitoring for T015
_test_start_times = {}
_slow_tests = []
_suite_start_time = None


def pytest_runtest_setup(item):
    """Capture test start time."""
    global _test_start_times
    _test_start_times[item.nodeid] = datetime.now()


def pytest_runtest_teardown(item):
    """Calculate test duration and warn if slow."""
    global _test_start_times, _slow_tests

    if item.nodeid not in _test_start_times:
        return

    start_time = _test_start_times[item.nodeid]
    duration = (datetime.now() - start_time).total_seconds()

    # Warn if individual test exceeds 5 seconds
    if duration > 5.0:
        _slow_tests.append((item.nodeid, duration))
        print(f"\n⚠️  SLOW TEST: {item.nodeid} took {duration:.2f}s (>5s threshold)")


def pytest_sessionfinish(session, exitstatus):
    """Report slowest tests and check total suite time."""
    global _slow_tests, _suite_start_time

    # Report slowest 10 tests
    if _slow_tests:
        print(f"\n{'=' * 60}")
        print(f"Slowest {min(10, len(_slow_tests))} Tests:")
        print(f"{'=' * 60}")

        sorted_slow = sorted(_slow_tests, key=lambda x: x[1], reverse=True)[:10]
        for test_name, duration in sorted_slow:
            print(f"  {duration:.2f}s - {test_name}")
        print()


# Feature 028: Test Infrastructure Resilience Fixtures

@pytest.fixture(scope="class")
def database_with_clean_schema(request):
    """
    Provide clean IRIS database with valid schema for test class.

    Validates schema before tests run, resets if needed.
    Registers cleanup handler to remove test data after class.

    Implements FR-001, FR-004 from Feature 028.
    """
    from tests.fixtures.database_state import TestDatabaseState, TestStateRegistry
    from tests.fixtures.database_cleanup import DatabaseCleanupHandler
    from tests.utils.schema_validator import SchemaValidator
    from tests.fixtures.schema_reset import SchemaResetter
    from common.iris_connection_manager import get_iris_connection

    # Validate schema
    validator = SchemaValidator()
    validation_result = validator.validate_schema()

    if not validation_result.is_valid:
        # Schema invalid - reset
        resetter = SchemaResetter()
        resetter.reset_schema()

    # Create test state
    test_class = request.cls.__name__ if request.cls else "unknown"
    test_state = TestDatabaseState.create_for_test(test_class)

    # Register state
    registry = TestStateRegistry()
    registry.register_state(test_state)

    # Get database connection
    conn = get_iris_connection()

    # Register cleanup handler - ALWAYS runs
    def cleanup():
        handler = DatabaseCleanupHandler(conn, test_state.test_run_id)
        handler.cleanup()
        registry.remove_state(test_state.test_run_id)

    request.addfinalizer(cleanup)

    yield conn, test_state

    # Cleanup runs here automatically via addfinalizer


@pytest.fixture(scope="session")
def validate_schema_once():
    """
    Validate database schema once at session start.

    Implements FR-015 from Feature 028 (pre-flight checks).
    """
    from tests.utils.preflight_checks import PreflightChecker

    checker = PreflightChecker()
    results = checker.run_all_checks()

    # Exit if critical checks fail
    if not all(r.passed for r in results):
        pytest.exit(
            "Pre-flight checks failed. Cannot proceed with tests.\n"
            "Run preflight checks manually: python tests/utils/preflight_checks.py",
            returncode=1
        )

    return results
