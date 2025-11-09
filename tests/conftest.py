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
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

# Add repository root to Python path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import framework modules
from iris_vector_rag.config.manager import ConfigurationManager

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


# ============================================================================
# Feature 035: Backend Mode Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def backend_configuration():
    """
    Session-scoped backend configuration fixture.

    Loads backend mode configuration from environment/config file.
    Logs configuration at session start per FR-012.

    Returns:
        BackendConfiguration instance
    """
    from iris_vector_rag.testing.backend_manager import (
        load_configuration,
        log_session_start,
    )

    config = load_configuration()
    log_session_start(config)
    return config


@pytest.fixture(scope="session")
def backend_mode(backend_configuration):
    """
    Session-scoped backend mode fixture.

    Returns:
        BackendMode enum value (COMMUNITY or ENTERPRISE)
    """
    return backend_configuration.mode


@pytest.fixture(scope="session")
def connection_pool(backend_configuration):
    """
    Session-scoped connection pool fixture.

    Creates connection pool with mode-appropriate limits:
    - Community: 1 connection
    - Enterprise: 999 connections

    Returns:
        ConnectionPool instance
    """
    from iris_vector_rag.testing.connection_pool import ConnectionPool

    return ConnectionPool(mode=backend_configuration.mode)


@pytest.fixture(scope="session")
def iris_devtools_bridge(backend_configuration):
    """
    Session-scoped iris-devtools bridge fixture.

    Provides access to iris-devtools container lifecycle and
    database state management operations.

    Returns:
        IrisDevToolsBridge instance

    Raises:
        IrisDevtoolsMissingError: If iris-devtools not available
    """
    from iris_vector_rag.testing.iris_devtools_bridge import IrisDevToolsBridge

    return IrisDevToolsBridge(
        iris_devtools_path=backend_configuration.iris_devtools_path
    )


@pytest.fixture
def iris_connection(connection_pool):
    """
    Function-scoped IRIS connection fixture.

    Acquires connection from pool, yields to test, then releases.
    Enforces mode-specific connection limits.

    Yields:
        Active IRIS database connection

    Raises:
        ConnectionPoolTimeout: If pool limit exceeded
    """
    with connection_pool.acquire(timeout=30.0) as conn:
        yield conn


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
    from iris_vector_rag.common.iris_connection_manager import get_iris_connection

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

    yield conn

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


# ============================================================================
# Feature 034: HybridGraphRAG Query Path Testing Fixtures
# ============================================================================


@pytest.fixture
def mock_iris_vector_graph_unavailable(mocker):
    """
    Mock iris_vector_graph as unavailable for testing graceful degradation.

    This fixture simulates the scenario where iris_vector_graph is not installed
    or cannot be imported, allowing tests to validate fallback behavior.

    Usage:
        def test_fallback(graphrag_pipeline, mock_iris_vector_graph_unavailable):
            # iris_vector_graph methods will raise AttributeError
            result = graphrag_pipeline.query("test", method="hybrid")
            assert result.metadata['retrieval_method'] == 'vector_fallback'
    """
    # Mock the IRIS_GRAPH_CORE_AVAILABLE flag if it exists
    try:
        mocker.patch('iris_rag.pipelines.hybrid_graphrag.IRIS_GRAPH_CORE_AVAILABLE', False)
    except AttributeError:
        pass  # Flag doesn't exist, no need to mock

    return mocker


@pytest.fixture
def mock_zero_results_retrieval(mocker):
    """
    Mock retrieval methods to return 0 results for testing fallback.

    Returns a callable that mocks a specific retrieval method to return empty results.

    Usage:
        def test_fallback(graphrag_pipeline, mock_zero_results_retrieval):
            mock_zero_results_retrieval(graphrag_pipeline, 'retrieve_via_hybrid_fusion')
            result = graphrag_pipeline.query("test", method="hybrid")
            assert result.metadata['retrieval_method'] == 'vector_fallback'
    """
    def mock_retrieval_method(pipeline, method_name):
        """Mock a specific retrieval method to return 0 results."""
        if hasattr(pipeline, 'retrieval_methods'):
            mocker.patch.object(
                pipeline.retrieval_methods,
                method_name,
                return_value=([], method_name.replace('retrieve_via_', ''))
            )
        else:
            # Fallback to mocking the private method directly
            mocker.patch.object(
                pipeline,
                f'_{method_name}',
                return_value=([], method_name.replace('retrieve_via_', ''))
            )

    return mock_retrieval_method


@pytest.fixture
def mock_connection_failure(mocker):
    """
    Mock iris_vector_graph connection failure for error handling tests.

    Returns a callable that mocks a specific retrieval method to raise ConnectionError.

    Usage:
        def test_error(graphrag_pipeline, mock_connection_failure):
            mock_connection_failure(graphrag_pipeline, 'retrieve_via_rrf')
            result = graphrag_pipeline.query("test", method="rrf")
            assert result.metadata['retrieval_method'] == 'vector_fallback'
    """
    def mock_retrieval_error(pipeline, method_name, error_msg="Connection failed"):
        """Mock a specific retrieval method to raise connection error."""
        if hasattr(pipeline, 'retrieval_methods'):
            mocker.patch.object(
                pipeline.retrieval_methods,
                method_name,
                side_effect=ConnectionError(error_msg)
            )
        else:
            mocker.patch.object(
                pipeline,
                f'_{method_name}',
                side_effect=ConnectionError(error_msg)
            )

    return mock_retrieval_error


# ============================================================================
# Feature 036: Pipeline Testing Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def basic_rag_pipeline(request):
    """
    Session-scoped fixture for BasicRAG pipeline.

    Creates BasicRAG pipeline with validation enabled for integration tests,
    disabled for contract tests. Tests API contracts, error handling, and dimension validation.

    Returns:
        BasicRAGPipeline instance

    Requirements: FR-001, FR-007
    """
    from iris_vector_rag import create_pipeline
    import os

    # Update port from environment if needed
    if 'IRIS_PORT' in os.environ and os.environ['IRIS_PORT'] != '1972':
        os.environ['IRIS_PORT'] = '1972'

    # Integration tests need validation, contract tests don't
    validate = 'integration' in str(request.node.fspath)

    return create_pipeline("basic", validate_requirements=validate)


@pytest.fixture(scope="session")
def crag_pipeline(request):
    """
    Session-scoped fixture for CRAG pipeline.

    Creates CRAG pipeline with validation for integration tests.
    Tests API contracts, error handling, fallback mechanisms, and dimension validation.

    Returns:
        CRAGPipeline instance

    Requirements: FR-001, FR-007, FR-015
    """
    from iris_vector_rag import create_pipeline
    import os

    if 'IRIS_PORT' in os.environ and os.environ['IRIS_PORT'] != '1972':
        os.environ['IRIS_PORT'] = '1972'

    validate = 'integration' in str(request.node.fspath)
    return create_pipeline("crag", validate_requirements=validate)


@pytest.fixture(scope="session")
def basic_rerank_pipeline(request):
    """
    Session-scoped fixture for BasicRerankRAG pipeline.

    Creates BasicRerankRAG pipeline with validation for integration tests.
    Tests API contracts, error handling, fallback mechanisms, and dimension validation.

    Returns:
        BasicRerankRAGPipeline instance

    Requirements: FR-001, FR-007, FR-016
    """
    from iris_vector_rag import create_pipeline
    import os

    if 'IRIS_PORT' in os.environ and os.environ['IRIS_PORT'] != '1972':
        os.environ['IRIS_PORT'] = '1972'

    validate = 'integration' in str(request.node.fspath)
    return create_pipeline("basic_rerank", validate_requirements=validate)


@pytest.fixture(scope="session")
def pylate_colbert_pipeline(request):
    """
    Session-scoped fixture for PyLateColBERT pipeline.

    Creates PyLateColBERT pipeline with validation for integration tests.
    Tests API contracts, error handling, fallback mechanisms, and dimension validation.

    Returns:
        PyLateColBERTPipeline instance

    Requirements: FR-001, FR-007, FR-015
    """
    from iris_vector_rag import create_pipeline
    import os

    if 'IRIS_PORT' in os.environ and os.environ['IRIS_PORT'] != '1972':
        os.environ['IRIS_PORT'] = '1972'

    validate = 'integration' in str(request.node.fspath)
    return create_pipeline("pylate_colbert", validate_requirements=validate)


@pytest.fixture
def sample_query():
    """
    Function-scoped fixture providing a sample test query.

    Returns:
        str: Sample query about diabetes symptoms

    Usage:
        def test_query(basic_rag_pipeline, sample_query):
            result = basic_rag_pipeline.query(sample_query)
            assert len(result['contexts']) > 0
    """
    return "What are the symptoms of diabetes?"


@pytest.fixture
def sample_documents():
    """
    Function-scoped fixture providing sample documents for testing.

    Returns:
        list: List of document dictionaries with doc_id, title, content, metadata

    Usage:
        def test_load(basic_rag_pipeline, sample_documents):
            result = basic_rag_pipeline.load_documents(sample_documents)
            assert result['documents_loaded'] > 0
    """
    import json
    import os

    # Load from sample data file
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_pmc_docs_basic.json')

    with open(data_path, 'r') as f:
        data = json.load(f)

    return data['documents']
