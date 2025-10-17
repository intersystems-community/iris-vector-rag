# Contract: Test Execution Requirements

**Feature**: 025-fixes-for-testing
**Contract Type**: Infrastructure
**Date**: 2025-10-03

## Purpose

Define requirements for pytest test execution with IRIS database integration.

## Requirements

### REQ-1: pytest Configuration

**MUST**: pytest.ini file contains `-p no:randomly` flag to disable pytest-randomly plugin

**Rationale**: pytest-randomly causes `ValueError: Seed must be between 0 and 2**32 - 1` errors in numpy/thinc libraries used by sentence-transformers.

**Validation**:
```bash
grep -- "-p no:randomly" pytest.ini
# Expected: addopts line contains -p no:randomly
```

### REQ-2: PYTHONPATH Configuration

**MUST**: Tests run with `PYTHONPATH=/Users/tdyar/ws/rag-templates` to enable imports

**Rationale**: Tests need to import iris_rag and common modules from repository root.

**Validation**:
```bash
export PYTHONPATH=/Users/tdyar/ws/rag-templates
pytest tests/unit/ --collect-only
# Expected: No ModuleNotFoundError
```

### REQ-3: IRIS Connection Fixture

**MUST**: E2E tests have access to module-scoped IRIS connection fixture

**Signature**:
```python
@pytest.fixture(scope="module")
def iris_connection(config_manager: ConfigurationManager) -> Connection:
    """Provide IRIS database connection for E2E tests.

    Yields:
        Connection: Active IRIS database connection

    Cleanup:
        Closes connection after all tests in module complete
    """
    pass
```

**Validation**:
```python
def test_iris_connection_available(iris_connection):
    """Validate IRIS connection fixture is available."""
    assert iris_connection is not None
    cursor = iris_connection.cursor()
    cursor.execute("SELECT 1")
    assert cursor.fetchone()[0] == 1
```

### REQ-4: IRIS Health Check

**MUST**: Tests verify IRIS is running before execution

**Implementation**:
```python
def pytest_sessionstart(session):
    """Verify IRIS is healthy before running tests."""
    from common.iris_port_discovery import discover_iris_port
    from iris_rag.config.manager import ConfigurationManager

    config = ConfigurationManager()
    port = discover_iris_port()

    if port is None:
        pytest.exit("IRIS database not running. Start with: docker-compose up -d")

    # Test connection
    conn_manager = ConnectionManager(config)
    conn = conn_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    if result[0] != 1:
        pytest.exit("IRIS database connection failed health check")
```

**Validation**:
```bash
# IRIS running
docker ps | grep iris
# Expected: Container running on port 11972, 21972, or 31972

# Health check passes
pytest tests/ --collect-only
# Expected: No "IRIS database not running" error
```

### REQ-5: Test Markers

**MUST**: Tests use appropriate pytest markers

**Markers**:
- `@pytest.mark.e2e`: End-to-end tests requiring full IRIS setup
- `@pytest.mark.integration`: Integration tests requiring database but not full pipeline
- `@pytest.mark.requires_database`: Any test requiring IRIS connection
- `@pytest.mark.unit`: Unit tests with mocks (no IRIS)

**Configuration** (pytest.ini):
```ini
[pytest]
markers =
    e2e: End-to-end tests with real IRIS database
    integration: Integration tests requiring database
    requires_database: Tests that need IRIS connection
    unit: Unit tests with mocked dependencies
```

**Validation**:
```python
# tests/e2e/test_basic_pipeline_e2e.py
@pytest.mark.e2e
@pytest.mark.requires_database
def test_basic_pipeline_workflow(basic_pipeline, sample_documents):
    """Full E2E test of BasicRAG pipeline."""
    pass
```

### REQ-6: Test Execution Time

**MUST**: Full test suite executes in < 2 minutes

**Current**: ~106 seconds for 206 passing tests
**Target**: < 120 seconds for 300+ tests

**Rationale**: Fast feedback loop for developers

**Validation**:
```bash
time pytest tests/unit/ tests/e2e/ -p no:randomly
# Expected: real time < 2m0s
```

## Contract Tests

Location: `tests/contract/test_pytest_execution_contract.py`

```python
"""Contract tests for pytest execution requirements."""

import pytest
import subprocess
import time


def test_pytest_ini_has_no_randomly_flag():
    """REQ-1: pytest.ini disables pytest-randomly."""
    with open("pytest.ini", "r") as f:
        content = f.read()
    assert "-p no:randomly" in content, "pytest.ini must contain '-p no:randomly' flag"


def test_pythonpath_allows_imports():
    """REQ-2: PYTHONPATH enables module imports."""
    import sys
    sys.path.insert(0, "/Users/tdyar/ws/rag-templates")

    # Should not raise ModuleNotFoundError
    import iris_rag
    import common
    assert iris_rag is not None
    assert common is not None


@pytest.mark.requires_database
def test_iris_connection_fixture_available(iris_connection):
    """REQ-3: IRIS connection fixture is available in E2E tests."""
    assert iris_connection is not None
    cursor = iris_connection.cursor()
    cursor.execute("SELECT 1")
    assert cursor.fetchone()[0] == 1


@pytest.mark.requires_database
def test_iris_health_check_passes():
    """REQ-4: IRIS database is healthy."""
    from common.iris_port_discovery import discover_iris_port
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.connection import ConnectionManager

    port = discover_iris_port()
    assert port is not None, "IRIS must be running"

    config = ConfigurationManager()
    conn_manager = ConnectionManager(config)
    conn = conn_manager.get_connection()

    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    assert result[0] == 1, "IRIS health check failed"


def test_pytest_markers_configured():
    """REQ-5: pytest markers are properly configured."""
    result = subprocess.run(
        ["pytest", "--markers"],
        capture_output=True,
        text=True
    )

    assert "e2e:" in result.stdout, "e2e marker must be configured"
    assert "integration:" in result.stdout, "integration marker must be configured"
    assert "requires_database:" in result.stdout, "requires_database marker must be configured"
    assert "unit:" in result.stdout, "unit marker must be configured"


def test_test_suite_execution_time():
    """REQ-6: Full test suite runs in < 2 minutes."""
    start = time.time()

    result = subprocess.run(
        ["pytest", "tests/unit/", "tests/e2e/", "-p", "no:randomly", "-q", "--tb=no"],
        capture_output=True,
        text=True
    )

    duration = time.time() - start

    # Allow some variance for CI/CD environments
    assert duration < 150, f"Test suite took {duration:.1f}s, must be < 150s (2.5 min max)"

    # Report actual time
    print(f"Test suite execution time: {duration:.1f}s")
```

## Success Criteria

All contract tests pass:
- ✅ pytest.ini configured correctly
- ✅ PYTHONPATH enables imports
- ✅ IRIS connection fixture available
- ✅ IRIS health check passes
- ✅ pytest markers configured
- ✅ Test suite runs in < 2 minutes

## Dependencies

- pytest 8.4.1+
- pytest-cov 6.1.1+
- IRIS database running (docker-compose up -d)
- Python 3.12+

## Related Contracts

- [coverage_reporting_contract.md](./coverage_reporting_contract.md) - Coverage configuration
- [test_isolation_contract.md](./test_isolation_contract.md) - Test cleanup requirements
- [api_alignment_contract.md](./api_alignment_contract.md) - API contract validation
