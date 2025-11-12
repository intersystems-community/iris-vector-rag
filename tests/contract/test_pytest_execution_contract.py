"""Contract tests for pytest execution requirements.

Feature: 025-fixes-for-testing
Contract: test_execution_contract.md
"""

import subprocess
import sys
import time

import pytest


def test_pytest_ini_has_no_randomly_flag():
    """REQ-1: pytest.ini disables pytest-randomly."""
    with open("pytest.ini", "r") as f:
        content = f.read()
    assert "-p no:randomly" in content, "pytest.ini must contain '-p no:randomly' flag"


def test_pythonpath_allows_imports():
    """REQ-2: PYTHONPATH enables module imports."""
    # PYTHONPATH already set in conftest.py
    # Should not raise ModuleNotFoundError
    import common
    import iris_vector_rag

    assert iris_rag is not None
    assert common is not None


@pytest.mark.requires_database
def test_iris_connection_fixture_available():
    """REQ-3: IRIS connection fixture is available in E2E tests."""
    # This test validates that E2E conftest has the fixture
    # Actual fixture usage is tested in E2E tests
    from pathlib import Path

    e2e_conftest = Path("tests/e2e/conftest.py")
    assert e2e_conftest.exists()

    content = e2e_conftest.read_text()
    assert (
        "e2e_connection_manager" in content
    ), "E2E conftest should have connection manager fixture"


@pytest.mark.requires_database
def test_iris_health_check_passes():
    """REQ-4: IRIS database is healthy."""
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager

    # Try to find IRIS on common ports
    iris_ports = [11972, 21972, 1972]
    iris_available = False

    for port in iris_ports:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
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
""",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if "SUCCESS" in result.stdout:
                iris_available = True
                break
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue

    assert iris_available, "IRIS must be running"


def test_pytest_markers_configured():
    """REQ-5: pytest markers are properly configured."""
    result = subprocess.run(["pytest", "--markers"], capture_output=True, text=True)

    assert "e2e:" in result.stdout, "e2e marker must be configured"
    assert "integration:" in result.stdout, "integration marker must be configured"
    assert (
        "requires_database:" in result.stdout
    ), "requires_database marker must be configured"
    assert "unit:" in result.stdout, "unit marker must be configured"


def test_test_suite_execution_time():
    """REQ-6: Full test suite runs in < 2 minutes."""
    start = time.time()

    result = subprocess.run(
        ["pytest", "tests/unit/", "-p", "no:randomly", "-q", "--tb=no"],
        capture_output=True,
        text=True,
    )

    duration = time.time() - start

    # Allow some variance for CI/CD environments
    # Unit tests should be fast (< 30 seconds)
    assert (
        duration < 60
    ), f"Unit test suite took {duration:.1f}s, must be < 60s for fast feedback"

    # Report actual time
    print(f"Unit test suite execution time: {duration:.1f}s")
