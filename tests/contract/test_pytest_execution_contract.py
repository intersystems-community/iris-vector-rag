"""Contract tests for pytest execution requirements.

Feature: 025-fixes-for-testing
Contract: test_execution_contract.md
"""

import os
import subprocess
import sys
import time

import pytest


def test_test_order_is_not_randomized(pytestconfig):
    """REQ-1: pytest-randomly does not reorder the suite.

    Asserted against the live config rather than the text of a config file: the
    settings moved to pyproject.toml when pytest.ini turned out to be inert, and
    a test that greps a file cannot tell whether pytest read it.
    """
    if not pytestconfig.pluginmanager.hasplugin("randomly"):
        return

    addopts = " ".join(pytestconfig.getini("addopts"))
    assert "no:randomly" in addopts, (
        "pytest-randomly is installed and active, so test order changes every "
        "run. Known order dependencies are recorded in "
        "tests/KNOWN_ORDER_DEPENDENCIES.md. Disable it with '-p no:randomly' in "
        "addopts."
    )


def test_pythonpath_allows_imports():
    """REQ-2: PYTHONPATH enables module imports."""
    # PYTHONPATH already set in conftest.py
    # Should not raise ModuleNotFoundError
    import iris_vector_rag
    from iris_vector_rag import common

    assert iris_vector_rag is not None
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

    # The port is the one this run was told to use (CI starts its throwaway container on
    # 1972 and sets IRIS_PORT); otherwise it is this project's container, 51972 (see
    # CLAUDE.md). Probing other projects' ports is how a test ends up asserting against
    # someone else's database.
    iris_ports = [int(os.environ.get("IRIS_PORT", "51972"))]
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


def test_pytest_markers_configured(pytestconfig):
    """REQ-5: pytest markers are properly configured.

    Reads the config pytest is running under, so it cannot pass by grepping a
    file pytest ignored. tests/unit/test_pytest_config_is_live.py checks the
    whole marker set; these four are the ones this contract names.
    """
    declared = {
        line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")
    }

    for marker in ("e2e", "integration", "requires_database", "unit"):
        assert marker in declared, (
            f"the {marker} marker is not registered in the live pytest config, "
            f"so with --strict-markers every test using it is a collection error"
        )


# The test runs the whole unit suite in a child process (about 35 s on a CI runner), so
# it needs more than the 30 s every other contract test gets; the thread timeout method
# would otherwise end the entire pytest process, not just this test.
@pytest.mark.timeout(300)
def test_test_suite_execution_time():
    """REQ-6: Full test suite runs in < 2 minutes."""
    start = time.time()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/",
            "-p",
            "no:randomly",
            "-q",
            "--tb=no",
        ],
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
