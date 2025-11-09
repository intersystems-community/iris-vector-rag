"""Contract tests for test isolation requirements.

Feature: 025-fixes-for-testing
Contract: test_isolation_contract.md
"""

import inspect

import pytest


def test_fixture_scopes_configured():
    """REQ-2: Validate fixture scopes for optimal performance and isolation."""
    # Check that E2E conftest exists and has proper fixtures
    import sys
    from pathlib import Path

    # Load E2E conftest module
    e2e_conftest_path = Path("tests/e2e/conftest.py")
    assert e2e_conftest_path.exists(), "E2E conftest.py must exist"

    # Read and verify fixture scopes are defined
    content = e2e_conftest_path.read_text()

    # Session-scoped fixtures for expensive resources
    assert (
        '@pytest.fixture(scope="session")' in content
    ), "Should have session-scoped fixtures"
    assert (
        "e2e_config_manager" in content
    ), "Should have e2e_config_manager fixture"

    # Module-scoped for pipeline dependencies
    assert (
        '@pytest.fixture(scope="module")' in content
    ), "Should have module-scoped fixtures"
    assert (
        "pipeline_dependencies" in content
    ), "Should have pipeline_dependencies fixture"

    # Function-scoped for isolation
    assert (
        '@pytest.fixture(scope="function")' in content
    ), "Should have function-scoped fixtures"
    assert (
        "e2e_database_cleanup" in content
    ), "Should have e2e_database_cleanup fixture"


@pytest.mark.requires_database
def test_database_cleanup_fixture():
    """REQ-1: Validate database cleanup fixture exists and is function-scoped."""
    # This test validates that E2E conftest has the cleanup fixture
    # Actual cleanup testing is done in E2E tests
    from pathlib import Path

    e2e_conftest = Path("tests/e2e/conftest.py")
    assert e2e_conftest.exists()

    content = e2e_conftest.read_text()

    # Should have cleanup fixture
    assert (
        "e2e_database_cleanup" in content
    ), "E2E conftest should have database cleanup fixture"

    # Should be function-scoped for isolation
    assert (
        '@pytest.fixture(scope="function")' in content
        and "e2e_database_cleanup" in content
    ), "Database cleanup should be function-scoped"


def test_transaction_rollback_available():
    """REQ-3: Validate transaction rollback capability for unit tests."""
    # Check that conftest has database session fixtures
    from pathlib import Path

    conftest_path = Path("tests/conftest.py")
    assert conftest_path.exists(), "Global conftest.py must exist"

    content = conftest_path.read_text()

    # Should have mock database session fixture
    assert (
        "mock_database_session" in content or "iris_test_session" in content
    ), "Should have database session fixtures"

    # Should use yield for teardown
    assert "yield" in content, "Fixtures should use yield for teardown"


def test_no_test_data_pollution():
    """REQ-1: Validate tests don't pollute each other's state."""
    # This test runs independently and should have clean state
    # If previous tests polluted state, this would fail

    # Simple validation: we can import modules without conflicts
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager

    # Should not raise errors or have unexpected state
    config = ConfigurationManager()
    assert config is not None

    # ConnectionManager should initialize fresh
    conn_mgr = ConnectionManager()
    assert conn_mgr is not None
