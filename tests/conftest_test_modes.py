"""
Test Mode Configuration for pytest
=================================

This configures pytest to understand our test modes and provides
fixtures that respect the current test mode.
"""

import pytest
import os
from tests.test_modes import MockController, TestMode


def pytest_configure(config):
    """Configure pytest with our custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests with mocks allowed")
    config.addinivalue_line("markers", "integration: Integration tests with mixed real/mock")
    config.addinivalue_line("markers", "e2e: End-to-end tests with real components only")
    config.addinivalue_line("markers", "requires_real_db: Test requires real database")
    config.addinivalue_line("markers", "requires_real_data: Test requires real data (1000+ docs)")
    config.addinivalue_line("markers", "no_mocks: Test must not use any mocks")


def pytest_runtest_setup(item):
    """Setup for each test - check mode compatibility."""
    current_mode = MockController.get_test_mode()
    
    # Check if test is compatible with current mode
    if current_mode == TestMode.E2E:
        # In E2E mode, skip unit tests that are mock-only
        if item.get_closest_marker("unit") and not item.get_closest_marker("e2e"):
            pytest.skip(f"Skipping unit test in E2E mode: {item.name}")
    
    elif current_mode == TestMode.UNIT:
        # In unit mode, skip E2E tests that require real components
        if item.get_closest_marker("e2e") and not item.get_closest_marker("unit"):
            pytest.skip(f"Skipping E2E test in unit mode: {item.name}")


@pytest.fixture(scope="session", autouse=True)
def configure_test_mode():
    """Auto-configure test mode based on environment or pytest args."""
    # Check for explicit mode setting
    mode_str = os.environ.get("RAG_TEST_MODE")
    
    if not mode_str:
        # Auto-detect based on available resources
        try:
            from common.iris_connection import IRISConnection
            conn = IRISConnection()
            conn.connect()
            conn.disconnect()
            # If we can connect, default to integration mode
            mode_str = "integration"
        except:
            # If no database, default to unit mode
            mode_str = "unit"
    
    try:
        mode = TestMode(mode_str)
    except ValueError:
        mode = TestMode.UNIT
    
    MockController.set_test_mode(mode)
    
    print(f"\n🧪 Test Mode: {mode.value.upper()}")
    print(f"🎭 Mocks Disabled: {MockController.are_mocks_disabled()}")
    print(f"🗄️  Real Database Required: {MockController.require_real_database()}")
    print(f"📊 Real Data Required: {MockController.require_real_data()}")


@pytest.fixture
def test_mode():
    """Fixture providing current test mode."""
    return MockController.get_test_mode()


@pytest.fixture
def mocks_disabled():
    """Fixture indicating if mocks are disabled."""
    return MockController.are_mocks_disabled()


@pytest.fixture
def ensure_no_mocks():
    """Fixture that ensures no mocks are used in this test."""
    if not MockController.are_mocks_disabled():
        pytest.skip("Test requires mocks to be disabled (use E2E mode)")
    yield
    # Could add post-test validation here if needed