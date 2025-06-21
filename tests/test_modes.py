"""
Test Mode Configuration
======================

This module provides clear test mode configuration to control when mocks are active
and when real end-to-end testing is performed.

Test Modes:
- UNIT: Fast unit tests with mocks (development)
- INTEGRATION: Integration tests with some real components
- E2E: Full end-to-end tests with real database and data (final validation)
"""

import os
import pytest
from enum import Enum
from typing import Optional


class TestMode(Enum):
    """Test execution modes."""
    UNIT = "unit"           # Fast tests with mocks
    INTEGRATION = "integration"  # Mixed real/mock tests
    E2E = "e2e"            # Full real end-to-end tests


class MockController:
    """
    Controls when mocks are active vs inactive.
    Provides clear visibility into mock state.
    """
    
    _current_mode: Optional[TestMode] = None
    _mocks_disabled = False
    
    @classmethod
    def set_test_mode(cls, mode: TestMode):
        """Set the current test mode."""
        cls._current_mode = mode
        cls._mocks_disabled = (mode == TestMode.E2E)
        
        # Set environment variable for other modules
        os.environ["RAG_TEST_MODE"] = mode.value
        os.environ["RAG_MOCKS_DISABLED"] = str(cls._mocks_disabled)
    
    @classmethod
    def get_test_mode(cls) -> TestMode:
        """Get current test mode."""
        if cls._current_mode is None:
            # Auto-detect from environment or default to UNIT
            mode_str = os.environ.get("RAG_TEST_MODE", "unit")
            try:
                cls._current_mode = TestMode(mode_str)
            except ValueError:
                cls._current_mode = TestMode.UNIT
        return cls._current_mode
    
    @classmethod
    def are_mocks_disabled(cls) -> bool:
        """Check if mocks are disabled (E2E mode)."""
        if cls._mocks_disabled is None:
            mode = cls.get_test_mode()
            cls._mocks_disabled = (mode == TestMode.E2E)
        return cls._mocks_disabled
    
    @classmethod
    def require_real_database(cls) -> bool:
        """Check if real database is required."""
        return cls.get_test_mode() in [TestMode.INTEGRATION, TestMode.E2E]
    
    @classmethod
    def require_real_data(cls) -> bool:
        """Check if real data is required."""
        return cls.get_test_mode() == TestMode.E2E
    
    @classmethod
    def skip_if_mocks_disabled(cls, reason: str = "Test requires mocks"):
        """Pytest skip decorator for tests that require mocks."""
        return pytest.mark.skipif(
            cls.are_mocks_disabled(),
            reason=f"{reason} (mocks disabled in {cls.get_test_mode().value} mode)"
        )
    
    @classmethod
    def skip_if_not_e2e(cls, reason: str = "Test requires E2E mode"):
        """Pytest skip decorator for E2E-only tests."""
        return pytest.mark.skipif(
            cls.get_test_mode() != TestMode.E2E,
            reason=f"{reason} (current mode: {cls.get_test_mode().value})"
        )
    
    @classmethod
    def skip_if_no_real_db(cls, reason: str = "Test requires real database"):
        """Pytest skip decorator for tests requiring real database."""
        return pytest.mark.skipif(
            not cls.require_real_database(),
            reason=f"{reason} (current mode: {cls.get_test_mode().value})"
        )


def mock_safe(mock_func):
    """
    Decorator to make mock usage safe.
    Raises error if mocks are disabled in E2E mode.
    """
    def wrapper(*args, **kwargs):
        if MockController.are_mocks_disabled():
            raise RuntimeError(
                f"Mocks are disabled in {MockController.get_test_mode().value} mode! "
                f"This test should not use mocks in E2E validation."
            )
        return mock_func(*args, **kwargs)
    return wrapper


# Pytest markers for different test types
pytest_markers = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration, 
    "e2e": pytest.mark.e2e,
    "requires_real_db": pytest.mark.requires_real_db,
    "requires_real_data": pytest.mark.requires_real_data,
    "no_mocks": pytest.mark.no_mocks
}