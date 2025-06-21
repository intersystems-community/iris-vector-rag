"""
Test Mode Validator
==================

Tests to validate that the mock control system works correctly.
This ensures the test mode framework properly controls mock usage.
"""

import os
import pytest
from unittest.mock import Mock, patch
from tests.test_modes import MockController, TestMode, mock_safe


class TestMockController:
    """Test the MockController functionality."""
    
    def setup_method(self):
        """Reset MockController state before each test."""
        MockController._current_mode = None
        MockController._mocks_disabled = None
        # Clear environment variables
        for var in ["RAG_TEST_MODE", "RAG_MOCKS_DISABLED"]:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_mode_is_unit(self):
        """Test that default mode is UNIT when no environment is set."""
        mode = MockController.get_test_mode()
        assert mode == TestMode.UNIT
        assert not MockController.are_mocks_disabled()
    
    def test_set_unit_mode(self):
        """Test setting UNIT mode."""
        MockController.set_test_mode(TestMode.UNIT)
        
        assert MockController.get_test_mode() == TestMode.UNIT
        assert not MockController.are_mocks_disabled()
        assert not MockController.require_real_database()
        assert not MockController.require_real_data()
        assert os.environ["RAG_TEST_MODE"] == "unit"
        assert os.environ["RAG_MOCKS_DISABLED"] == "False"
    
    def test_set_integration_mode(self):
        """Test setting INTEGRATION mode."""
        MockController.set_test_mode(TestMode.INTEGRATION)
        
        assert MockController.get_test_mode() == TestMode.INTEGRATION
        assert not MockController.are_mocks_disabled()
        assert MockController.require_real_database()
        assert not MockController.require_real_data()
        assert os.environ["RAG_TEST_MODE"] == "integration"
        assert os.environ["RAG_MOCKS_DISABLED"] == "False"
    
    def test_set_e2e_mode(self):
        """Test setting E2E mode."""
        MockController.set_test_mode(TestMode.E2E)
        
        assert MockController.get_test_mode() == TestMode.E2E
        assert MockController.are_mocks_disabled()
        assert MockController.require_real_database()
        assert MockController.require_real_data()
        assert os.environ["RAG_TEST_MODE"] == "e2e"
        assert os.environ["RAG_MOCKS_DISABLED"] == "True"
    
    def test_environment_variable_detection(self):
        """Test that mode is detected from environment variables."""
        os.environ["RAG_TEST_MODE"] = "integration"
        
        # Reset internal state to force re-detection
        MockController._current_mode = None
        MockController._mocks_disabled = None
        
        mode = MockController.get_test_mode()
        assert mode == TestMode.INTEGRATION
    
    def test_invalid_environment_variable_defaults_to_unit(self):
        """Test that invalid environment variable defaults to UNIT."""
        os.environ["RAG_TEST_MODE"] = "invalid_mode"
        
        # Reset internal state to force re-detection
        MockController._current_mode = None
        MockController._mocks_disabled = None
        
        mode = MockController.get_test_mode()
        assert mode == TestMode.UNIT
    
    def test_skip_decorators(self):
        """Test the skip decorator functionality."""
        # Test skip_if_mocks_disabled in UNIT mode
        MockController.set_test_mode(TestMode.UNIT)
        decorator = MockController.skip_if_mocks_disabled()
        assert decorator.mark.name == "skipif"
        assert not decorator.mark.args[0]  # Should not skip in UNIT mode
        
        # Test skip_if_mocks_disabled in E2E mode
        MockController.set_test_mode(TestMode.E2E)
        decorator = MockController.skip_if_mocks_disabled()
        assert decorator.mark.name == "skipif"
        assert decorator.mark.args[0]  # Should skip in E2E mode
        
        # Test skip_if_not_e2e in UNIT mode
        MockController.set_test_mode(TestMode.UNIT)
        decorator = MockController.skip_if_not_e2e()
        assert decorator.mark.name == "skipif"
        assert decorator.mark.args[0]  # Should skip in UNIT mode
        
        # Test skip_if_not_e2e in E2E mode
        MockController.set_test_mode(TestMode.E2E)
        decorator = MockController.skip_if_not_e2e()
        assert decorator.mark.name == "skipif"
        assert not decorator.mark.args[0]  # Should not skip in E2E mode


class TestMockSafeDecorator:
    """Test the mock_safe decorator functionality."""
    
    def setup_method(self):
        """Reset MockController state before each test."""
        MockController._current_mode = None
        MockController._mocks_disabled = None
    
    def test_mock_safe_allows_mocks_in_unit_mode(self):
        """Test that mock_safe allows mocks in UNIT mode."""
        MockController.set_test_mode(TestMode.UNIT)
        
        @mock_safe
        def mock_function():
            return Mock()
        
        # Should not raise an error
        result = mock_function()
        assert isinstance(result, Mock)
    
    def test_mock_safe_allows_mocks_in_integration_mode(self):
        """Test that mock_safe allows mocks in INTEGRATION mode."""
        MockController.set_test_mode(TestMode.INTEGRATION)
        
        @mock_safe
        def mock_function():
            return Mock()
        
        # Should not raise an error
        result = mock_function()
        assert isinstance(result, Mock)
    
    def test_mock_safe_blocks_mocks_in_e2e_mode(self):
        """Test that mock_safe blocks mocks in E2E mode."""
        MockController.set_test_mode(TestMode.E2E)
        
        @mock_safe
        def mock_function():
            return Mock()
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            mock_function()
        
        assert "Mocks are disabled in e2e mode" in str(exc_info.value)
        assert "This test should not use mocks in E2E validation" in str(exc_info.value)
    
    def test_mock_safe_with_arguments(self):
        """Test that mock_safe works with function arguments."""
        MockController.set_test_mode(TestMode.UNIT)
        
        @mock_safe
        def mock_function_with_args(arg1, arg2, kwarg1=None):
            return Mock(return_value=f"{arg1}-{arg2}-{kwarg1}")
        
        result = mock_function_with_args("test", "value", kwarg1="keyword")
        assert isinstance(result, Mock)
        assert result.return_value == "test-value-keyword"


class TestModeIntegration:
    """Test integration between different components of the test mode system."""
    
    def setup_method(self):
        """Reset MockController state before each test."""
        MockController._current_mode = None
        MockController._mocks_disabled = None
    
    def test_e2e_mode_integration(self):
        """Test complete E2E mode integration."""
        # Set E2E mode
        MockController.set_test_mode(TestMode.E2E)
        
        # Verify all E2E characteristics
        assert MockController.get_test_mode() == TestMode.E2E
        assert MockController.are_mocks_disabled()
        assert MockController.require_real_database()
        assert MockController.require_real_data()
        
        # Verify mock_safe blocks mocks
        @mock_safe
        def should_fail():
            return Mock()
        
        with pytest.raises(RuntimeError):
            should_fail()
    
    def test_unit_mode_integration(self):
        """Test complete UNIT mode integration."""
        # Set UNIT mode
        MockController.set_test_mode(TestMode.UNIT)
        
        # Verify all UNIT characteristics
        assert MockController.get_test_mode() == TestMode.UNIT
        assert not MockController.are_mocks_disabled()
        assert not MockController.require_real_database()
        assert not MockController.require_real_data()
        
        # Verify mock_safe allows mocks
        @mock_safe
        def should_work():
            return Mock()
        
        result = should_work()
        assert isinstance(result, Mock)
    
    def test_mode_switching(self):
        """Test switching between different modes."""
        # Start in UNIT mode
        MockController.set_test_mode(TestMode.UNIT)
        assert not MockController.are_mocks_disabled()
        
        # Switch to E2E mode
        MockController.set_test_mode(TestMode.E2E)
        assert MockController.are_mocks_disabled()
        
        # Switch to INTEGRATION mode
        MockController.set_test_mode(TestMode.INTEGRATION)
        assert not MockController.are_mocks_disabled()
        assert MockController.require_real_database()
        
        # Switch back to UNIT mode
        MockController.set_test_mode(TestMode.UNIT)
        assert not MockController.are_mocks_disabled()
        assert not MockController.require_real_database()


class TestMockControlValidation:
    """Test validation of mock control in different scenarios."""
    
    def setup_method(self):
        """Reset MockController state before each test."""
        MockController._current_mode = None
        MockController._mocks_disabled = None
    
    def test_mock_detection_in_unit_tests(self):
        """Test that mocks are properly detected and allowed in unit tests."""
        MockController.set_test_mode(TestMode.UNIT)
        
        # This should work fine
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            assert os.path.exists("fake_path")
            mock_exists.assert_called_once_with("fake_path")
    
    def test_mock_detection_in_e2e_tests(self):
        """Test that mocks are properly detected and blocked in E2E tests."""
        MockController.set_test_mode(TestMode.E2E)
        
        # Direct mock usage should still work (we can't prevent all mocking)
        # but mock_safe decorator should prevent it
        @mock_safe
        def try_to_mock():
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                return os.path.exists("fake_path")
        
        with pytest.raises(RuntimeError):
            try_to_mock()
    
    def test_environment_consistency(self):
        """Test that environment variables stay consistent with mode."""
        # Test each mode
        for mode in [TestMode.UNIT, TestMode.INTEGRATION, TestMode.E2E]:
            MockController.set_test_mode(mode)
            
            # Check environment variables match
            assert os.environ["RAG_TEST_MODE"] == mode.value
            assert os.environ["RAG_MOCKS_DISABLED"] == str(MockController.are_mocks_disabled())
    
    def test_cross_module_consistency(self):
        """Test that mode settings are consistent across module imports."""
        MockController.set_test_mode(TestMode.E2E)
        
        # Import the module again to test consistency
        from tests.test_modes import MockController as ImportedController
        
        assert ImportedController.get_test_mode() == TestMode.E2E
        assert ImportedController.are_mocks_disabled()


@pytest.mark.unit
class TestUnitModeSpecific:
    """Tests that should only run in unit mode."""
    
    def test_unit_mode_mock_usage(self):
        """Test that demonstrates proper mock usage in unit mode."""
        # This test should be skipped in E2E mode
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked_value"
        
        assert mock_obj.method() == "mocked_value"
        mock_obj.method.assert_called_once()


@pytest.mark.e2e
class TestE2EModeSpecific:
    """Tests that should only run in E2E mode."""
    
    def test_e2e_mode_real_components(self):
        """Test that demonstrates real component usage in E2E mode."""
        # This test should be skipped in unit mode
        # In real E2E mode, this would test actual components
        assert MockController.get_test_mode() == TestMode.E2E
        assert MockController.are_mocks_disabled()


@pytest.mark.integration
class TestIntegrationModeSpecific:
    """Tests that should run in integration mode."""
    
    def test_integration_mode_mixed_usage(self):
        """Test that demonstrates mixed real/mock usage in integration mode."""
        # Explicitly set integration mode for this test
        MockController.set_test_mode(TestMode.INTEGRATION)
        
        # This would typically use real database but mock external services
        assert MockController.require_real_database()
        assert not MockController.require_real_data()


class TestMockControllerEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Reset MockController state before each test."""
        MockController._current_mode = None
        MockController._mocks_disabled = None
    
    def test_none_mode_handling(self):
        """Test handling when mode is None."""
        # Clear environment variables that might affect mode detection
        old_env = os.environ.get("RAG_TEST_MODE")
        if "RAG_TEST_MODE" in os.environ:
            del os.environ["RAG_TEST_MODE"]
        
        MockController._current_mode = None
        MockController._mocks_disabled = None
        
        try:
            # Should default to UNIT when no environment is set
            mode = MockController.get_test_mode()
            assert mode == TestMode.UNIT
        finally:
            # Restore environment
            if old_env:
                os.environ["RAG_TEST_MODE"] = old_env
    
    def test_invalid_environment_handling(self):
        """Test handling of invalid environment variables."""
        os.environ["RAG_TEST_MODE"] = "not_a_real_mode"
        MockController._current_mode = None
        
        # Should default to UNIT
        mode = MockController.get_test_mode()
        assert mode == TestMode.UNIT
    
    def test_mocks_disabled_cache_consistency(self):
        """Test that mocks_disabled cache stays consistent."""
        MockController.set_test_mode(TestMode.UNIT)
        assert not MockController.are_mocks_disabled()
        
        # Change mode
        MockController.set_test_mode(TestMode.E2E)
        assert MockController.are_mocks_disabled()
        
        # Cache should update correctly
        MockController.set_test_mode(TestMode.UNIT)
        assert not MockController.are_mocks_disabled()