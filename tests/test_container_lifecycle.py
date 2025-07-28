"""
Test suite for ContainerLifecycleManager to verify COMPOSE_FILE change detection and container restart behavior.

This test suite verifies that the ContainerLifecycleManager correctly:
1. Detects changes to the COMPOSE_FILE environment variable
2. Restarts the Docker container when the compose file changes
3. Maintains container state when no changes occur
4. Handles error scenarios gracefully

Following TDD principles with proper test isolation and comprehensive coverage.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from tests.utils.container_lifecycle_manager import ContainerLifecycleManager
from tests.utils.compose_file_tracker import ComposeFileTracker


class TestContainerLifecycleManager:
    """Test suite for ContainerLifecycleManager COMPOSE_FILE change detection."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Store original environment
        self.original_compose_file = os.environ.get('COMPOSE_FILE')
        
        # Create temporary state file for isolated testing
        self.temp_state_file = tempfile.NamedTemporaryFile(delete=False).name
        
    def teardown_method(self):
        """Clean up test environment after each test."""
        # Restore original environment
        if self.original_compose_file is not None:
            os.environ['COMPOSE_FILE'] = self.original_compose_file
        elif 'COMPOSE_FILE' in os.environ:
            del os.environ['COMPOSE_FILE']
            
        # Clean up temporary state file
        if os.path.exists(self.temp_state_file):
            os.remove(self.temp_state_file)

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_initial_container_startup_with_default_compose_file(self, mock_container_manager_class):
        """Test initial container startup when no COMPOSE_FILE is set."""
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        mock_container_manager.is_iris_running.return_value = False
        mock_container_manager.get_connection_params.return_value = {'host': 'localhost', 'port': 1972}
        
        # Ensure no COMPOSE_FILE is set
        if 'COMPOSE_FILE' in os.environ:
            del os.environ['COMPOSE_FILE']
            
        # Create lifecycle manager with custom state file
        with patch.object(ComposeFileTracker, '_get_default_state_file', return_value=self.temp_state_file):
            lifecycle_mgr = ContainerLifecycleManager()
            
            # Act
            connection_params = lifecycle_mgr.ensure_correct_container_running()
            
            # Assert
            mock_container_manager.start_iris.assert_called_once()
            mock_container_manager.wait_for_health.assert_called_once()
            assert connection_params == {'host': 'localhost', 'port': 1972}
            
            # Verify container manager was initialized with default compose file
            # When no COMPOSE_FILE is set, ComposeFileTracker returns 'docker-compose.yml' as default
            mock_container_manager_class.assert_called_with('docker-compose.yml')

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_container_restart_when_compose_file_changes(self, mock_container_manager_class):
        """Test that container is restarted when COMPOSE_FILE environment variable changes."""
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        mock_container_manager.is_iris_running.return_value = True
        mock_container_manager.get_connection_params.return_value = {'host': 'localhost', 'port': 1972}
        
        # Set initial COMPOSE_FILE
        os.environ['COMPOSE_FILE'] = 'docker-compose.yml'
        
        # Create lifecycle manager and run first time to establish state
        with patch.object(ComposeFileTracker, '_get_default_state_file', return_value=self.temp_state_file):
            lifecycle_mgr = ContainerLifecycleManager()
            
            # First run - should not restart (establishing baseline)
            mock_container_manager.is_iris_running.return_value = True
            lifecycle_mgr.ensure_correct_container_running()
            
            # Reset mocks for the actual test
            mock_container_manager.reset_mock()
            mock_container_manager_class.reset_mock()
            
            # Act - Change COMPOSE_FILE environment variable
            os.environ['COMPOSE_FILE'] = 'docker-compose.licensed.yml'
            
            # Create new lifecycle manager to detect the change
            lifecycle_mgr2 = ContainerLifecycleManager()
            mock_container_manager.is_iris_running.return_value = True
            connection_params = lifecycle_mgr2.ensure_correct_container_running()
            
            # Assert
            # Should stop existing container
            mock_container_manager.stop_iris.assert_called_once()
            # Should start new container
            mock_container_manager.start_iris.assert_called_once()
            # Should wait for health
            mock_container_manager.wait_for_health.assert_called_once()
            # Should return connection params
            assert connection_params == {'host': 'localhost', 'port': 1972}
            
            # Verify container manager was re-initialized with new compose file
            mock_container_manager_class.assert_called_with('docker-compose.licensed.yml')

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_no_restart_when_compose_file_unchanged(self, mock_container_manager_class):
        """Test that container is not restarted when COMPOSE_FILE remains unchanged."""
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        mock_container_manager.is_iris_running.return_value = True
        mock_container_manager.get_connection_params.return_value = {'host': 'localhost', 'port': 1972}
        
        # Set COMPOSE_FILE
        os.environ['COMPOSE_FILE'] = 'docker-compose.yml'
        
        # Create lifecycle manager and establish state
        with patch.object(ComposeFileTracker, '_get_default_state_file', return_value=self.temp_state_file):
            lifecycle_mgr = ContainerLifecycleManager()
            lifecycle_mgr.ensure_correct_container_running()
            
            # Reset mocks
            mock_container_manager.reset_mock()
            
            # Act - Run again with same COMPOSE_FILE
            lifecycle_mgr2 = ContainerLifecycleManager()
            connection_params = lifecycle_mgr2.ensure_correct_container_running()
            
            # Assert
            # Should NOT stop container
            mock_container_manager.stop_iris.assert_not_called()
            # Should NOT start container (already running)
            mock_container_manager.start_iris.assert_not_called()
            # Should still wait for health
            mock_container_manager.wait_for_health.assert_called_once()
            # Should return connection params
            assert connection_params == {'host': 'localhost', 'port': 1972}

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_container_restart_error_handling(self, mock_container_manager_class):
        """Test error handling during container restart when compose file changes."""
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        mock_container_manager.is_iris_running.return_value = True
        mock_container_manager.get_connection_params.return_value = {'host': 'localhost', 'port': 1972}
        
        # Set initial COMPOSE_FILE and establish state
        os.environ['COMPOSE_FILE'] = 'docker-compose.yml'
        
        with patch.object(ComposeFileTracker, '_get_default_state_file', return_value=self.temp_state_file):
            lifecycle_mgr = ContainerLifecycleManager()
            # First run should succeed to establish baseline state
            lifecycle_mgr.ensure_correct_container_running()
            
            # Reset mocks and set up failure for the restart scenario
            mock_container_manager.reset_mock()
            mock_container_manager.is_iris_running.return_value = True
            mock_container_manager.start_iris.side_effect = Exception("Container start failed")
            
            # Act & Assert - Change COMPOSE_FILE and expect exception
            os.environ['COMPOSE_FILE'] = 'docker-compose.licensed.yml'
            
            lifecycle_mgr2 = ContainerLifecycleManager()
            with pytest.raises(Exception, match="Container start failed"):
                lifecycle_mgr2.ensure_correct_container_running()
            
            # Verify that stop was called before the failure
            mock_container_manager.stop_iris.assert_called_once()

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_iris_container_fixture_behavior_with_compose_file_change(self, mock_container_manager_class):
        """Test the iris_container fixture behavior when COMPOSE_FILE changes."""
        # This test simulates the actual usage pattern described in the task
        
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        mock_container_manager.is_iris_running.return_value = False
        mock_container_manager.get_connection_params.return_value = {'host': 'localhost', 'port': 1972}
        
        with patch.object(ComposeFileTracker, '_get_default_state_file', return_value=self.temp_state_file):
            # Step 1: Set COMPOSE_FILE to docker-compose.yml
            os.environ['COMPOSE_FILE'] = 'docker-compose.yml'
            
            # Step 2: Run iris_container fixture (simulated)
            lifecycle_mgr1 = ContainerLifecycleManager()
            connection_params1 = lifecycle_mgr1.ensure_correct_container_running()
            
            # Verify first container start
            assert mock_container_manager.start_iris.call_count == 1
            assert connection_params1 == {'host': 'localhost', 'port': 1972}
            
            # Reset mocks for second run
            mock_container_manager.reset_mock()
            mock_container_manager.is_iris_running.return_value = True  # Container now running
            
            # Step 3: Change COMPOSE_FILE to docker-compose.licensed.yml
            os.environ['COMPOSE_FILE'] = 'docker-compose.licensed.yml'
            
            # Step 4: Run iris_container fixture again (simulated)
            lifecycle_mgr2 = ContainerLifecycleManager()
            connection_params2 = lifecycle_mgr2.ensure_correct_container_running()
            
            # Step 5: Assert that container was restarted
            mock_container_manager.stop_iris.assert_called_once()
            mock_container_manager.start_iris.assert_called_once()
            mock_container_manager.wait_for_health.assert_called_once()
            assert connection_params2 == {'host': 'localhost', 'port': 1972}
            
            # Verify container manager was re-initialized with new compose file
            mock_container_manager_class.assert_called_with('docker-compose.licensed.yml')

    def test_compose_file_tracker_state_persistence(self):
        """Test that ComposeFileTracker correctly persists and detects state changes."""
        # Arrange
        tracker = ComposeFileTracker(self.temp_state_file)
        
        # Test initial state (no previous state file)
        os.environ['COMPOSE_FILE'] = 'docker-compose.yml'
        
        # Act & Assert - First run should detect change from None to docker-compose.yml
        # This is expected behavior when no state file exists
        assert tracker.has_compose_file_changed()
        assert tracker.get_last_compose_file() == 'docker-compose.yml'
        
        # Second call with same environment should not detect change
        assert not tracker.has_compose_file_changed()
        
        # Change environment variable
        os.environ['COMPOSE_FILE'] = 'docker-compose.licensed.yml'
        
        # Should detect change
        assert tracker.has_compose_file_changed()
        assert tracker.get_current_compose_file() == 'docker-compose.licensed.yml'
        assert tracker.get_last_compose_file() == 'docker-compose.licensed.yml'
        
        # No change on subsequent call
        assert not tracker.has_compose_file_changed()

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_container_lifecycle_manager_connection_params_delegation(self, mock_container_manager_class):
        """Test that ContainerLifecycleManager correctly delegates connection parameter requests."""
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        expected_params = {'host': 'test-host', 'port': 9999, 'username': 'test'}
        mock_container_manager.get_connection_params.return_value = expected_params
        
        lifecycle_mgr = ContainerLifecycleManager()
        
        # Act
        actual_params = lifecycle_mgr.get_connection_params()
        
        # Assert
        assert actual_params == expected_params
        mock_container_manager.get_connection_params.assert_called_once()

    @patch('tests.utils.container_lifecycle_manager.ContainerManager')
    def test_container_lifecycle_manager_running_status_delegation(self, mock_container_manager_class):
        """Test that ContainerLifecycleManager correctly delegates container running status checks."""
        # Arrange
        mock_container_manager = Mock()
        mock_container_manager_class.return_value = mock_container_manager
        mock_container_manager.is_iris_running.return_value = True
        
        lifecycle_mgr = ContainerLifecycleManager()
        
        # Act
        is_running = lifecycle_mgr.is_container_running()
        
        # Assert
        assert is_running is True
        mock_container_manager.is_iris_running.assert_called_once()