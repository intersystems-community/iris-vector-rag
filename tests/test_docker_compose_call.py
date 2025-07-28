"""Test suite for verifying ContainerLifecycleManager docker-compose file usage."""

import os
import pytest
from unittest.mock import patch, Mock, call
from tests.utils.container_lifecycle_manager import ContainerLifecycleManager


class TestDockerComposeCall:
    """Test that ContainerLifecycleManager calls the correct docker-compose file."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Store original environment to restore later
        self.original_compose_file = os.environ.get('COMPOSE_FILE')
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        # Restore original environment
        if self.original_compose_file is not None:
            os.environ['COMPOSE_FILE'] = self.original_compose_file
        elif 'COMPOSE_FILE' in os.environ:
            del os.environ['COMPOSE_FILE']
    
    @patch('tests.utils.container_manager.subprocess.run')
    @patch('tests.utils.container_lifecycle_manager.ComposeFileTracker')
    def test_ensure_correct_container_running_uses_licensed_compose_file(
        self, mock_tracker_class, mock_subprocess_run
    ):
        """Test that ensure_correct_container_running uses the correct docker-compose file."""
        # Arrange
        os.environ['COMPOSE_FILE'] = 'docker-compose.licensed.yml'
        
        # Mock the tracker to indicate no compose file change (normal startup case)
        mock_tracker = Mock()
        mock_tracker.has_compose_file_changed.return_value = False
        mock_tracker_class.return_value = mock_tracker
        
        # Mock subprocess.run to avoid actual docker commands
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Mock container health checks and running status
        with patch('tests.utils.container_manager.ContainerManager.is_container_healthy') as mock_health, \
             patch('tests.utils.container_manager.ContainerManager.is_iris_running') as mock_running:
            mock_health.return_value = True
            mock_running.return_value = False  # Container not running, so it should start
            
            # Create lifecycle manager
            lifecycle_mgr = ContainerLifecycleManager()
            
            # Act
            lifecycle_mgr.ensure_correct_container_running()
        
        # Assert
        # Verify subprocess.run was called with the correct docker-compose file
        expected_call = call([
            "docker-compose", "-f", "docker-compose.licensed.yml", "up", "-d", "iris_db"
        ], check=True)
        
        mock_subprocess_run.assert_called_with(expected_call.args[0], check=True)
    
    @patch('tests.utils.container_manager.subprocess.run')
    @patch('tests.utils.container_lifecycle_manager.ComposeFileTracker')
    def test_ensure_correct_container_running_uses_default_compose_file_when_env_not_set(
        self, mock_tracker_class, mock_subprocess_run
    ):
        """Test that ensure_correct_container_running uses default docker-compose.yml when COMPOSE_FILE not set."""
        # Arrange
        # Ensure COMPOSE_FILE is not set
        if 'COMPOSE_FILE' in os.environ:
            del os.environ['COMPOSE_FILE']
        
        # Mock the tracker to indicate no compose file change (normal startup case)
        mock_tracker = Mock()
        mock_tracker.has_compose_file_changed.return_value = False
        mock_tracker_class.return_value = mock_tracker
        
        # Mock subprocess.run to avoid actual docker commands
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Mock container health checks and running status
        with patch('tests.utils.container_manager.ContainerManager.is_container_healthy') as mock_health, \
             patch('tests.utils.container_manager.ContainerManager.is_iris_running') as mock_running:
            mock_health.return_value = True
            mock_running.return_value = False  # Container not running, so it should start
            
            # Create lifecycle manager
            lifecycle_mgr = ContainerLifecycleManager()
            
            # Act
            lifecycle_mgr.ensure_correct_container_running()
        
        # Assert
        # Verify subprocess.run was called with the default docker-compose file
        expected_call = call([
            "docker-compose", "-f", "docker-compose.yml", "up", "-d", "iris_db"
        ], check=True)
        
        mock_subprocess_run.assert_called_with(expected_call.args[0], check=True)
    
    @patch('tests.utils.container_manager.subprocess.run')
    @patch('tests.utils.container_lifecycle_manager.ComposeFileTracker')
    def test_ensure_correct_container_running_with_compose_file_change(
        self, mock_tracker_class, mock_subprocess_run
    ):
        """Test that ensure_correct_container_running handles compose file changes correctly."""
        # Arrange
        os.environ['COMPOSE_FILE'] = 'docker-compose.licensed.yml'
        
        # Mock the tracker to indicate compose file HAS changed (restart scenario)
        mock_tracker = Mock()
        mock_tracker.has_compose_file_changed.return_value = True
        mock_tracker.get_current_compose_file.return_value = 'docker-compose.licensed.yml'
        mock_tracker_class.return_value = mock_tracker
        
        # Mock subprocess.run to avoid actual docker commands
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Mock container health and running status
        with patch('tests.utils.container_manager.ContainerManager.is_container_healthy') as mock_health, \
             patch('tests.utils.container_manager.ContainerManager.is_iris_running') as mock_running:
            mock_health.return_value = True
            # Container is running initially, then not running after stop
            mock_running.side_effect = [True, False]
            
            # Create lifecycle manager
            lifecycle_mgr = ContainerLifecycleManager()
            
            # Act
            lifecycle_mgr.ensure_correct_container_running()
        
        # Assert
        # Verify subprocess.run was called for both stop and start operations
        expected_calls = [
            call(["docker-compose", "-f", "docker-compose.licensed.yml", "down"], check=True),
            call(["docker-compose", "-f", "docker-compose.licensed.yml", "up", "-d", "iris_db"], check=True)
        ]
        
        # Check that both calls were made
        assert mock_subprocess_run.call_count == 2
        actual_calls = mock_subprocess_run.call_args_list
        
        # Verify the stop call
        assert actual_calls[0] == expected_calls[0]
        # Verify the start call  
        assert actual_calls[1] == expected_calls[1]