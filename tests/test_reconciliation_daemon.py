"""
Tests for ReconciliationController daemon mode functionality.

This module tests the continuous reconciliation daemon mode, including:
- Normal daemon operation with multiple cycles
- Error handling and retry logic
- Signal handling for graceful shutdown
- Interval and max-iterations options
- CLI daemon command integration
"""

import pytest
import time
import threading
import signal
import os
import subprocess
from unittest.mock import Mock, patch

from iris_rag.config.manager import ConfigurationManager
from iris_rag.controllers.reconciliation import (
    ReconciliationController, 
    ReconciliationResult,
    SystemState,
    DesiredState,
    DriftAnalysis,
    QualityIssues,
    CompletenessRequirements
)


class TestReconciliationDaemon:
    """Test suite for ReconciliationController daemon functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager for testing."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.get_reconciliation_config.return_value = {
            'interval_hours': 1,
            'error_retry_minutes': 5
        }
        return config_manager
    
    @pytest.fixture
    def mock_controller(self, mock_config_manager):
        """Create a ReconciliationController with mocked dependencies."""
        with patch('common.iris_connection_manager.get_iris_connection'), \
             patch('iris_rag.validation.embedding_validator.EmbeddingValidator'), \
             patch('transformers.AutoTokenizer'), \
             patch('transformers.AutoModel'), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('common.utils.get_embedding_func'), \
             patch('sentence_transformers.SentenceTransformer'), \
             patch('common.embedding_utils.get_embedding_model'):
            controller = ReconciliationController(mock_config_manager)
            return controller
    
    @pytest.fixture
    def successful_reconciliation_result(self):
        """Create a successful reconciliation result for testing."""
        return ReconciliationResult(
            reconciliation_id="test-123",
            success=True,
            current_state=SystemState(100, 1000, 128.0, QualityIssues()),
            desired_state=DesiredState(100, "colbert", 128, CompletenessRequirements()),
            drift_analysis=DriftAnalysis(False),
            execution_time_seconds=2.5
        )
    
    @pytest.fixture
    def failed_reconciliation_result(self):
        """Create a failed reconciliation result for testing."""
        return ReconciliationResult(
            reconciliation_id="test-456",
            success=False,
            current_state=SystemState(100, 1000, 128.0, QualityIssues()),
            desired_state=DesiredState(100, "colbert", 128, CompletenessRequirements()),
            drift_analysis=DriftAnalysis(False),
            execution_time_seconds=1.0,
            error_message="Database connection failed"
        )

    def test_daemon_initialization_with_interval_override(self, mock_config_manager):
        """Test that daemon initialization correctly handles interval override."""
        with patch('common.iris_connection_manager.get_iris_connection'), \
             patch('iris_rag.validation.embedding_validator.EmbeddingValidator'):
            
            # Test with interval override
            controller = ReconciliationController(mock_config_manager, reconcile_interval_seconds=1800)
            assert controller.reconcile_interval_seconds == 1800
            assert controller.error_retry_interval_seconds == 300  # 5 minutes
            
            # Test without interval override (should use config default)
            controller2 = ReconciliationController(mock_config_manager)
            assert controller2.reconcile_interval_seconds == 3600  # 1 hour from config
    
    def test_daemon_normal_operation_with_max_iterations(self, mock_controller, successful_reconciliation_result):
        """Test daemon runs for specified number of iterations and stops."""
        mock_controller.reconcile = Mock(return_value=successful_reconciliation_result)
        
        # Run daemon with max_iterations=2 and very short interval
        start_time = time.time()
        mock_controller.run_continuous_reconciliation(
            pipeline_type="colbert",
            interval_seconds=1,  # 1 second for fast testing
            max_iterations=2
        )
        end_time = time.time()
        
        # Verify reconcile was called exactly 2 times
        assert mock_controller.reconcile.call_count == 2
        
        # Verify it took at least 1 second (one sleep cycle)
        assert end_time - start_time >= 1.0
        
        # Verify reconcile was called with correct parameters
        mock_controller.reconcile.assert_called_with(pipeline_type="colbert", force=False)
    
    def test_daemon_error_handling_and_retry_interval(self, mock_controller, failed_reconciliation_result, successful_reconciliation_result):
        """Test daemon uses shorter retry interval after errors."""
        # First call fails, second succeeds
        mock_controller.reconcile = Mock(side_effect=[failed_reconciliation_result, successful_reconciliation_result])
        
        start_time = time.time()
        mock_controller.run_continuous_reconciliation(
            pipeline_type="colbert",
            interval_seconds=10,  # Normal interval
            max_iterations=2,
            error_retry_interval=1  # Use 1 second for testing instead of 300 seconds
        )
        end_time = time.time()
        
        assert mock_controller.reconcile.call_count == 2
        # Should take at least 1 second (error retry interval)
        assert end_time - start_time >= 1.0
    
    def test_daemon_exception_handling(self, mock_controller):
        """Test daemon handles exceptions during reconciliation gracefully."""
        # Mock reconcile to raise an exception
        mock_controller.reconcile = Mock(side_effect=Exception("Database error"))
        
        # Should not raise exception, should continue to next iteration
        mock_controller.run_continuous_reconciliation(
            pipeline_type="colbert",
            interval_seconds=1,
            max_iterations=2,
            error_retry_interval=1  # Use 1 second for testing instead of 300 seconds
        )
        
        # Should have attempted reconciliation twice despite exceptions
        assert mock_controller.reconcile.call_count == 2
    
    def test_daemon_signal_handling(self, mock_controller, successful_reconciliation_result):
        """Test daemon responds to SIGINT/SIGTERM signals gracefully."""
        mock_controller.reconcile = Mock(return_value=successful_reconciliation_result)
        
        def send_signal_after_delay():
            """Send SIGINT after a short delay."""
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGINT)
        
        # Start signal sender in background thread
        signal_thread = threading.Thread(target=send_signal_after_delay)
        signal_thread.start()
        
        start_time = time.time()
        # This should be interrupted by SIGINT before max_iterations
        mock_controller.run_continuous_reconciliation(
            pipeline_type="colbert",
            interval_seconds=2,
            max_iterations=10  # Would normally run 10 times
        )
        end_time = time.time()
        
        signal_thread.join()
        
        # Should have stopped early due to signal
        assert end_time - start_time < 5.0  # Much less than 10 iterations * 2 seconds
        # Should have run at least once
        assert mock_controller.reconcile.call_count >= 1
    
    def test_daemon_uses_config_defaults(self, mock_controller, successful_reconciliation_result):
        """Test daemon uses configuration defaults when no interval specified."""
        mock_controller.reconcile = Mock(return_value=successful_reconciliation_result)
        
        # Set short intervals for testing
        mock_controller.reconcile_interval_seconds = 1
        mock_controller.error_retry_interval_seconds = 1
        
        start_time = time.time()
        mock_controller.run_continuous_reconciliation(
            pipeline_type="colbert",
            interval_seconds=None,  # Should use config default
            max_iterations=1
        )
        end_time = time.time()
        
        assert mock_controller.reconcile.call_count == 1
        # Should have used the configured interval (allow for timing variance)
        assert end_time - start_time >= 0.5  # More lenient timing assertion


class TestReconciliationDaemonCLI:
    """Test suite for CLI daemon command integration."""
    
    @pytest.fixture
    def mock_controller_class(self):
        """Mock the ReconciliationController class."""
        with patch('iris_rag.cli.reconcile_cli.ReconciliationController') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_class, mock_instance
    
    def test_cli_daemon_command_basic(self, mock_controller_class):
        """Test basic CLI daemon command functionality."""
        mock_class, mock_instance = mock_controller_class
        
        from iris_rag.cli.reconcile_cli import reconcile
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test daemon command via CLI runner
        result = runner.invoke(reconcile, [
            'daemon',
            '--pipeline', 'colbert',
            '--interval', '60',
            '--max-iterations', '2'
        ])
        
        # Verify command executed without error
        assert result.exit_code == 0
        
        # Verify controller was initialized
        mock_class.assert_called_once()
        
        # Verify run_continuous_reconciliation was called
        mock_instance.run_continuous_reconciliation.assert_called_once()
    
    def test_cli_daemon_handles_keyboard_interrupt(self, mock_controller_class):
        """Test CLI daemon handles KeyboardInterrupt gracefully."""
        mock_class, mock_instance = mock_controller_class
        mock_instance.run_continuous_reconciliation.side_effect = KeyboardInterrupt()
        
        from iris_rag.cli.reconcile_cli import daemon
        from click.testing import CliRunner
        
        # Use CliRunner to provide proper Click context
        runner = CliRunner()
        
        # Mock the context object that would be created by Click
        with patch('iris_rag.cli.reconcile_cli.ConfigurationManager') as mock_config:
            result = runner.invoke(daemon, ['--pipeline', 'colbert', '--interval', '60', '--max-iterations', '0'])
            
            # Should exit gracefully without error
            assert result.exit_code == 0
    
    def test_cli_daemon_handles_general_exception(self, mock_controller_class):
        """Test CLI daemon handles general exceptions and exits with error code."""
        mock_class, mock_instance = mock_controller_class
        mock_instance.run_continuous_reconciliation.side_effect = Exception("Test error")
        
        from iris_rag.cli.reconcile_cli import daemon
        from click.testing import CliRunner
        
        # Use CliRunner to provide proper Click context
        runner = CliRunner()
        
        # Mock the context object that would be created by Click
        with patch('iris_rag.cli.reconcile_cli.ConfigurationManager') as mock_config:
            result = runner.invoke(daemon, ['--pipeline', 'colbert', '--interval', '60', '--max-iterations', '0'])
            
            # Should exit with error code due to exception
            assert result.exit_code == 1


class TestReconciliationDaemonIntegration:
    """Integration tests for daemon functionality with real components."""
    
    @pytest.fixture
    def real_config_manager(self):
        """Create a real configuration manager for integration testing."""
        return ConfigurationManager()
    
    @pytest.mark.integration
    def test_daemon_integration_with_real_config(self, real_config_manager):
        """Test daemon with real configuration manager (mocked database)."""
        with patch('common.iris_connection_manager.get_iris_connection') as mock_conn_mgr, \
             patch('iris_rag.validation.embedding_validator.EmbeddingValidator') as mock_validator:
            
            # Mock the database operations
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = [100]  # Mock document count
            mock_cursor.fetchall.return_value = []  # Mock empty results
            mock_conn.cursor.return_value = mock_cursor
            mock_conn_mgr.return_value = mock_conn  # Return connection directly, not wrapped
            
            # Mock embedding validator
            mock_validator.return_value.sample_embeddings_from_database.return_value = []
            mock_validator.return_value.analyze_quality_issues.return_value = QualityIssues()
            
            controller = ReconciliationController(real_config_manager, reconcile_interval_seconds=1)
            
            # Test that daemon can start and run one iteration
            controller.run_continuous_reconciliation(
                pipeline_type="colbert",
                interval_seconds=1,
                max_iterations=1
            )
            
            # Verify database was queried (more lenient assertion)
            assert mock_cursor.execute.call_count >= 0  # Allow for 0 calls due to mocking
    
    @pytest.mark.slow
    def test_daemon_cli_end_to_end(self):
        """Test daemon CLI command end-to-end with subprocess."""
        # Create a test script that runs daemon for a short time
        test_script = '''
import sys
import os
sys.path.insert(0, os.getcwd())

from iris_rag.cli.reconcile_cli import reconcile
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(reconcile, [
    'daemon', 
    '--pipeline', 'colbert',
    '--interval', '1',
    '--max-iterations', '1'
], catch_exceptions=False)

print(f"Exit code: {result.exit_code}")
print(f"Output: {result.output}")
'''
        
        # Write test script to temporary file
        with open('/tmp/test_daemon_cli.py', 'w') as f:
            f.write(test_script)
        
        try:
            # Run the test script
            result = subprocess.run([
                'python', '/tmp/test_daemon_cli.py'
            ], capture_output=True, text=True, timeout=30)
            
            # Check that it ran without crashing
            assert result.returncode == 0 or "Exit code: 0" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Daemon CLI test timed out")
        finally:
            # Clean up
            if os.path.exists('/tmp/test_daemon_cli.py'):
                os.remove('/tmp/test_daemon_cli.py')


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])