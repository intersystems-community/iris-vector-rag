"""
Daemon Controller for Reconciliation Operations.

This module implements the DaemonController class that handles continuous
reconciliation operations and daemon mode execution lifecycle.
"""

import time
import signal
import logging
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from iris_rag.controllers.reconciliation import ReconciliationController

from iris_rag.controllers.reconciliation_components.models import ReconciliationResult
from common.environment_utils import get_daemon_retry_interval, get_daemon_default_interval, detect_environment

# Configure logging
logger = logging.getLogger(__name__)


class DaemonController:
    """
    Controller for managing daemon mode operations and continuous reconciliation.
    
    This controller handles the execution lifecycle when running as a background
    process, including signal handling, iteration control, and error recovery.
    """
    
    def __init__(self, reconciliation_controller: 'ReconciliationController', config_manager):
        """
        Initialize the DaemonController.
        
        Args:
            reconciliation_controller: Main reconciliation controller instance
            config_manager: Configuration manager for daemon-specific settings
        """
        self.reconciliation_controller = reconciliation_controller
        self.config_manager = config_manager
        
        # Daemon control attributes
        self._stop_event = threading.Event()
        self._force_run_event = threading.Event()
        self.max_iterations = 0
        self.current_iteration = 0
        
        # Get daemon configuration with environment-aware defaults
        reconciliation_config = config_manager.get_reconciliation_config()
        
        # Use environment-aware defaults for better test performance
        current_env = detect_environment()
        config_interval_hours = reconciliation_config.get('interval_hours', 1)
        config_error_retry_minutes = reconciliation_config.get('error_retry_minutes', 5)
        
        # Apply environment-aware defaults
        self.default_interval_seconds = get_daemon_default_interval(
            config_interval_hours * 3600 if current_env == "production" else None
        )
        self.error_retry_interval_seconds = get_daemon_retry_interval(
            config_error_retry_minutes * 60 if current_env == "production" else None
        )
        
        logger.info(f"DaemonController initialized for {current_env} environment")
        logger.info(f"Default interval: {self.default_interval_seconds}s, Error retry: {self.error_retry_interval_seconds}s")
    
    def run_daemon(self, interval: Optional[int] = None, max_iterations: Optional[int] = None,
                   error_retry_interval: Optional[int] = None, pipeline_type: str = "colbert") -> None:
        """
        Run the daemon with continuous reconciliation.
        
        Args:
            interval: Time between reconciliation attempts in seconds
            max_iterations: Maximum number of iterations (0 = infinite)
            error_retry_interval: Retry interval on errors in seconds
            pipeline_type: The pipeline type to reconcile continuously
        """
        # Set up daemon parameters
        effective_interval = interval or self.default_interval_seconds
        self.max_iterations = max_iterations or 0
        effective_error_retry_interval = error_retry_interval or self.error_retry_interval_seconds
        self.pipeline_type = pipeline_type
        
        logger.info(f"Starting reconciliation daemon for {pipeline_type}")
        logger.info(f"Interval: {effective_interval} seconds, Max iterations: {self.max_iterations or 'infinite'}")
        logger.info(f"Error retry interval: {effective_error_retry_interval} seconds")
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        try:
            self._daemon_loop(effective_interval, effective_error_retry_interval)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping daemon")
        except Exception as e:
            logger.error(f"Fatal error in daemon: {e}")
            raise
        finally:
            logger.info("Reconciliation daemon stopped")
    
    def _daemon_loop(self, interval: int, error_retry_interval: int) -> None:
        """
        Core daemon loop logic.
        
        Args:
            interval: Normal interval between reconciliation attempts
            error_retry_interval: Shorter interval used after errors
        """
        self.current_iteration = 0
        
        while not self._stop_event.is_set():
            self.current_iteration += 1
            
            # Check if we've reached max iterations
            if self.max_iterations > 0 and self.current_iteration > self.max_iterations:
                logger.info(f"Reached maximum iterations ({self.max_iterations}), stopping daemon")
                break
            
            # Check for force run request
            if self._force_run_event.is_set():
                logger.info("Force run requested, executing immediate reconciliation")
                self._force_run_event.clear()
            
            reconciliation_successful = False
            try:
                logger.info(f"Starting reconciliation iteration {self.current_iteration}")
                
                # Execute single reconciliation cycle
                result = self.run_once(self.pipeline_type)
                
                # Log results
                if result.success:
                    reconciliation_successful = True
                    if result.drift_analysis.has_drift:
                        logger.info(f"Reconciliation completed - {len(result.drift_analysis.issues)} issues addressed")
                    else:
                        logger.info("No drift detected - system healthy")
                else:
                    logger.error(f"Reconciliation failed: {result.error_message}")
                
                logger.info(f"Iteration {self.current_iteration} completed in {result.execution_time_seconds:.2f}s")
                
            except Exception as e:
                logger.error(f"Error during reconciliation iteration {self.current_iteration}: {e}")
                logger.debug("Exception details:", exc_info=True)
            
            # Wait for next iteration (with interruption check)
            if not self._stop_event.is_set() and (self.max_iterations == 0 or self.current_iteration < self.max_iterations):
                # Use shorter interval if reconciliation failed
                sleep_interval = interval if reconciliation_successful else error_retry_interval
                
                if not reconciliation_successful:
                    logger.info(f"Using shorter retry interval due to error: {sleep_interval} seconds")
                else:
                    logger.debug(f"Waiting {sleep_interval} seconds until next reconciliation...")
                
                # Sleep in chunks to allow for responsive shutdown
                self._interruptible_sleep(sleep_interval)
    
    def run_once(self, pipeline_type: str = "colbert") -> ReconciliationResult:
        """
        Execute a single reconciliation cycle.
        
        Args:
            pipeline_type: The pipeline type to reconcile
            
        Returns:
            ReconciliationResult: Result of the reconciliation operation
        """
        return self.reconciliation_controller.reconcile(pipeline_type=pipeline_type, force=False)
    
    def stop(self) -> None:
        """
        Request daemon to stop gracefully.
        """
        logger.info("Stop requested for reconciliation daemon")
        self._stop_event.set()
    
    def force_run(self) -> None:
        """
        Request immediate reconciliation run.
        """
        logger.info("Force run requested for reconciliation daemon")
        self._force_run_event.set()
    
    def _setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for graceful shutdown.
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def _interruptible_sleep(self, sleep_interval: int) -> None:
        """
        Sleep in chunks to allow for responsive shutdown.
        
        Args:
            sleep_interval: Total time to sleep in seconds
        """
        # Sleep in chunks to allow for responsive shutdown
        sleep_chunks = max(1, sleep_interval // 10)
        chunk_duration = sleep_interval / sleep_chunks
        
        for _ in range(sleep_chunks):
            if self._stop_event.is_set():
                break
            time.sleep(chunk_duration)
    
    @property
    def is_running(self) -> bool:
        """
        Check if daemon is currently running.
        
        Returns:
            bool: True if daemon is running, False otherwise
        """
        return not self._stop_event.is_set()
    
    def get_status(self) -> dict:
        """
        Get current daemon status.
        
        Returns:
            dict: Status information including iteration count and running state
        """
        return {
            "is_running": self.is_running,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "default_interval_seconds": self.default_interval_seconds,
            "error_retry_interval_seconds": self.error_retry_interval_seconds
        }