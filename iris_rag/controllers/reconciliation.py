"""
Reconciliation Controller for Generalized RAG Pipeline State Management.

This module implements the core ReconciliationController class that provides
automatic data integrity management across all RAG pipeline implementations.
It follows the Desired-State Reconciliation pattern to ensure consistent,
reliable data states.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Any

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.validation.embedding_validator import EmbeddingValidator
from iris_rag.controllers.reconciliation_components.models import (
    SystemState,
    CompletenessRequirements,
    DesiredState,
    DriftAnalysis,
    ReconciliationResult,
    QualityIssues
)
from iris_rag.controllers.reconciliation_components.state_observer import StateObserver
from iris_rag.controllers.reconciliation_components.drift_analyzer import DriftAnalyzer
from iris_rag.controllers.reconciliation_components.document_service import DocumentService
from iris_rag.controllers.reconciliation_components.remediation_engine import RemediationEngine
from iris_rag.controllers.reconciliation_components.convergence_verifier import ConvergenceVerifier
from iris_rag.controllers.reconciliation_components.daemon_controller import DaemonController

# Configure logging
logger = logging.getLogger(__name__)


class ReconciliationController:
    """
    Core controller for orchestrating reconciliation operations across RAG pipelines.
    
    This controller implements the Desired-State Reconciliation pattern to ensure
    data integrity and consistency across all RAG pipeline implementations.
    """
    
    def __init__(self, config_manager: ConfigurationManager, reconcile_interval_seconds: Optional[int] = None):
        """
        Initialize the ReconciliationController.
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            reconcile_interval_seconds: Override default reconciliation interval (for daemon mode)
        """
        self.config_manager = config_manager
        self.connection_manager = ConnectionManager(config_manager)
        self.embedding_validator = EmbeddingValidator(config_manager)
        
        # Initialize StateObserver with required dependencies
        self.state_observer = StateObserver(
            self.config_manager,
            self.connection_manager,
            self.embedding_validator
        )
        
        # Initialize DriftAnalyzer
        self.drift_analyzer = DriftAnalyzer()
        
        # Initialize DocumentService
        self.document_service = DocumentService(self.connection_manager, self.config_manager)
        
        # Initialize RemediationEngine
        self.remediation_engine = RemediationEngine(
            self.config_manager,
            self.connection_manager,
            self.document_service,
            self.embedding_validator
        )
        
        # Initialize ConvergenceVerifier
        self.convergence_verifier = ConvergenceVerifier(
            self.state_observer,
            self.drift_analyzer
        )
        
        # Initialize DaemonController
        self.daemon_controller = DaemonController(self, self.config_manager)
        
        # Get reconciliation configuration with optional interval override
        reconciliation_config = config_manager.get_reconciliation_config()
        self.reconcile_interval_seconds = (
            reconcile_interval_seconds or
            reconciliation_config.get('interval_hours', 1) * 3600
        )
        self.error_retry_interval_seconds = reconciliation_config.get('error_retry_minutes', 5) * 60
        
        logger.info("ReconciliationController initialized")
    
    
    def reconcile(self, pipeline_type: str = "colbert", force: bool = False) -> ReconciliationResult:
        """
        Execute the main reconciliation process for a specific pipeline type.
        
        This method orchestrates the complete reconciliation workflow:
        1. Observe current system state
        2. Get desired state configuration
        3. Analyze drift between current and desired states
        4. Execute remediation actions if drift is detected
        5. Verify convergence to desired state
        
        Args:
            pipeline_type: The pipeline type to reconcile (e.g., "colbert", "basic", "noderag")
            force: Whether to force reconciliation even if no drift is detected
            
        Returns:
            ReconciliationResult: Complete result of the reconciliation operation
        """
        reconciliation_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting reconciliation operation {reconciliation_id} for pipeline: {pipeline_type}")
        
        try:
            # Phase 1: Observe current state
            current_state = self.state_observer.observe_current_state()
            
            # Phase 2: Get desired state for the specific pipeline
            desired_state = self.state_observer.get_desired_state(pipeline_type)
            
            # Phase 3: Analyze drift
            drift_analysis = self.drift_analyzer.analyze_drift(current_state, desired_state)
            
            # Phase 4: Reconcile drift (if needed)
            actions_taken = []
            if drift_analysis.has_drift or force:
                actions_taken = self.remediation_engine.remediate_drift(drift_analysis, desired_state, current_state)
            else:
                logger.info("No drift detected, skipping reconciliation actions")
            
            # Phase 5: Verify convergence
            convergence_check = self.convergence_verifier.check_convergence(desired_state, actions_taken)
            
            execution_time = time.time() - start_time
            
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                success=True,
                current_state=current_state,
                desired_state=desired_state,
                drift_analysis=drift_analysis,
                actions_taken=actions_taken,
                convergence_check=convergence_check,
                execution_time_seconds=execution_time
            )
            
            logger.info(f"Reconciliation {reconciliation_id} for {pipeline_type} completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Reconciliation failed for {pipeline_type}: {e}"
            logger.error(error_msg)
            
            # Return failed result
            return ReconciliationResult(
                reconciliation_id=reconciliation_id,
                success=False,
                current_state=SystemState(0, 0, 0.0, QualityIssues()),
                desired_state=DesiredState(0, "", 0, CompletenessRequirements()),
                drift_analysis=DriftAnalysis(False),
                execution_time_seconds=execution_time,
                error_message=error_msg
            )
    
    def analyze_drift_only(self, pipeline_type: str = "colbert") -> Dict[str, Any]:
        """
        Analyze drift without executing reconciliation actions (for dry-run mode).
        
        Args:
            pipeline_type: The pipeline type to analyze drift for
            
        Returns:
            Dict containing current state, desired state, and drift analysis
        """
        logger.info(f"Analyzing drift for {pipeline_type} pipeline (dry-run mode)")
        
        try:
            # Phase 1: Observe current state
            current_state = self.state_observer.observe_current_state()
            
            # Phase 2: Get desired state for the specific pipeline
            desired_state = self.state_observer.get_desired_state(pipeline_type)
            
            # Phase 3: Analyze drift
            drift_analysis = self.drift_analyzer.analyze_drift(current_state, desired_state)
            
            logger.info(f"Drift analysis completed for {pipeline_type}: "
                       f"{len(drift_analysis.issues)} issues detected")
            
            return {
                "current_state": current_state,
                "desired_state": desired_state,
                "drift_analysis": drift_analysis
            }
            
        except Exception as e:
            logger.error(f"Error during drift analysis for {pipeline_type}: {e}")
            raise
    
    def get_status(self, pipeline_type: str = "colbert") -> Dict[str, Any]:
        """
        Get the current status of the reconciliation system.
        
        Args:
            pipeline_type: The pipeline type to get status for
            
        Returns:
            Dict containing current system status information
        """
        try:
            current_state = self.state_observer.observe_current_state()
            desired_state = self.state_observer.get_desired_state(pipeline_type)
            drift_analysis = self.drift_analyzer.analyze_drift(current_state, desired_state)
            daemon_status = self.daemon_controller.get_status()
            
            return {
                "pipeline_type": pipeline_type,
                "current_state": current_state,
                "desired_state": desired_state,
                "drift_detected": drift_analysis.has_drift,
                "drift_issues_count": len(drift_analysis.issues),
                "daemon_status": daemon_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "pipeline_type": pipeline_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_daemon(self, interval: Optional[int] = None, max_iterations: Optional[int] = None,
                   error_retry_interval: Optional[int] = None) -> None:
        """
        Run the daemon with continuous reconciliation.
        
        Args:
            interval: Time between reconciliation attempts in seconds
            max_iterations: Maximum number of iterations (0 = infinite)
            error_retry_interval: Retry interval on errors in seconds
        """
        return self.daemon_controller.run_daemon(interval, max_iterations, error_retry_interval)
    
    def run_continuous_reconciliation(self, pipeline_type: str = "colbert",
                                    interval_seconds: Optional[int] = None,
                                    max_iterations: int = 0) -> None:
        """
        Run continuous reconciliation in daemon mode.
        
        Args:
            pipeline_type: The pipeline type to reconcile continuously
            interval_seconds: Time between reconciliation attempts in seconds (uses config default if None)
            max_iterations: Maximum number of iterations (0 = infinite)
        """
        # Delegate to daemon controller with pipeline type
        return self.daemon_controller.run_daemon(interval_seconds, max_iterations, None, pipeline_type)
    
    def stop_daemon(self) -> None:
        """
        Request daemon to stop gracefully.
        """
        return self.daemon_controller.stop()
    
    def force_run_daemon(self) -> None:
        """
        Request immediate reconciliation run.
        """
        return self.daemon_controller.force_run()


if __name__ == "__main__":
    """Simple test to verify the controller can be instantiated and called."""
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create a basic configuration manager for testing
        config_manager = ConfigurationManager()
        
        # Create and test the reconciliation controller
        controller = ReconciliationController(config_manager)
        
        print("ReconciliationController instantiated successfully")
        
        # Test a reconciliation call
        result = controller.reconcile()
        
        print(f"Reconciliation test completed:")
        print(f"  ID: {result.reconciliation_id}")
        print(f"  Success: {result.success}")
        print(f"  Execution time: {result.execution_time_seconds:.2f}s")
        print(f"  Drift detected: {result.drift_analysis.has_drift}")
        print(f"  Issues found: {len(result.drift_analysis.issues)}")
        
        if result.error_message:
            print(f"  Error: {result.error_message}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()