"""
Convergence Verifier Component for Reconciliation Framework

This module contains the ConvergenceVerifier class responsible for verifying
that the system has converged to the desired state after remediation actions.
It re-observes the system state and re-analyzes drift to determine if
convergence has been achieved.
"""

import logging
from typing import List

from .models import (
    DesiredState,
    ReconciliationAction,
    ConvergenceCheck,
    DriftIssue
)
from .state_observer import StateObserver
from .drift_analyzer import DriftAnalyzer

logger = logging.getLogger(__name__)


class ConvergenceVerifier:
    """
    Verifies system convergence to desired state after remediation actions.
    
    The ConvergenceVerifier is responsible for determining whether the system
    has successfully converged to the desired state by re-observing the current
    state and re-analyzing drift after remediation actions have been taken.
    """
    
    def __init__(self, state_observer: StateObserver, drift_analyzer: DriftAnalyzer):
        """
        Initialize the ConvergenceVerifier.
        
        Args:
            state_observer: StateObserver instance for observing current system state
            drift_analyzer: DriftAnalyzer instance for analyzing drift between states
        """
        self.state_observer = state_observer
        self.drift_analyzer = drift_analyzer
        logger.debug("ConvergenceVerifier initialized")
    
    def check_convergence(self, desired_state: DesiredState, actions_taken: List[ReconciliationAction]) -> ConvergenceCheck:
        """
        Verify that the system has converged to the desired state.
        
        This method re-observes the current system state and re-analyzes drift
        to determine if the system has successfully converged to the desired state
        after remediation actions have been taken.
        
        Args:
            desired_state: Desired target state
            actions_taken: List of reconciliation actions that were executed
            
        Returns:
            ConvergenceCheck: Results of convergence verification
        """
        logger.debug("Verifying convergence by re-observing system state")
        
        try:
            # Re-observe the current state after reconciliation actions
            current_state = self.state_observer.observe_current_state()
            
            # Re-analyze drift with the new state
            drift_analysis = self.drift_analyzer.analyze_drift(current_state, desired_state)
            
            # Convergence is achieved if no drift is detected
            converged = not drift_analysis.has_drift
            remaining_issues = drift_analysis.issues if drift_analysis.has_drift else []
            
            convergence_check = ConvergenceCheck(
                converged=converged,
                remaining_issues=remaining_issues
            )
            
            if converged:
                logger.info("✅ Convergence achieved - system state matches desired state")
            else:
                logger.warning(f"⚠️ Convergence not achieved - {len(remaining_issues)} issues remain")
                for issue in remaining_issues:
                    logger.warning(f"  - {issue.issue_type}: {issue.description}")
            
            return convergence_check
            
        except Exception as e:
            logger.error(f"Error during convergence verification: {e}")
            return ConvergenceCheck(
                converged=False,
                remaining_issues=[DriftIssue(
                    issue_type="verification_error",
                    severity="high",
                    description=f"Failed to verify convergence: {e}",
                    recommended_action="Check system state and retry reconciliation"
                )]
            )