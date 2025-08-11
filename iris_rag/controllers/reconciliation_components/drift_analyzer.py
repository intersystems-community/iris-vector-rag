"""
Drift Analyzer Component for Reconciliation Framework

This module contains the DriftAnalyzer class responsible for analyzing drift
between current and desired system states. It identifies various types of
drift issues including mock embeddings, low diversity, missing embeddings,
and document count discrepancies.
"""

import logging
from typing import List

from .models import SystemState, DesiredState, DriftIssue, DriftAnalysis

logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """
    Analyzes drift between current and desired system states.

    The DriftAnalyzer is responsible for detecting various types of drift issues
    that can occur in RAG pipeline systems, including data quality issues,
    completeness problems, and configuration mismatches.
    """

    def __init__(self):
        """Initialize the DriftAnalyzer."""
        logger.debug("DriftAnalyzer initialized")

    def analyze_drift(self, current_state: SystemState, desired_state: DesiredState) -> DriftAnalysis:
        """
        Analyze drift between current and desired states.

        Args:
            current_state: Current system state
            desired_state: Desired target state

        Returns:
            DriftAnalysis: Analysis of detected drift issues
        """
        logger.debug("Analyzing drift between current and desired states")

        issues = []

        # Check for low diversity embeddings first, as they can be mistaken for mock embeddings
        if current_state.quality_issues.avg_diversity_score < desired_state.diversity_threshold:
            issues.append(
                DriftIssue(
                    issue_type="low_diversity_embeddings",
                    severity="high",
                    description=f"Low embedding diversity detected (score: {current_state.quality_issues.avg_diversity_score:.3f}, threshold: {desired_state.diversity_threshold})",
                    affected_count=current_state.quality_issues.low_diversity_document_count,
                    recommended_action="Regenerate low-diversity embeddings",
                )
            )

        # Check for mock embeddings contamination (only if not already flagged as low diversity)
        elif current_state.quality_issues.mock_embeddings_detected:
            issues.append(
                DriftIssue(
                    issue_type="mock_contamination",
                    severity="critical",
                    description=f"Mock embeddings detected in {current_state.quality_issues.mock_document_count} documents",
                    affected_count=current_state.quality_issues.mock_document_count,
                    recommended_action="Clear and regenerate embeddings for affected documents",
                )
            )

        # Basic drift detection: missing embeddings
        if (
            desired_state.completeness_requirements.require_all_docs
            and current_state.documents_without_any_embeddings > 0
        ):
            issues.append(
                DriftIssue(
                    issue_type="missing_embeddings",
                    severity="high",
                    description=f"{current_state.documents_without_any_embeddings} documents found with no token embeddings at all.",
                    affected_count=current_state.documents_without_any_embeddings,
                    recommended_action="Generate token embeddings for these documents.",
                )
            )

        # Check for incomplete token embeddings
        if (
            desired_state.completeness_requirements.require_token_embeddings
            and current_state.documents_with_incomplete_embeddings_count > 0
        ):
            issues.append(
                DriftIssue(
                    issue_type="incomplete_token_embeddings",
                    severity="medium",
                    description=f"{current_state.documents_with_incomplete_embeddings_count} documents found with an insufficient number of token embeddings.",
                    affected_count=current_state.documents_with_incomplete_embeddings_count,
                    recommended_action="Generate or complete token embeddings for these documents.",
                )
            )

        # Check document count drift
        if current_state.total_documents < desired_state.target_document_count:
            missing_docs = desired_state.target_document_count - current_state.total_documents
            issues.append(
                DriftIssue(
                    issue_type="insufficient_documents",
                    severity="medium",
                    description=f"Only {current_state.total_documents} documents found, need {desired_state.target_document_count}",
                    affected_count=missing_docs,
                    recommended_action="Ingest additional documents",
                )
            )

        has_drift = len(issues) > 0

        drift_analysis = DriftAnalysis(has_drift=has_drift, issues=issues)

        logger.info(f"Drift analysis complete: {len(issues)} issues detected")
        for issue in issues:
            logger.warning(f"Drift issue: {issue.issue_type} - {issue.description}")

        return drift_analysis
