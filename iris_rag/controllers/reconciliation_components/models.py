"""
Data Models for Reconciliation Framework

This module contains all dataclass definitions and type definitions used by the
reconciliation controller and its components. These models represent the core
data structures for desired-state reconciliation operations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from iris_rag.validation.embedding_validator import EmbeddingQualityIssues


# Backward compatibility alias
QualityIssues = EmbeddingQualityIssues


@dataclass
class SystemState:
    """Represents the current observed state of the system."""
    total_documents: int
    total_token_embeddings: int
    avg_embedding_size: float
    quality_issues: QualityIssues
    documents_without_any_embeddings: int = 0
    documents_with_incomplete_embeddings_count: int = 0  # New field for incomplete
    observed_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompletenessRequirements:
    """Defines completeness requirements for the desired state."""
    require_all_docs: bool = True
    require_token_embeddings: bool = False
    min_embedding_quality_score: float = 0.8
    min_embeddings_per_doc: int = 5  # Default, will be overridden by config


@dataclass
class DesiredState:
    """Represents the desired target state for the system."""
    target_document_count: int
    embedding_model: str
    vector_dimensions: int
    completeness_requirements: CompletenessRequirements
    diversity_threshold: float = 0.7
    schema_version: str = "2.1"


@dataclass
class DriftIssue:
    """Represents a specific drift issue detected during analysis."""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_count: int = 0
    recommended_action: str = ""


@dataclass
class DriftAnalysis:
    """Results of drift analysis between current and desired states."""
    has_drift: bool
    issues: List[DriftIssue] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReconciliationAction:
    """Represents an action to be taken during reconciliation."""
    action_type: str
    description: str
    estimated_duration_seconds: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceCheck:
    """Results of convergence verification after reconciliation."""
    converged: bool
    remaining_issues: List[DriftIssue] = field(default_factory=list)
    verification_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReconciliationResult:
    """Complete result of a reconciliation operation."""
    reconciliation_id: str
    success: bool
    current_state: SystemState
    desired_state: DesiredState
    drift_analysis: DriftAnalysis
    actions_taken: List[ReconciliationAction] = field(default_factory=list)
    convergence_check: Optional[ConvergenceCheck] = None
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)