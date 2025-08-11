"""
Declarative State Management for RAG Templates.

This module provides a declarative interface to the existing reconciliation
system, making it easier to specify desired states and ensure the system
converges to them.
"""

import logging
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict

from iris_rag.config.manager import ConfigurationManager
from iris_rag.controllers.reconciliation import ReconciliationController
from iris_rag.controllers.reconciliation_components.models import (
    DesiredState,
    CompletenessRequirements,
    ReconciliationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class DeclarativeStateSpec:
    """Declarative specification for RAG system state."""

    # Document requirements
    document_count: int
    document_source: Optional[str] = None
    document_selection: Optional[Dict[str, Any]] = None

    # Embedding requirements
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    force_regenerate: bool = False

    # Quality requirements
    min_embedding_diversity: float = 0.1
    max_contamination_ratio: float = 0.05
    validation_mode: str = "strict"

    # Pipeline configuration
    pipeline_type: str = "basic"
    chunk_size: int = 512
    chunk_overlap: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeclarativeStateSpec":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DeclarativeStateSpec":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("state", data))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DeclarativeStateSpec":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data.get("state", data))

    def to_desired_state(self) -> DesiredState:
        """Convert to reconciliation DesiredState."""
        return DesiredState(
            target_document_count=self.document_count,
            embedding_model=self.embedding_model,
            vector_dimensions=self.embedding_dimension,
            completeness_requirements=CompletenessRequirements(
                require_all_docs=True,
                require_token_embeddings=(self.pipeline_type == "colbert"),
                min_embedding_quality_score=0.8,
                min_embeddings_per_doc=5,
            ),
            diversity_threshold=self.min_embedding_diversity,
        )


class DeclarativeStateManager:
    """
    Manages declarative state for RAG systems.

    This provides a simpler interface to the reconciliation system,
    focused on declaring desired states rather than imperative operations.
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None, auto_sync: bool = False):
        """
        Initialize the declarative state manager.

        Args:
            config_manager: Configuration manager (creates default if None)
            auto_sync: Whether to automatically sync on state changes
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.reconciliation_controller = ReconciliationController(self.config_manager)
        self.auto_sync = auto_sync
        self._current_spec: Optional[DeclarativeStateSpec] = None

        logger.info("DeclarativeStateManager initialized")

    def declare_state(self, spec: Union[DeclarativeStateSpec, Dict, str]) -> "DeclarativeStateManager":
        """
        Declare the desired state.

        Args:
            spec: State specification (DeclarativeStateSpec, dict, or path to YAML/JSON)

        Returns:
            Self for chaining
        """
        # Parse specification
        if isinstance(spec, str):
            path = Path(spec)
            if path.suffix in [".yaml", ".yml"]:
                self._current_spec = DeclarativeStateSpec.from_yaml(path)
            elif path.suffix == ".json":
                self._current_spec = DeclarativeStateSpec.from_json(path)
            else:
                raise ValueError(f"Unknown file type: {path.suffix}")
        elif isinstance(spec, dict):
            self._current_spec = DeclarativeStateSpec.from_dict(spec)
        elif isinstance(spec, DeclarativeStateSpec):
            self._current_spec = spec
        else:
            raise TypeError(f"Invalid spec type: {type(spec)}")

        logger.info(
            f"Declared state: {self._current_spec.document_count} documents, "
            f"pipeline: {self._current_spec.pipeline_type}"
        )

        # Auto-sync if enabled
        if self.auto_sync:
            self.sync()

        return self

    def sync(self, force: bool = False, dry_run: bool = False) -> ReconciliationResult:
        """
        Sync system to declared state.

        Args:
            force: Force reconciliation even if no drift detected
            dry_run: Only analyze drift without making changes

        Returns:
            ReconciliationResult with sync details
        """
        if not self._current_spec:
            raise RuntimeError("No state declared. Call declare_state() first.")

        # Convert to desired state
        desired_state = self._current_spec.to_desired_state()

        # Set desired state in controller
        self.reconciliation_controller.desired_state = desired_state

        logger.info(f"Syncing to declared state (force={force}, dry_run={dry_run})")

        if dry_run:
            # Just analyze drift
            current_state = self.reconciliation_controller.state_observer.observe()
            drift = self.reconciliation_controller.drift_analyzer.analyze_drift(current_state, desired_state)

            return ReconciliationResult(
                run_id=f"dry_run_{int(time.time())}",
                pipeline_type=self._current_spec.pipeline_type,
                drift_analysis=drift,
                actions_taken=[],
                converged=False,
                final_state=current_state,
            )

        # Perform actual reconciliation
        return self.reconciliation_controller.reconcile(pipeline_type=self._current_spec.pipeline_type, force=force)

    def get_drift_report(self) -> Dict[str, Any]:
        """
        Get current drift report without syncing.

        Returns:
            Drift analysis as dictionary
        """
        if not self._current_spec:
            raise RuntimeError("No state declared.")

        desired_state = self._current_spec.to_desired_state()
        current_state = self.reconciliation_controller.state_observer.observe()
        drift = self.reconciliation_controller.drift_analyzer.analyze_drift(current_state, desired_state)

        return {
            "has_drift": drift.has_drift,
            "summary": drift.summary,
            "completeness_issues": [asdict(i) for i in drift.completeness_issues],
            "quality_issues": asdict(drift.quality_issues) if drift.quality_issues else None,
            "document_count_drift": drift.document_count_drift,
        }

    def ensure_state(self, spec: Union[DeclarativeStateSpec, Dict, str], timeout: int = 300) -> ReconciliationResult:
        """
        Ensure system reaches declared state within timeout.

        Args:
            spec: State specification
            timeout: Maximum time to wait for convergence (seconds)

        Returns:
            ReconciliationResult

        Raises:
            TimeoutError: If state not achieved within timeout
        """
        import time

        start_time = time.time()

        # Declare and sync
        self.declare_state(spec)
        result = self.sync()

        # Wait for convergence
        while not result.converged and (time.time() - start_time) < timeout:
            time.sleep(5)
            result = self.sync()

        if not result.converged:
            raise TimeoutError(f"Failed to achieve declared state within {timeout}s")

        return result

    def validate_state(self) -> bool:
        """
        Validate current state matches declaration.

        Returns:
            True if current state matches declaration
        """
        drift_report = self.get_drift_report()
        return not drift_report["has_drift"]


# Convenience functions for common patterns


def ensure_documents(count: int, source: Optional[str] = None, pipeline: str = "basic") -> ReconciliationResult:
    """
    Ensure exact number of documents in system.

    Args:
        count: Desired document count
        source: Document source directory
        pipeline: Pipeline type to use

    Returns:
        ReconciliationResult
    """
    manager = DeclarativeStateManager()

    spec = DeclarativeStateSpec(document_count=count, document_source=source, pipeline_type=pipeline)

    return manager.ensure_state(spec)


def sync_from_file(path: str, auto_sync: bool = True) -> DeclarativeStateManager:
    """
    Create manager from state file and optionally sync.

    Args:
        path: Path to YAML or JSON state file
        auto_sync: Whether to sync immediately

    Returns:
        Configured DeclarativeStateManager
    """
    manager = DeclarativeStateManager(auto_sync=auto_sync)
    manager.declare_state(path)
    return manager


# Test integration helpers


def create_test_state(doc_count: int = 10, pipeline: str = "basic") -> DeclarativeStateSpec:
    """
    Create a test state specification.

    Args:
        doc_count: Number of test documents
        pipeline: Pipeline type

    Returns:
        DeclarativeStateSpec for testing
    """
    return DeclarativeStateSpec(
        document_count=doc_count,
        pipeline_type=pipeline,
        validation_mode="lenient",  # More forgiving for tests
        min_embedding_diversity=0.05,  # Lower threshold for test data
        max_contamination_ratio=0.1,  # Allow more mocks in tests
    )
