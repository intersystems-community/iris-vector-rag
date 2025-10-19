"""Data models for Makefile audit tool."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import re


class Severity(Enum):
    """Issue severity levels (from NFR-002)."""

    BLOCKING = "blocking"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(Enum):
    """Issue categories mapped to functional requirements."""

    HARDCODED_CONFIG = "hardcoded_config"  # FR-001
    MANUAL_SCHEMA_INIT = "manual_schema_init"  # FR-002
    MISSING_PREREQS = "missing_prereqs"  # FR-003
    INCONSISTENT_PATTERN = "inconsistent_pattern"  # FR-004
    PIPELINE_TEST_FAILURE = "pipeline_test_failure"  # FR-005
    DOCUMENTATION_GAP = "documentation_gap"  # FR-006
    ENV_VAR_PRECEDENCE = "env_var_precedence"  # Edge case


@dataclass(frozen=True)
class MakeTarget:
    """Represents a single make target."""

    name: str
    line_number: int
    dependencies: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    env_variables: Dict[str, str] = field(default_factory=dict)
    help_text: Optional[str] = None
    phony: bool = False

    def __post_init__(self):
        """Validate target data."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.name):
            raise ValueError(
                f"Invalid target name '{self.name}' - must match ^[a-zA-Z0-9_-]+$"
            )
        if self.line_number <= 0:
            raise ValueError(f"Line number must be positive, got {self.line_number}")


@dataclass(frozen=True)
class AuditIssue:
    """Represents a detected configuration inconsistency."""

    target_name: str
    line_number: int
    category: IssueCategory
    severity: Severity
    description: str
    current_value: str = ""
    recommended_fix: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class AuditReport:
    """Aggregates all audit findings."""

    makefile_path: str
    total_lines: int
    total_targets: int
    issues: List[AuditIssue]
    generated_at: datetime
    auditor_version: str = "1.0.0"

    @property
    def blocking_count(self) -> int:
        """Count of blocking issues."""
        return sum(1 for i in self.issues if i.severity == Severity.BLOCKING)

    @property
    def high_count(self) -> int:
        """Count of high severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        """Count of medium severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        """Count of low severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.LOW)

    @property
    def total_issues(self) -> int:
        """Total count of all issues."""
        return len(self.issues)

    def issues_by_severity(self) -> Dict[Severity, List[AuditIssue]]:
        """Group issues by severity level."""
        result = {severity: [] for severity in Severity}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    def issues_by_category(self) -> Dict[IssueCategory, List[AuditIssue]]:
        """Group issues by category."""
        result = {category: [] for category in IssueCategory}
        for issue in self.issues:
            result[issue.category].append(issue)
        return result
