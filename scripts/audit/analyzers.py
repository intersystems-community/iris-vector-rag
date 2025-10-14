"""Issue detection analyzers for Makefile targets."""

import re
from typing import List

from models import AuditIssue, IssueCategory, MakeTarget, Severity


def analyze_hardcoded_config(target: MakeTarget) -> List[AuditIssue]:
    """Detect hardcoded IRIS_PORT, IRIS_HOST, etc. (FR-001)."""
    issues = []

    # Check for hardcoded IRIS_PORT without ${} syntax
    for cmd in target.commands:
        # Pattern: IRIS_PORT=<number> without ${IRIS_PORT
        if re.search(r"IRIS_PORT=\d+", cmd) and "${IRIS_PORT" not in cmd:
            issues.append(
                AuditIssue(
                    target_name=target.name,
                    line_number=target.line_number,
                    category=IssueCategory.HARDCODED_CONFIG,
                    severity=Severity.HIGH,
                    description=f"Hardcoded IRIS_PORT overrides .env configuration in {target.name}",
                    current_value=cmd.strip(),
                    recommended_fix="Use ${IRIS_PORT:-1972} to allow .env override",
                    rationale="FR-001: Must use environment variables from .env file",
                )
            )

        # Pattern: IRIS_HOST=<hostname> without ${IRIS_HOST
        if (
            re.search(r"IRIS_HOST=[a-zA-Z0-9.-]+", cmd)
            and "${IRIS_HOST" not in cmd
        ):
            issues.append(
                AuditIssue(
                    target_name=target.name,
                    line_number=target.line_number,
                    category=IssueCategory.HARDCODED_CONFIG,
                    severity=Severity.MEDIUM,
                    description=f"Hardcoded IRIS_HOST in {target.name}",
                    current_value=cmd.strip(),
                    recommended_fix="Use ${IRIS_HOST:-localhost} to allow .env override",
                    rationale="FR-001: Must use environment variables from .env file",
                )
            )

    return issues


def analyze_manual_schema_init(target: MakeTarget) -> List[AuditIssue]:
    """Detect manual schema initialization scripts (FR-002)."""
    issues = []

    # Violation patterns from research.md
    violation_patterns = [
        (
            r"scripts/test-db/initialize_clean_schema\.py",
            "initialize_clean_schema.py bypasses framework auto-setup",
        ),
        (
            r"scripts/.*setup.*schema.*\.py",
            "Manual schema setup script bypasses framework",
        ),
        (r"IRIS\s*<\s*.*\.sql", "Direct SQL execution bypasses framework"),
    ]

    for cmd in target.commands:
        for pattern, reason in violation_patterns:
            if re.search(pattern, cmd):
                issues.append(
                    AuditIssue(
                        target_name=target.name,
                        line_number=target.line_number,
                        category=IssueCategory.MANUAL_SCHEMA_INIT,
                        severity=Severity.HIGH,
                        description=f"Manual schema initialization in {target.name}: {reason}",
                        current_value=cmd.strip(),
                        recommended_fix="Remove manual schema init - let framework auto-setup handle it via create_pipeline(auto_setup=True)",
                        rationale="FR-002: Framework auto-setup must create required schema, not make targets",
                    )
                )

    return issues


def analyze_missing_prereqs(target: MakeTarget) -> List[AuditIssue]:
    """Detect missing prerequisite verification (FR-003)."""
    issues = []

    # Check if target uses database but doesn't depend on setup-db
    uses_database = any(
        keyword in cmd.lower()
        for cmd in target.commands
        for keyword in ["iris", "database", "db", "sql"]
    )

    if uses_database and "setup-db" not in target.dependencies:
        # Check if it's a setup target itself (avoid false positive)
        if "setup" not in target.name.lower():
            issues.append(
                AuditIssue(
                    target_name=target.name,
                    line_number=target.line_number,
                    category=IssueCategory.MISSING_PREREQS,
                    severity=Severity.MEDIUM,
                    description=f"Target {target.name} uses database but doesn't depend on setup-db",
                    current_value=f"{target.name}: {' '.join(target.dependencies)}",
                    recommended_fix=f"{target.name}: setup-db {' '.join(target.dependencies)}",
                    rationale="FR-003: Targets must verify prerequisites before attempting operations",
                )
            )

    return issues


def analyze_inconsistent_patterns(targets: List[MakeTarget]) -> List[AuditIssue]:
    """Detect inconsistent patterns across similar targets (FR-004)."""
    issues = []

    # Group targets by name similarity (e.g., test-ragas-*)
    ragas_targets = [t for t in targets if "ragas" in t.name]

    # Check if RAGAS targets have consistent env variable handling
    if len(ragas_targets) > 1:
        iris_port_configs = {}
        for target in ragas_targets:
            for cmd in target.commands:
                if "IRIS_PORT" in cmd:
                    iris_port_configs[target.name] = cmd.strip()

        # If multiple RAGAS targets set IRIS_PORT differently, flag as inconsistent
        if len(set(iris_port_configs.values())) > 1:
            for target_name, config in iris_port_configs.items():
                issues.append(
                    AuditIssue(
                        target_name=target_name,
                        line_number=next(
                            t.line_number
                            for t in ragas_targets
                            if t.name == target_name
                        ),
                        category=IssueCategory.INCONSISTENT_PATTERN,
                        severity=Severity.MEDIUM,
                        description=f"Inconsistent IRIS_PORT configuration across RAGAS targets",
                        current_value=config,
                        recommended_fix="Standardize all RAGAS targets to use ${IRIS_PORT:-1972}",
                        rationale="FR-004: Similar targets must use consistent configuration patterns",
                    )
                )

    return issues


def analyze_pipeline_test_failures(target: MakeTarget) -> List[AuditIssue]:
    """Detect pipeline tests without proper setup (FR-005)."""
    issues = []

    # Check if target tests a pipeline (graphrag, crag, etc.)
    tests_pipeline = any(
        pipeline in " ".join(target.commands).lower()
        for pipeline in ["graphrag", "crag", "hybrid"]
    )

    if tests_pipeline:
        # Check if auto_setup=True is used (allows framework to create tables)
        has_auto_setup = any(
            "auto_setup" in cmd or "auto-setup" in cmd
            for cmd in target.commands
        )

        if not has_auto_setup:
            issues.append(
                AuditIssue(
                    target_name=target.name,
                    line_number=target.line_number,
                    category=IssueCategory.PIPELINE_TEST_FAILURE,
                    severity=Severity.MEDIUM,
                    description=f"Pipeline test {target.name} may not use framework auto-setup",
                    current_value=f"Target tests {target.name} without explicit auto_setup",
                    recommended_fix="Ensure test script uses create_pipeline(auto_setup=True)",
                    rationale="FR-005: Pipeline tests must allow framework auto-setup to create required tables",
                )
            )

    return issues


def analyze_documentation_gaps(target: MakeTarget) -> List[AuditIssue]:
    """Detect missing or inaccurate help text (FR-006)."""
    issues = []

    # Check if target has help text
    if not target.help_text and target.phony:
        # PHONY targets should have help text (they're user-facing)
        issues.append(
            AuditIssue(
                target_name=target.name,
                line_number=target.line_number,
                category=IssueCategory.DOCUMENTATION_GAP,
                severity=Severity.LOW,
                description=f"Target {target.name} lacks help text documentation",
                current_value=f"No ## comment before {target.name}:",
                recommended_fix=f"Add ## comment describing {target.name}",
                rationale="FR-006: Make target help text must accurately reflect purpose and prerequisites",
            )
        )

    return issues


def analyze_env_var_precedence(target: MakeTarget) -> List[AuditIssue]:
    """Detect environment variable precedence issues (edge case)."""
    issues = []

    for cmd in target.commands:
        # Check for export VAR=value without checking if already set
        # Pattern: export IRIS_PORT=11972 (should be export IRIS_PORT=${IRIS_PORT:-11972})
        if re.search(r"export\s+([A-Z_]+)=([^\s;$]+)", cmd):
            match = re.search(r"export\s+([A-Z_]+)=([^\s;$]+)", cmd)
            var_name = match.group(1)
            if "${" not in cmd:  # No variable expansion
                issues.append(
                    AuditIssue(
                        target_name=target.name,
                        line_number=target.line_number,
                        category=IssueCategory.ENV_VAR_PRECEDENCE,
                        severity=Severity.MEDIUM,
                        description=f"Environment variable {var_name} exported without precedence check",
                        current_value=cmd.strip(),
                        recommended_fix=f"export {var_name}=${{{var_name}:-{match.group(2)}}}",
                        rationale="Edge case: Environment variables should allow .env and shell overrides",
                    )
                )

    return issues


def analyze_all_issues(targets: List[MakeTarget]) -> List[AuditIssue]:
    """Run all analyzers and collect issues."""
    issues = []

    for target in targets:
        issues.extend(analyze_hardcoded_config(target))
        issues.extend(analyze_manual_schema_init(target))
        issues.extend(analyze_missing_prereqs(target))
        issues.extend(analyze_pipeline_test_failures(target))
        issues.extend(analyze_documentation_gaps(target))
        issues.extend(analyze_env_var_precedence(target))

    # Inconsistent patterns analyzer works across all targets
    issues.extend(analyze_inconsistent_patterns(targets))

    return issues
