"""Contract tests for requirement-task mapping validation per MAP-001.

These tests define the expected behavior of the task mapping validator.
They must fail initially and pass once the validator is implemented.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path


class TestTaskMappingContract:
    """Contract tests for MAP-001 requirement-task mapping behavior."""

    def test_MAP001_requirement_extraction(self):
        """Verify system can extract requirements from spec.md."""
        spec_content = """# Feature Specification

## Requirements

### Functional Requirements
- **FR-001**: System MUST validate user input
- **FR-002**: System MUST log all transactions
- **FR-003**: System MUST support concurrent users

### Non-Functional Requirements
- **NFR-001**: Response time MUST be under 2 seconds
"""

        # This should fail until validator implements extract_requirements
        from scripts.validate_task_mapping import extract_requirements

        requirements = extract_requirements(spec_content)

        assert len(requirements) == 4
        assert "FR-001" in requirements
        assert "FR-002" in requirements
        assert "FR-003" in requirements
        assert "NFR-001" in requirements

    def test_MAP001_task_extraction(self):
        """Verify system can extract tasks from tasks.md."""
        tasks_content = """# Tasks

## T001: Implement input validation
Implements requirement FR-001

## T002: Add transaction logging
Addresses FR-002

## T003: Performance optimization
Related to NFR-001
"""

        # This should fail until validator implements extract_tasks
        from scripts.validate_task_mapping import extract_tasks

        tasks = extract_tasks(tasks_content)

        assert len(tasks) == 3
        assert tasks["T001"]["requirements"] == ["FR-001"]
        assert tasks["T002"]["requirements"] == ["FR-002"]
        assert tasks["T003"]["requirements"] == ["NFR-001"]

    def test_MAP001_gap_detection(self):
        """Verify system finds requirements without tasks."""
        requirements = ["FR-001", "FR-002", "FR-003", "NFR-001"]
        tasks = {
            "T001": {"requirements": ["FR-001"]},
            "T002": {"requirements": ["FR-002"]},
            # FR-003 and NFR-001 have no tasks
        }

        # This should fail until validator implements find_gaps
        from scripts.validate_task_mapping import find_gaps

        gaps = find_gaps(requirements, tasks)

        assert len(gaps) == 2
        assert "FR-003" in gaps
        assert "NFR-001" in gaps

    def test_MAP001_edge_case_detection(self):
        """Verify system validates edge cases have corresponding tests."""
        spec_content = """# Feature Specification

## Edge Cases
- What happens when input is empty?
- How does system handle connection timeout?
- What if database server crashes?
"""

        tasks_content = """# Tasks

## T010: Test empty input handling
Tests edge case for empty input

## T011: Test network resilience
Tests network failure scenarios
"""

        # This should fail until validator implements validate_edge_cases
        from scripts.validate_task_mapping import validate_edge_cases

        edge_case_gaps = validate_edge_cases(spec_content, tasks_content)

        # Should find at least 1 gap (timeout and/or database crash not covered)
        assert len(edge_case_gaps) >= 1
        # At least one of timeout or database should be in gaps
        has_expected_gap = any(("timeout" in gap.lower() or "database" in gap.lower() or "crashes" in gap.lower())
                              for gap in edge_case_gaps)
        assert has_expected_gap, f"Expected gap not found in: {edge_case_gaps}"

    def test_MAP001_report_generation(self):
        """Verify system generates mapping report."""
        mapping_data = {
            "total_requirements": 5,
            "mapped_requirements": 3,
            "gaps": ["FR-003", "NFR-002"],
            "coverage_percentage": 60.0,
            "edge_case_gaps": ["Network timeout handling"]
        }

        # This should fail until validator implements generate_mapping_report
        from scripts.validate_task_mapping import generate_mapping_report

        report = generate_mapping_report(mapping_data)

        # Report should contain key metrics
        assert "Requirement-Task Mapping Report" in report
        assert "Total Requirements: 5" in report
        assert "Mapped Requirements: 3" in report
        assert "Coverage: 60.0%" in report
        assert "Missing Requirements:" in report
        assert "FR-003" in report
        assert "NFR-002" in report
        assert "Edge Case Gaps:" in report
        assert "Network timeout handling" in report