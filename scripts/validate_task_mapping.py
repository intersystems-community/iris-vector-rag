#!/usr/bin/env python3
"""Validate that all requirements have corresponding tasks.

This script ensures that:
1. All functional requirements (FR-*) have implementation tasks
2. All non-functional requirements (NFR-*) have tasks
3. Edge cases defined in spec have test tasks
4. No requirements are left unimplemented

Usage:
    python scripts/validate_task_mapping.py [--spec SPEC_FILE] [--tasks TASKS_FILE]
"""

import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json


def extract_requirements(spec_content: str) -> List[str]:
    """Extract requirement IDs from spec.md content."""
    requirements = []

    # Find FR-XXX and NFR-XXX patterns
    # Look for patterns like **FR-001**, FR-001:, or just FR-001
    pattern = r'\b((?:FR|NFR)-\d{3})\b'

    matches = re.findall(pattern, spec_content)
    requirements = list(set(matches))  # Remove duplicates

    return sorted(requirements)


def extract_tasks(tasks_content: str) -> Dict[str, Dict]:
    """Extract tasks and their linked requirements from tasks.md."""
    tasks = {}

    # Split content into task sections
    # Tasks typically start with ## T### or ### T###
    task_pattern = r'^#{2,3}\s+(T\d{3})[:\s].*$'
    task_splits = re.split(task_pattern, tasks_content, flags=re.MULTILINE)

    # Process pairs (task_id, task_content)
    for i in range(1, len(task_splits), 2):
        if i + 1 < len(task_splits):
            task_id = task_splits[i]
            task_content = task_splits[i + 1]

            # Find referenced requirements in task content
            req_pattern = r'\b((?:FR|NFR)-\d{3})\b'
            requirements = list(set(re.findall(req_pattern, task_content)))

            tasks[task_id] = {
                "requirements": requirements,
                "content": task_content[:200]  # First 200 chars for context
            }

    return tasks


def find_gaps(requirements: List[str], tasks: Dict[str, Dict]) -> List[str]:
    """Find requirements without corresponding tasks."""
    # Collect all requirements referenced in tasks
    covered_requirements = set()
    for task_data in tasks.values():
        covered_requirements.update(task_data["requirements"])

    # Find gaps
    gaps = []
    for req in requirements:
        if req not in covered_requirements:
            gaps.append(req)

    return sorted(gaps)


def extract_edge_cases(spec_content: str) -> List[str]:
    """Extract edge cases from spec content."""
    edge_cases = []

    # Look for Edge Cases section
    edge_section_pattern = r'#+\s*Edge Cases.*?(?=^#|\Z)'
    edge_match = re.search(edge_section_pattern, spec_content, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    if edge_match:
        edge_content = edge_match.group(0)

        # Extract bullet points or questions
        # Look for lines starting with -, *, or containing ?
        lines = edge_content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or '?' in line):
                # Clean up the line
                cleaned = re.sub(r'^[-*]\s*', '', line)
                if cleaned and len(cleaned) > 10:  # Skip very short lines
                    edge_cases.append(cleaned)

    return edge_cases


def validate_edge_cases(spec_content: str, tasks_content: str) -> List[str]:
    """Find edge cases without corresponding test tasks."""
    edge_cases = extract_edge_cases(spec_content)
    tasks_lower = tasks_content.lower()

    gaps = []
    for edge_case in edge_cases:
        # Create searchable keywords from edge case
        # Remove common words and punctuation
        keywords = re.findall(r'\b\w{4,}\b', edge_case.lower())
        important_keywords = [k for k in keywords if k not in
                            ['what', 'when', 'where', 'how', 'happens', 'does', 'should', 'system']]

        # Check if any important keyword appears in tasks
        found = False
        for keyword in important_keywords[:3]:  # Check first 3 important keywords
            if keyword in tasks_lower:
                found = True
                break

        if not found:
            gaps.append(edge_case)

    return gaps


def generate_mapping_report(mapping_data: Dict) -> str:
    """Generate a formatted mapping report."""
    report = []
    report.append("=" * 70)
    report.append("Requirement-Task Mapping Report")
    report.append("=" * 70)

    report.append(f"\nTotal Requirements: {mapping_data['total_requirements']}")
    report.append(f"Mapped Requirements: {mapping_data['mapped_requirements']}")
    report.append(f"Coverage: {mapping_data['coverage_percentage']:.1f}%")

    if mapping_data['gaps']:
        report.append("\n" + "-" * 70)
        report.append("Missing Requirements:")
        report.append("-" * 70)
        for gap in mapping_data['gaps']:
            report.append(f"  ❌ {gap}")

    if mapping_data.get('edge_case_gaps'):
        report.append("\n" + "-" * 70)
        report.append("Edge Case Gaps:")
        report.append("-" * 70)
        for gap in mapping_data['edge_case_gaps']:
            report.append(f"  ⚠️  {gap}")

    if mapping_data.get('task_summary'):
        report.append("\n" + "-" * 70)
        report.append("Task Summary:")
        report.append("-" * 70)
        for task_id, task_info in sorted(mapping_data['task_summary'].items())[:10]:
            if task_info['requirements']:
                report.append(f"  {task_id}: {', '.join(task_info['requirements'])}")

    report.append("\n" + "=" * 70)
    return "\n".join(report)


def find_spec_and_tasks(base_path: Path) -> Tuple[Path, Path]:
    """Find spec.md and tasks.md files in the project."""
    # Look in specs directory
    specs_dir = base_path / "specs"
    if specs_dir.exists():
        # Find most recent feature directory
        feature_dirs = [d for d in specs_dir.iterdir() if d.is_dir() and d.name.startswith("0")]
        if feature_dirs:
            # Sort by name (assuming numbered features)
            latest_feature = sorted(feature_dirs)[-1]

            spec_file = latest_feature / "spec.md"
            tasks_file = latest_feature / "tasks.md"

            if spec_file.exists() and tasks_file.exists():
                return spec_file, tasks_file

    # Fallback to looking in current directory
    spec_file = base_path / "spec.md"
    tasks_file = base_path / "tasks.md"

    return spec_file, tasks_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate requirement-task mapping"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="Path to spec.md file"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        help="Path to tasks.md file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--fail-on-gaps",
        action="store_true",
        help="Exit with non-zero code if gaps found"
    )

    args = parser.parse_args()

    # Find files
    base_path = Path.cwd()
    if args.spec and args.tasks:
        spec_file = args.spec
        tasks_file = args.tasks
    else:
        spec_file, tasks_file = find_spec_and_tasks(base_path)

    # Validate files exist
    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}", file=sys.stderr)
        sys.exit(1)

    if not tasks_file.exists():
        print(f"Error: Tasks file not found: {tasks_file}", file=sys.stderr)
        sys.exit(1)

    # Read files
    spec_content = spec_file.read_text()
    tasks_content = tasks_file.read_text()

    # Extract and analyze
    requirements = extract_requirements(spec_content)
    tasks = extract_tasks(tasks_content)
    gaps = find_gaps(requirements, tasks)
    edge_case_gaps = validate_edge_cases(spec_content, tasks_content)

    # Calculate metrics
    total_requirements = len(requirements)
    mapped_requirements = total_requirements - len(gaps)
    coverage_percentage = (mapped_requirements / total_requirements * 100) if total_requirements > 0 else 100

    # Prepare results
    mapping_data = {
        "total_requirements": total_requirements,
        "mapped_requirements": mapped_requirements,
        "gaps": gaps,
        "coverage_percentage": coverage_percentage,
        "edge_case_gaps": edge_case_gaps,
        "task_summary": tasks
    }

    # Output results
    if args.json:
        # Simplified JSON for output
        json_data = {
            "total_requirements": total_requirements,
            "mapped_requirements": mapped_requirements,
            "coverage_percentage": coverage_percentage,
            "missing_requirements": gaps,
            "edge_case_gaps": edge_case_gaps,
            "task_count": len(tasks)
        }
        print(json.dumps(json_data, indent=2))
    else:
        report = generate_mapping_report(mapping_data)
        print(report)

    # Exit code
    if args.fail_on_gaps and (gaps or edge_case_gaps):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()