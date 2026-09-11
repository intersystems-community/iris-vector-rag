#!/usr/bin/env python3
"""Check that every requirement in a feature's spec is claimed by a task.

Reads `specs/<feature>/spec.md` and `specs/<feature>/tasks.md`, collects the
`FR-nnn` / `NFR-nnn` identifiers each declares, and reports the requirements no
task cites.

Two things this got wrong for a long time, both fixed here and pinned by
`tests/unit/test_validate_task_mapping.py`:

* It read tasks only as markdown headings (`## T001 ...`). Every spec in this
  repo writes them as checkboxes (`- [ ] T001 ...`), the format
  `.specify/templates/tasks-template.md` mandates, so the check reported 0.0%
  coverage and every requirement "Missing" for features that were fully mapped.
  Both formats are read now.
* It graded whatever `sorted(specs/*)[-1]` happened to be -- work on 082 was
  checked against 083's spec. The feature comes from the branch name now, and a
  branch that names no feature (master, a housekeeping branch) is skipped rather
  than compared against an unrelated spec.

`--fail-on-gaps` acts on requirement gaps only. Edge-case matching is keyword
guesswork and stays advisory: it prints, it never sets the exit code.

Usage:
    python scripts/validate_task_mapping.py [--spec SPEC] [--tasks TASKS]
    python scripts/validate_task_mapping.py --fail-on-gaps
    python scripts/validate_task_mapping.py --branch 082-table-shape --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REQUIREMENT = re.compile(r"\b((?:FR|NFR)-\d{3})\b")

# `- [ ] T001 ...`, `- [x] T001 ...`, `* [X] T1004 ...`
CHECKBOX_TASK = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*(T\d+)\b")
# `## T001: ...`, `### T001 ...`
HEADING_TASK = re.compile(r"^#{2,6}\s+(T\d+)\b")
# Any markdown heading, which ends a checkbox task's run of continuation lines.
HEADING = re.compile(r"^#{1,6}\s")

# A branch that owns a feature starts with the feature number: `082-table-shape`,
# and also `082-table-shape-fixup` for follow-up work on the same feature.
FEATURE_BRANCH = re.compile(r"^(\d{3})-")


def extract_requirements(spec_content: str) -> List[str]:
    """Every requirement identifier the spec declares, deduplicated and sorted."""
    return sorted(set(REQUIREMENT.findall(spec_content)))


def requirement_text(spec_content: str) -> Dict[str, str]:
    """Map each requirement to the line that declares it, for the gap report."""
    text: Dict[str, str] = {}
    for line in spec_content.splitlines():
        for req in REQUIREMENT.findall(line):
            if req not in text:
                cleaned = re.sub(r"^[\s\-*#>]+", "", line).strip()
                cleaned = cleaned.replace("**", "")
                cleaned = re.sub(rf"^{re.escape(req)}\s*[:.\-]?\s*", "", cleaned)
                text[req] = cleaned
    return text


def extract_tasks(tasks_content: str) -> Dict[str, Dict]:
    """Tasks and the requirements each one cites.

    A task starts at either a checkbox line or a `T###` heading, and owns every
    line after it up to the next task or the next heading -- so a requirement
    cited on an indented sub-bullet belongs to the task above it.
    """
    tasks: Dict[str, Dict] = {}
    current: Optional[str] = None
    lines: Dict[str, List[str]] = {}

    for line in tasks_content.splitlines():
        start = CHECKBOX_TASK.match(line) or HEADING_TASK.match(line)
        if start:
            current = start.group(1)
            lines.setdefault(current, []).append(line)
            continue
        if current is None:
            continue
        if HEADING.match(line):
            current = None
            continue
        lines[current].append(line)

    for task_id, body in lines.items():
        content = "\n".join(body)
        tasks[task_id] = {
            "requirements": sorted(set(REQUIREMENT.findall(content))),
            "content": content[:200],
        }
    return tasks


def find_gaps(requirements: List[str], tasks: Dict[str, Dict]) -> List[str]:
    """Requirements no task cites."""
    covered = {req for task in tasks.values() for req in task["requirements"]}
    return sorted(req for req in requirements if req not in covered)


def extract_edge_cases(spec_content: str) -> List[str]:
    """Extract edge cases from spec content."""
    edge_cases = []

    edge_section_pattern = r"#+\s*Edge Cases.*?(?=^#|\Z)"
    edge_match = re.search(
        edge_section_pattern, spec_content, re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    if edge_match:
        edge_content = edge_match.group(0)

        for line in edge_content.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("*") or "?" in line):
                cleaned = re.sub(r"^[-*]\s*", "", line)
                if cleaned and len(cleaned) > 10:
                    edge_cases.append(cleaned)

    return edge_cases


def validate_edge_cases(spec_content: str, tasks_content: str) -> List[str]:
    """Edge cases whose keywords appear nowhere in the tasks. Advisory only."""
    edge_cases = extract_edge_cases(spec_content)
    tasks_lower = tasks_content.lower()

    gaps = []
    for edge_case in edge_cases:
        keywords = re.findall(r"\b\w{4,}\b", edge_case.lower())
        important_keywords = [
            k
            for k in keywords
            if k
            not in [
                "what",
                "when",
                "where",
                "how",
                "happens",
                "does",
                "should",
                "system",
            ]
        ]

        found = False
        for keyword in important_keywords[:3]:
            if keyword in tasks_lower:
                found = True
                break

        if not found:
            gaps.append(edge_case)

    return gaps


def analyze(spec_content: str, tasks_content: str, feature: str | None = None) -> Dict:
    """Everything the report and the exit code are derived from."""
    requirements = extract_requirements(spec_content)
    tasks = extract_tasks(tasks_content)
    gaps = find_gaps(requirements, tasks)
    total = len(requirements)
    mapped = total - len(gaps)

    return {
        "feature": feature,
        "total_requirements": total,
        "mapped_requirements": mapped,
        "gaps": gaps,
        "gap_text": {req: requirement_text(spec_content).get(req, "") for req in gaps},
        "coverage_percentage": (mapped / total * 100) if total else 100.0,
        "edge_case_gaps": validate_edge_cases(spec_content, tasks_content),
        "task_summary": tasks,
        # Only requirement gaps fail. Edge-case matching is keyword guesswork.
        "fails": bool(gaps),
    }


def generate_mapping_report(mapping_data: Dict) -> str:
    """Generate a formatted mapping report."""
    report = []
    report.append("=" * 70)
    report.append("Requirement-Task Mapping Report")
    report.append("=" * 70)

    if mapping_data.get("feature"):
        report.append(f"\nFeature: {mapping_data['feature']}")

    report.append(f"\nTotal Requirements: {mapping_data['total_requirements']}")
    report.append(f"Mapped Requirements: {mapping_data['mapped_requirements']}")
    report.append(f"Tasks: {len(mapping_data.get('task_summary', {}))}")
    report.append(f"Coverage: {mapping_data['coverage_percentage']:.1f}%")

    if mapping_data["gaps"]:
        report.append("\n" + "-" * 70)
        report.append("Missing Requirements: no task cites these")
        report.append("-" * 70)
        for gap in mapping_data["gaps"]:
            text = mapping_data.get("gap_text", {}).get(gap, "")
            report.append(f"  ❌ {gap}" + (f" — {text}" if text else ""))

    if mapping_data.get("edge_case_gaps"):
        report.append("\n" + "-" * 70)
        report.append("Edge Case Gaps (advisory: never fails the check)")
        report.append("-" * 70)
        for gap in mapping_data["edge_case_gaps"]:
            report.append(f"  ⚠️  {gap}")

    report.append("\n" + "=" * 70)
    return "\n".join(report)


def current_branch(base_path: Path) -> Optional[str]:
    """The checked-out branch name, or None outside a git work tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=base_path,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def find_spec_dir(specs_dir: Path, branch: Optional[str]) -> Optional[Path]:
    """The feature directory the branch is working on, if there is one.

    A branch names its feature by number, so `082-table-shape-fixup` still checks
    082. A branch with no feature number (master, a housekeeping branch) names no
    feature, and a feature with no `tasks.md` yet has nothing to compare.
    """
    if not branch or not specs_dir.is_dir():
        return None

    match = FEATURE_BRANCH.match(branch)
    if not match:
        return None

    exact = specs_dir / branch
    candidates = [exact] if exact.is_dir() else []
    candidates += sorted(
        d
        for d in specs_dir.iterdir()
        if d.is_dir() and d.name.startswith(f"{match.group(1)}-") and d != exact
    )

    for candidate in candidates:
        if (candidate / "spec.md").exists() and (candidate / "tasks.md").exists():
            return candidate
    return None


def find_spec_and_tasks(
    base_path: Path, branch: Optional[str] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate the current feature's spec and tasks, or (None, None) if there isn't one."""
    feature_dir = find_spec_dir(
        base_path / "specs", branch if branch is not None else current_branch(base_path)
    )
    if feature_dir is None:
        return None, None
    return feature_dir / "spec.md", feature_dir / "tasks.md"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate requirement-task mapping")
    parser.add_argument("--spec", type=Path, help="Path to spec.md file")
    parser.add_argument("--tasks", type=Path, help="Path to tasks.md file")
    parser.add_argument(
        "--branch",
        help="Branch name to derive the feature from (default: the checked-out branch)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--fail-on-gaps",
        action="store_true",
        help="Exit non-zero if a requirement has no task (edge cases stay advisory)",
    )

    args = parser.parse_args()

    base_path = Path.cwd()
    if args.spec and args.tasks:
        spec_file: Optional[Path] = args.spec
        tasks_file: Optional[Path] = args.tasks
        feature = args.spec.parent.name
    else:
        branch = args.branch if args.branch is not None else current_branch(base_path)
        spec_file, tasks_file = find_spec_and_tasks(base_path, branch)
        feature = spec_file.parent.name if spec_file else None

        if spec_file is None:
            message = (
                f"Skipped: no feature spec for branch {branch!r} "
                "(no specs/<number>-* directory with both spec.md and tasks.md)."
            )
            print(
                json.dumps({"skipped": True, "reason": message})
                if args.json
                else message
            )
            sys.exit(0)

    for path, label in ((spec_file, "Spec"), (tasks_file, "Tasks")):
        if not path.exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    mapping_data = analyze(spec_file.read_text(), tasks_file.read_text(), feature)

    if args.json:
        print(
            json.dumps(
                {
                    "feature": mapping_data["feature"],
                    "total_requirements": mapping_data["total_requirements"],
                    "mapped_requirements": mapping_data["mapped_requirements"],
                    "coverage_percentage": mapping_data["coverage_percentage"],
                    "missing_requirements": mapping_data["gaps"],
                    "edge_case_gaps": mapping_data["edge_case_gaps"],
                    "task_count": len(mapping_data["task_summary"]),
                },
                indent=2,
            )
        )
    else:
        print(generate_mapping_report(mapping_data))

    sys.exit(1 if args.fail_on_gaps and mapping_data["fails"] else 0)


if __name__ == "__main__":
    main()
