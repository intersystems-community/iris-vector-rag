"""The requirement-task mapping check, tested against the task format specs use.

The check ran on every `spec.md` / `tasks.md` commit and reported "Coverage: 0.0%"
with every requirement "Missing" for a feature whose tasks all cited their
requirement inline. It only recognised tasks written as markdown headings
(`## T001 ...`), while `.specify/templates/tasks-template.md` -- and therefore
every spec in this repo -- writes them as checkboxes (`- [ ] T001 ...`). Nothing
noticed, because the hook was advisory: it printed 0.0% and exited 0.

These tests pin the three things that made it useless: which task lines it reads,
which feature it reads, and whether a gap can fail a commit.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_task_mapping.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_task_mapping", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vtm = _load_module()


CHECKBOX_TASKS = """\
# Tasks: Something

## Phase 1: Setup

- [ ] T001 Create the module in iris_vector_rag/thing.py
- [X] T002 [P] [US1] Read the catalog once (FR-001, FR-002)
- [ ] T003 [P] [US1] Name the identifier column (FR-003)
"""

HEADING_TASKS = """\
# Tasks: Something

## T001: Create the module

Covers FR-001.

### T002 Read the catalog

Covers FR-002 and FR-003.
"""


# --- which task lines the check reads ---------------------------------------


def test_checkbox_tasks_are_read():
    """The format `.specify/templates/tasks-template.md` mandates."""
    tasks = vtm.extract_tasks(CHECKBOX_TASKS)

    assert sorted(tasks) == ["T001", "T002", "T003"]


def test_a_checkbox_task_carries_the_requirements_named_on_its_line():
    tasks = vtm.extract_tasks(CHECKBOX_TASKS)

    assert sorted(tasks["T002"]["requirements"]) == ["FR-001", "FR-002"]
    assert tasks["T003"]["requirements"] == ["FR-003"]
    assert tasks["T001"]["requirements"] == []


def test_a_completed_checkbox_still_counts():
    """`- [X]` is a task that shipped, not a task that vanished."""
    tasks = vtm.extract_tasks("- [X] T009 Done thing (FR-007)\n")

    assert tasks["T009"]["requirements"] == ["FR-007"]


def test_heading_tasks_are_still_read():
    """The old format keeps working; this is an addition, not a replacement."""
    tasks = vtm.extract_tasks(HEADING_TASKS)

    assert sorted(tasks) == ["T001", "T002"]
    assert tasks["T001"]["requirements"] == ["FR-001"]
    assert sorted(tasks["T002"]["requirements"]) == ["FR-002", "FR-003"]


def test_a_task_owns_the_sub_bullets_indented_under_it():
    """A requirement cited on a continuation line belongs to the task above it."""
    content = """\
- [ ] T001 Write the resolver in iris_vector_rag/storage/shape_resolver.py
  - one catalog read per table (FR-004)
- [ ] T002 Next thing
"""
    tasks = vtm.extract_tasks(content)

    assert tasks["T001"]["requirements"] == ["FR-004"]
    assert tasks["T002"]["requirements"] == []


def test_task_ids_longer_than_three_digits_are_read():
    tasks = vtm.extract_tasks("- [ ] T1004 A big feature (FR-011)\n")

    assert tasks["T1004"]["requirements"] == ["FR-011"]


# --- coverage and gaps ------------------------------------------------------


def test_a_fully_mapped_feature_reports_no_gaps():
    spec = "**FR-001**: a\n**FR-002**: b\n**FR-003**: c\n"

    gaps = vtm.find_gaps(
        vtm.extract_requirements(spec), vtm.extract_tasks(CHECKBOX_TASKS)
    )

    assert gaps == []


def test_the_gap_report_names_the_requirement_and_quotes_it():
    """ "FR-004 Missing" is a fact; the sentence is what tells you what to do."""
    spec = (
        "- **FR-001**: read the catalog once\n- **FR-004**: cache the resolved shape\n"
    )
    tasks = "- [ ] T001 Read the catalog (FR-001)\n"

    report = vtm.generate_mapping_report(vtm.analyze(spec, tasks))

    assert "FR-004" in report
    assert "cache the resolved shape" in report
    assert "FR-001" not in report.split("Missing Requirements")[1]


def test_edge_case_gaps_are_reported_but_do_not_count_as_gaps():
    """Edge-case matching is keyword guesswork; it must not fail a commit.

    `find_gaps` is what `--fail-on-gaps` acts on, so the keyword heuristic stays
    advisory: it prints, and the exit code ignores it.
    """
    spec = """\
- **FR-001**: read the catalog once

### Edge Cases

- What happens when the quantum flux capacitor desynchronises entirely?
"""
    tasks = "- [ ] T001 Read the catalog (FR-001)\n"
    result = vtm.analyze(spec, tasks)

    assert result["gaps"] == []
    assert result["edge_case_gaps"]
    assert result["fails"] is False


def test_a_requirement_with_no_task_fails():
    result = vtm.analyze("- **FR-002**: b\n", "- [ ] T001 Something else\n")

    assert result["gaps"] == ["FR-002"]
    assert result["fails"] is True


# --- which feature the check reads -----------------------------------------


@pytest.fixture
def specs_tree(tmp_path: Path) -> Path:
    for name in ("081-query-options", "082-table-shape", "083-iris-session"):
        (tmp_path / "specs" / name).mkdir(parents=True)
        (tmp_path / "specs" / name / "spec.md").write_text("- **FR-001**: a\n")
        (tmp_path / "specs" / name / "tasks.md").write_text("- [ ] T001 a (FR-001)\n")
    return tmp_path


def test_the_branch_names_the_feature_being_checked(specs_tree):
    """Not `sorted(dirs)[-1]`: work on 082 was graded against 083's spec."""
    found = vtm.find_spec_dir(specs_tree / "specs", "082-table-shape")

    assert found == specs_tree / "specs" / "082-table-shape"


def test_a_branch_numbered_like_a_feature_matches_that_feature(specs_tree):
    """`082-table-shape-fixup` is still work on 082."""
    found = vtm.find_spec_dir(specs_tree / "specs", "082-table-shape-fixup")

    assert found == specs_tree / "specs" / "082-table-shape"


def test_a_branch_that_names_no_feature_matches_nothing(specs_tree):
    """On master there is no feature under review, so there is nothing to check."""
    assert vtm.find_spec_dir(specs_tree / "specs", "master") is None
    assert vtm.find_spec_dir(specs_tree / "specs", "housekeeping-tooling") is None


def test_a_feature_dir_without_tasks_yet_matches_nothing(tmp_path):
    """Between /speckit.specify and /speckit.tasks there is nothing to compare."""
    (tmp_path / "specs" / "084-query-plan").mkdir(parents=True)
    (tmp_path / "specs" / "084-query-plan" / "spec.md").write_text("- **FR-001**: a\n")

    assert vtm.find_spec_dir(tmp_path / "specs", "084-query-plan") is None


# --- end to end, the way the hook runs it ----------------------------------


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_no_matching_feature_exits_zero_and_says_so(specs_tree):
    """A commit on master must not fail because it touched a spec file."""
    result = _run(specs_tree, "--branch", "master", "--fail-on-gaps")

    assert result.returncode == 0, result.stderr
    assert "no feature spec" in result.stdout.lower(), result.stdout


def test_a_mapped_feature_exits_zero(specs_tree):
    result = _run(specs_tree, "--branch", "082-table-shape", "--fail-on-gaps")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "100.0%" in result.stdout, result.stdout


def test_a_gap_in_the_current_feature_fails_the_run(specs_tree):
    spec = specs_tree / "specs" / "082-table-shape" / "spec.md"
    spec.write_text("- **FR-001**: a\n- **FR-002**: uncovered thing\n")

    result = _run(specs_tree, "--branch", "082-table-shape", "--fail-on-gaps")

    assert result.returncode == 1
    assert "FR-002" in result.stdout, result.stdout
    assert "uncovered thing" in result.stdout, result.stdout


def test_without_fail_on_gaps_a_gap_only_reports(specs_tree):
    spec = specs_tree / "specs" / "082-table-shape" / "spec.md"
    spec.write_text("- **FR-001**: a\n- **FR-002**: uncovered thing\n")

    result = _run(specs_tree, "--branch", "082-table-shape")

    assert result.returncode == 0
    assert "FR-002" in result.stdout


def test_json_output_names_the_feature_it_checked(specs_tree):
    import json

    result = _run(specs_tree, "--branch", "082-table-shape", "--json")

    payload = json.loads(result.stdout)
    assert payload["feature"] == "082-table-shape"
    assert payload["coverage_percentage"] == 100.0


def test_the_repos_own_specs_are_read_by_the_checkbox_parser():
    """Guards against the real regression: 082's tasks must not read as zero."""
    tasks = REPO_ROOT / "specs" / "082-table-shape" / "tasks.md"
    if not tasks.exists():
        pytest.skip("082 spec not present on this branch")

    parsed = vtm.extract_tasks(tasks.read_text())

    assert len(parsed) > 20, len(parsed)
    cited = {req for task in parsed.values() for req in task["requirements"]}
    assert len(cited) >= 10, sorted(cited)
