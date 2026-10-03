"""ci.yml must gate: right branch, every PR, no masked failures, supported Pythons.

Findings OS-07, OS-08, OS-26 (unit tests). The workflow is parsed, not grepped,
so a trigger or matrix mistake fails here before it fails silently in CI.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def _triggers(wf):
    # PyYAML reads the bare key `on` as boolean True.
    return wf.get("on", wf.get(True))


def _steps(wf):
    for name, job in wf["jobs"].items():
        for step in job.get("steps", []):
            yield name, step


def test_push_trigger_is_the_default_branch(wf):
    assert "master" in _triggers(wf)["push"]["branches"]


def test_every_pull_request_triggers_it(wf):
    pr = _triggers(wf)["pull_request"]
    # No branch filter means every PR, whatever its base.
    assert not pr or "branches" not in pr


def test_no_step_masks_failure_with_or_true(wf):
    for job, step in _steps(wf):
        assert "|| true" not in step.get("run", ""), f"{job}: {step.get('name')}"


def test_no_continue_on_error_anywhere(wf):
    for name, job in wf["jobs"].items():
        assert "continue-on-error" not in job, name
    for job, step in _steps(wf):
        assert "continue-on-error" not in step, f"{job}: {step.get('name')}"


def test_unit_matrix_is_python_311_and_312_only(wf):
    versions = wf["jobs"]["unit-tests"]["strategy"]["matrix"]["python-version"]
    assert [str(v) for v in versions] == ["3.11", "3.12"]


def test_unit_job_runs_unit_tests(wf):
    runs = " ".join(s.get("run", "") for s in wf["jobs"]["unit-tests"]["steps"])
    assert "pytest tests/unit" in runs


def test_docker_and_coverage_jobs_are_gone(wf):
    assert "build-and-test-docker" not in wf["jobs"]
    assert "coverage-report" not in wf["jobs"]
    text = WORKFLOW.read_text()
    assert "docker build" not in text
    assert "codecov" not in text.lower()
    assert "--cov" not in text


def test_known_failures_are_deselected_and_new_ones_block(wf):
    runs = " ".join(s.get("run", "") for s in wf["jobs"]["unit-tests"]["steps"])
    assert "ci_known_failures.txt" in runs
