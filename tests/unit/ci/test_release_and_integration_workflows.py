"""release.yml checks its tag, runs what it claims, and every action is pinned;
the integration tests run in CI on a pinned throwaway container.

Findings OS-19, OS-26 (integration half), OS-09 (ci-honesty stage 2).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"
#: A release-style tag (v1, v4.2, v3.97.9) or a full commit SHA. Branches
#: (`main`, `master`, `release/v1`) move under us and are not pins.
PINNED = re.compile(r"^(v\d+(\.\d+){0,2}|[0-9a-f]{40})$")


def _load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _steps(wf):
    for job_name, job in wf["jobs"].items():
        for step in job.get("steps", []):
            yield job_name, step


def test_every_action_is_pinned_to_a_tag_or_sha():
    for path in WORKFLOWS.glob("*.yml"):
        for job, step in _steps(yaml.safe_load(path.read_text())):
            uses = step.get("uses")
            if not uses or uses.startswith("./"):
                continue
            ref = uses.partition("@")[2]
            assert PINNED.match(ref), f"{path.name}:{job}: {uses} is not pinned"


def test_release_checks_the_tag_against_the_package_version():
    wf = _load("release.yml")
    text = "\n".join(s.get("run", "") for _, s in _steps(wf))
    assert "pyproject.toml" in text
    # The check runs before anything is tested or built, so a wrong tag fails first.
    first = next(iter(wf["jobs"].values()))
    assert any(
        "pyproject" in s.get("run", "") and "tag" in s.get("run", "").lower()
        for s in first["steps"]
    ), "the first job must check the tag"


def test_release_check_rejects_a_mismatch():
    """Run the step's script for real against a wrong tag."""
    import os
    import subprocess

    wf = _load("release.yml")
    step = next(
        s
        for _, s in _steps(wf)
        if "pyproject" in s.get("run", "") and "tag" in s.get("run", "").lower()
    )
    root = WORKFLOWS.parents[1]
    env = {**os.environ, "RELEASE_TAG": "v0.0.0-not-this", "GITHUB_REF_NAME": "v0.0.0"}
    bad = subprocess.run(
        ["bash", "-c", step["run"]], cwd=root, env=env, capture_output=True, text=True
    )
    assert bad.returncode != 0, bad.stdout
    import tomllib

    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    env = {**os.environ, "RELEASE_TAG": f"v{version}", "GITHUB_REF_NAME": f"v{version}"}
    good = subprocess.run(
        ["bash", "-c", step["run"]], cwd=root, env=env, capture_output=True, text=True
    )
    assert good.returncode == 0, good.stdout + good.stderr


def test_release_no_longer_ignores_the_mcp_schema_contract():
    assert "test_mcp_tool_schemas" not in (WORKFLOWS / "release.yml").read_text()


def test_release_quarantines_crag_through_the_known_failures_list():
    """Not `--ignore`: an ignored file hides new failures as well as old ones."""
    text = (WORKFLOWS / "release.yml").read_text()
    assert "--ignore=tests/contract/test_crag" not in text
    assert "ci_known_failures.txt" in text


def test_release_uses_no_continue_on_error_or_or_true():
    text = (WORKFLOWS / "release.yml").read_text()
    assert "continue-on-error" not in text
    assert "|| true" not in text


def test_pr_integration_installs_no_phantom_extra():
    assert "evaluation" not in (WORKFLOWS / "pr-integration.yml").read_text()


# -- integration tests in CI (OS-26) ------------------------------------------------


@pytest.fixture(scope="module")
def ci():
    return _load("ci.yml")


def test_ci_has_an_integration_job(ci):
    assert "integration-tests" in ci["jobs"]


def test_integration_job_pins_its_throwaway_container(ci):
    job = ci["jobs"]["integration-tests"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "start-iris.sh" in runs
    pytest_step = next(
        s for s in job["steps"] if "tests/integration" in s.get("run", "")
    )
    assert pytest_step["env"]["IRIS_CONTAINER_NAME"] == "iris"
    assert "ci_known_failures.txt" in pytest_step["run"]
    assert "|| true" not in pytest_step["run"]


def test_integration_job_is_a_real_gate(ci):
    job = ci["jobs"]["integration-tests"]
    assert "continue-on-error" not in job
    assert all("continue-on-error" not in s for s in job["steps"])


def test_every_ignored_integration_file_still_exists(ci):
    """An `--ignore` of a deleted file hides nothing and rots; fail so it is removed."""
    root = WORKFLOWS.parents[1]
    job = ci["jobs"]["integration-tests"]
    run = next(s for s in job["steps"] if "tests/integration" in s.get("run", ""))[
        "run"
    ]
    for ignored in re.findall(r"--ignore=(\S+)", run):
        assert (
            root / ignored
        ).is_file(), f"{ignored} no longer exists; drop the --ignore"


def test_the_ivg_4x_upgrade_e2e_runs_in_the_integration_job(ci):
    """It once was ignored because it did not collect; it collects and passes, so it runs."""
    root = WORKFLOWS.parents[1]
    job = ci["jobs"]["integration-tests"]
    run = next(s for s in job["steps"] if "tests/integration" in s.get("run", ""))[
        "run"
    ]
    assert "test_ivg_4x_upgrade_e2e" not in run
    assert (root / "tests/integration/test_ivg_4x_upgrade_e2e.py").is_file()


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_the_deselect_command_survives_comments_in_the_known_failures_file(workflow):
    """The owner comments in ci_known_failures.txt must not become --deselect args."""
    import subprocess

    root = WORKFLOWS.parents[1]
    text = (WORKFLOWS / workflow).read_text()
    line = next(
        l.strip() for l in text.splitlines() if l.strip().startswith("DESELECT=")
    )
    out = subprocess.run(
        ["bash", "-c", line + '; printf "%s\\n" $DESELECT'],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out, "no deselect arguments produced"
    assert all(a.startswith("--deselect=tests/") and "::" in a for a in out), out
