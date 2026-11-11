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


# -- the tests that need a real API key ---------------------------------------------

NEEDS_KEY = WORKFLOWS.parents[1] / "tests" / "contract" / "ci_needs_llm_key.txt"
KNOWN_FAILURES = WORKFLOWS.parents[1] / "tests" / "contract" / "ci_known_failures.txt"


def _ids(path):
    return [
        l.strip()
        for l in path.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


def _integration_step(ci):
    job = ci["jobs"]["integration-tests"]
    return next(s for s in job["steps"] if "tests/integration" in s.get("run", ""))


def test_integration_tests_use_the_api_key_secret_when_there_is_one(ci):
    env = _integration_step(ci)["env"]
    assert "secrets.OPENAI_API_KEY" in env["OPENAI_API_KEY"]
    assert "secrets.OPENAI_API_KEY" in env["HAS_LLM_KEY"]


def _pytest_runs(ci, tmp_path, has_key, fail=False):
    """Run the step's script with a stand-in `python`; one argument list per call."""
    import os
    import subprocess

    stub = tmp_path / "python"
    stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$ARGS_OUT"\necho "---" >> "$ARGS_OUT"\n'
        '[ "$STUB_FAIL" = "1" ] && exit 1\nexit 0\n'
    )
    stub.chmod(0o755)
    out = tmp_path / "args"
    result = subprocess.run(
        ["bash", "-e", "-c", _integration_step(ci)["run"]],
        cwd=WORKFLOWS.parents[1],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "ARGS_OUT": str(out),
            "HAS_LLM_KEY": has_key,
            "STUB_FAIL": "1" if fail else "0",
        },
        capture_output=True,
        text=True,
    )
    runs = [r.split() for r in out.read_text().split("---\n") if r.strip()]
    return runs, result


def test_without_a_key_the_tests_that_need_one_are_skipped_and_said_so(ci, tmp_path):
    """Fork PRs and the private mirror have no secret; the run says what it left out."""
    runs, result = _pytest_runs(ci, tmp_path, "false")
    needs_key = _ids(NEEDS_KEY)
    assert needs_key
    assert len(runs) == 1
    assert all(f"--deselect={i}" in runs[0] for i in needs_key)
    assert f"{len(needs_key)} tests that need a real key were skipped" in result.stdout


def test_with_a_key_the_tests_that_need_one_run_in_a_process_of_their_own(ci, tmp_path):
    """Connections they leave open took the Community licences later tests needed.

    A process's connections are released when it exits, so the main run sees the
    licences a keyless run sees, and the key tests start from a fresh process.
    """
    runs, result = _pytest_runs(ci, tmp_path, "true")
    needs_key = _ids(NEEDS_KEY)
    assert len(runs) == 2
    main, keyed = runs
    assert all(f"--deselect={i}" in main for i in needs_key)
    assert all(f"--deselect={i}" in main for i in _ids(KNOWN_FAILURES))
    assert all(i in keyed for i in needs_key)
    assert not any(a.startswith("--deselect") for a in keyed)
    assert "skipped" not in result.stdout


def test_a_failing_main_run_still_runs_the_key_tests_and_fails_the_step(ci, tmp_path):
    runs, result = _pytest_runs(ci, tmp_path, "true", fail=True)
    assert len(runs) == 2
    assert result.returncode != 0


def test_a_test_is_either_quarantined_or_waiting_for_a_key_not_both():
    assert not set(_ids(NEEDS_KEY)) & set(_ids(KNOWN_FAILURES))


def test_the_gate_reports_a_stale_entry_in_the_needs_key_list(tmp_path, monkeypatch):
    """pytest ignores a --deselect that matches nothing, so the gate must check both lists."""
    import importlib.util
    import sys

    root = WORKFLOWS.parents[1]
    spec = importlib.util.spec_from_file_location(
        "antipatterns", root / "scripts" / "gates" / "antipatterns.py"
    )
    gate = importlib.util.module_from_spec(spec)
    # @dataclass looks its module up in sys.modules.
    monkeypatch.setitem(sys.modules, "antipatterns", gate)
    spec.loader.exec_module(gate)
    (tmp_path / "tests" / "contract").mkdir(parents=True)
    (tmp_path / "tests" / "contract" / "ci_needs_llm_key.txt").write_text(
        "tests/integration/test_gone.py::test_was_here\n"
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    findings = gate.stale_deselect_entry()

    assert [f.location for f in findings] == ["tests/contract/ci_needs_llm_key.txt:1"]


def test_the_key_is_checked_before_the_integration_tests_run(ci):
    steps = ci["jobs"]["integration-tests"]["steps"]
    runs = [s.get("run", "") for s in steps]
    check = next(i for i, r in enumerate(runs) if "check-llm-key.sh" in r)
    tests = next(i for i, r in enumerate(runs) if "tests/integration" in r)
    assert check < tests
    env = steps[check]["env"]
    assert "secrets.OPENAI_API_KEY" in env["OPENAI_API_KEY"]
    assert "secrets.OPENAI_API_KEY" in env["HAS_LLM_KEY"]


def test_one_slow_llm_call_cannot_outlast_the_per_test_timeout(ci):
    """The client defaults (60 s, 2 retries) allow 180 s per call, past the 120 s limit.

    A stalled OpenAI call in a fixture then ran into the test timeout instead of
    failing as an LLM timeout.
    """
    step = _integration_step(ci)
    per_test = int(re.search(r"--timeout=(\d+)", step["run"]).group(1))
    timeout = float(step["env"]["RAG_LLM__TIMEOUT"])
    retries = int(step["env"]["RAG_LLM__MAX_RETRIES"])
    assert timeout * (retries + 1) < per_test


def test_a_test_that_times_out_fails_alone(ci):
    """--timeout-method=thread ends the whole pytest process, so no test is reported."""
    assert "--timeout-method=signal" in _integration_step(ci)["run"]
