"""`make iris-reset` recovers a wedged project container, and only that container.

The container and port come from the declaration in CLAUDE.md, as in start-iris.sh, so the
target cannot be pointed at another project's container. Only CI's throwaway container is
passed explicitly, and only when CI is set. `--dry-run` prints the plan these tests read.
Decision: docs/remediation/2026-10-review/tickets/G9-verify-fixes-needing-iris.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "iris-reset.sh"
CONTAINER = "iris-vector-rag-iris"
PORT = "51972"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "CLAUDE.md").write_text(
        f"# fake\n\nIRIS_CONTAINER: {CONTAINER}\nIRIS_PORT: {PORT}\n"
    )
    return tmp_path


def run(project: Path, *args: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=project,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(project), **env},
    )


def plan(result: subprocess.CompletedProcess) -> dict:
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(
        line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line
    )


def test_the_declaration_in_claude_md_is_the_target(project):
    got = plan(run(project))
    assert got["container"] == CONTAINER
    assert got["port"] == PORT


def test_steps_stop_cleanly_restart_wait_then_warm_up(project):
    steps = [
        line for line in run(project).stdout.splitlines() if line.startswith("step")
    ]
    names = [line.split("=", 1)[1].split()[0] for line in steps]
    assert names == ["graceful-stop", "restart", "wait-healthy", "warm-up"]


def test_restart_and_health_use_idt_not_a_reimplementation(project):
    out = run(project).stdout
    assert "idt container restart" in out
    assert "idt container test-connection" in out


@pytest.mark.parametrize("other", ["los-iris", "opsreview-iris", "aihub-iris-116"])
def test_another_projects_container_is_refused(project, other):
    result = run(project, "--container", other)
    assert result.returncode != 0
    assert other in result.stderr and CONTAINER in result.stderr


def test_ci_may_name_its_own_throwaway_container(project):
    got = plan(run(project, "--container", "iris", "--port", "1972", CI="true"))
    assert (got["container"], got["port"]) == ("iris", "1972")


def test_environment_cannot_redirect_the_target(project):
    """IRIS_CONTAINER_NAME in the shell must not re-aim a reset at another container."""
    result = run(project, IRIS_CONTAINER_NAME="los-iris")
    assert plan(result)["container"] == CONTAINER


def test_no_declaration_means_no_target_and_no_default(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# nothing declared\n")
    result = run(tmp_path)
    assert result.returncode != 0
    assert "IRIS_CONTAINER" in result.stderr


def test_the_makefile_exposes_the_target():
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "iris-reset:" in makefile
    assert "scripts/ci/iris-reset.sh" in makefile
    assert "iris-reset" in makefile.split(".PHONY:", 1)[1].splitlines()[0]
