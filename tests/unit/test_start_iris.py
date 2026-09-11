"""`scripts/ci/start-iris.sh` decides what to start, and what never to destroy.

The script defaulted to a container named `iris` on port 1972 and opened with
`docker rm -f`. On a CI runner that is a pristine database; on this machine it
deletes a container belonging to another project and then binds the port
`opsreview-iris` owns. So it could not be run locally at all, and the "reproduce
the CI step" instruction in the release notes was one no one could follow.

Now the container and the port come from the declaration in CLAUDE.md, and an
existing container is reused or started, never recreated, unless `--recreate`
says so in as many words. `--dry-run` prints the plan, which is what these tests
read.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "start-iris.sh"

CONTAINER = "iris-vector-rag-iris"
PORT = "51972"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "bin").mkdir()
    (tmp_path / "CLAUDE.md").write_text(
        f"# fake\n\nIRIS_CONTAINER: {CONTAINER}\nIRIS_PORT: {PORT}\n"
    )
    stub_docker(tmp_path)
    return tmp_path


def stub_docker(project: Path, running: str = "", existing: str = "") -> None:
    """`docker ps` (running) and `docker ps -a` (existing), nothing else."""
    (project / "bin" / "running.txt").write_text(running + "\n" if running else "")
    (project / "bin" / "existing.txt").write_text(
        (existing or running) + "\n" if (existing or running) else ""
    )
    docker = project / "bin" / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        f'here="{project / "bin"}"\n'
        'if [[ "$1" == "ps" ]]; then\n'
        '  if [[ "$*" == *" -a"* ]]; then cat "$here/existing.txt"; '
        'else cat "$here/running.txt"; fi\n'
        "fi\n"
        "exit 0\n"
    )
    docker.chmod(0o755)


def run(project: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=project,
        capture_output=True,
        text=True,
        env={"PATH": f"{project / 'bin'}:/usr/bin:/bin", "HOME": str(project)},
    )


def plan(project: Path, *args: str) -> dict:
    result = run(project, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(
        line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line
    )


# --- where the container and port come from ---------------------------------


def test_the_declaration_in_claude_md_is_the_default(project):
    """Not `iris` on 1972 -- that container and that port belong to other projects."""
    assert plan(project)["container"] == CONTAINER
    assert plan(project)["port"] == PORT


def test_flags_override_the_declaration(project):
    """CI runs a throwaway container; it says which one on the command line."""
    resolved = plan(project, "--container", "iris", "--port", "1972")

    assert resolved["container"] == "iris"
    assert resolved["port"] == "1972"


def test_the_image_can_be_overridden(project):
    assert "iris-community" in plan(project)["image"]
    assert plan(project, "--image", "foo/bar:1")["image"] == "foo/bar:1"


def test_an_undeclared_project_with_no_flags_refuses_to_guess(tmp_path):
    """No declaration and no `--container` used to mean "delete the one called iris"."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "CLAUDE.md").write_text("# fake\n")
    stub_docker(tmp_path)

    result = run(tmp_path)

    assert result.returncode == 1
    assert "IRIS_CONTAINER" in result.stderr, result.stderr


# --- what it does to a container that already exists ------------------------


def test_a_missing_container_is_created(project):
    stub_docker(project, running="los-iris")

    assert plan(project)["action"] == "create"


def test_a_running_container_is_reused_not_recreated(project):
    """This is the line between a CI step and deleting this project's database."""
    stub_docker(project, running=f"los-iris\n{CONTAINER}")

    assert plan(project)["action"] == "reuse"


def test_a_stopped_container_is_started_not_recreated(project):
    stub_docker(project, running="los-iris", existing=f"los-iris\n{CONTAINER}")

    assert plan(project)["action"] == "start"


def test_recreate_is_the_only_way_to_destroy_data(project):
    stub_docker(project, running=CONTAINER)

    assert plan(project, "--recreate")["action"] == "recreate"


def test_recreate_on_a_missing_container_just_creates_it(project):
    stub_docker(project, running="")

    assert plan(project, "--recreate")["action"] == "recreate"


# --- the CI call sites ------------------------------------------------------


def test_ci_workflows_name_the_container_they_start():
    """A workflow that relied on the old defaults would now start this project's."""
    for name in ("release.yml", "pr-integration.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text()
        for line in workflow.splitlines():
            if "start-iris.sh" in line:
                assert "--container" in line and "--port" in line, (name, line)
