"""The IRIS container pre-flight check, which this project did not have.

`/speckit.implement` opens by running `.specify/scripts/bash/check-container.sh`
and refusing to touch a task if it exits 1. That file did not exist here, so the
pre-flight silently did nothing for an entire feature, and the container in use
was whatever the shell happened to reach.

The one behavioural change from the copy this is adapted from: a project that
declares no container FAILS instead of passing. Passing on no declaration is how
the check stayed invisible -- there was nothing to notice. Declaring
`IRIS_CONTAINER: none` is the way to say a project needs no container, and that
still passes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".specify" / "scripts" / "bash" / "check-container.sh"

CONTAINER = "iris-vector-rag-iris"
PORT = "51972"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A repo root with a CLAUDE.md and a stubbed `docker` on PATH."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "CLAUDE.md").write_text("# fake project\n")
    return tmp_path


def stub_docker(project: Path, running: str = "", ports: str = "") -> None:
    """Install a `docker` that reports exactly the names and ports we say."""
    (project / "bin" / "names.txt").write_text(running + "\n" if running else "")
    (project / "bin" / "ports.txt").write_text(ports + "\n" if ports else "")
    docker = project / "bin" / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        f'here="{project / "bin"}"\n'
        'case "$1" in\n'
        '  ps) cat "$here/names.txt" ;;\n'
        '  port) cat "$here/ports.txt" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    docker.chmod(0o755)


def declare(project: Path, container: str | None, port: str | None = None) -> None:
    lines = ["# fake project", ""]
    if container is not None:
        lines.append(f"IRIS_CONTAINER: {container}")
    if port is not None:
        lines.append(f"IRIS_PORT: {port}")
    (project / "CLAUDE.md").write_text("\n".join(lines) + "\n")


def run(project: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": f"{project / 'bin'}:/usr/bin:/bin",
        "HOME": str(project),
        "CLAUDE_PROJECT_DIR": str(project),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
    )


# --- the declaration itself -------------------------------------------------


def test_no_declaration_blocks(project):
    """The failure this project actually had: nothing declared, nothing checked."""
    stub_docker(project, running=CONTAINER)

    result = run(project)

    assert result.returncode == 1, result.stdout
    assert "IRIS_CONTAINER" in result.stdout, result.stdout


def test_declaring_none_passes(project):
    """A project that needs no container says so, and says it in one place."""
    declare(project, "none")
    stub_docker(project)

    result = run(project)

    assert result.returncode == 0, result.stdout


def test_a_missing_claude_md_blocks(tmp_path):
    (tmp_path / "bin").mkdir()
    stub_docker(tmp_path, running=CONTAINER)

    result = run(tmp_path)

    assert result.returncode == 1


# --- the container ----------------------------------------------------------


def test_a_running_declared_container_passes(project):
    declare(project, CONTAINER)
    stub_docker(project, running=f"other-iris\n{CONTAINER}\nsomething-else")

    result = run(project)

    assert result.returncode == 0, result.stdout
    assert CONTAINER in result.stdout


def test_a_stopped_container_blocks_and_says_how_to_start_it(project):
    declare(project, CONTAINER)
    stub_docker(project, running="los-iris\nopsreview-iris")

    result = run(project)

    assert result.returncode == 1
    assert f"docker start {CONTAINER}" in result.stdout, result.stdout


def test_a_name_that_only_contains_the_declared_one_is_not_a_match(project):
    """`iris-vector-rag-iris-old` is a different container, not a near miss."""
    declare(project, CONTAINER)
    stub_docker(project, running=f"{CONTAINER}-old")

    result = run(project)

    assert result.returncode == 1


# --- the port, which is the container-crossing hazard -----------------------


def test_the_declared_port_must_be_the_one_the_container_publishes(project):
    """Wrong port, right name: the run would talk to another project's data."""
    declare(project, CONTAINER, PORT)
    stub_docker(project, running=CONTAINER, ports="1972/tcp -> 0.0.0.0:1972")

    result = run(project)

    assert result.returncode == 1
    assert PORT in result.stdout, result.stdout


def test_the_matching_port_passes(project):
    declare(project, CONTAINER, PORT)
    stub_docker(
        project,
        running=CONTAINER,
        ports=f"1972/tcp -> 0.0.0.0:{PORT}\n52773/tcp -> 0.0.0.0:52773",
    )

    result = run(project)

    assert result.returncode == 0, result.stdout


def test_no_port_declared_checks_only_the_name(project):
    declare(project, CONTAINER)
    stub_docker(project, running=CONTAINER, ports="")

    result = run(project)

    assert result.returncode == 0, result.stdout


# --- json, the way a gate consumes it --------------------------------------


def test_json_reports_ok(project):
    import json

    declare(project, CONTAINER, PORT)
    stub_docker(project, running=CONTAINER, ports=f"1972/tcp -> 0.0.0.0:{PORT}")

    result = run(project, "--json")

    payload = json.loads(result.stdout)
    assert payload == {"status": "ok", "container": CONTAINER, "port": PORT}


def test_json_reports_blocked_with_the_reason(project):
    import json

    declare(project, CONTAINER)
    stub_docker(project, running="")

    result = run(project, "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["container"] == CONTAINER
    assert "docker start" in payload["hint"]


# --- this repo's own declaration -------------------------------------------


def test_this_repo_declares_its_own_container_and_port():
    """The prose said `iris-vector-rag-iris` / 51972; now a script can read it."""
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text()

    assert f"\nIRIS_CONTAINER: {CONTAINER}\n" in claude_md
    assert f"\nIRIS_PORT: {PORT}\n" in claude_md
