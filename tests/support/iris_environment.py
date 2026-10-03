"""Where the test suite's IRIS comes from.

One rule, from docs/remediation/2026-10-review/tickets/G9-verify-fixes-needing-iris.md:

* `IRIS_CONTAINER_NAME` set   -> that pinned container, which must be the one CLAUDE.md
  declares (CI may pin its own throwaway container; `CI` is set there).
* `IRIS_CONTAINER_NAME` unset -> an ephemeral iris-devtester container, removed at the end.

There is no third branch. An IRIS that cannot be reached raises `IrisUnreachable`; it never
turns into a mock and the test is never skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

DEFAULT_IMAGE = "intersystemsdc/iris-community:2026.1"
RESET_HINT = "Recover a wedged project container with: make iris-reset"


class IrisUnreachable(RuntimeError):
    """The IRIS a test needs cannot be reached."""


class ForeignContainerError(RuntimeError):
    """`IRIS_CONTAINER_NAME` names a container this project does not declare."""


@dataclass(frozen=True)
class IrisTarget:
    container: str
    host: str
    port: int
    namespace: str
    username: str
    password: str
    ephemeral: bool

    def environment(self) -> dict:
        """The IRIS_* variables (iris-devtester's names) the code under test reads."""
        return {
            "IRIS_HOST": self.host,
            "IRIS_PORT": str(self.port),
            "IRIS_USERNAME": self.username,
            "IRIS_PASSWORD": self.password,
            "IRIS_NAMESPACE": self.namespace,
        }

    @property
    def connection_string(self) -> str:
        return f"iris://{self.username}:{self.password}@{self.host}:{self.port}/{self.namespace}"


def declared_container(repo_root: Path) -> str:
    """The container CLAUDE.md declares as `IRIS_CONTAINER: <name>`."""
    claude_md = Path(repo_root) / "CLAUDE.md"
    text = claude_md.read_text() if claude_md.exists() else ""
    match = re.search(r"^IRIS_CONTAINER:\s*(\S+)", text, re.MULTILINE)
    if not match:
        raise IrisUnreachable(
            f"{claude_md} has no 'IRIS_CONTAINER: <name>' line, so there is no project "
            "container to pin. Declare one, or unset IRIS_CONTAINER_NAME to use an "
            "ephemeral container."
        )
    return match.group(1)


def docker_port(name: str) -> Optional[int]:
    """Host port Docker publishes for the container's 1972, or None."""
    try:
        out = subprocess.run(
            ["docker", "port", name, "1972/tcp"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = (
        re.search(r":(\d+)\s*$", out.stdout.strip().splitlines()[0])
        if out.stdout.strip()
        else None
    )
    return int(match.group(1)) if out.returncode == 0 and match else None


def _credentials(env: Mapping[str, str]) -> dict:
    return {
        "namespace": env.get("IRIS_NAMESPACE") or "USER",
        "username": env.get("IRIS_USERNAME") or "_SYSTEM",
        "password": env.get("IRIS_PASSWORD") or "SYS",
    }


def resolve_pinned(
    env: Mapping[str, str],
    *,
    declared: str,
    docker_port: Callable[[str], Optional[int]] = docker_port,
) -> Optional[IrisTarget]:
    """The pinned container, or None when the suite should start an ephemeral one."""
    name = (env.get("IRIS_CONTAINER_NAME") or "").strip()
    if not name:
        return None
    if name != declared and not env.get("CI"):
        raise ForeignContainerError(
            f"IRIS_CONTAINER_NAME={name!r} is not this project's container ({declared!r}). "
            "The test suite only attaches to the container CLAUDE.md declares; "
            "another project's container is never a test database."
        )
    port = docker_port(name)
    if port is None and env.get("IRIS_PORT"):
        port = int(env["IRIS_PORT"])
    if port is None:
        raise IrisUnreachable(
            f"Container {name!r} publishes no port 1972 and IRIS_PORT is not set. "
            f"Is it running? docker start {name}. {RESET_HINT}"
        )
    return IrisTarget(name, "localhost", port, ephemeral=False, **_credentials(env))


def _dbapi_connect(target: IrisTarget) -> Any:
    import iris

    return iris.dbapi.connect(
        hostname=target.host,
        port=target.port,
        namespace=target.namespace,
        username=target.username,
        password=target.password,
        sharedmemory=False,
    )


def connect(
    target: IrisTarget,
    *,
    connector: Callable[[IrisTarget], Any] = _dbapi_connect,
    attempts: int = 6,
    delay: float = 5.0,
) -> Any:
    """A live connection, after a warm-up query. Retries a server that is still coming up."""
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(delay)
        conn = None
        try:
            conn = connector(target)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return conn
        except Exception as exc:  # noqa: BLE001 - reported below with the target
            last = exc
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
    hint = "" if target.ephemeral else f" {RESET_HINT}"
    raise IrisUnreachable(
        f"IRIS at {target.host}:{target.port} (container {target.container}) did not answer "
        f"{attempts} warm-up attempt(s): {last!r}.{hint}"
    ) from last


class EphemeralContainer:
    """A throwaway IRIS container, removed with its volume by `stop()`."""

    def __init__(self, name: str):
        self.name = name

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", "-v", self.name], capture_output=True)


def start_ephemeral(env: Mapping[str, str] = os.environ):
    """Start a throwaway IRIS container. Returns `(EphemeralContainer, IrisTarget)`.

    Started with the docker CLI on a port Docker picks, the way scripts/ci/start-iris.sh does:
    iris-devtester's `IRISContainer.start()` waits on the Docker log stream, which times out
    here. IRIS 2026.1's post-start init (irissqlcli via embedded Python) crashes on amd64 and
    takes IRIS down, so it is skipped with IRIS_INIT=1 and the passwords are unexpired below.
    """
    name = f"ivr-test-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    image = env.get("IRIS_IMAGE") or DEFAULT_IMAGE
    run = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "-p",
            "0:1972",
            "-e",
            "IRIS_INIT=1",
            image,
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        raise IrisUnreachable(f"docker could not start {image}: {run.stderr.strip()}")
    container = EphemeralContainer(name)
    try:
        port = _wait_for_port(name)
        _unexpire_passwords(name)
    except BaseException:
        container.stop()
        raise
    return container, IrisTarget(
        name, "localhost", port, ephemeral=True, **_credentials({})
    )


def _wait_for_port(name: str, attempts: int = 30, delay: float = 2.0) -> int:
    for _ in range(attempts):
        port = docker_port(name)
        if port is not None:
            return port
        time.sleep(delay)
    raise IrisUnreachable(f"Ephemeral container {name} never published port 1972.")


def _unexpire_passwords(name: str, attempts: int = 36, delay: float = 5.0) -> None:
    script = (
        'do ##class(Security.Users).UnExpireUserPasswords("*")\n'
        'write "PASSWORDS_UNEXPIRED",!\nhalt\n'
    )
    for _ in range(attempts):
        out = subprocess.run(
            ["docker", "exec", "-i", name, "iris", "session", "IRIS", "-U%SYS"],
            input=script,
            capture_output=True,
            text=True,
        )
        if "PASSWORDS_UNEXPIRED" in out.stdout:
            return
        time.sleep(delay)
    raise IrisUnreachable(
        f"Ephemeral container {name} never accepted the password unexpire step."
    )
