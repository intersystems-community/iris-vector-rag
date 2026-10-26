"""The AI Hub lab for spec 093's ConfigStore tests.

`aihub_iris` starts the registered throwaway `ivr-aihub-lab` (IRIS for Health Community
2026.3 AI build, ports 56972/56773) unless it is already running, loads the Wallet secret,
the three `AI.LLM` entries and the two under-privileged users from
`tests/fixtures/aihub/ivrsetup.mac`, and removes all of it at session end. A container the
fixture started is removed too; one it found running is left running.

Test modules import the fixture by name. The tests carry the `aihub` marker: GitHub's
runners cannot pull the internal image, so CI deselects them.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator

import pytest

LAB = Path(__file__).resolve().parents[1] / "fixtures" / "aihub"
CONTAINER = "ivr-aihub-lab"
PORT = 56972
READY_TIMEOUT = 240

LAB_USERS = {
    "admin": ("_SYSTEM", "SYS"),
    "no_wallet": ("ivr_nowallet", "ivr-pw"),
    "no_native": ("ivr_nonative", "ivr-pw"),
}


def _docker(*args: str, stdin: str = "", check: bool = True) -> str:
    result = subprocess.run(
        ["docker", *args], input=stdin, capture_output=True, text=True, timeout=120
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _running() -> bool:
    out = _docker("ps", "--filter", f"name=^{CONTAINER}$", "--format", "{{.Names}}")
    return out.strip() == CONTAINER


def _run_routine(name: str) -> str:
    """Load both routines and run ^<name> in %SYS, returning the terminal output.

    Setup starts by calling ^ivrteardown, so both are always loaded.
    """
    loads = ""
    for routine in ("ivrteardown", "ivrsetup"):
        _docker("cp", str(LAB / f"{routine}.mac"), f"{CONTAINER}:/tmp/{routine}.mac")
        loads += f'do $system.OBJ.Load("/tmp/{routine}.mac","ck-d")\n'
    return _docker(
        "exec",
        "-i",
        CONTAINER,
        "iris",
        "session",
        "IRIS",
        "-U",
        "%SYS",
        stdin=f"{loads}do ^{name}\nhalt\n",
    )


def _set_up_until_ready() -> None:
    """Run the setup routine until it reports SETUP-OK; a fresh instance takes a while."""
    deadline = time.monotonic() + READY_TIMEOUT
    last = ""
    while time.monotonic() < deadline:
        try:
            last = _run_routine("ivrsetup")
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            last = str(exc)
        if "SETUP-OK" in last:
            return
        time.sleep(5)
    raise RuntimeError(f"{CONTAINER} did not set up in {READY_TIMEOUT}s: {last[-500:]}")


@pytest.fixture(scope="session")
def aihub_iris() -> Iterator[Dict[str, object]]:
    """Connection settings for the lab, keyed by role (admin, no_wallet, no_native)."""
    compose = ["compose", "-f", str(LAB / "docker-compose.yml")]
    started = not _running()
    if started:
        try:
            _docker(*compose, "up", "-d")
        except (RuntimeError, FileNotFoundError) as exc:
            pytest.fail(f"cannot start {CONTAINER} (internal image, needs VPN): {exc}")
    _set_up_until_ready()
    try:
        yield {
            "host": "localhost",
            "port": PORT,
            "namespace": "USER",
            "users": LAB_USERS,
        }
    finally:
        _run_routine("ivrteardown")
        if started:
            _docker(*compose, "down", "-v", check=False)
