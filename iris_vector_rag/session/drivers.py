"""The substitutable means of reaching IRIS.

Driver selection used to be several import-time ``try/except`` blocks in different
modules, so which driver a process used depended on which module imported first and
nothing said which had won. Here it is a decision with a log line, and the seven
operations a session needs are the whole of what a driver has to provide -- which is
why a test can supply one that touches no network.

This module is the only place in the library permitted to call the driver's own
connect (see the quality gate in ``scripts/gates/antipatterns.py``).

Feature: 083-iris-session (FR-006, FR-007, FR-018)
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from iris_vector_rag.session.errors import (
    HandshakeTimeout,
    IRISUnreachable,
    NoUsableDriver,
    wrap_connect_failure,
)
from iris_vector_rag.session.identity import SessionIdentity

logger = logging.getLogger(__name__)

PRECHECK_TIMEOUT = 2.0


def default_connect_timeout() -> float:
    return float(os.environ.get("IRIS_CONNECT_TIMEOUT", "10"))


@runtime_checkable
class Driver(Protocol):
    """What a session needs from whatever reaches IRIS."""

    name: str

    def available(self) -> bool:
        """Can this driver be used in this process? Must not connect."""

    def connect(self, identity: SessionIdentity) -> Any:
        """A live connection, or raise."""

    def ping(self, connection: Any) -> bool:
        """Is this connection still usable? False rather than raising."""

    def close(self, connection: Any) -> None:
        """Release a connection. Never raises."""

    def password_expired(self, error: BaseException) -> bool:
        """Is this failure the expired-default-password state?"""

    def recover_password(self, identity: SessionIdentity) -> bool:
        """Clear that state. True if the attempt was made successfully."""


def _import_iris() -> Any:
    try:
        import iris

        return iris
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.debug("iris module not importable: %s: %s", type(exc).__name__, exc)
        return None


def _ping(connection: Any) -> bool:
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            cursor.close()
        return True
    except Exception:
        return False


def _close(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        pass


class EmbeddedDriver:
    """IRIS reached from inside the instance, through the embedded runtime."""

    name = "embedded"

    def _runtime_state(self) -> Optional[str]:
        iris = _import_iris()
        if iris is None or not hasattr(iris, "runtime") or not hasattr(iris, "dbapi"):
            return None
        try:
            return iris.runtime.get().state
        except Exception:
            return None

    def _install_dir(self) -> Optional[str]:
        return os.environ.get("IRISINSTALLDIR") or os.environ.get(
            "ISC_PACKAGE_INSTALLDIR"
        )

    def unavailable_because(self) -> Optional[str]:
        state = self._runtime_state()
        if state is None:
            return "no iris.runtime/iris.dbapi in this process"
        if state in ("embedded-kernel", "embedded-local"):
            return None
        if state == "unavailable" and self._install_dir():
            return None
        return f"embedded runtime state is {state!r}"

    def available(self) -> bool:
        return self.unavailable_because() is None

    def connect(self, identity: SessionIdentity) -> Any:
        iris = _import_iris()
        if iris is None:  # pragma: no cover - available() gates this
            raise NoUsableDriver(identity, "iris module not importable")
        state = self._runtime_state()
        if state == "unavailable":
            install_dir = self._install_dir()
            logger.debug(
                "configuring embedded-local runtime from IRISINSTALLDIR=%s", install_dir
            )
            iris.runtime.configure(mode="embedded", install_dir=install_dir)
        try:
            return iris.dbapi.connect(namespace=identity.namespace)
        except Exception as exc:
            raise wrap_connect_failure(identity, exc) from exc

    def ping(self, connection: Any) -> bool:
        return _ping(connection)

    def close(self, connection: Any) -> None:
        _close(connection)

    def password_expired(self, error: BaseException) -> bool:
        # A process running inside the instance is already past authentication.
        return False

    def recover_password(self, identity: SessionIdentity) -> bool:
        return False


class NetworkDriver:
    """IRIS reached over the SuperServer port, with the hardening this took.

    The handshake runs on a daemon thread because the driver's connect has no
    timeout of its own and can block forever; a daemon thread is abandoned at
    interpreter exit rather than joined.
    """

    name = "network"

    def __init__(self, connect_timeout: Optional[float] = None) -> None:
        self._connect_timeout = connect_timeout

    @property
    def connect_timeout(self) -> float:
        if self._connect_timeout is not None:
            return self._connect_timeout
        return default_connect_timeout()

    def unavailable_because(self) -> Optional[str]:
        iris = _import_iris()
        if iris is None:
            return "iris module not importable"
        if hasattr(iris, "dbapi") and hasattr(iris.dbapi, "connect"):
            return None
        if hasattr(iris, "createConnection") or hasattr(iris, "connect"):
            return None
        return "iris module has no usable connect method"

    def available(self) -> bool:
        return self.unavailable_because() is None

    def _precheck(self, identity: SessionIdentity) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(PRECHECK_TIMEOUT)
            try:
                probe.connect((identity.host, identity.port))
            except (ConnectionRefusedError, OSError, socket.timeout) as exc:
                raise IRISUnreachable(
                    identity,
                    f"port {identity.port} not reachable on {identity.host} "
                    f"(check IRIS is running)",
                ) from exc

    def _raw_connect(self, identity: SessionIdentity) -> Any:
        iris = _import_iris()
        if hasattr(iris, "dbapi") and hasattr(iris.dbapi, "connect"):
            return iris.dbapi.connect(
                hostname=identity.host,
                port=identity.port,
                namespace=identity.namespace,
                username=identity.username,
                password=identity.password,
                sharedmemory=False,
            )
        if hasattr(iris, "createConnection"):
            return iris.createConnection(
                identity.host,
                identity.port,
                identity.namespace,
                identity.username,
                identity.password,
            )
        return iris.connect(
            hostname=identity.host,
            port=identity.port,
            namespace=identity.namespace,
            username=identity.username,
            password=identity.password,
        )

    def connect(self, identity: SessionIdentity) -> Any:
        self._precheck(identity)
        budget = self.connect_timeout
        outcome: Dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["connection"] = self._raw_connect(identity)
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                outcome["error"] = exc

        worker = threading.Thread(target=run, name="iris-connect", daemon=True)
        worker.start()
        worker.join(budget)
        if worker.is_alive():
            raise HandshakeTimeout(
                identity, f"driver handshake did not complete within {budget:.0f}s"
            )
        if "error" in outcome:
            raise wrap_connect_failure(identity, outcome["error"])
        logger.info("connected to IRIS at %s", identity.label)
        return outcome["connection"]

    def ping(self, connection: Any) -> bool:
        return _ping(connection)

    def close(self, connection: Any) -> None:
        _close(connection)

    def password_expired(self, error: BaseException) -> bool:
        text = str(error).lower()
        return "password change required" in text or "expired" in text

    def recover_password(self, identity: SessionIdentity) -> bool:
        """Clear the "password change required" state a fresh container ships with.

        ``UnExpireUserPasswords`` is the documented, version-stable way to do it,
        and running it beats waiting out the container's own 55-second delay.
        """
        container = self._container_publishing(identity.port)
        if not container:
            return False
        logger.info("clearing expired password state in container %s", container)
        script = 'Do ##class(Security.Users).UnExpireUserPasswords("*")\nHalt\n'
        try:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    container,
                    "iris",
                    "session",
                    "IRIS",
                    "-U",
                    "%SYS",
                ],
                input=script.encode(),
                capture_output=True,
                timeout=15,
            )
            return True
        except Exception as exc:
            logger.debug("password recovery failed: %s: %s", type(exc).__name__, exc)
            return False

    def _container_publishing(self, port: int) -> Optional[str]:
        try:
            listed = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}\\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if listed.returncode != 0:
            return None
        for line in listed.stdout.splitlines():
            if str(port) in line and "tcp" in line:
                return line.split("\t")[0].strip()
        return None


def select_driver(
    identity: SessionIdentity,
    prefer_embedded: bool = True,
    connect_timeout: Optional[float] = None,
) -> Driver:
    """Choose a driver, say which and why, and name everything tried on failure."""
    embedded = EmbeddedDriver()
    network = NetworkDriver(connect_timeout=connect_timeout)
    candidates: List[Any] = (
        [embedded, network] if prefer_embedded else [network, embedded]
    )

    rejected: List[str] = []
    for candidate in candidates:
        if candidate.available():
            logger.info(
                "IRIS driver: %s selected for %s", candidate.name, identity.label
            )
            return candidate
        reason = candidate.unavailable_because() or "unavailable"
        rejected.append(f"{candidate.name} ({reason})")

    raise NoUsableDriver(identity, "no usable driver; tried " + ", ".join(rejected))
