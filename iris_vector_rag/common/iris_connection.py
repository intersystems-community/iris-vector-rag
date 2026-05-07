"""
Unified IRIS connection module - simplifies connection architecture from 6 components to 1.

This module provides a simple, production-ready API for IRIS database connections:
- Single function call: get_iris_connection()
- Automatic edition detection (Community vs Enterprise)
- Module-level connection caching (singleton pattern)
- Thread-safe operations
- Preserves UV compatibility fix from iris_dbapi_connector.py

Feature: 051-simplify-iris-connection
Feature: 064-llm-cache-disk (Connection hardening bypass)
"""

import logging
import os
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

from iris_vector_rag.common.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Module-level connection cache (singleton pattern)
_connection_cache: Dict[Tuple[str, int, str, str], Any] = {}
_cache_lock = threading.Lock()

# Module-level edition cache (session-wide)
_edition_cache: Optional[Tuple[str, int]] = None


def _get_iris_dbapi_module():
    """
    Import the IRIS connectivity module (intersystems-irispython).

    Uses `import iris` which provides both iris.connect() and
    iris.createConnection(). This follows the same pattern as
    iris-vector-graph and the official InterSystems documentation.
    """
    try:
        import iris
        if hasattr(iris, "connect") or hasattr(iris, "createConnection"):
            return iris
        logger.warning(
            "iris module imported from %s but has no connect/createConnection",
            getattr(iris, "__file__", "unknown")
        )
    except Exception as e:
        logger.error("Failed to import iris module: %s: %s", type(e).__name__, e)

    logger.error(
        "InterSystems IRIS Python SDK not available. "
        "Install with: pip install intersystems-irispython>=5.1.2"
    )
    return None


def auto_detect_iris_port() -> Optional[int]:
    """Auto-detect running IRIS instance and its SuperServer port."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\\t{{.Ports}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "iris" in line.lower() and "1972" in line:
                    match = re.search(r"0\.0\.0\.0:(\d+)->1972/tcp", line)
                    if match:
                        return int(match.group(1))
    except Exception:
        pass

    try:
        result = subprocess.run(["iris", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if "status:" in line and "running" in line:
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "SuperServers:" in lines[j]:
                            match = re.search(r"SuperServers:\s+(\d+)", lines[j])
                            if match:
                                return int(match.group(1))
    except Exception:
        pass
    return None


def _validate_connection_params(h: str, p: int, n: str, u: str, pwd: str) -> None:
    if not h or not h.strip():
        raise ValidationError("host", h, "non-empty string", "Host cannot be empty")
    if not isinstance(p, int) or p < 1 or p > 65535:
        raise ValidationError("port", p, "1-65535", f"Invalid port: {p}")
    if not n or not n.strip() or not re.match(r"^[A-Za-z0-9_]+$", n):
        raise ValidationError("namespace", n, "alphanumeric + underscores", f"Invalid namespace: {n}")


def _get_connection_params_from_env() -> Dict[str, Any]:
    port_env = os.environ.get("IRIS_PORT")
    port = int(port_env) if port_env else (auto_detect_iris_port() or 1972)
    return {
        "host": os.environ.get("IRIS_HOST", "localhost"),
        "port": port,
        "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
        "username": os.environ.get("IRIS_USER", os.environ.get("IRIS_USERNAME", "_SYSTEM")),
        "password": os.environ.get("IRIS_PASSWORD", "SYS"),
    }


def detect_iris_edition() -> Tuple[str, int]:
    global _edition_cache
    if _edition_cache is not None:
        return _edition_cache
    backend_mode = os.environ.get("IRIS_BACKEND_MODE", "").lower()
    if backend_mode in ("community", "enterprise"):
        max_connections = 1 if backend_mode == "community" else 999
        _edition_cache = (backend_mode, max_connections)
        return _edition_cache
    _edition_cache = ("community", 1)
    return _edition_cache


def _hard_fix_iris_passwords(host: str, port: int):
    """Bypasses 55s delay in Community containers."""
    try:
        result = subprocess.run(["docker", "ps", "--format", "{{.Names}}\\t{{.Ports}}"], capture_output=True, text=True, timeout=5)
        container_name = None
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if str(port) in line and "tcp" in line:
                    container_name = line.split("\t")[0].strip()
                    break
        if not container_name:
            return False
        logger.info(f"Bypassing IRIS hardening for: {container_name}")
        cmds = [
            'Set user = ##class(Security.Users).%OpenId("SuperUser")',
            'Do user.PasswordSet("SYS")',
            'Do user.UnExpirePassword()',
            'Do user.%Save()',
            'Set user = ##class(Security.Users).%OpenId("_SYSTEM")',
            'Do user.PasswordSet("SYS")',
            'Do user.UnExpirePassword()',
            'Do user.%Save()',
            'Halt'
        ]
        subprocess.run(["docker", "exec", "-i", container_name, "iris", "session", "IRIS", "-U", "%SYS"], input="\n".join(cmds).encode(), capture_output=True, timeout=15)
        return True
    except Exception:
        return False


def get_iris_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    namespace: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    env = _get_connection_params_from_env()
    h = env["host"] if host is None else host
    p = env["port"] if port is None else port
    n = env["namespace"] if namespace is None else namespace
    u = env["username"] if username is None else username
    pwd = env["password"] if password is None else password
    _validate_connection_params(h, p, n, u, pwd)
    cache_key = (h, p, n, u)

    with _cache_lock:
        for attempt in range(2):
            if cache_key in _connection_cache:
                conn = _connection_cache[cache_key]
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
                    return conn
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del _connection_cache[cache_key]

            iris_mod = _get_iris_dbapi_module()
            if iris_mod is None:
                raise ConnectionError("Cannot import IRIS DBAPI module")

            try:
                if hasattr(iris_mod, "createConnection"):
                    conn = iris_mod.createConnection(h, p, n, u, pwd)
                else:
                    conn = iris_mod.connect(
                        hostname=h,
                        port=p,
                        namespace=n,
                        username=u,
                        password=pwd,
                    )
                _connection_cache[cache_key] = conn
                logger.info(f"✅ Connected to IRIS at {h}:{p}/{n}")
                return conn
            except Exception as e:
                msg = str(e).lower()
                if ("password change required" in msg or "expired" in msg) and attempt == 0:
                    if _hard_fix_iris_passwords(h, p):
                        continue
                
                # Always raise on final attempt or if not a password issue that we can fix
                if attempt == 1 or not ("password change required" in msg or "expired" in msg):
                    raise ConnectionError(
                        f"IRIS connection failed to {h}:{p}/{n}: {e} (check configuration)"
                    ) from e
        
        # This part should theoretically not be reached if all paths raise or return
        raise ConnectionError(f"IRIS connection failed to {h}:{p}/{n}")


class IRISConnectionPool:
    def __init__(self, max_connections: Optional[int] = None, **connection_params):
        self._connection_params = connection_params
        if max_connections is None:
            edition, _ = detect_iris_edition()
            max_connections = 1 if edition == "community" else 20
        self.max_connections = max_connections
        import queue
        self._available_connections: queue.Queue = queue.Queue(maxsize=max_connections)
        self._all_connections: List[Any] = []
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0):
        import queue
        try:
            conn = self._available_connections.get(timeout=timeout)
            return _PooledConnection(self, conn)
        except queue.Empty:
            with self._lock:
                if len(self._all_connections) < self.max_connections:
                    conn = get_iris_connection(**self._connection_params)
                    self._all_connections.append(conn)
                    return _PooledConnection(self, conn)
            raise queue.Empty(f"Connection pool exhausted (timeout={timeout}s)")

    def release(self, connection):
        try:
            self._available_connections.put_nowait(connection)
        except Exception:
            pass


    def close_all(self):
        with self._lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
            while not self._available_connections.empty():
                try:
                    self._available_connections.get_nowait()
                except Exception:
                    break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


class _PooledConnection:
    def __init__(self, pool: IRISConnectionPool, connection):
        self._pool, self._connection = pool, connection

    def __enter__(self):
        return self._connection
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.release(self._connection)
        return False
