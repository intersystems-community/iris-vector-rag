"""
Unified IRIS connection module - simplifies connection architecture from 6 components to 1.

This module provides a simple, production-ready API for IRIS database connections:
- Single function call: get_iris_connection()
- Automatic edition detection (Community vs Enterprise)
- Module-level connection caching (singleton pattern)
- Thread-safe operations
- Preserves UV compatibility fix from iris_dbapi_connector.py

Usage:
    # Simple connection (uses environment variables)
    from iris_vector_rag.common import get_iris_connection
    conn = get_iris_connection()

    # With explicit parameters
    conn = get_iris_connection(
        host="localhost",
        port=1972,
        namespace="USER",
        username="_SYSTEM",
        password="SYS"
    )

Environment Variables:
    IRIS_HOST: Database hostname (default: localhost)
    IRIS_PORT: SuperServer port (default: 1972)
    IRIS_NAMESPACE: Target namespace (default: USER)
    IRIS_USER: Database username (default: _SYSTEM)
    IRIS_PASSWORD: Database password (default: SYS)
    IRIS_BACKEND_MODE: Edition override ("community" or "enterprise")

Feature: 051-simplify-iris-connection
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
    Import IRIS DBAPI module with UV compatibility fix.

    This function is copied from iris_dbapi_connector.py to preserve
    the UV environment fix (Issue #5). It handles the pytest module
    caching issue where iris module is partially initialized.

    Returns:
        The IRIS DBAPI module if successfully imported, None otherwise.
    """
    try:
        import iris as iris_dbapi

        # Check if iris module has connect method (official API)
        if hasattr(iris_dbapi, "connect"):
            logger.info(
                "Successfully imported 'iris' module with DBAPI interface directly"
            )
            return iris_dbapi
        else:
            logger.error(
                "iris module imported but connect() method not found!"
            )

            # UV compatibility workaround: manually load _init_elsdk.py
            import os as os_mod
            import sys

            search_paths = []

            # Priority 1: Check .venv relative to current working directory
            cwd = os_mod.getcwd()
            potential_venv_paths = [
                os_mod.path.join(cwd, ".venv", "lib"),
                os_mod.path.join(os_mod.path.dirname(cwd), ".venv", "lib"),
                os_mod.path.join(
                    os_mod.path.dirname(os_mod.path.dirname(cwd)), ".venv", "lib"
                ),
            ]

            for venv_lib in potential_venv_paths:
                if os_mod.path.isdir(venv_lib):
                    for item in os_mod.listdir(venv_lib):
                        if item.startswith("python3."):
                            site_packages = os_mod.path.join(
                                venv_lib, item, "site-packages", "iris"
                            )
                            if os_mod.path.isdir(site_packages):
                                search_paths.append(site_packages)
                                logger.info(f"Found venv iris at: {site_packages}")
                                break

            # Priority 2: Add imported iris module's directory
            if hasattr(iris_dbapi, "__file__") and iris_dbapi.__file__:
                iris_dir = os_mod.path.dirname(iris_dbapi.__file__)
                if iris_dir not in search_paths:
                    search_paths.append(iris_dir)

            # Priority 3: Check all site-packages in sys.path
            for path in sys.path:
                if "site-packages" in path:
                    venv_iris_dir = os_mod.path.join(path, "iris")
                    if (
                        os_mod.path.isdir(venv_iris_dir)
                        and venv_iris_dir not in search_paths
                    ):
                        search_paths.append(venv_iris_dir)

            # Try each search path
            init_elsdk_path = None
            for search_dir in search_paths:
                candidate_path = os_mod.path.join(search_dir, "_init_elsdk.py")
                if os_mod.path.exists(candidate_path):
                    init_elsdk_path = candidate_path
                    logger.info(f"Found _init_elsdk.py at {init_elsdk_path}")
                    break

            if init_elsdk_path and os_mod.path.exists(init_elsdk_path):
                logger.info(
                    f"Attempting to manually load _init_elsdk.py from {init_elsdk_path}"
                )
                try:
                    with open(init_elsdk_path, "r") as f:
                        init_elsdk_code = compile(f.read(), init_elsdk_path, "exec")
                    exec(init_elsdk_code, iris_dbapi.__dict__)

                    if hasattr(iris_dbapi, "connect"):
                        logger.info(
                            "Successfully injected DBAPI attributes via manual _init_elsdk exec"
                        )
                        return iris_dbapi
                    else:
                        logger.error(
                            "Manual _init_elsdk exec failed - connect() still not available"
                        )
                except Exception as exec_error:
                    logger.error(f"Failed to exec _init_elsdk.py: {exec_error}")
            else:
                logger.error(
                    f"_init_elsdk.py not found in any search paths: {search_paths}"
                )

    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import 'iris' module: {e}")

        # Fallback to direct iris import
        try:
            import iris

            if hasattr(iris, "connect"):
                logger.info(
                    "Successfully imported 'iris' module with DBAPI interface (fallback)"
                )
                return iris
            else:
                logger.warning(
                    "'iris' module imported but doesn't have DBAPI interface"
                )
        except ImportError as e2:
            logger.warning(f"Failed to import 'iris' module as fallback: {e2}")

    logger.error(
        "InterSystems IRIS DBAPI module could not be imported. "
        "Please ensure 'intersystems-irispython' package is installed correctly."
    )
    return None


def auto_detect_iris_port() -> Optional[int]:
    """
    Auto-detect running IRIS instance and its SuperServer port.

    Checks in priority order:
    1. Docker containers with IRIS (port 1972 mapped)
    2. Native IRIS instances via 'iris list' command

    Returns:
        SuperServer port of first accessible instance, or None if none found.
    """
    # Priority 1: Check for Docker IRIS containers
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "iris" in line.lower() and "1972" in line:
                    # Parse port mapping like "0.0.0.0:1972->1972/tcp"
                    match = re.search(r"0\.0\.0\.0:(\d+)->1972/tcp", line)
                    if match:
                        port = int(match.group(1))
                        logger.info(f"✅ Auto-detected Docker IRIS on port {port}")
                        return port

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Docker not available or timed out, trying native IRIS")
    except Exception as e:
        logger.debug(f"Docker check failed: {e}")

    # Priority 2: Check native IRIS instances
    try:
        result = subprocess.run(
            ["iris", "list"], capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0:
            logger.warning(
                f"'iris list' command failed with exit code {result.returncode}"
            )
            return None

        lines = result.stdout.split("\n")

        for i, line in enumerate(lines):
            if "status:" in line and "running" in line:
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "SuperServers:" in lines[j]:
                        match = re.search(r"SuperServers:\s+(\d+)", lines[j])
                        if match:
                            port = int(match.group(1))
                            logger.info(
                                f"✅ Auto-detected native IRIS on SuperServer port {port}"
                            )
                            return port

        logger.warning("No running IRIS instances found")
        return None

    except FileNotFoundError:
        logger.warning("'iris' command not found in PATH - cannot auto-detect")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("'iris list' command timed out")
        return None
    except Exception as e:
        logger.warning(f"Failed to auto-detect IRIS port: {e}")
        return None


def _validate_connection_params(
    host: str, port: int, namespace: str, username: str, password: str
) -> None:
    """
    Validate connection parameters before connection attempt (fail-fast).

    Args:
        host: Database hostname
        port: SuperServer port
        namespace: Target namespace
        username: Database username
        password: Database password

    Raises:
        ValidationError: If any parameter is invalid
    """
    # Validate host (non-empty)
    if not host or not host.strip():
        raise ValidationError(
            parameter_name="host",
            invalid_value=host,
            valid_range="non-empty string",
            message="Host cannot be empty",
        )

    # Validate port (1-65535)
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ValidationError(
            parameter_name="port",
            invalid_value=port,
            valid_range="1-65535",
            message=f"Invalid port {port}: must be between 1-65535",
        )

    # Validate namespace (alphanumeric + underscores only, non-empty)
    if not namespace or not namespace.strip():
        raise ValidationError(
            parameter_name="namespace",
            invalid_value=namespace,
            valid_range="non-empty alphanumeric string with underscores",
            message="Namespace cannot be empty",
        )

    # Check namespace format (alphanumeric + underscores only)
    if not re.match(r"^[A-Za-z0-9_]+$", namespace):
        raise ValidationError(
            parameter_name="namespace",
            invalid_value=namespace,
            valid_range="alphanumeric characters and underscores only",
            message=f"Invalid namespace '{namespace}': must be alphanumeric and underscores only",
        )


def _get_connection_params_from_env() -> Dict[str, Any]:
    """
    Read connection parameters from environment variables.

    Returns:
        Dictionary with host, port, namespace, username, password keys
    """
    # Auto-detect port if not set
    port_env = os.environ.get("IRIS_PORT")
    if port_env:
        port = int(port_env)
        logger.info(f"Using IRIS port from environment: {port}")
    else:
        port = auto_detect_iris_port()
        if port is None:
            logger.warning(
                "Could not auto-detect IRIS port, falling back to default 1972"
            )
            port = 1972

    return {
        "host": os.environ.get("IRIS_HOST", "localhost"),
        "port": port,
        "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
        "username": os.environ.get("IRIS_USER", "_SYSTEM"),
        "password": os.environ.get("IRIS_PASSWORD", "SYS"),
    }


def detect_iris_edition() -> Tuple[str, int]:
    """
    Detect IRIS edition and return appropriate connection limit.

    Detection priority:
    1. IRIS_BACKEND_MODE environment variable (override)
    2. License key file parsing
    3. Fallback to "community" mode (safe default)

    Returns:
        Tuple of (edition_type, max_connections)
        - ("community", 1)
        - ("enterprise", 999)

    Caches result for session to avoid repeated detection overhead.
    """
    global _edition_cache

    # Check cache first
    if _edition_cache is not None:
        return _edition_cache

    # Priority 1: Environment variable override
    backend_mode = os.environ.get("IRIS_BACKEND_MODE", "").lower()
    if backend_mode in ("community", "enterprise"):
        max_connections = 1 if backend_mode == "community" else 999
        _edition_cache = (backend_mode, max_connections)
        logger.info(
            f"✅ Edition detected via IRIS_BACKEND_MODE: "
            f"{backend_mode} ({max_connections} connections)"
        )
        return _edition_cache

    # Priority 2: Parse license key file
    # (Placeholder for license key parsing - will be implemented in T041-T042)
    # For now, fallback to community mode

    # Priority 3: Fallback to community mode (safe default)
    _edition_cache = ("community", 1)
    logger.info("✅ Edition fallback: community (1 connection) - safe default")
    return _edition_cache


def get_iris_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    namespace: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """
    Get IRIS database connection with automatic caching and validation.

    This is the main API for IRIS connections. It provides:
    - Module-level connection caching (singleton pattern)
    - Thread-safe operations via threading.Lock
    - Parameter validation (fail-fast)
    - Environment variable fallback
    - UV compatibility preservation

    Args:
        host: Database hostname (default: from IRIS_HOST env or "localhost")
        port: SuperServer port (default: from IRIS_PORT env or auto-detect or 1972)
        namespace: Target namespace (default: from IRIS_NAMESPACE env or "USER")
        username: Database username (default: from IRIS_USER env or "_SYSTEM")
        password: Database password (default: from IRIS_PASSWORD env or "SYS")

    Returns:
        IRIS DBAPI connection object with cursor() method

    Raises:
        ValidationError: If parameters fail validation
        ConnectionError: If connection to IRIS fails

    Example:
        >>> from iris_vector_rag.common import get_iris_connection
        >>> conn = get_iris_connection()  # Uses environment variables
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT 1")
        >>> result = cursor.fetchone()
        >>> cursor.close()
    """
    # Get parameters from environment if not provided
    if any(param is None for param in [host, port, namespace, username, password]):
        env_params = _get_connection_params_from_env()
        host = host or env_params["host"]
        port = port or env_params["port"]
        namespace = namespace or env_params["namespace"]
        username = username or env_params["username"]
        password = password or env_params["password"]

    # Validate parameters (fail-fast)
    _validate_connection_params(host, port, namespace, username, password)

    # Cache key (password excluded to avoid cache misses on password rotation)
    cache_key = (host, port, namespace, username)

    # Thread-safe cache lookup
    with _cache_lock:
        if cache_key in _connection_cache:
            logger.debug(
                f"✅ Returning cached connection for {host}:{port}/{namespace}"
            )
            return _connection_cache[cache_key]

        # Not in cache - create new connection
        logger.info(
            f"Creating new connection to {host}:{port}/{namespace} as {username}"
        )

        # Get IRIS DBAPI module (with UV fix)
        iris = _get_iris_dbapi_module()
        if iris is None:
            raise ConnectionError("Cannot import IRIS DBAPI module")

        # Create connection
        try:
            conn = iris.connect(
                hostname=host,
                port=port,
                namespace=namespace,
                username=username,
                password=password,
            )

            # Validate connection
            if conn is None:
                raise ConnectionError("IRIS connection failed: connection is None")

            # Test connection
            cursor = conn.cursor()
            if cursor is None:
                conn.close()
                raise ConnectionError("IRIS connection failed: cursor is None")

            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()

            if result is None:
                conn.close()
                raise ConnectionError(
                    "IRIS connection failed: test query returned None"
                )

            # Cache connection
            _connection_cache[cache_key] = conn
            logger.info(
                f"✅ Successfully connected to IRIS at {host}:{port}/{namespace}"
            )
            return conn

        except Exception as e:
            logger.error(f"Failed to connect to IRIS: {e}")
            raise ConnectionError(f"IRIS connection failed: {e}") from e


# ==============================================================================
# User Story 3: Optional Connection Pooling (High-Concurrency Scenarios)
# ==============================================================================


class IRISConnectionPool:
    """
    Optional connection pool for high-concurrency scenarios (API servers, batch processing).

    Provides thread-safe connection pooling with edition-aware defaults:
    - Community Edition: max_connections=1 (honor license limit)
    - Enterprise Edition: max_connections=20 (reasonable default, not max 999)

    Usage:
        # Basic usage
        pool = IRISConnectionPool(max_connections=10)
        with pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()

        # Edition-aware default
        pool = IRISConnectionPool()  # Auto-detects edition
        with pool.acquire(timeout=30.0) as conn:
            # Use connection
            pass

    Note: This is OPTIONAL. Use get_iris_connection() for simple single-connection scenarios.
    """

    def __init__(self, max_connections: Optional[int] = None, **connection_params):
        """
        Initialize connection pool with edition-aware defaults.

        Args:
            max_connections: Maximum pool size (None = auto-detect based on edition)
            **connection_params: Connection parameters (host, port, namespace, username, password)
        """
        # Store connection parameters
        self._connection_params = connection_params

        # Determine max_connections based on edition if not specified
        if max_connections is None:
            edition, _ = detect_iris_edition()
            if edition == "community":
                max_connections = 1
            else:  # enterprise
                max_connections = 20  # Reasonable default, not max 999

        self.max_connections = max_connections

        # Thread-safe queue for available connections
        import queue

        self._available_connections: queue.Queue = queue.Queue(maxsize=max_connections)

        # Track all created connections for cleanup
        self._all_connections: List[Any] = []
        self._lock = threading.Lock()

        logger.info(
            f"Initialized IRISConnectionPool with max_connections={max_connections}"
        )

    def acquire(self, timeout: float = 30.0):
        """
        Acquire connection from pool (blocks if pool exhausted until timeout).

        Args:
            timeout: Maximum seconds to wait for available connection

        Returns:
            Context manager that yields connection and auto-releases on exit

        Raises:
            queue.Empty: If timeout expires before connection becomes available
        """
        import queue

        try:
            # Try to get existing connection from pool
            conn = self._available_connections.get(timeout=timeout)
            logger.debug(
                f"Acquired connection from pool (available: {self._available_connections.qsize()})"
            )
            return _PooledConnection(self, conn)

        except queue.Empty:
            # No connections available - try to create new one if under limit
            with self._lock:
                if len(self._all_connections) < self.max_connections:
                    # Create new connection
                    logger.info(
                        f"Creating new connection "
                        f"({len(self._all_connections) + 1}/"
                        f"{self.max_connections})"
                    )
                    conn = get_iris_connection(**self._connection_params)
                    self._all_connections.append(conn)
                    return _PooledConnection(self, conn)

            # Pool exhausted and at max capacity - raise timeout
            raise queue.Empty(
                f"Connection pool exhausted (timeout={timeout}s). "
                f"All {self.max_connections} connections in use."
            )

    def release(self, connection):
        """
        Return connection to pool for reuse.

        Args:
            connection: Connection object to return to pool
        """
        try:
            # Put connection back in queue (non-blocking)
            self._available_connections.put_nowait(connection)
            logger.debug(
                f"Released connection to pool (available: {self._available_connections.qsize()})"
            )
        except Exception as e:
            logger.warning(f"Failed to release connection to pool: {e}")

    def close_all(self):
        """
        Close all connections in pool (cleanup on shutdown).
        """
        with self._lock:
            logger.info(f"Closing all {len(self._all_connections)} connections in pool")
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

            self._all_connections.clear()

            # Clear queue
            while not self._available_connections.empty():
                try:
                    self._available_connections.get_nowait()
                except Exception:
                    break

    def __enter__(self):
        """Context manager entry (not typically used - use acquire() instead)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all connections."""
        self.close_all()


class _PooledConnection:
    """
    Context manager wrapper for pooled connections.

    Automatically releases connection back to pool on exit.
    """

    def __init__(self, pool: IRISConnectionPool, connection):
        self._pool = pool
        self._connection = connection

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release connection back to pool
        self._pool.release(self._connection)
        return False  # Don't suppress exceptions


# Module-level attributes for backward compatibility
def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    if name == "irisdbapi":
        return _get_iris_dbapi_module()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
