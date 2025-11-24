"""
Connector for InterSystems IRIS using Python DBAPI
"""

import logging
import os
import subprocess
import re

logger = logging.getLogger(__name__)


def auto_detect_iris_port():
    """
    Auto-detect running IRIS instance and its SuperServer port.

    Checks in priority order:
    1. Docker containers with IRIS (port 1972 mapped)
    2. Native IRIS instances via 'iris list' command

    Returns:
        int: SuperServer port of first accessible instance, or None if none found.
    """
    # Priority 1: Check for Docker IRIS containers
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Look for IRIS containers with port mappings
            for line in result.stdout.split('\n'):
                if 'iris' in line.lower() and '1972' in line:
                    # Parse port mapping like "0.0.0.0:1972->1972/tcp"
                    # Extract the external port (first number)
                    match = re.search(r'0\.0\.0\.0:(\d+)->1972/tcp', line)
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
            ["iris", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            logger.warning(f"'iris list' command failed with exit code {result.returncode}")
            return None

        # Parse output for running instances
        # Format: "status:       running, since ..."
        # Then next section: "SuperServers: <port>"
        lines = result.stdout.split('\n')

        for i, line in enumerate(lines):
            if 'status:' in line and 'running' in line:
                # Found a running instance, look for SuperServers port in next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    if 'SuperServers:' in lines[j]:
                        # Extract port number
                        match = re.search(r'SuperServers:\s+(\d+)', lines[j])
                        if match:
                            port = int(match.group(1))
                            logger.info(f"✅ Auto-detected native IRIS on SuperServer port {port}")
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


def _get_iris_dbapi_module():
    """
    Attempts to import and return the appropriate IRIS DBAPI module.

    Based on PyPI documentation for intersystems-irispython package:
    - The main import is 'import iris'
    - DBAPI functionality is accessed through the iris module
    - The package provides both native connections and DBAPI interface

    Returns:
        The IRIS DBAPI module if successfully imported, None otherwise.
    """
    try:
        import iris as iris_dbapi

        # DEBUG: Log what we got
        logger.debug(f"Imported iris module: type={type(iris_dbapi)}, has__file__={hasattr(iris_dbapi, '__file__')}")
        logger.debug(f"iris_dbapi.__file__={getattr(iris_dbapi, '__file__', 'NO __file__')}")
        logger.debug(f"hasattr(iris_dbapi, 'connect')={hasattr(iris_dbapi, 'connect')}")
        if hasattr(iris_dbapi, 'connect'):
            logger.debug(f"iris_dbapi.connect={iris_dbapi.connect}")

        # Check if iris module has connect method (official API)
        if hasattr(iris_dbapi, "connect"):
            # The iris module provides the DBAPI interface
            logger.info("Successfully imported 'iris' module with DBAPI interface directly")
            return iris_dbapi
        else:
            logger.error("iris module imported but connect() method not found!")
            logger.error(f"Available attributes: {[x for x in dir(iris_dbapi) if not x.startswith('_')][:20]}")

            # Workaround for pytest module caching issue:
            # During pytest collection, iris may be imported when PYTEST_CURRENT_TEST is set,
            # causing partial initialization (only 3 attributes: current_dir, file_name_elsdk, os)
            # Manually load _init_elsdk.py to inject DBAPI attributes into iris module
            import os as os_mod
            import sys

            # Try to find _init_elsdk.py in multiple locations:
            # 1. Same directory as the imported iris module
            # 2. Virtual environment site-packages (if available)
            # 3. All site-packages in sys.path

            search_paths = []

            # Priority 1: Try to find .venv relative to current working directory or project root
            # This handles the case where uv doesn't add .venv to sys.path
            cwd = os_mod.getcwd()
            potential_venv_paths = [
                os_mod.path.join(cwd, '.venv', 'lib'),
                os_mod.path.join(os_mod.path.dirname(cwd), '.venv', 'lib'),
                os_mod.path.join(os_mod.path.dirname(os_mod.path.dirname(cwd)), '.venv', 'lib'),
            ]

            for venv_lib in potential_venv_paths:
                if os_mod.path.isdir(venv_lib):
                    # Find python3.X/site-packages/iris
                    for item in os_mod.listdir(venv_lib):
                        if item.startswith('python3.'):
                            site_packages = os_mod.path.join(venv_lib, item, 'site-packages', 'iris')
                            if os_mod.path.isdir(site_packages):
                                search_paths.append(site_packages)
                                logger.info(f"Found venv iris at: {site_packages}")
                                break

            # Priority 2: Add the actual imported iris module's directory
            if hasattr(iris_dbapi, '__file__') and iris_dbapi.__file__:
                iris_dir = os_mod.path.dirname(iris_dbapi.__file__)
                if iris_dir not in search_paths:
                    search_paths.append(iris_dir)

            # Priority 3: Check all site-packages directories in sys.path
            for path in sys.path:
                if 'site-packages' in path:
                    venv_iris_dir = os_mod.path.join(path, 'iris')
                    if os_mod.path.isdir(venv_iris_dir) and venv_iris_dir not in search_paths:
                        search_paths.append(venv_iris_dir)

            # Try each search path
            init_elsdk_path = None
            for search_dir in search_paths:
                candidate_path = os_mod.path.join(search_dir, '_init_elsdk.py')
                if os_mod.path.exists(candidate_path):
                    init_elsdk_path = candidate_path
                    logger.info(f"Found _init_elsdk.py at {init_elsdk_path}")
                    break

            if init_elsdk_path and os_mod.path.exists(init_elsdk_path):
                logger.info(f"Attempting to manually load _init_elsdk.py from {init_elsdk_path}")
                try:
                    with open(init_elsdk_path, 'r') as f:
                        init_elsdk_code = compile(f.read(), init_elsdk_path, 'exec')
                    exec(init_elsdk_code, iris_dbapi.__dict__)

                    # Check if connect method is now available
                    if hasattr(iris_dbapi, "connect"):
                        logger.info("Successfully injected DBAPI attributes via manual _init_elsdk exec")
                        return iris_dbapi
                    else:
                        logger.error("Manual _init_elsdk exec failed - connect() still not available")
                except Exception as exec_error:
                    logger.error(f"Failed to exec _init_elsdk.py: {exec_error}")
            else:
                logger.error(f"_init_elsdk.py not found in any search paths: {search_paths}")

        # If neither condition is met, fall through to the except/fallback logic below
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import 'iris' module (circular import issue): {e}")

        # Fallback to direct iris import for older installations
        try:
            import iris

            if hasattr(iris, "connect"):
                logger.info(
                    "Successfully imported 'iris' module with DBAPI interface (fallback)"
                )
                return iris
            else:
                logger.warning(
                    "'iris' module imported but doesn't appear to have DBAPI interface (no 'connect' method)"
                )
        except ImportError as e2:
            logger.warning(f"Failed to import 'iris' module as fallback: {e2}")

    # All import attempts failed
    logger.error(
        "InterSystems IRIS DBAPI module could not be imported. "
        "The 'iris' module was found but doesn't have the expected 'connect' method. "
        "Please ensure the 'intersystems-irispython' package is installed correctly. "
        "DBAPI connections will not be available."
    )
    return None


def get_iris_dbapi_connection():
    """
    Establishes a connection to InterSystems IRIS using direct iris.connect().

    This replaces the problematic DBAPI connection that had SSL errors.
    Uses direct iris.connect() which is proven to work reliably.

    Reads connection parameters from environment variables:
    - IRIS_HOST
    - IRIS_PORT
    - IRIS_NAMESPACE
    - IRIS_USER
    - IRIS_PASSWORD

    Returns:
        A direct IRIS connection object or None if connection fails.
    """
    # Use existing UV-compatible fallback logic
    iris = _get_iris_dbapi_module()
    if iris is None:
        logger.error("Cannot import intersystems_iris.dbapi module")
        return None

    # Get connection parameters from environment with auto-detection fallback
    host = os.environ.get("IRIS_HOST", "localhost")

    # Auto-detect port if not set in environment
    port_env = os.environ.get("IRIS_PORT")
    if port_env:
        port = int(port_env)
        logger.info(f"Using IRIS port from environment: {port}")
    else:
        port = auto_detect_iris_port()
        if port is None:
            logger.warning("Could not auto-detect IRIS port, falling back to default 1972")
            port = 1972

    namespace = os.environ.get("IRIS_NAMESPACE", "USER")
    user = os.environ.get("IRIS_USER", "_SYSTEM")
    password = os.environ.get("IRIS_PASSWORD", "SYS")

    # Retry connection with exponential backoff for transient errors
    max_retries = 3
    retry_delay = 0.5  # Start with 500ms delay

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                import time
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {retry_delay}s delay")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

            logger.info(
                f"Attempting IRIS connection to {host}:{port}/{namespace} as user {user}"
            )

            # Use direct iris.connect() with keyword arguments (PyPI documentation format)
            args = {
                'hostname': host,
                'port': port,
                'namespace': namespace,
                'username': user,
                'password': password
            }
            conn = iris.connect(**args)

            # Validate the connection
            if conn is None:
                logger.error("Direct IRIS connection failed: connection is None")
                if attempt < max_retries - 1:
                    continue
                return None

            # Test the connection with a simple query
            try:
                cursor = conn.cursor()
                if cursor is None:
                    logger.error("Direct IRIS connection failed: cursor is None")
                    conn.close()
                    if attempt < max_retries - 1:
                        continue
                    return None

                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()

                if result is None:
                    logger.error("Direct IRIS connection failed: test query returned None")
                    conn.close()
                    if attempt < max_retries - 1:
                        continue
                    return None

            except Exception as test_e:
                logger.error(f"Direct IRIS connection validation failed: {test_e}")
                try:
                    conn.close()
                except:
                    pass
                if attempt < max_retries - 1:
                    continue
                return None

            logger.info("✅ Successfully connected to IRIS using direct iris.connect()")
            return conn

        except Exception as e:
            logger.error(f"Direct IRIS connection failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            return None

    return None


# Lazy-loaded DBAPI module - initialized only when needed
_cached_irisdbapi = None


def get_iris_dbapi_module():
    """
    Get the IRIS DBAPI module with lazy loading to avoid circular imports.

    This function caches the module after first successful import to avoid
    repeated import attempts.

    Returns:
        The IRIS DBAPI module if available, None otherwise.
    """
    global _cached_irisdbapi

    if _cached_irisdbapi is None:
        _cached_irisdbapi = _get_iris_dbapi_module()

    return _cached_irisdbapi


# For backward compatibility, provide irisdbapi as a property-like access
@property
def irisdbapi():
    """Backward compatibility property for accessing the IRIS DBAPI module."""
    return get_iris_dbapi_module()


# Make irisdbapi available as module attribute through __getattr__
def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    if name == "irisdbapi":
        return get_iris_dbapi_module()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    # Basic test for the connection
    # Ensure environment variables are set (e.g., in a .env file or system-wide)
    # Example:
    # export IRIS_HOST="your_iris_host"
    # export IRIS_PORT="1972"
    # export IRIS_NAMESPACE="USER"
    # export IRIS_USER="your_user"
    # export IRIS_PASSWORD="your_password"
    logging.basicConfig(level=logging.INFO)
    connection = get_iris_dbapi_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT %Version FROM %SYSTEM.Version")
            version = cursor.fetchone()
            logger.info(f"IRIS Version (DBAPI): {version[0]}")
            cursor.close()
        except Exception as e:
            logger.error(f"Error during DBAPI test query: {e}")
        finally:
            connection.close()
            logger.info("DBAPI connection closed.")
    else:
        logger.warning("DBAPI connection could not be established for testing.")
