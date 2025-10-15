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

        # Check if iris_dbapi module has _DBAPI submodule with connect method
        if hasattr(iris_dbapi, "_DBAPI") and hasattr(iris_dbapi._DBAPI, "connect"):
            # The _DBAPI submodule provides the DBAPI interface
            logger.info("Successfully imported 'iris' module with DBAPI interface")
            return iris_dbapi._DBAPI
        elif hasattr(iris_dbapi, "connect"):
            # The iris_dbapi module itself provides the DBAPI interface
            logger.info("Successfully imported 'iris' module with DBAPI interface")
            return iris_dbapi
        else:
            logger.warning(
                "'iris' module imported but doesn't appear to have DBAPI interface (no 'connect' method)"
            )
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
    # Use direct iris import instead of DBAPI
    try:
        import iris
    except ImportError as e:
        logger.error(f"Cannot import iris module: {e}")
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

            # Use direct iris.connect() - this avoids SSL issues
            conn = iris.connect(host, port, namespace, user, password)

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
