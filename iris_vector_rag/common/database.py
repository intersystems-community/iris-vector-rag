"""
Database connectivity module for RAG Templates.
Provides simple database connection checking functionality.
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


def wait_for_database(
    max_attempts: int = 30, wait_seconds: int = 2, database_url: Optional[str] = None
) -> bool:
    """
    Wait for database to become available.

    Args:
        max_attempts: Maximum number of connection attempts
        wait_seconds: Seconds to wait between attempts
        database_url: Database URL to connect to (optional)

    Returns:
        True if database is available, False otherwise
    """
    # Get database URL from environment if not provided
    if not database_url:
        database_url = os.getenv("DATABASE_URL", "iris://localhost:1974/USER")

    logger.info(f"Waiting for database at {database_url}")

    for attempt in range(max_attempts):
        try:
            # Try to import and use IRIS connection
            # Parse the database URL to get connection parameters
            # Format: iris://username:password@host:port/namespace
            import urllib.parse

            from .iris_client import IRISClient

            parsed = urllib.parse.urlparse(database_url)

            host = parsed.hostname or "localhost"
            port = parsed.port or 1974
            username = parsed.username or "demo"
            password = parsed.password or "demo"
            namespace = parsed.path.lstrip("/") or "USER"

            # Create IRIS client and test connection
            client = IRISClient(
                host=host,
                port=port,
                username=username,
                password=password,
                namespace=namespace,
            )

            # Test the connection with a simple query
            result = client.execute_query("SELECT 1")
            if result:
                logger.info("Database connection successful")
                return True

        except ImportError:
            # Fallback to basic socket check if IRIS client is not available
            try:
                import socket
                import urllib.parse

                parsed = urllib.parse.urlparse(database_url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 1974

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    logger.info(f"Database port {port} is accessible on {host}")
                    return True

            except Exception as e:
                logger.warning(f"Socket connection check failed: {e}")

        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")

        if attempt < max_attempts - 1:
            logger.info(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    logger.error(f"Failed to connect to database after {max_attempts} attempts")
    return False


def check_database_health() -> dict:
    """
    Check database health and return status information.

    Returns:
        Dictionary with health status information
    """
    try:
        from .iris_client import IRISClient

        # Get connection parameters from environment
        database_url = os.getenv("DATABASE_URL", "iris://localhost:1974/USER")

        import urllib.parse

        parsed = urllib.parse.urlparse(database_url)

        host = parsed.hostname or "localhost"
        port = parsed.port or 1972
        username = parsed.username or "demo"
        password = parsed.password or "demo"
        namespace = parsed.path.lstrip("/") or "USER"

        # Create IRIS client and test connection
        client = IRISClient(
            host=host,
            port=port,
            username=username,
            password=password,
            namespace=namespace,
        )

        # Test with a simple query
        result = client.execute_query("SELECT 1")

        return {
            "status": "healthy" if result else "unhealthy",
            "database_url": f"iris://{host}:{port}/{namespace}",
            "connection_test": "passed" if result else "failed",
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "connection_test": "failed"}
