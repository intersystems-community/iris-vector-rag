"""
IRIS edition detection and validation.

Detects IRIS database edition (Community vs Enterprise) and validates
backend configuration against detected edition.

Feature: 035-make-2-modes
"""

from enum import Enum
from typing import Any

from iris_rag.testing.exceptions import (
    EditionDetectionError,
    EditionMismatchError,
)


class IRISEdition(Enum):
    """
    Detected IRIS database edition.

    Values determined by querying $SYSTEM.License.LicenseType() at runtime.
    """

    COMMUNITY = "community"
    ENTERPRISE = "enterprise"


def detect_iris_edition(connection: Any) -> IRISEdition:
    """
    Detect IRIS edition from active database connection.

    Executes SQL query: SELECT $SYSTEM.License.LicenseType()

    Args:
        connection: Active IRIS database connection

    Returns:
        Detected IRISEdition (COMMUNITY or ENTERPRISE)

    Raises:
        EditionDetectionError: If edition detection fails

    Examples:
        >>> conn = get_iris_connection()
        >>> edition = detect_iris_edition(conn)
        >>> print(edition)
        IRISEdition.COMMUNITY
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT $SYSTEM.License.LicenseType()")
        result = cursor.fetchone()

        if result is None:
            raise EditionDetectionError(
                "Edition detection query returned no results. "
                "Verify IRIS connection is active."
            )

        license_type = result[0]

        # Community edition returns "Community"
        if "community" in license_type.lower():
            return IRISEdition.COMMUNITY

        # Enterprise editions return "Enterprise" or "Enterprise Advanced"
        if "enterprise" in license_type.lower():
            return IRISEdition.ENTERPRISE

        # Unknown license type
        raise EditionDetectionError(
            f"Unrecognized IRIS license type: {license_type}\n"
            "Expected 'Community' or 'Enterprise'"
        )

    except EditionDetectionError:
        # Re-raise our own exceptions
        raise

    except Exception as e:
        # Wrap other exceptions
        raise EditionDetectionError(
            f"Failed to detect IRIS edition: {e}"
        ) from e
