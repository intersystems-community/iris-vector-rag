"""
Contract tests for IRIS edition detection.

These tests define the expected API contract for:
- Detecting IRIS Community vs Enterprise edition via SQL query (FR-008)
- Error handling when edition detection fails

Contract: specs/035-make-2-modes/contracts/backend_config_contract.yaml
Status: Tests MUST FAIL until Phase 3.3 implementation
"""

from unittest.mock import MagicMock, patch

import pytest

# These imports will fail until implementation - expected behavior for TDD
from iris_vector_rag.testing.validators import (
    EditionDetectionError,
    IRISEdition,
    detect_iris_edition,
)


@pytest.mark.contract
@pytest.mark.requires_backend_mode
@pytest.mark.requires_database
class TestIRISEditionDetection:
    """Contract tests for detect_iris_edition() operation."""

    def test_detect_community_edition(self):
        """
        Detects COMMUNITY edition from SQL query.

        Given: IRIS database connection returning LicenseType() = "Community"
        When: detect_iris_edition(connection) called
        Then: Returns IRISEdition.COMMUNITY

        Requirement: FR-008
        """
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Mock SQL query result: $SYSTEM.License.LicenseType()
        mock_cursor.fetchone.return_value = ("Community",)

        edition = detect_iris_edition(mock_connection)

        assert edition == IRISEdition.COMMUNITY

        # Verify SQL query was executed
        mock_cursor.execute.assert_called_once()
        sql_call = mock_cursor.execute.call_args[0][0]
        assert "$SYSTEM.License.LicenseType()" in sql_call

    def test_detect_enterprise_edition(self):
        """
        Detects ENTERPRISE edition from SQL query.

        Given: IRIS database connection returning LicenseType() = "Enterprise"
        When: detect_iris_edition(connection) called
        Then: Returns IRISEdition.ENTERPRISE

        Requirement: FR-008
        """
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Mock SQL query result: $SYSTEM.License.LicenseType()
        mock_cursor.fetchone.return_value = ("Enterprise",)

        edition = detect_iris_edition(mock_connection)

        assert edition == IRISEdition.ENTERPRISE

        # Verify SQL query was executed
        mock_cursor.execute.assert_called_once()

    def test_detect_enterprise_edition_advanced(self):
        """
        Detects ENTERPRISE from "Enterprise Advanced" license.

        Given: IRIS database connection returning LicenseType() = "Enterprise Advanced"
        When: detect_iris_edition(connection) called
        Then: Returns IRISEdition.ENTERPRISE

        Requirement: FR-008
        """
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Mock SQL query result
        mock_cursor.fetchone.return_value = ("Enterprise Advanced",)

        edition = detect_iris_edition(mock_connection)

        assert edition == IRISEdition.ENTERPRISE

    def test_detection_failure_sql_error(self):
        """
        Raises EditionDetectionError on SQL failure.

        Given: IRIS database connection that raises exception on SQL query
        When: detect_iris_edition(connection) called
        Then: Raises EditionDetectionError with clear message

        Requirement: FR-008
        """
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Simulate SQL error
        mock_cursor.execute.side_effect = Exception("SQL execution failed")

        with pytest.raises(EditionDetectionError) as exc_info:
            detect_iris_edition(mock_connection)

        error_msg = str(exc_info.value)
        assert "edition" in error_msg.lower()
        assert "detect" in error_msg.lower()

    def test_detection_failure_empty_result(self):
        """
        Raises EditionDetectionError when query returns no rows.

        Given: IRIS database connection returning empty result
        When: detect_iris_edition(connection) called
        Then: Raises EditionDetectionError

        Requirement: FR-008
        """
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Simulate empty result
        mock_cursor.fetchone.return_value = None

        with pytest.raises(EditionDetectionError) as exc_info:
            detect_iris_edition(mock_connection)

        error_msg = str(exc_info.value)
        assert "edition" in error_msg.lower()

    def test_detection_failure_unknown_license_type(self):
        """
        Raises EditionDetectionError for unknown license types.

        Given: IRIS database connection returning LicenseType() = "Unknown"
        When: detect_iris_edition(connection) called
        Then: Raises EditionDetectionError with helpful message

        Requirement: FR-008
        """
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Mock unknown license type
        mock_cursor.fetchone.return_value = ("Unknown License Type",)

        with pytest.raises(EditionDetectionError) as exc_info:
            detect_iris_edition(mock_connection)

        error_msg = str(exc_info.value)
        assert "unknown" in error_msg.lower() or "unrecognized" in error_msg.lower()
