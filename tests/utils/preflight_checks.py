"""
Pre-flight validation checks for test infrastructure.

Validates prerequisites before test execution:
- IRIS database connectivity
- API keys present
- Schema tables exist

Implements FR-015, FR-016, NFR-003 from spec.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv


@dataclass
class PreflightCheckResult:
    """Result of a single pre-flight check."""

    check_name: str
    passed: bool
    message: str
    remediation: Optional[str] = None
    duration_ms: int = 0


class PreflightChecker:
    """Validates test environment prerequisites."""

    def __init__(self):
        """Initialize checker and load environment."""
        load_dotenv()

    def check_iris_connectivity(self) -> PreflightCheckResult:
        """
        Verify IRIS database is accessible.

        Automatically detects and remediates password change requirements.

        Returns:
            PreflightCheckResult with connectivity status
        """
        start = time.time()

        try:
            from common.iris_connection_manager import get_iris_connection

            conn = get_iris_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            if result and result[0] == 1:
                duration_ms = int((time.time() - start) * 1000)
                return PreflightCheckResult(
                    check_name="IRIS Connectivity",
                    passed=True,
                    message=f"Connected to IRIS at {os.getenv('IRIS_HOST', 'localhost')}:{os.getenv('IRIS_PORT', '11972')}",
                    duration_ms=duration_ms,
                )
            else:
                duration_ms = int((time.time() - start) * 1000)
                return PreflightCheckResult(
                    check_name="IRIS Connectivity",
                    passed=False,
                    message="Connection established but SELECT 1 failed",
                    remediation="Check IRIS instance health",
                    duration_ms=duration_ms,
                )

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)

            # Check if this is a password change issue
            if "Password change required" in str(e) or "password change required" in str(e).lower():
                try:
                    from tests.utils.iris_password_reset import reset_iris_password_if_needed

                    if reset_iris_password_if_needed(e):
                        # Retry connection after password reset
                        from common.iris_connection_manager import get_iris_connection
                        conn = get_iris_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()

                        if result and result[0] == 1:
                            duration_ms = int((time.time() - start) * 1000)
                            return PreflightCheckResult(
                                check_name="IRIS Connectivity",
                                passed=True,
                                message=f"Connected to IRIS (password reset automatically) at {os.getenv('IRIS_HOST', 'localhost')}:{os.getenv('IRIS_PORT', '11972')}",
                                duration_ms=duration_ms,
                            )
                except Exception as reset_error:
                    return PreflightCheckResult(
                        check_name="IRIS Connectivity",
                        passed=False,
                        message=f"Password reset attempted but failed: {str(reset_error)}",
                        remediation="Manual password reset required: docker exec -it iris_db_rag_templates bash",
                        duration_ms=duration_ms,
                    )

            return PreflightCheckResult(
                check_name="IRIS Connectivity",
                passed=False,
                message=f"Failed to connect to IRIS: {str(e)}",
                remediation="Run 'docker-compose up -d' to start IRIS database",
                duration_ms=duration_ms,
            )

    def check_api_keys(self) -> PreflightCheckResult:
        """
        Verify required API keys are present.

        Returns:
            PreflightCheckResult with API key status
        """
        start = time.time()

        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key:
            duration_ms = int((time.time() - start) * 1000)
            return PreflightCheckResult(
                check_name="API Keys",
                passed=True,
                message="OPENAI_API_KEY is configured",
                duration_ms=duration_ms,
            )
        else:
            duration_ms = int((time.time() - start) * 1000)
            return PreflightCheckResult(
                check_name="API Keys",
                passed=False,
                message="OPENAI_API_KEY not found in environment",
                remediation="Add OPENAI_API_KEY to .env file",
                duration_ms=duration_ms,
            )

    def check_schema_tables(self) -> PreflightCheckResult:
        """
        List existing RAG schema tables.

        Note: This doesn't fail if tables don't exist (they may be created
        automatically), but provides information about current state.

        Returns:
            PreflightCheckResult with schema status
        """
        start = time.time()

        try:
            from common.iris_connection_manager import get_iris_connection

            conn = get_iris_connection()
            cursor = conn.cursor()

            # Query for RAG schema tables
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG'
                ORDER BY TABLE_NAME
            """)

            tables = [row[0] for row in cursor.fetchall()]
            duration_ms = int((time.time() - start) * 1000)

            if tables:
                return PreflightCheckResult(
                    check_name="Schema Tables",
                    passed=True,
                    message=f"Found {len(tables)} RAG tables: {', '.join(tables)}",
                    duration_ms=duration_ms,
                )
            else:
                return PreflightCheckResult(
                    check_name="Schema Tables",
                    passed=True,  # Not a failure - tables may be created automatically
                    message="No RAG tables found (will be created automatically)",
                    duration_ms=duration_ms,
                )

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            return PreflightCheckResult(
                check_name="Schema Tables",
                passed=False,
                message=f"Failed to query schema tables: {str(e)}",
                remediation="Verify IRIS connection and permissions",
                duration_ms=duration_ms,
            )

    def run_all_checks(self) -> List[PreflightCheckResult]:
        """
        Run all pre-flight checks.

        Returns:
            List of PreflightCheckResult for all checks

        Performance target: <2 seconds (NFR-003)
        """
        start = time.time()

        results = [
            self.check_iris_connectivity(),
            self.check_api_keys(),
            self.check_schema_tables(),
        ]

        total_duration = time.time() - start

        # Log performance validation
        if total_duration >= 2.0:
            print(f"WARNING: Pre-flight checks took {total_duration:.2f}s, exceeds 2s target (NFR-003)")

        return results

    def print_results(self, results: List[PreflightCheckResult]) -> None:
        """
        Print formatted pre-flight check results.

        Args:
            results: List of check results to display
        """
        print("\n=== Pre-flight Checks ===")

        for result in results:
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.check_name}: {result.message} ({result.duration_ms}ms)")

            if result.remediation:
                print(f"  → Remediation: {result.remediation}")

        total_duration = sum(r.duration_ms for r in results)
        all_passed = all(r.passed for r in results)

        print(f"\nTotal: {total_duration}ms")
        print(f"Status: {'PASS' if all_passed else 'FAIL'}\n")


if __name__ == "__main__":
    """Quick validation of pre-flight checks."""
    checker = PreflightChecker()
    results = checker.run_all_checks()
    checker.print_results(results)

    # Exit with error if any check failed
    if not all(r.passed for r in results):
        exit(1)
