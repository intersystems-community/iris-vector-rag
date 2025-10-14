"""
IRIS password reset utility for test infrastructure.

Automatically detects and remediates password change requirements.
Implements automatic password reset for Feature 028.
"""

import os
import subprocess
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class IRISPasswordResetHandler:
    """
    Handles automatic detection and remediation of IRIS password change requirements.

    When IRIS requires a password change (common in Docker containers), this handler:
    1. Detects the "Password change required" error
    2. Resets the password using Docker exec
    3. Updates environment variables
    4. Retries the connection
    """

    def __init__(self, container_name: str = "iris_db_rag_templates"):
        """
        Initialize password reset handler.

        Args:
            container_name: Name of IRIS Docker container
        """
        self.container_name = container_name
        self.default_user = "_SYSTEM"
        self.default_password = "SYS"

    def detect_password_change_required(self, error_message: str) -> bool:
        """
        Detect if error is due to password change requirement.

        Args:
            error_message: Error message from connection attempt

        Returns:
            True if password change is required
        """
        password_change_indicators = [
            "Password change required",
            "password change required",
            "PASSWORD_CHANGE_REQUIRED",
            "User must change password",
        ]

        return any(indicator in error_message for indicator in password_change_indicators)

    def reset_iris_password(self, username: str = None, new_password: str = None) -> Tuple[bool, str]:
        """
        Reset IRIS password using Docker exec.

        Args:
            username: Username to reset (defaults to _SYSTEM)
            new_password: New password (defaults to SYS)

        Returns:
            Tuple of (success, message)
        """
        username = username or self.default_user
        new_password = new_password or self.default_password

        try:
            # Check if container is running
            check_cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)

            if self.container_name not in result.stdout:
                return False, f"Container {self.container_name} not running. Start with: docker-compose up -d"

            # Reset password using iris session
            # This uses the IRIS ObjectScript command to change password
            reset_cmd = [
                "docker", "exec", "-i", self.container_name,
                "iris", "session", "IRIS", "-U", "%SYS",
                "##class(Security.Users).ChangePassword('{}','{}')".format(username, new_password)
            ]

            logger.info(f"Resetting IRIS password for user {username}...")

            result = subprocess.run(
                reset_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                input=new_password + "\n"  # Provide current password if needed
            )

            if result.returncode == 0:
                # Update environment variables
                os.environ["IRIS_USERNAME"] = username
                os.environ["IRIS_PASSWORD"] = new_password

                # Wait for password change to take effect
                time.sleep(2)

                return True, f"Password reset successful for user {username}"
            else:
                # Try alternative method using docker exec with SQL
                alt_cmd = [
                    "docker", "exec", "-i", self.container_name,
                    "sh", "-c",
                    f"echo \"ALTER USER {username} IDENTIFY BY '{new_password}'\" | iris sql IRIS -U %SYS"
                ]

                result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=30)

                if "error" not in result.stdout.lower() and "error" not in result.stderr.lower():
                    os.environ["IRIS_USERNAME"] = username
                    os.environ["IRIS_PASSWORD"] = new_password
                    time.sleep(2)
                    return True, f"Password reset successful (via SQL) for user {username}"
                else:
                    return False, f"Password reset failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Password reset timed out after 30 seconds"
        except FileNotFoundError:
            return False, "Docker command not found. Ensure Docker is installed and in PATH"
        except Exception as e:
            return False, f"Password reset failed: {str(e)}"

    def reset_password_via_management_portal(self) -> Tuple[bool, str]:
        """
        Alternative: Reset password via IRIS Management Portal REST API.

        Returns:
            Tuple of (success, message)
        """
        try:
            import requests

            # Get management portal URL
            host = os.environ.get("IRIS_HOST", "localhost")
            mgmt_port = os.environ.get("IRIS_MGMT_PORT", "52773")

            # Try to reset via REST API
            # Note: This requires the Management Portal to be accessible
            url = f"http://{host}:{mgmt_port}/api/mgmnt/v1/user/password"

            # This is a simplified approach - actual implementation would need proper auth
            return False, "Management Portal API password reset not implemented. Use Docker exec method."

        except ImportError:
            return False, "requests package not available"
        except Exception as e:
            return False, f"Management Portal reset failed: {str(e)}"

    def auto_remediate_password_issue(self, error: Exception) -> bool:
        """
        Automatically detect and remediate password change requirement.

        Args:
            error: Exception from connection attempt

        Returns:
            True if remediation successful, False otherwise
        """
        error_msg = str(error)

        if not self.detect_password_change_required(error_msg):
            return False

        logger.warning("⚠️  IRIS password change required. Attempting automatic remediation...")

        success, message = self.reset_iris_password()

        if success:
            logger.info(f"✓ {message}")
            logger.info("Connection should now work. Retrying...")
            return True
        else:
            logger.error(f"✗ {message}")
            logger.error("Manual intervention required:")
            logger.error(f"  1. docker exec -it {self.container_name} bash")
            logger.error(f"  2. iris session IRIS -U %SYS")
            logger.error(f"  3. Do ##class(Security.Users).ChangePassword('{self.default_user}','{self.default_password}')")
            return False


def reset_iris_password_if_needed(error: Exception, max_retries: int = 1) -> bool:
    """
    Convenience function to reset IRIS password if needed.

    Args:
        error: Exception from connection attempt
        max_retries: Maximum number of reset attempts

    Returns:
        True if password was reset successfully
    """
    handler = IRISPasswordResetHandler()

    for attempt in range(max_retries):
        if handler.auto_remediate_password_issue(error):
            return True

        if attempt < max_retries - 1:
            logger.warning(f"Retry {attempt + 1}/{max_retries} for password reset...")
            time.sleep(3)

    return False


if __name__ == "__main__":
    """Quick test of password reset functionality."""
    logging.basicConfig(level=logging.INFO)

    handler = IRISPasswordResetHandler()
    success, message = handler.reset_iris_password()

    print(f"\n{'='*60}")
    print(f"Password Reset Test")
    print(f"{'='*60}")
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Message: {message}")
    print(f"{'='*60}\n")

    if not success:
        exit(1)
