#!/usr/bin/env python3
"""
Environment Manager - Smart Environment Detection and Activation

This module automatically detects and uses the best available Python environment:
1. UV environment (.venv) if available and working
2. Current active environment if it has required packages
3. System Python as fallback

Developers can override by setting FORCE_ENV environment variable or by
activating their preferred environment before running scripts.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Smart environment detection and management."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.uv_env_path = self.project_root / ".venv"
        self.current_env = os.environ.get("VIRTUAL_ENV")
        self.force_env = os.environ.get("FORCE_ENV")  # Allow developer override

    def get_best_python_executable(self) -> str:
        """
        Get the best Python executable to use.

        Returns:
            Path to Python executable
        """
        # Developer override
        if self.force_env:
            if self.force_env == "system":
                return sys.executable
            elif self.force_env == "uv" and self.uv_env_path.exists():
                return str(self.uv_env_path / "bin" / "python")
            else:
                logger.warning(
                    f"FORCE_ENV={self.force_env} not valid, using auto-detection"
                )

        # If already in a virtual environment and it has required packages, use it
        if self.current_env and self._check_environment_has_iris():
            logger.debug(f"Using current virtual environment: {self.current_env}")
            return sys.executable

        # Check UV environment if available
        if self.uv_env_path.exists():
            uv_python = self.uv_env_path / "bin" / "python"
            if uv_python.exists() and self._check_environment_has_iris(str(uv_python)):
                logger.debug(f"Using UV environment: {self.uv_env_path}")
                return str(uv_python)

        # Fallback to current Python
        logger.debug("Using current Python executable")
        return sys.executable

    def _check_environment_has_iris(self, python_exe: Optional[str] = None) -> bool:
        """
        Check if an environment has the required IRIS packages.

        Args:
            python_exe: Python executable to check (None for current)

        Returns:
            True if environment has required packages
        """
        if python_exe is None:
            python_exe = sys.executable

        try:
            # Quick check for intersystems_irispython package
            result = subprocess.run(
                [
                    python_exe,
                    "-c",
                    "try: import iris; print(hasattr(iris, 'connect')); except ImportError: import iris; print(hasattr(iris, 'connect'))",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            return result.returncode == 0 and "True" in result.stdout

        except Exception as e:
            logger.debug(f"Environment check failed for {python_exe}: {e}")
            return False

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the current environment setup.

        Returns:
            Dictionary with environment information
        """
        best_python = self.get_best_python_executable()

        info = {
            "best_python": best_python,
            "current_python": sys.executable,
            "current_venv": self.current_env,
            "uv_env_exists": self.uv_env_path.exists(),
            "force_env": self.force_env,
            "iris_available": self._check_environment_has_iris(),
            "best_iris_available": self._check_environment_has_iris(best_python),
        }

        return info

    def ensure_iris_available(self) -> bool:
        """
        Ensure IRIS packages are available in the best environment.

        Returns:
            True if IRIS is available, False otherwise
        """
        best_python = self.get_best_python_executable()
        return self._check_environment_has_iris(best_python)

    def run_in_best_environment(
        self, script_path: str, args: list = None
    ) -> subprocess.CompletedProcess:
        """
        Run a script in the best available environment.

        Args:
            script_path: Path to Python script
            args: Additional arguments

        Returns:
            CompletedProcess result
        """
        best_python = self.get_best_python_executable()
        cmd = [best_python, script_path]
        if args:
            cmd.extend(args)

        logger.debug(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root)


def get_iris_connection_in_best_env():
    """
    Get an IRIS connection using the best available environment.
    This function automatically handles environment switching if needed.
    """
    env_manager = EnvironmentManager()

    # If we're already in the best environment, use direct import
    if env_manager.get_best_python_executable() == sys.executable:
        try:
            import iris

            if hasattr(iris, "connect"):
                return iris.connect(
                    hostname=os.environ.get("IRIS_HOST", "localhost"),
                    port=int(os.environ.get("IRIS_PORT", "1974")),
                    namespace=os.environ.get("IRIS_NAMESPACE", "USER"),
                    username=os.environ.get("IRIS_USERNAME", "SuperUser"),
                    password=os.environ.get("IRIS_PASSWORD", "SYS"),
                )
        except Exception as e:
            logger.warning(f"Direct IRIS import failed: {e}")

    # Otherwise, we need to spawn a subprocess in the correct environment
    logger.info("Switching to optimal environment for IRIS connection")
    raise RuntimeError(
        "IRIS connection requires environment switch. "
        "Please use the connection manager or run in the correct environment."
    )


def check_environment_status():
    """Check and report environment status."""
    env_manager = EnvironmentManager()
    info = env_manager.get_environment_info()

    print("=== Environment Status ===")
    print(f"Current Python: {info['current_python']}")
    print(f"Best Python: {info['best_python']}")
    print(f"Current VENV: {info['current_venv'] or 'None'}")
    print(f"UV Environment: {'Available' if info['uv_env_exists'] else 'Not found'}")
    print(f"Force Environment: {info['force_env'] or 'Auto-detect'}")
    print(f"IRIS in Current: {'✅' if info['iris_available'] else '❌'}")
    print(f"IRIS in Best: {'✅' if info['best_iris_available'] else '❌'}")

    if not info["best_iris_available"]:
        print("\n⚠️  Warning: IRIS packages not available in any environment")
        print("   Consider running: uv sync  # if using UV")
        print("   Or: pip install intersystems-irispython")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Environment Manager")
    parser.add_argument("--check", action="store_true", help="Check environment status")
    parser.add_argument("--run", help="Run script in best environment")
    parser.add_argument("args", nargs="*", help="Arguments for script")

    args = parser.parse_args()

    if args.check:
        check_environment_status()
    elif args.run:
        env_manager = EnvironmentManager()
        result = env_manager.run_in_best_environment(args.run, args.args)
        sys.exit(result.returncode)
    else:
        parser.print_help()
