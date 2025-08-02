"""
CLI entry point for Quick Start wizard.

This module provides the main entry point for running the Quick Start CLI wizard
as a Python module using `python -m quick_start.cli`.
"""

import sys
from .wizard import QuickStartCLIWizard


def main():
    """Main entry point for the CLI wizard."""
    wizard = QuickStartCLIWizard(interactive=True)
    result = wizard.run()
    
    # Exit with appropriate code based on result
    if result.get("status") == "success":
        sys.exit(0)
    elif result.get("status") in ["help_displayed", "profiles_listed"]:
        sys.exit(0)
    elif result.get("status") == "cancelled":
        sys.exit(130)  # Standard exit code for SIGINT
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()