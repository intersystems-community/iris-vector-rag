#!/usr/bin/env python
"""
Run tests with real PMC documents following TDD principles.

This script runs the tests with real PMC documents in two phases:
1. RED phase: Tests are expected to fail (by setting TDD_PHASE=RED)
2. GREEN phase: Tests should pass after implementing the solution (by setting TDD_PHASE=GREEN)

Usage:
    python run_tdd_real_pmc_tests.py [--phase RED|GREEN]
"""

import os
import sys
import subprocess
import logging
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"test_results/tdd_real_pmc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def run_tests_with_phase(phase):
    """Run tests with the specified TDD phase."""
    logger.info(f"Running real PMC tests in TDD {phase} phase")
    
    # Ensure test results directory exists
    os.makedirs("test_results", exist_ok=True)
    
    # Set TDD phase environment variable
    env = os.environ.copy()
    env["TDD_PHASE"] = phase
    
    # Command to run tests
    cmd = [
        "poetry", "run", "pytest", "-xvs",
        "-c", "tests/pytest_real_pmc.ini",
        "tests/test_all_with_real_pmc_1000.py",
        "--no-header", "--no-summary", "-v"
    ]
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    # Run tests and capture output
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on test failure
        )
        
        # Log test output
        logger.info(f"Test output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Test errors:\n{result.stderr}")
        
        # Check results based on phase
        if phase == "RED" and result.returncode != 0:
            logger.info("✅ RED phase tests failed as expected (good for TDD)")
            return True
        elif phase == "GREEN" and result.returncode == 0:
            logger.info("✅ GREEN phase tests passed as expected")
            return True
        elif phase == "RED" and result.returncode == 0:
            logger.error("❌ RED phase tests unexpectedly passed (not following TDD)")
            return False
        elif phase == "GREEN" and result.returncode != 0:
            logger.error("❌ GREEN phase tests failed (implementation incomplete)")
            return False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def main():
    """Main function to run TDD tests."""
    parser = argparse.ArgumentParser(description="Run real PMC tests following TDD principles")
    parser.add_argument("--phase", choices=["RED", "GREEN"], default="RED", 
                      help="TDD phase (RED: tests should fail, GREEN: tests should pass)")
    args = parser.parse_args()
    
    # Save test metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "tdd_phase": args.phase,
        "description": f"Running real PMC tests in {args.phase} phase of TDD",
    }
    
    with open(f"test_results/tdd_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Run tests with specified phase
    success = run_tests_with_phase(args.phase)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
