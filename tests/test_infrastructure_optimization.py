#!/usr/bin/env python3
"""
Test script to validate infrastructure optimization features.

This script tests the container reuse, data reset, and other optimization features
without running the full comprehensive test suite.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a shell command with timeout"""
    logger.info(f"Executing: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {command}")
        raise

def check_container_running() -> bool:
    """Check if IRIS container is running"""
    result = run_command("docker-compose ps iris_db --format json")
    return result.returncode == 0 and '"State":"running"' in result.stdout

def check_container_healthy() -> bool:
    """Check if IRIS container is healthy"""
    result = run_command("docker-compose ps iris_db --format json")
    return result.returncode == 0 and '"Health":"healthy"' in result.stdout

def test_container_lifecycle():
    """Test container start/stop/reuse lifecycle"""
    logger.info("Testing container lifecycle...")
    
    # Ensure clean start
    logger.info("Cleaning up any existing containers...")
    run_command("docker-compose down -v")
    
    # Start fresh container
    logger.info("Starting fresh IRIS container...")
    result = run_command("docker-compose up -d iris_db", timeout=180)
    if result.returncode != 0:
        logger.error("Failed to start IRIS container")
        return False
    
    # Wait for healthy status
    logger.info("Waiting for container to become healthy...")
    max_wait = 120
    wait_interval = 5
    elapsed = 0
    
    while elapsed < max_wait:
        if check_container_healthy():
            logger.info("Container is healthy")
            break
        time.sleep(wait_interval)
        elapsed += wait_interval
        logger.info(f"Waiting... ({elapsed}/{max_wait}s)")
    
    if elapsed >= max_wait:
        logger.error("Container failed to become healthy")
        return False
    
    # Test reuse detection
    logger.info("Testing container reuse detection...")
    if not check_container_running():
        logger.error("Container should be running but isn't detected")
        return False
    
    if not check_container_healthy():
        logger.error("Container should be healthy but isn't detected")
        return False
    
    logger.info("Container lifecycle test passed")
    return True

def test_script_flags():
    """Test the script flags without running full test"""
    logger.info("Testing script flags...")
    
    # Test help flag
    result = run_command("./scripts/run_comprehensive_dbapi_test.sh --help")
    if result.returncode != 0:
        logger.error("Help flag failed")
        return False
    
    if "--reuse-iris" not in result.stdout:
        logger.error("Help output doesn't contain --reuse-iris flag")
        return False
    
    if "--reset-data" not in result.stdout:
        logger.error("Help output doesn't contain --reset-data flag")
        return False
    
    logger.info("Script flags test passed")
    return True

def test_makefile_targets():
    """Test that new Makefile targets exist"""
    logger.info("Testing Makefile targets...")
    
    # Test help output contains new targets
    result = run_command("make help")
    if result.returncode != 0:
        logger.error("Make help failed")
        return False
    
    required_targets = [
        "test-dbapi-comprehensive-reuse",
        "test-dbapi-comprehensive-reuse-reset", 
        "test-dbapi-dev",
        "test-dbapi-dev-reset"
    ]
    
    for target in required_targets:
        if target not in result.stdout:
            logger.error(f"Makefile help doesn't contain target: {target}")
            return False
    
    logger.info("Makefile targets test passed")
    return True

def test_environment_variables():
    """Test environment variable handling"""
    logger.info("Testing environment variables...")
    
    # Set test environment variables
    test_env = os.environ.copy()
    test_env.update({
        'IRIS_REUSE_MODE': 'true',
        'IRIS_RESET_DATA': 'true',
        'TEST_DOCUMENT_COUNT': '100'
    })
    
    # Import the test runner class to verify it reads the variables
    try:
        from tests.test_comprehensive_dbapi_rag_system import ComprehensiveDBAPITestRunner
        
        # Temporarily set environment
        old_env = {}
        for key, value in test_env.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            runner = ComprehensiveDBAPITestRunner()
            
            if not runner.reuse_iris:
                logger.error("IRIS_REUSE_MODE not properly read")
                return False
            
            if not runner.reset_data:
                logger.error("IRIS_RESET_DATA not properly read")
                return False
            
            if runner.test_document_count != 100:
                logger.error("TEST_DOCUMENT_COUNT not properly read")
                return False
            
        finally:
            # Restore environment
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        
    except ImportError as e:
        logger.error(f"Failed to import test runner: {e}")
        return False
    
    logger.info("Environment variables test passed")
    return True

def cleanup():
    """Cleanup test resources"""
    logger.info("Cleaning up test resources...")
    run_command("docker-compose down -v")

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("INFRASTRUCTURE OPTIMIZATION VALIDATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Script Flags", test_script_flags),
        ("Makefile Targets", test_makefile_targets),
        ("Environment Variables", test_environment_variables),
        ("Container Lifecycle", test_container_lifecycle),
    ]
    
    passed = 0
    failed = 0
    
    try:
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                if test_func():
                    logger.info(f"✅ {test_name} test PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} test FAILED")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ {test_name} test FAILED with exception: {e}")
                failed += 1
    
    finally:
        cleanup()
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total:  {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All infrastructure optimization tests PASSED!")
        return 0
    else:
        logger.error(f"💥 {failed} test(s) FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())