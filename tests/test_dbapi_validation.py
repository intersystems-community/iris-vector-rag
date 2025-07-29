#!/usr/bin/env python3
"""
DBAPI Validation Test

A simple validation test to ensure the comprehensive DBAPI test infrastructure is working correctly.
This test validates the basic components without running the full comprehensive test.
"""

import os
import sys
import logging
import traceback

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

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        # Test connection manager imports
        from common.connection_manager import get_connection_manager, set_global_connection_type
        from common.iris_dbapi_connector import irisdbapi
        logger.info("✓ Connection manager imports successful")
        
        # Test utility imports
        from common.utils import get_embedding_func, get_llm_func
        logger.info("✓ Utility imports successful")
        
        # Test core pipeline imports
        try:
            from iris_rag.pipelines.basic import BasicRAGPipeline
            logger.info("✓ BasicRAG pipeline import successful")
        except ImportError as e:
            logger.warning(f"BasicRAG pipeline import failed: {e}")
        
        # Test test utility imports
        from tests.utils import load_pmc_documents
        logger.info("✓ Test utility imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_dbapi_availability():
    """Test DBAPI module availability"""
    logger.info("Testing DBAPI availability...")
    
    try:
        from common.iris_dbapi_connector import irisdbapi
        
        if irisdbapi:
            logger.info(f"✓ DBAPI module available: {irisdbapi}")
            return True
        else:
            logger.warning("DBAPI module not available - this is expected if intersystems-irispython is not installed")
            return False
            
    except Exception as e:
        logger.error(f"DBAPI availability test failed: {e}")
        return False

def test_docker_availability():
    """Test Docker availability"""
    logger.info("Testing Docker availability...")
    
    try:
        import subprocess
        
        # Check if docker command is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✓ Docker available: {result.stdout.strip()}")
        else:
            logger.warning("Docker command failed")
            return False
        
        # Check if docker-compose is available
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✓ Docker Compose available: {result.stdout.strip()}")
        else:
            logger.warning("Docker Compose command failed")
            return False
        
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✓ Docker daemon is running")
            return True
        else:
            logger.warning("Docker daemon is not running")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("Docker commands timed out")
        return False
    except FileNotFoundError:
        logger.warning("Docker commands not found")
        return False
    except Exception as e:
        logger.error(f"Docker availability test failed: {e}")
        return False

def test_file_structure():
    """Test that required files exist"""
    logger.info("Testing file structure...")
    
    required_files = [
        'docker-compose.yml',
        'common/db_init_complete.sql',
        'tests/test_comprehensive_dbapi_rag_system.py',
        'scripts/run_comprehensive_dbapi_test.sh',
        'common/connection_manager.py',
        'common/iris_dbapi_connector.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            logger.info(f"✓ {file_path} exists")
        else:
            logger.error(f"✗ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    return True

def test_script_permissions():
    """Test that shell scripts have execute permissions"""
    logger.info("Testing script permissions...")
    
    script_path = os.path.join(project_root, 'scripts/run_comprehensive_dbapi_test.sh')
    
    if os.path.exists(script_path):
        if os.access(script_path, os.X_OK):
            logger.info("✓ Shell script has execute permissions")
            return True
        else:
            logger.warning("Shell script does not have execute permissions")
            logger.info("Run: chmod +x scripts/run_comprehensive_dbapi_test.sh")
            return False
    else:
        logger.error("Shell script not found")
        return False

def test_environment_variables():
    """Test environment variable handling"""
    logger.info("Testing environment variables...")
    
    # Test setting and getting environment variables
    test_vars = {
        'TEST_DOCUMENT_COUNT': '100',
        'IRIS_HOST': 'localhost',
        'IRIS_PORT': '1972',
        'RAG_CONNECTION_TYPE': 'dbapi'
    }
    
    for var, value in test_vars.items():
        os.environ[var] = value
        retrieved = os.environ.get(var)
        if retrieved == value:
            logger.info(f"✓ Environment variable {var} = {retrieved}")
        else:
            logger.error(f"✗ Environment variable {var} failed: expected {value}, got {retrieved}")
            return False
    
    return True

def run_validation():
    """Run all validation tests"""
    logger.info("=" * 60)
    logger.info("DBAPI VALIDATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("DBAPI Availability", test_dbapi_availability),
        ("Docker Availability", test_docker_availability),
        ("Script Permissions", test_script_permissions),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validation tests passed! Ready to run comprehensive DBAPI test.")
        return True
    else:
        logger.warning("⚠️  Some validation tests failed. Please fix issues before running comprehensive test.")
        
        # Provide helpful suggestions
        if not results.get("DBAPI Availability", False):
            logger.info("\n💡 To install DBAPI support:")
            logger.info("   pip install intersystems-irispython")
        
        if not results.get("Docker Availability", False):
            logger.info("\n💡 To fix Docker issues:")
            logger.info("   - Ensure Docker is installed and running")
            logger.info("   - Check Docker daemon status: sudo systemctl status docker")
            logger.info("   - Add user to docker group: sudo usermod -aG docker $USER")
        
        if not results.get("Script Permissions", False):
            logger.info("\n💡 To fix script permissions:")
            logger.info("   chmod +x scripts/run_comprehensive_dbapi_test.sh")
        
        return False

def main():
    """Main entry point"""
    success = run_validation()
    
    if success:
        logger.info("\n🚀 Ready to run comprehensive test:")
        logger.info("   make test-dbapi-comprehensive")
        logger.info("   # or")
        logger.info("   ./scripts/run_comprehensive_dbapi_test.sh")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()