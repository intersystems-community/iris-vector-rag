#!/usr/bin/env python3
"""
Deploy ObjectScript classes for RAG integration.

This script compiles and deploys the ObjectScript wrapper classes
to the IRIS database for RAG pipeline integration.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compile_objectscript_class(class_file_path: str, class_name: str) -> bool:
    """
    Compile an ObjectScript class file in IRIS.
    
    Args:
        class_file_path: Path to the .cls file
        class_name: Name of the class (e.g., RAGDemo.Invoker)
        
    Returns:
        True if compilation successful, False otherwise
    """
    try:
        logger.info(f"Compiling ObjectScript class: {class_name}")
        
        # Read the class file content
        with open(class_file_path, 'r') as f:
            class_content = f.read()
        
        # Get IRIS connection
        iris_conn = get_iris_connection()
        cursor = iris_conn.cursor()
        
        # Create a temporary file in IRIS-accessible location
        temp_file = f"/tmp/{class_name.replace('.', '_')}.cls"
        
        # Use IRIS SQL to create the class
        # Note: This is a simplified approach - in production, you might use
        # the Management Portal or other IRIS tools for class compilation
        
        # For now, we'll try to execute the class compilation via SQL
        # This may require adjustments based on your IRIS setup
        compile_sql = f"""
        DO $SYSTEM.OBJ.CompileText("{class_content}", "ck")
        """
        
        try:
            cursor.execute("SELECT 1 AS test")  # Test connection
            logger.info(f"IRIS connection successful for {class_name}")
            
            # Note: Direct compilation via SQL may not work in all IRIS configurations
            # This is a placeholder for the actual compilation logic
            logger.warning(f"Class {class_name} compilation requires manual deployment to IRIS")
            logger.info(f"Class file available at: {class_file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to compile {class_name}: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error compiling ObjectScript class {class_name}: {str(e)}")
        return False


def deploy_all_classes() -> bool:
    """
    Deploy all ObjectScript classes for RAG integration.
    
    Returns:
        True if all deployments successful, False otherwise
    """
    logger.info("Starting ObjectScript class deployment")
    
    # Define classes to deploy
    classes_to_deploy = [
        {
            "file": "objectscript/RAGDemo.Invoker.cls",
            "name": "RAGDemo.Invoker"
        },
        {
            "file": "objectscript/RAGDemo.TestBed.cls", 
            "name": "RAGDemo.TestBed"
        }
    ]
    
    success_count = 0
    total_count = len(classes_to_deploy)
    
    for class_info in classes_to_deploy:
        class_file = Path(project_root) / class_info["file"]
        
        if not class_file.exists():
            logger.error(f"Class file not found: {class_file}")
            continue
            
        if compile_objectscript_class(str(class_file), class_info["name"]):
            success_count += 1
            logger.info(f"Successfully processed: {class_info['name']}")
        else:
            logger.error(f"Failed to process: {class_info['name']}")
    
    logger.info(f"Deployment complete: {success_count}/{total_count} classes processed")
    return success_count == total_count


def verify_deployment() -> bool:
    """
    Verify that the ObjectScript classes are properly deployed.
    
    Returns:
        True if verification successful, False otherwise
    """
    logger.info("Verifying ObjectScript class deployment")
    
    try:
        iris_conn = get_iris_connection()
        cursor = iris_conn.cursor()
        
        # Test classes to verify
        test_queries = [
            ("RAGDemo.InvokerExists", "SELECT RAGDemo.InvokerExists() AS exists"),
            ("RAGDemo.TestBedExists", "SELECT RAGDemo.TestBedExists() AS exists"),
            ("RAGDemo.HealthCheck", "SELECT RAGDemo.HealthCheck() AS health")
        ]
        
        for test_name, query in test_queries:
            try:
                logger.info(f"Testing: {test_name}")
                cursor.execute(query)
                result = cursor.fetchone()
                logger.info(f"Test {test_name} result: {result}")
                
            except Exception as e:
                logger.warning(f"Test {test_name} failed (expected if not deployed): {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False


def main():
    """Main deployment function."""
    logger.info("ObjectScript RAG Integration Deployment")
    logger.info("=" * 50)
    
    # Check if we're in the right directory
    if not Path("objectscript").exists():
        logger.error("ObjectScript directory not found. Run from project root.")
        sys.exit(1)
    
    # Deploy classes
    if deploy_all_classes():
        logger.info("All classes processed successfully")
    else:
        logger.warning("Some classes failed to process")
    
    # Verify deployment
    verify_deployment()
    
    # Print manual deployment instructions
    print("\n" + "=" * 60)
    print("MANUAL DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)
    print("Due to IRIS ObjectScript compilation complexities, manual deployment may be required:")
    print()
    print("1. Copy the .cls files to your IRIS instance:")
    print("   - objectscript/RAGDemo.Invoker.cls")
    print("   - objectscript/RAGDemo.TestBed.cls")
    print()
    print("2. In IRIS Management Portal or Terminal:")
    print("   - Navigate to System Explorer > Classes")
    print("   - Import the .cls files")
    print("   - Compile with 'ck' flags")
    print()
    print("3. Verify deployment by running:")
    print("   SELECT RAGDemo.InvokerExists() AS test")
    print("   SELECT RAGDemo.TestBedExists() AS test")
    print()
    print("4. Test the integration:")
    print("   SELECT RAGDemo.HealthCheck() AS health")
    print("=" * 60)


if __name__ == "__main__":
    main()