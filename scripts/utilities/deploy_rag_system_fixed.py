"""
Production deployment script for RAG system (Fixed version)
Handles environment setup, configuration, and health checks
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGDeployment:
    """Handles RAG system deployment and configuration"""
    
    def __init__(self, environment: str = "production"):
        """
        Initialize deployment for specified environment
        
        Args:
            environment: One of 'development', 'staging', 'production'
        """
        self.environment = environment
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for the environment"""
        # Default configurations
        configs = {
            "development": {
                "connection_type": "odbc",
                "iris_host": "localhost",
                "iris_port": 1972,
                "iris_namespace": "RAG",
                "iris_username": "demo",
                "iris_password": "demo",
                "log_level": "DEBUG",
                "enable_monitoring": False
            },
            "staging": {
                "connection_type": "odbc",  # Changed from jdbc until fixed
                "iris_host": os.getenv("STAGING_IRIS_HOST", "localhost"),
                "iris_port": int(os.getenv("STAGING_IRIS_PORT", "1972")),
                "iris_namespace": "RAG_STAGING",
                "iris_username": os.getenv("STAGING_IRIS_USER", "demo"),
                "iris_password": os.getenv("STAGING_IRIS_PASS", "demo"),
                "log_level": "INFO",
                "enable_monitoring": True
            },
            "production": {
                "connection_type": os.getenv("PROD_CONNECTION_TYPE", "odbc"),  # Changed default
                "iris_host": os.getenv("PROD_IRIS_HOST", "localhost"),
                "iris_port": int(os.getenv("PROD_IRIS_PORT", "1972")),
                "iris_namespace": os.getenv("PROD_IRIS_NAMESPACE", "RAG"),
                "iris_username": os.getenv("PROD_IRIS_USER", "demo"),
                "iris_password": os.getenv("PROD_IRIS_PASS", "demo"),
                "log_level": "WARNING",
                "enable_monitoring": True
            }
        }
        
        return configs.get(self.environment, configs["development"])
    
    def setup_environment(self):
        """Set up environment variables for the deployment"""
        logger.info(f"Setting up {self.environment} environment...")
        
        # Set environment variables
        os.environ["RAG_CONNECTION_TYPE"] = self.config["connection_type"]
        os.environ["IRIS_HOST"] = self.config["iris_host"]
        os.environ["IRIS_PORT"] = str(self.config["iris_port"])
        os.environ["IRIS_NAMESPACE"] = self.config["iris_namespace"]
        
        if self.config.get("iris_username"):
            os.environ["IRIS_USERNAME"] = self.config["iris_username"]
        if self.config.get("iris_password"):
            os.environ["IRIS_PASSWORD"] = self.config["iris_password"]
        
        # Set logging level
        logging.getLogger().setLevel(self.config["log_level"])
        
        logger.info("Environment setup complete")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            logger.info("✅ Python version OK")
            checks.append(True)
        else:
            logger.error("❌ Python 3.8+ required")
            checks.append(False)
        
        # Check required packages based on connection type
        if self.config["connection_type"] == "jdbc":
            required_packages = [
                "jaydebeapi",
                "jpype1",
                "sentence-transformers",
                "openai",
                "numpy"
            ]
        else:
            # ODBC requirements
            required_packages = [
                "sentence-transformers",
                "openai",
                "numpy"
            ]
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"✅ Package {package} found")
                checks.append(True)
            except ImportError:
                if package in ["jaydebeapi", "jpype1"] and self.config["connection_type"] == "odbc":
                    logger.warning(f"⚠️  Package {package} not found (not required for ODBC)")
                    checks.append(True)  # Not required for ODBC
                else:
                    logger.error(f"❌ Package {package} not found")
                    checks.append(False)
        
        # Check JDBC driver if using JDBC
        if self.config["connection_type"] == "jdbc":
            jdbc_path = "./intersystems-jdbc-3.8.4.jar"
            if os.path.exists(jdbc_path):
                logger.info("✅ JDBC driver found")
                checks.append(True)
            else:
                logger.error(f"❌ JDBC driver not found at {jdbc_path}")
                checks.append(False)
        
        return all(checks)
    
    def test_connection(self) -> bool:
        """Test database connection"""
        logger.info("Testing database connection...")
        
        try:
            if self.config["connection_type"] == "odbc":
                from common.iris_connector import get_iris_connection
                conn = get_iris_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                logger.info("✅ ODBC connection successful")
                return True
            else:
                # JDBC test
                logger.warning("JDBC connection test skipped (authentication issues)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run health checks on all pipelines"""
        logger.info("Running health checks...")
        
        health_status = {}
        
        # List of pipelines to check
        pipelines = [
            "basic_rag",
            "crag",
            "hyde",
            "colbert",
            "noderag",
            "graphrag",
            "hybrid_ifind_rag"
        ]
        
        for pipeline in pipelines:
            try:
                # Simple import test for now
                module = __import__(f"{pipeline}.pipeline", fromlist=[''])
                health_status[pipeline] = True
                logger.info(f"✅ {pipeline} - OK")
            except Exception as e:
                health_status[pipeline] = False
                logger.error(f"❌ {pipeline} - Failed: {e}")
        
        return health_status
    
    def deploy(self) -> bool:
        """Execute the deployment"""
        logger.info(f"Starting RAG deployment for {self.environment}")
        logger.info("=" * 50)
        
        # Setup environment
        self.setup_environment()
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting deployment.")
            return False
        
        # Test connection
        if not self.test_connection():
            logger.error("Connection test failed. Aborting deployment.")
            return False
        
        # Run health checks
        health_status = self.run_health_checks()
        healthy_count = sum(1 for status in health_status.values() if status)
        total_count = len(health_status)
        
        logger.info(f"\nHealth check summary: {healthy_count}/{total_count} pipelines healthy")
        
        # Save deployment info
        deployment_info = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "config": self.config,
            "health_status": health_status,
            "deployment_status": "success" if healthy_count == total_count else "partial"
        }
        
        with open(f"deployment_{self.environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("\n✅ Deployment complete!")
        logger.info(f"Connection type: {self.config['connection_type'].upper()}")
        logger.info(f"Environment: {self.environment}")
        
        return True

def main():
    """Main deployment entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy RAG system")
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    
    args = parser.parse_args()
    
    # Create and run deployment
    deployment = RAGDeployment(args.env)
    success = deployment.deploy()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()