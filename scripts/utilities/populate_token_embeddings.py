#!/usr/bin/env python3
"""
Script to populate missing ColBERT token embeddings using SetupOrchestrator.

This script uses the SetupOrchestrator to run the ColBERT pipeline setup,
which includes generating missing token embeddings in the RAG.DocumentTokenEmbeddings table.
"""

import logging
import sys
from dotenv import load_dotenv

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.validation.orchestrator import SetupOrchestrator


def setup_logging():
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main function to populate token embeddings."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ColBERT token embeddings population")
    
    try:
        # Initialize configuration manager
        logger.info("Initializing configuration manager")
        config_manager = ConfigurationManager()
        
        # Initialize connection manager
        logger.info("Initializing connection manager")
        connection_manager = ConnectionManager(config_manager=config_manager)
        
        # Initialize setup orchestrator
        logger.info("Initializing setup orchestrator")
        orchestrator = SetupOrchestrator(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        # Run ColBERT pipeline setup with auto-fix enabled
        logger.info("Running ColBERT pipeline setup to generate token embeddings")
        validation_report = orchestrator.setup_pipeline("colbert", auto_fix=True)
        
        # Check results
        if validation_report.overall_valid:
            logger.info("✅ ColBERT pipeline setup completed successfully!")
            logger.info("Token embeddings have been populated.")
        else:
            logger.warning("⚠️ ColBERT pipeline setup completed with some issues:")
            for issue in validation_report.issues:
                logger.warning(f"  - {issue}")
        
        logger.info("Script execution completed")
        
    except Exception as e:
        logger.error(f"❌ Error during token embeddings population: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()