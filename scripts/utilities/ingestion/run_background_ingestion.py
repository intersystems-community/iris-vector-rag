#!/usr/bin/env python3
"""
Background Ingestion Runner for IRIS RAG Templates

This script runs the document ingestion process in the background with proper
logging and monitoring capabilities.
"""

import logging
import sys
import os
import time
import signal
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.loader_varchar_fixed import process_and_load_documents
from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

# Import ColBERT encoder from centralized utils
from common.utils import get_colbert_doc_encoder_func

# Configure logging for background operation
def setup_logging():
    """Set up comprehensive logging for background operation."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ingestion_background_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Background ingestion logging started. Log file: {log_file}")
    return logger, log_file

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def check_database_connection():
    """Verify database connection before starting."""
    logger = logging.getLogger(__name__)
    try:
        conn = get_iris_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            current_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            logger.info(f"Database connection verified. Current document count: {current_count}")
            return True, current_count
        else:
            logger.error("Failed to establish database connection")
            return False, 0
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False, 0

def run_ingestion_process():
    """Run the main ingestion process."""
    logger = logging.getLogger(__name__)
    
    # Check database connection first
    db_ok, initial_count = check_database_connection()
    if not db_ok:
        logger.error("Cannot proceed without database connection")
        return False
    
    # Set up data directory - using the 100k dataset
    pmc_directory = "data/pmc_100k_downloaded"
    if not os.path.exists(pmc_directory):
        logger.error(f"PMC data directory not found: {pmc_directory}")
        return False
    
    logger.info(f"Starting ingestion from directory: {pmc_directory}")
    logger.info(f"Initial document count in database: {initial_count}")
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model()
        logger.info("Embedding model initialized")
        
        # Create embedding function from model
        def embedding_func(texts):
            return embedding_model.encode(texts).tolist()
        
        # Initialize ColBERT document encoder for token embeddings
        try:
            colbert_encoder = ColBERTDocEncoder(mock=False)
            logger.info("ColBERT document encoder initialized")
            
            # Create ColBERT encoder function that matches expected signature
            def colbert_doc_encoder_func(document_text):
                return colbert_encoder.encode(document_text)
                
        except Exception as e:
            logger.warning(f"Failed to initialize ColBERT encoder: {e}")
            logger.warning("Proceeding without ColBERT token embeddings")
            colbert_doc_encoder_func = None
        
        # Run the ingestion process with a high limit to process all remaining documents
        # Using a limit of 50000 to process the remaining ~47,100 documents
        result = process_and_load_documents(
            pmc_directory=pmc_directory,
            embedding_func=embedding_func,
            colbert_doc_encoder_func=colbert_doc_encoder_func,
            limit=50000,  # Process remaining documents
            batch_size=100,  # Larger batch size for efficiency
            use_mock=False
        )
        
        if result.get("success"):
            logger.info("✅ Ingestion process completed successfully!")
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   - Processed: {result.get('processed_count', 0)} documents")
            logger.info(f"   - Loaded: {result.get('loaded_doc_count', 0)} documents")
            logger.info(f"   - Token embeddings: {result.get('loaded_token_count', 0)}")
            logger.info(f"   - Errors: {result.get('error_count', 0)}")
            logger.info(f"   - Duration: {result.get('duration_seconds', 0):.2f} seconds")
            logger.info(f"   - Rate: {result.get('documents_per_second', 0):.2f} docs/sec")
            
            # Check final count
            db_ok, final_count = check_database_connection()
            if db_ok:
                logger.info(f"📈 Database document count increased from {initial_count} to {final_count}")
                logger.info(f"📈 Net documents added: {final_count - initial_count}")
            
            return True
        else:
            logger.error(f"❌ Ingestion process failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error during ingestion: {e}")
        return False

def main():
    """Main execution function."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up logging
    logger, log_file = setup_logging()
    
    logger.info("🚀 Starting background ingestion process...")
    logger.info(f"📝 Process ID: {os.getpid()}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📋 Log file: {log_file}")
    
    start_time = time.time()
    
    try:
        success = run_ingestion_process()
        
        duration = time.time() - start_time
        if success:
            logger.info(f"🎉 Background ingestion completed successfully in {duration:.2f} seconds")
            sys.exit(0)
        else:
            logger.error(f"💥 Background ingestion failed after {duration:.2f} seconds")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Fatal error in background ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()