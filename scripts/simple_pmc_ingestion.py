#!/usr/bin/env python3
"""
Simple PMC Document Ingestion - Working version
Processes PMC documents using existing infrastructure
"""

import sys
import time
import logging
import glob
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/simple_ingestion_{int(time.time())}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def process_pmc_files():
    """
    REAL ingestion using loader_fixed with embeddings and schema ensure.
    This simple script now performs true ingestion into IRIS.
    """
    import os
    import glob
    import json
    from datetime import datetime

    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    pmc_directory = "data/downloaded_pmc_docs"
    pmc_files = glob.glob(os.path.join(pmc_directory, "*.xml"))
    logger.info(f"Found {len(pmc_files)} PMC files to process (REAL ingestion)")

    if not pmc_files:
        logger.error("No PMC files found!")
        return {"total_processed": 0, "successful": 0, "failed": 0}

    try:
        from data.loader_fixed import process_and_load_documents
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.embeddings.manager import EmbeddingManager

        config_manager = ConfigurationManager()
        embedding_manager = EmbeddingManager(config_manager)
        embedding_func = embedding_manager.embed_texts

        limit = max(2000, len(pmc_files))
        batch_size = 50

        logger.info(f"Starting REAL ingestion with limit={limit}, batch_size={batch_size}")
        stats = process_and_load_documents(
            pmc_directory=pmc_directory,
            connection=None,
            embedding_func=embedding_func,
            db_config=None,
            limit=limit,
            batch_size=batch_size,
        )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"simple_ingestion_summary_{ts}.json"
        try:
            with open(summary_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Wrote ingestion summary to {summary_path}")
        except Exception as werr:
            logger.warning(f"Failed to write ingestion summary: {werr}")

        results = {
            "total_processed": stats.get("processed_count", 0),
            "successful": stats.get("loaded_doc_count", 0),
            "failed": stats.get("error_count", 0),
            "loaded_chunk_count": stats.get("loaded_chunk_count", 0),
            "documents_per_second": stats.get("documents_per_second", 0),
        }
        logger.info(f"REAL ingestion completed. Results: {results}")
        return results

    except Exception as e:
        logger.error(f"REAL ingestion failed: {e}")
        return {"total_processed": 0, "successful": 0, "failed": 0}

def main():
    """Main function."""
    logger.info("Starting simple PMC document ingestion...")
    
    try:
        results = process_pmc_files()
        logger.info(f"Ingestion completed successfully! Processed {results['total_processed']} documents")
        return 0
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())