#!/usr/bin/env python3
import logging
import sys
import time
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/ingestion_{int(time.time())}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting PMC document ingestion...")

    # Use simple ingestion approach
    logger.info("Beginning large-scale ingestion of PMC documents...")

    try:
        results = process_pmc_files()
        logger.info(
            f"Ingestion completed! Processed {results.get('total_processed', 0)} documents"
        )
        return 0
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


def process_pmc_files():
    """
    REAL ingestion using loader_fixed with embeddings and schema ensure.
    Processes PMC XML files, generates embeddings, and loads into IRIS.
    """
    import glob
    import json
    import os
    from datetime import datetime

    # Ensure output directory exists
    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    pmc_directory = (
        os.getenv("EVAL_DATASET_DIR")
        or os.getenv("EVAL_PMC_DIR")
        or "data/downloaded_pmc_docs"
    )
    pmc_files = glob.glob(os.path.join(pmc_directory, "*.xml"))
    logger.info(
        f"Using PMC directory: {pmc_directory} - Found {len(pmc_files)} PMC files to process"
    )

    if not pmc_files:
        logger.error("No PMC files found!")
        return {"total_processed": 0, "successful": 0, "failed": 0}

    try:
        # Import real loader and embedding manager
        from data.loader_fixed import process_and_load_documents
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.embeddings.manager import EmbeddingManager

        # Initialize embedding backend
        config_manager = ConfigurationManager()
        embedding_manager = EmbeddingManager(config_manager)
        embedding_func = embedding_manager.embed_texts

        # Run end-to-end processing and loading
        limit = max(2000, len(pmc_files))  # ensure we cover all current files
        batch_size = 50

        logger.info(
            f"Starting REAL ingestion with limit={limit}, batch_size={batch_size}"
        )
        stats = process_and_load_documents(
            pmc_directory=pmc_directory,
            connection=None,  # let loader create and manage connection
            embedding_func=embedding_func,  # REAL embeddings
            db_config=None,
            limit=limit,
            batch_size=batch_size,
        )

        # Persist summary report
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"ingestion_summary_{ts}.json"
        try:
            with open(summary_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Wrote ingestion summary to {summary_path}")
        except Exception as werr:
            logger.warning(f"Failed to write ingestion summary: {werr}")

        # Map to legacy return keys for compatibility
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


if __name__ == "__main__":
    sys.exit(main())
