#!/usr/bin/env python
"""
Start GraphRAG indexing with DSPy entity extraction enabled.

This script bridges the configuration gap between memory_config.yaml (which has
config nested under rag_memory_config.knowledge_extraction.entity_extraction)
and EntityExtractionService (which expects it at the top level).
"""
import sys
import os
import logging
from pathlib import Path

# Add rag-templates to path
sys.path.insert(0, str(Path(__file__).parent))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.services.entity_extraction import EntityExtractionService
from iris_rag.pipelines.graphrag import GraphRAGPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_config_for_dspy():
    """
    Load configuration and bridge the gap between memory_config.yaml structure
    and EntityExtractionService expectations.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", "memory_config.yaml")
    logger.info(f"Loading configuration from {config_path}")

    config_manager = ConfigurationManager(config_path=config_path)

    # Extract entity extraction config from nested structure
    entity_config = (
        config_manager.get("rag_memory_config", {})
        .get("knowledge_extraction", {})
        .get("entity_extraction", {})
    )

    # Inject at top level for EntityExtractionService
    config_manager._config["entity_extraction"] = entity_config

    logger.info("✅ Configuration bridged for DSPy entity extraction")
    logger.info(f"   Method: {entity_config.get('method')}")
    logger.info(f"   Entity types: {entity_config.get('entity_types')}")
    logger.info(f"   DSPy enabled: {entity_config.get('llm', {}).get('use_dspy')}")
    logger.info(f"   Model: {entity_config.get('llm', {}).get('model')}")

    return config_manager


def main():
    """Start GraphRAG indexing with DSPy-enhanced entity extraction."""
    logger.info("=" * 80)
    logger.info("Starting GraphRAG Indexing with DSPy Entity Extraction")
    logger.info("=" * 80)

    try:
        # Set up configuration
        config_manager = setup_config_for_dspy()

        # Initialize connection
        connection_manager = ConnectionManager(config_manager)
        logger.info("✅ Connection manager initialized")

        # Test entity extraction service
        extractor = EntityExtractionService(config_manager, connection_manager)
        logger.info("✅ Entity extraction service initialized")
        logger.info(f"   Enabled types: {extractor.enabled_types}")
        logger.info(f"   DSPy enabled: {extractor.config.get('llm', {}).get('use_dspy', False)}")

        # Initialize GraphRAG pipeline
        logger.info("\nInitializing GraphRAG pipeline...")
        pipeline = GraphRAGPipeline(
            config_manager=config_manager,
            connection_manager=connection_manager
        )

        # Run indexing
        logger.info("\n" + "=" * 80)
        logger.info("Starting indexing...")
        logger.info("=" * 80)

        # TODO: Add your indexing logic here
        # Example:
        # from iris_rag.core.models import Document
        # documents = load_documents_from_somewhere()
        # results = pipeline.index_documents(documents)

        logger.info("\n⚠️  NOTE: Add your indexing logic to this script")
        logger.info("   Example: Load documents and call pipeline.index_documents()")

        return 0

    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
