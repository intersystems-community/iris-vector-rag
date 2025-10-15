#!/usr/bin/env python
"""
Test DSPy entity extraction with a sample TrakCare ticket.
"""
import sys
import logging
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.services.entity_extraction import EntityExtractionService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample TrakCare ticket
SAMPLE_TICKET = """
User cannot access TrakCare appointment module. Getting error: 'Access Denied - User permissions not configured'.
Customer: Austin Health
Product: TrakCare 2019.1
Module: Appointment booking

Resolution: Updated user permissions in TrakCare admin console to grant access to appointment module.
Configured role: Receptionist with booking rights.
"""

def main():
    logger.info("=" * 80)
    logger.info("DSPy Entity Extraction Test")
    logger.info("=" * 80)

    try:
        # Initialize configuration from memory_config.yaml
        import os
        config_path = os.path.join(os.path.dirname(__file__), "config", "memory_config.yaml")
        config_manager = ConfigurationManager(config_path=config_path)
        logger.info(f"✅ Configuration loaded from {config_path}")

        # Override config to use the correct path for entity extraction
        # EntityExtractionService expects config at top level, but memory_config.yaml has it nested
        entity_config = config_manager.get("rag_memory_config", {}).get("knowledge_extraction", {}).get("entity_extraction", {})

        # Inject the entity_extraction config at the top level for EntityExtractionService
        config_manager._config["entity_extraction"] = entity_config

        logger.info(f"📝 Entity extraction config:")
        logger.info(f"   Method: {entity_config.get('method')}")
        logger.info(f"   Entity types: {entity_config.get('entity_types')}")
        logger.info(f"   DSPy enabled: {entity_config.get('llm', {}).get('use_dspy')}")

        # Initialize connection (optional for this test)
        try:
            connection_manager = ConnectionManager(config_manager)
            logger.info(f"✅ Connection manager initialized")
        except:
            connection_manager = None
            logger.warning("⚠️  Connection manager not available (OK for test)")

        # Initialize entity extraction service
        extractor = EntityExtractionService(config_manager, connection_manager)
        logger.info(f"✅ Entity extraction service initialized")
        logger.info(f"   Method: {extractor.method}")
        logger.info(f"   Enabled types: {extractor.enabled_types}")

        # Check if DSPy is enabled
        use_dspy = extractor.config.get("llm", {}).get("use_dspy", False)
        logger.info(f"   DSPy enabled: {use_dspy}")

        # Create a test document
        doc = Document(
            id="test_doc_1",
            page_content=SAMPLE_TICKET,
            metadata={"source": "test"}
        )

        # Extract entities
        logger.info("\n" + "-" * 80)
        logger.info("Extracting entities from sample ticket...")
        logger.info("-" * 80)
        entities = extractor.extract_entities(doc)

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info(f"RESULTS: Extracted {len(entities)} entities")
        logger.info("=" * 80)

        for i, entity in enumerate(entities, 1):
            logger.info(f"\n{i}. Entity:")
            logger.info(f"   Text: {entity.text}")
            logger.info(f"   Type: {entity.entity_type}")
            logger.info(f"   Confidence: {entity.confidence:.2f}")
            logger.info(f"   Method: {entity.metadata.get('method', 'unknown')}")

        # Verify target met
        logger.info("\n" + "=" * 80)
        if len(entities) >= 4:
            logger.info(f"✅ SUCCESS: Extracted {len(entities)} entities (target: 4+)")
        else:
            logger.warning(f"⚠️  WARNING: Only {len(entities)} entities extracted (target: 4+)")

        # Extract relationships
        logger.info("\n" + "-" * 80)
        logger.info("Extracting relationships...")
        logger.info("-" * 80)
        relationships = extractor.extract_relationships(entities, doc)

        logger.info(f"\nExtracted {len(relationships)} relationships")
        for i, rel in enumerate(relationships, 1):
            logger.info(f"{i}. {rel.relationship_type}")

        if len(relationships) >= 2:
            logger.info(f"\n✅ SUCCESS: Extracted {len(relationships)} relationships (target: 2+)")
        else:
            logger.warning(f"\n⚠️  WARNING: Only {len(relationships)} relationships (target: 2+)")

        return 0 if len(entities) >= 4 else 1

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
