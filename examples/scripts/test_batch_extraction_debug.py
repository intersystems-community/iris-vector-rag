#!/usr/bin/env python3
"""
DEBUG version - Show actual JSON output from batch extraction.
"""
import sys
import json
import logging
from pathlib import Path

# Add rag-templates to path
sys.path.insert(0, str(Path(__file__).parent / "iris_rag"))

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.dspy_modules.batch_entity_extraction import BatchEntityExtractionModule

# Configure logging to show everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Just 3 tickets for quick test
SAMPLE_TICKETS = [
    {
        "id": "I123456",
        "text": "User reports unable to access TrakCare patient portal. Error message 'Authentication Failed' appears after entering credentials. Using TrakCare v2023.1 on Chrome browser.",
    },
    {
        "id": "I123457",
        "text": "Lab results not displaying in TrakCare EMR module. Technician unable to view test results for patient in Radiology department. System shows loading spinner indefinitely.",
    },
    {
        "id": "I123458",
        "text": "Prescription module crashing when trying to prescribe medications. Doctor reports TrakCare Pharmacy module freezes when searching for drug formulary. Impacts patient care workflow.",
    },
]


def setup_dspy():
    """Configure DSPy for entity extraction."""
    config_path = Path(__file__).parent / "config" / "memory_config.yaml"
    config_manager = ConfigurationManager(str(config_path))

    # Bridge the config structure
    entity_config = (
        config_manager.get("rag_memory_config", {})
        .get("knowledge_extraction", {})
        .get("entity_extraction", {})
    )
    config_manager._config["entity_extraction"] = entity_config

    # Enable DSPy
    if "llm" not in config_manager._config["entity_extraction"]:
        config_manager._config["entity_extraction"]["llm"] = {}
    config_manager._config["entity_extraction"]["llm"]["use_dspy"] = True
    config_manager._config["entity_extraction"]["llm"]["model"] = "qwen2.5:7b"

    # Configure DSPy
    from iris_vector_rag.dspy_modules.entity_extraction_module import configure_dspy_for_ollama
    configure_dspy_for_ollama(model_name="qwen2.5:7b")

    return config_manager


def main():
    """Test batch extraction with JSON debugging."""
    logger.info("="*80)
    logger.info("Batch Extraction JSON Debug Test")
    logger.info("="*80)

    logger.info("\n🔧 Setting up DSPy...")
    setup_dspy()

    logger.info("🔧 Initializing BatchEntityExtractionModule...")
    module = BatchEntityExtractionModule()

    logger.info(f"\n🚀 Testing batch extraction with {len(SAMPLE_TICKETS)} tickets...")

    try:
        # Call batch extraction
        from iris_vector_rag.dspy_modules.entity_extraction_module import configure_dspy_for_ollama
        import dspy as dspy_module

        # Ensure DSPy is configured
        if dspy_module.settings.lm is None:
            configure_dspy_for_ollama(model_name="qwen2.5:7b")

        # Prepare tickets
        batch_input = json.dumps([
            {"ticket_id": t["id"], "text": t["text"]}
            for t in SAMPLE_TICKETS
        ])

        logger.info("Calling DSPy ChainOfThought...")
        prediction = module.extract(
            tickets_batch=batch_input,
            entity_types="PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION"
        )

        # Show RAW output from LLM
        logger.info("\n" + "="*80)
        logger.info("RAW LLM OUTPUT:")
        logger.info("="*80)
        print(prediction.batch_results)
        logger.info("="*80)

        # Try to parse it
        logger.info("\nAttempting JSON parse...")
        try:
            data = json.loads(prediction.batch_results)
            logger.info(f"✅ SUCCESS! Parsed {len(data)} results")
            for result in data:
                ticket_id = result.get("ticket_id", "UNKNOWN")
                num_entities = len(result.get("entities", []))
                logger.info(f"  {ticket_id}: {num_entities} entities")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON PARSE FAILED: {e}")
            logger.error(f"   Error at line {e.lineno} column {e.colno}")
            logger.error(f"   Error position: {e.pos}")

            # Show context around error
            if e.pos:
                start = max(0, e.pos - 100)
                end = min(len(prediction.batch_results), e.pos + 100)
                context = prediction.batch_results[start:end]
                logger.error(f"\n   Context around error:\n   {context}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
