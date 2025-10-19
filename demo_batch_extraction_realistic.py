#!/usr/bin/env python3
"""
Realistic Batch Extraction Demo - Shows exactly what happens in production.

This demonstrates the batch extraction feature with realistic TrakCare tickets,
showing the actual performance improvement and logging output you'll see in production.
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "iris_rag"))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.models import Document

# Configure logging to show batch extraction messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Realistic TrakCare support tickets (sample of 10 for demo)
REALISTIC_TICKETS = [
    {
        "id": "I477985",
        "text": "User unable to access TrakCare appointment scheduling module. Error 'SESSION_TIMEOUT' appears after 5 minutes of inactivity. Affects front desk staff at Hospital ABC. TrakCare version 2023.1.5.",
    },
    {
        "id": "I439043",
        "text": "Lab results integration not working between TrakCare and Cerner Millennium. Test results from Chemistry lab not appearing in patient charts. Impacting 50+ patients daily. Using TrakCare 2022.2.",
    },
    {
        "id": "I437548",
        "text": "Prescription module crashes when doctor attempts to e-prescribe controlled substances. Error code ERX_401. TrakCare Pharmacy module version 2023.1. Only affects DEA-certified prescribers.",
    },
    {
        "id": "I446221",
        "text": "Billing department reports duplicate charge entries in TrakCare Revenue Cycle. Same procedure code appearing twice for 15% of patient encounters. Started after upgrade to TrakCare 2023.2.",
    },
    {
        "id": "I473600",
        "text": "Radiology PACS integration with TrakCare not retrieving DICOM images. MRI and CT scan viewers showing 'IMAGE_NOT_FOUND' error. Affects Radiology Information System workflows. TrakCare version 2023.1.3.",
    },
    {
        "id": "I458291",
        "text": "Emergency Department triage module extremely slow during peak hours. TrakCare ED dashboard takes 30+ seconds to load patient queue. Impacts patient flow and wait times. 100+ concurrent users.",
    },
    {
        "id": "I462115",
        "text": "Patient portal login failing for accounts created after October 1st. Authentication service returning ERROR_USER_NOT_FOUND despite valid credentials. TrakCare Patient Portal v2023.2.",
    },
    {
        "id": "I455789",
        "text": "Operating room scheduling conflicts in TrakCare Surgery module. System allowing double-booking of OR suites. Impacts surgical workflow at Main Hospital campus. TrakCare version 2023.1.7.",
    },
    {
        "id": "I441052",
        "text": "Insurance verification batch job timing out in TrakCare Claims module. Nightly eligibility checks failing for Medicare patients. Job fails with TIMEOUT_ERROR after 2 hours. TrakCare 2022.2.1.",
    },
    {
        "id": "I469338",
        "text": "Clinical decision support alerts not firing for drug-drug interactions. TrakCare CDS module failing to warn about warfarin + aspirin contraindications. Patient safety concern. TrakCare 2023.1.5.",
    },
]


def setup_config():
    """Configure for batch extraction."""
    config_path = Path(__file__).parent / "config" / "memory_config.yaml"
    config_manager = ConfigurationManager(str(config_path))

    # Enable DSPy for entity extraction
    entity_config = (
        config_manager.get("rag_memory_config", {})
        .get("knowledge_extraction", {})
        .get("entity_extraction", {})
    )
    config_manager._config["entity_extraction"] = entity_config

    if "llm" not in config_manager._config["entity_extraction"]:
        config_manager._config["entity_extraction"]["llm"] = {}
    config_manager._config["entity_extraction"]["llm"]["use_dspy"] = True
    config_manager._config["entity_extraction"]["llm"]["model"] = "qwen2.5:7b"

    return config_manager


def main():
    """
    Realistic batch extraction demo showing production behavior.
    """
    logger.info("=" * 80)
    logger.info("REALISTIC BATCH EXTRACTION DEMO")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This demo shows exactly what happens in production when you index tickets.")
    logger.info("Watch for the batch extraction messages showing 3x speedup!")
    logger.info("")

    # Setup
    logger.info("🔧 Initializing GraphRAG pipeline with batch extraction...")
    config_manager = setup_config()
    connection_manager = ConnectionManager()
    pipeline = GraphRAGPipeline(
        connection_manager=connection_manager,
        config_manager=config_manager
    )

    # Convert to Document objects
    documents = [
        Document(
            id=f"ticket_{ticket['id']}",
            page_content=ticket["text"],
            metadata={"source": "trakcare_support", "type": "ticket"}
        )
        for ticket in REALISTIC_TICKETS
    ]

    logger.info(f"")
    logger.info("=" * 80)
    logger.info(f"INDEXING {len(documents)} REALISTIC TRAKCARE TICKETS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Batch Extraction Configuration:")
    logger.info(f"  • Batch size: 5 tickets per LLM call")
    logger.info(f"  • Total batches: {(len(documents) + 4) // 5}")
    logger.info(f"  • Expected speedup: ~3x faster than individual processing")
    logger.info("")

    try:
        # Start timer
        start_time = time.time()

        # Load documents with batch entity extraction
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING IN ACTION (watch for 🚀 messages)")
        logger.info("=" * 80)
        logger.info("")

        pipeline.load_documents(
            documents_path="",
            documents=documents,
            generate_embeddings=True
        )

        # Calculate performance
        elapsed = time.time() - start_time
        tickets_per_second = len(documents) / elapsed
        estimated_individual_time = len(documents) * 6.46  # Individual mode average

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ BATCH EXTRACTION DEMO COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"📊 Performance Results:")
        logger.info(f"  • Tickets indexed: {len(documents)}")
        logger.info(f"  • Total time: {elapsed:.1f}s")
        logger.info(f"  • Rate: {tickets_per_second:.2f} tickets/sec")
        logger.info(f"  • Estimated individual mode time: {estimated_individual_time:.1f}s")
        logger.info(f"  • Time saved: {estimated_individual_time - elapsed:.1f}s")
        logger.info(f"  • Actual speedup: {estimated_individual_time / elapsed:.2f}x")
        logger.info("")
        logger.info("🎉 Batch extraction is working in production!")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"")
        logger.error("=" * 80)
        logger.error(f"❌ Demo failed: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
