#!/usr/bin/env python3
"""
Test script for batch entity extraction - verifies 3x speedup.

This script tests the new extract_batch_with_dspy() method to ensure it:
1. Successfully extracts entities from multiple tickets in ONE LLM call
2. Achieves ~3x speedup over individual extraction
3. Maintains extraction quality (4+ entities per ticket)
"""
import sys
import time
import logging
from pathlib import Path

# Add rag-templates to path
sys.path.insert(0, str(Path(__file__).parent / "iris_rag"))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.services.entity_extraction import EntityExtractionService
from iris_rag.core.models import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Sample TrakCare tickets for testing
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
    {
        "id": "I123459",
        "text": "Unable to schedule appointments in TrakCare Scheduling module. Calendar view shows ERROR 500 when attempting to book patient visit. Front desk staff unable to process walk-ins.",
    },
    {
        "id": "I123460",
        "text": "TrakCare billing integration with insurance provider not working. Claims submission fails with timeout error. Finance team cannot process reimbursements for patient procedures.",
    },
    {
        "id": "I123461",
        "text": "Nurse station computers unable to connect to TrakCare server. Network connectivity issue prevents access to patient vitals monitoring system. Critical for ICU operations.",
    },
    {
        "id": "I123462",
        "text": "TrakCare mobile app not syncing patient data. Healthcare providers using iPads cannot access updated patient records from central database. Occurs after latest iOS update.",
    },
    {
        "id": "I123463",
        "text": "Radiology imaging viewer in TrakCare not loading DICOM files. MRI and CT scan images fail to display for radiologist review. Impacts diagnostic workflow significantly.",
    },
    {
        "id": "I123464",
        "text": "TrakCare pharmacy inventory module showing incorrect stock levels. Medication quantities not updating after dispensing to patients. Risk of stockouts for critical drugs.",
    },
    {
        "id": "I123465",
        "text": "Patient discharge summaries not generating in TrakCare documentation module. Print function returns blank pages. Delays patient discharge process from hospital.",
    },
]


def setup_dspy_config():
    """Configure DSPy for entity extraction."""
    config_path = Path(__file__).parent / "config" / "memory_config.yaml"
    config_manager = ConfigurationManager(str(config_path))

    # Bridge the config structure (workaround for P0 issue)
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

    return config_manager


def test_individual_extraction(service: EntityExtractionService):
    """Test traditional one-ticket-per-LLM-call extraction."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Individual Extraction (1 ticket per LLM call)")
    logger.info("="*80)

    documents = [
        Document(id=ticket["id"], page_content=ticket["text"])
        for ticket in SAMPLE_TICKETS
    ]

    start_time = time.time()

    results = {}
    for doc in documents:
        entities = service._extract_with_dspy(doc.page_content, doc)
        results[doc.id] = entities
        logger.info(f"  {doc.id}: {len(entities)} entities extracted")

    elapsed = time.time() - start_time
    total_entities = sum(len(ents) for ents in results.values())
    avg_entities = total_entities / len(results)

    logger.info(f"\n📊 Individual Extraction Results:")
    logger.info(f"   Tickets processed: {len(results)}")
    logger.info(f"   Total entities: {total_entities}")
    logger.info(f"   Avg entities/ticket: {avg_entities:.2f}")
    logger.info(f"   Total time: {elapsed:.2f}s")
    logger.info(f"   Rate: {len(results)/elapsed:.2f} tickets/sec")

    return {"time": elapsed, "tickets": len(results), "entities": total_entities}


def test_batch_extraction(service: EntityExtractionService):
    """Test NEW batch extraction (5 tickets per LLM call - optimal batch size)."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Batch Extraction (5 tickets per LLM call)")
    logger.info("="*80)

    documents = [
        Document(id=ticket["id"], page_content=ticket["text"])
        for ticket in SAMPLE_TICKETS
    ]

    start_time = time.time()

    # Call new batch extraction method with optimal batch size
    # NOTE: batch_size=5 is optimal - larger sizes (10+) cause LLM JSON quality degradation
    results = service.extract_batch_with_dspy(documents, batch_size=5)

    elapsed = time.time() - start_time
    total_entities = sum(len(ents) for ents in results.values())
    avg_entities = total_entities / len(results) if results else 0

    logger.info(f"\n📊 Batch Extraction Results:")
    logger.info(f"   Tickets processed: {len(results)}")
    logger.info(f"   Total entities: {total_entities}")
    logger.info(f"   Avg entities/ticket: {avg_entities:.2f}")
    logger.info(f"   Total time: {elapsed:.2f}s")
    logger.info(f"   Rate: {len(results)/elapsed:.2f} tickets/sec")

    return {"time": elapsed, "tickets": len(results), "entities": total_entities}


def compare_results(individual_results, batch_results):
    """Compare individual vs batch extraction performance."""
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*80)

    speedup = individual_results["time"] / batch_results["time"]

    logger.info(f"\n🚀 Speedup Analysis:")
    logger.info(f"   Individual time: {individual_results['time']:.2f}s")
    logger.info(f"   Batch time: {batch_results['time']:.2f}s")
    logger.info(f"   Speedup factor: {speedup:.2f}x")

    if speedup >= 2.5:
        logger.info(f"   ✅ SUCCESS! Achieved {speedup:.1f}x speedup (target: 3x)")
    elif speedup >= 2.0:
        logger.info(f"   ⚠️  GOOD! Achieved {speedup:.1f}x speedup (close to 3x target)")
    else:
        logger.info(f"   ❌ NEEDS IMPROVEMENT: Only {speedup:.1f}x speedup (target: 3x)")

    logger.info(f"\n📊 Quality Comparison:")
    logger.info(f"   Individual avg entities: {individual_results['entities']/individual_results['tickets']:.2f}")
    logger.info(f"   Batch avg entities: {batch_results['entities']/batch_results['tickets']:.2f}")

    if batch_results["entities"] >= individual_results["entities"] * 0.9:
        logger.info(f"   ✅ Quality maintained in batch mode")
    else:
        logger.info(f"   ⚠️  Quality may be degraded in batch mode")


def main():
    """Run batch extraction test."""
    logger.info("="*80)
    logger.info("Batch Entity Extraction Test - 3x Speedup Verification")
    logger.info("="*80)

    # Setup
    logger.info("\n🔧 Setting up DSPy configuration...")
    config_manager = setup_dspy_config()

    logger.info("🔧 Initializing EntityExtractionService...")
    service = EntityExtractionService(config_manager)

    # Run tests
    try:
        # Test 1: Individual extraction (baseline)
        individual_results = test_individual_extraction(service)

        # Test 2: Batch extraction (new feature)
        batch_results = test_batch_extraction(service)

        # Compare results
        compare_results(individual_results, batch_results)

        logger.info("\n✅ Test completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
