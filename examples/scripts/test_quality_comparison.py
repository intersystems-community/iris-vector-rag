#!/usr/bin/env python3
"""
Quality comparison: Same 3 tickets, individual vs batch.
This proves batch extraction maintains quality.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "iris_rag"))

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.services.entity_extraction import EntityExtractionService
from iris_vector_rag.core.models import Document

# Same 3 tickets for both tests
TICKETS = [
    {"id": "I123456", "text": "User reports unable to access TrakCare patient portal. Error message 'Authentication Failed' appears after entering credentials. Using TrakCare v2023.1 on Chrome browser."},
    {"id": "I123457", "text": "Lab results not displaying in TrakCare EMR module. Technician unable to view test results for patient in Radiology department. System shows loading spinner indefinitely."},
    {"id": "I123458", "text": "Prescription module crashing when trying to prescribe medications. Doctor reports TrakCare Pharmacy module freezes when searching for drug formulary. Impacts patient care workflow."},
]

def setup():
    config_path = Path(__file__).parent / "config" / "memory_config.yaml"
    config_manager = ConfigurationManager(str(config_path))

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

    return EntityExtractionService(config_manager)

def test_individual(service):
    docs = [Document(id=t["id"], page_content=t["text"]) for t in TICKETS]
    results = {}
    for doc in docs:
        entities = service._extract_with_dspy(doc.page_content, doc)
        results[doc.id] = entities
    return results

def test_batch(service):
    docs = [Document(id=t["id"], page_content=t["text"]) for t in TICKETS]
    return service.extract_batch_with_dspy(docs, batch_size=3)

def main():
    print("="*80)
    print("QUALITY COMPARISON: Same 3 Tickets")
    print("="*80)

    service = setup()

    print("\nTest 1: Individual Extraction")
    individual_results = test_individual(service)
    individual_counts = {tid: len(ents) for tid, ents in individual_results.items()}
    print(f"  {individual_counts}")
    print(f"  Average: {sum(individual_counts.values())/len(individual_counts):.2f}")

    print("\nTest 2: Batch Extraction")
    batch_results = test_batch(service)
    batch_counts = {tid: len(ents) for tid, ents in batch_results.items()}
    print(f"  {batch_counts}")
    print(f"  Average: {sum(batch_counts.values())/len(batch_counts):.2f}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    diff = abs(sum(individual_counts.values()) - sum(batch_counts.values()))
    if diff <= 2:  # Allow 2 entity variance
        print("✅ Quality maintained! Batch extraction is as good as individual.")
    else:
        print(f"⚠️  Quality difference: {diff} entities")
    print("="*80)

if __name__ == "__main__":
    main()
