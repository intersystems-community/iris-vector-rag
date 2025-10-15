"""
OPTIMIZED: Batch Entity Extraction with DSPy.

Process multiple tickets in a single LLM call for 3-5x speedup.
"""
import dspy
import logging
import json
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class BatchEntityExtractionSignature(dspy.Signature):
    """Extract entities from MULTIPLE tickets in one LLM call."""

    tickets_batch = dspy.InputField(
        desc="JSON array of tickets. Each has: ticket_id, text. Extract entities for ALL tickets."
    )
    entity_types = dspy.InputField(
        desc="PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION"
    )

    batch_results = dspy.OutputField(
        desc="""JSON array of extraction results. One per ticket. Each result MUST have:
- ticket_id: The ticket ID
- entities: Array of {text, type, confidence} - AT LEAST 4 entities
- relationships: Array of {source, target, type, confidence} - AT LEAST 2 relationships

Example: [
  {
    "ticket_id": "I123456",
    "entities": [{"text": "TrakCare", "type": "PRODUCT", "confidence": 0.95}, ...],
    "relationships": [{"source": "user", "target": "TrakCare", "type": "accesses", "confidence": 0.9}]
  }
]"""
    )


class BatchEntityExtractionModule(dspy.Module):
    """Process 5-10 tickets per LLM call for massive speedup."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(BatchEntityExtractionSignature)
        logger.info("Initialized BATCH Entity Extraction Module (5-10 tickets/call)")

    def forward(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract entities from a batch of tickets.

        Args:
            tickets: List of dicts with 'id' and 'text' keys

        Returns:
            List of extraction results (one per ticket)
        """
        try:
            # Prepare batch input
            batch_input = json.dumps([
                {"ticket_id": t["id"], "text": t["text"]}
                for t in tickets
            ])

            # Single LLM call for entire batch
            prediction = self.extract(
                tickets_batch=batch_input,
                entity_types="PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION"
            )

            # Parse batch results
            results = json.loads(prediction.batch_results)

            logger.info(f"✅ Batch extracted {len(tickets)} tickets in ONE LLM call")
            return results

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            # Return empty results for all tickets
            return [
                {"ticket_id": t["id"], "entities": [], "relationships": []}
                for t in tickets
            ]
