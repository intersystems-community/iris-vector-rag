r"""
OPTIMIZED: Batch Entity Extraction with DSPy.

Process multiple tickets in a single LLM call for 3-5x speedup.

Enhanced with JSON retry logic (T025) to fix the 0.7% JSON parsing failure rate
observed in production where LLMs generate invalid escape sequences like \N, \i, etc.
"""
import dspy
import logging
import json
from typing import List, Dict, Any, Optional

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
    """Process 5-10 tickets per LLM call for massive speedup with JSON retry logic."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(BatchEntityExtractionSignature)
        logger.info("Initialized BATCH Entity Extraction Module (5-10 tickets/call) with JSON retry logic")

    def _parse_json_with_retry(
        self, json_str: str, max_attempts: int = 3, context: str = "Batch JSON parsing"
    ) -> Optional[List[Dict[str, Any]]]:
        r"""
        Parse JSON with retry and repair logic for LLM-generated invalid escape sequences.

        Fixes the 0.7% JSON parsing failure rate observed in production where LLMs
        generate invalid escape sequences like \N, \i, etc.

        This method is copied from entity_extraction.py:_parse_json_with_retry()
        to provide consistent JSON parsing across all DSPy modules.

        Args:
            json_str: JSON string to parse
            max_attempts: Maximum number of parsing attempts with repair
            context: Context string for logging

        Returns:
            Parsed JSON data as list of dicts, or None if all attempts fail
        """
        for attempt in range(max_attempts):
            try:
                # Attempt to parse JSON
                data = json.loads(json_str)

                # Ensure it's a list
                if not isinstance(data, list):
                    logger.warning(f"{context}: Expected list, got {type(data)}. Wrapping in list.")
                    data = [data]

                if attempt > 0:
                    logger.info(f"{context}: Successfully parsed after {attempt} repair attempts")

                return data

            except json.JSONDecodeError as e:
                if attempt < max_attempts - 1:
                    # Try to repair common LLM JSON errors
                    logger.warning(
                        f"{context}: JSON parse failed on attempt {attempt + 1}/{max_attempts}: {e}"
                    )
                    logger.debug(f"Invalid JSON (first 200 chars): {json_str[:200]}")

                    # Apply repair strategies
                    original_str = json_str

                    # Strategy 1: Fix trailing commas (common LLM error)
                    json_str = json_str.replace(',]', ']').replace(',}', '}')

                    # Strategy 2: Fix invalid escape sequences
                    # Replace \N with \\N, \i with \\i, etc.
                    # But preserve valid escapes: \n, \t, \r, \", \\, \/, \b, \f
                    valid_escapes = {'n', 't', 'r', '"', '\\', '/', 'b', 'f', 'u'}

                    # Find all backslash sequences and fix invalid ones
                    repaired = []
                    i = 0
                    while i < len(json_str):
                        if json_str[i] == '\\' and i + 1 < len(json_str):
                            next_char = json_str[i + 1]
                            if next_char not in valid_escapes:
                                # Invalid escape - add extra backslash
                                repaired.append('\\\\')
                                repaired.append(next_char)
                                i += 2
                            else:
                                # Valid escape - keep as is
                                repaired.append('\\')
                                i += 1
                        else:
                            repaired.append(json_str[i])
                            i += 1

                    json_str = ''.join(repaired)

                    if json_str != original_str:
                        logger.debug(f"Applied JSON repair (attempt {attempt + 1})")
                    else:
                        logger.debug(f"No repair pattern matched (attempt {attempt + 1})")

                else:
                    # Final attempt failed
                    logger.error(
                        f"{context}: Failed to parse JSON after {max_attempts} attempts: {e}"
                    )
                    logger.debug(f"Final JSON (first 500 chars): {json_str[:500]}")
                    return None

        return None

    def forward(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract entities from a batch of tickets with JSON retry logic (T025).

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

            # Parse batch results with retry logic (T025)
            results = self._parse_json_with_retry(
                prediction.batch_results,
                max_attempts=3,
                context=f"Batch extraction ({len(tickets)} tickets)"
            )

            if results is None:
                logger.error(f"Failed to parse batch results after retry attempts")
                # Return empty results for all tickets
                return [
                    {"ticket_id": t["id"], "entities": [], "relationships": []}
                    for t in tickets
                ]

            # Add batch-level logging (T025 requirement)
            total_entities = sum(len(r.get("entities", [])) for r in results)
            total_relationships = sum(len(r.get("relationships", [])) for r in results)
            logger.info(
                f"✅ Batch extracted {len(tickets)} tickets in ONE LLM call: "
                f"{total_entities} entities, {total_relationships} relationships"
            )

            return results

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            # Return empty results for all tickets
            return [
                {"ticket_id": t["id"], "entities": [], "relationships": []}
                for t in tickets
            ]
