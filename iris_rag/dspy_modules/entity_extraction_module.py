"""
DSPy-powered Entity Extraction for TrakCare Support Tickets.

This module provides optimized entity and relationship extraction using DSPy
with TrakCare-specific entity types and domain knowledge.
"""
import dspy
import logging
import json
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class EntityExtractionSignature(dspy.Signature):
    """
    Extract structured entities and relationships from TrakCare support tickets.

    Focuses on high-quality extraction with 4+ entities per ticket including:
    - Products (TrakCare, IRIS, Cache, HealthShare)
    - Users (role names, user types, access levels)
    - Modules (Appointment, Lab, Patient, Pharmacy, etc.)
    - Errors (error codes, error messages, exceptions)
    - Actions (login, access, configure, activate, etc.)
    """

    ticket_text = dspy.InputField(
        desc="TrakCare support ticket text (summary + description + resolution)"
    )
    entity_types = dspy.InputField(
        desc="Comma-separated list of entity types to extract: PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION"
    )

    entities = dspy.OutputField(
        desc="""List of extracted entities as JSON array. Each entity MUST have:
- text: The exact entity text from ticket
- type: One of PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION
- confidence: 0.0-1.0 confidence score

Example: [{"text": "TrakCare", "type": "PRODUCT", "confidence": 0.95}, {"text": "appointment module", "type": "MODULE", "confidence": 0.90}]

CRITICAL: Extract AT LEAST 3-5 entities per ticket. Look for products, modules, error messages, user roles, and actions."""
    )

    relationships = dspy.OutputField(
        desc="""List of relationships between entities as JSON array. Each relationship MUST have:
- source: Entity text (from entities list)
- target: Entity text (from entities list)
- type: Relationship type (uses, has_error, affects, configures, accesses, belongs_to)
- confidence: 0.0-1.0 confidence score

Example: [{"source": "user", "target": "TrakCare", "type": "accesses", "confidence": 0.90}]

CRITICAL: Extract AT LEAST 2-3 relationships per ticket showing how entities interact."""
    )


class TrakCareEntityExtractionModule(dspy.Module):
    """
    DSPy module for extracting entities and relationships from TrakCare tickets.

    Uses ChainOfThought reasoning to maximize entity extraction quality and
    ensure we get 4+ entities per ticket with proper relationships.
    """

    # TrakCare-specific entity types for domain-specific extraction
    TRAKCARE_ENTITY_TYPES = [
        "PRODUCT",        # TrakCare, IRIS, Cache, HealthShare, Ensemble
        "USER",           # user, admin, clinician, receptionist, nurse
        "MODULE",         # appointment, lab, patient, pharmacy, orders
        "ERROR",          # error code, exception, failure message
        "ACTION",         # login, access, configure, create, update, delete
        "ORGANIZATION",   # hospital name, department, facility
        "VERSION",        # software version numbers
    ]

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityExtractionSignature)
        logger.info("Initialized TrakCare Entity Extraction Module with DSPy Chain of Thought")

    def forward(self, ticket_text: str, entity_types: Optional[List[str]] = None) -> dspy.Prediction:
        """
        Extract entities and relationships from ticket text.

        Args:
            ticket_text: Support ticket content
            entity_types: Optional list of entity types to extract. Defaults to all TrakCare types.

        Returns:
            dspy.Prediction with 'entities' and 'relationships' fields
        """
        # Use provided entity types or default to TrakCare types
        if entity_types is None:
            entity_types = self.TRAKCARE_ENTITY_TYPES

        entity_types_str = ", ".join(entity_types)

        try:
            # Perform DSPy chain of thought extraction
            prediction = self.extract(
                ticket_text=ticket_text,
                entity_types=entity_types_str
            )

            # Parse JSON from DSPy output
            entities = self._parse_entities(prediction.entities)
            relationships = self._parse_relationships(prediction.relationships)

            # Validate extraction quality
            if len(entities) < 2:
                logger.warning(
                    f"Low entity count ({len(entities)}) - DSPy should extract 4+ entities. "
                    f"Consider retraining or adjusting prompt."
                )

            # Create validated prediction
            validated_prediction = dspy.Prediction(
                entities=json.dumps(entities),
                relationships=json.dumps(relationships),
                entity_count=len(entities),
                relationship_count=len(relationships)
            )

            logger.info(
                f"Extracted {len(entities)} entities and {len(relationships)} relationships via DSPy"
            )

            return validated_prediction

        except Exception as e:
            logger.error(f"DSPy entity extraction failed: {e}")
            # Return empty extraction on failure
            return dspy.Prediction(
                entities="[]",
                relationships="[]",
                entity_count=0,
                relationship_count=0,
                error=str(e)
            )

    def _parse_entities(self, entities_str: str) -> List[Dict[str, Any]]:
        """Parse entities from DSPy JSON output with validation."""
        try:
            # Try to parse as JSON
            entities = json.loads(entities_str)

            # Validate structure
            validated_entities = []
            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                # Ensure required fields
                if "text" not in entity or "type" not in entity:
                    continue

                # Ensure confidence field
                if "confidence" not in entity:
                    entity["confidence"] = 0.8  # Default confidence

                # Validate entity type
                if entity["type"] not in self.TRAKCARE_ENTITY_TYPES:
                    logger.debug(f"Unknown entity type: {entity['type']}, keeping anyway")

                validated_entities.append(entity)

            return validated_entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entities JSON: {e}. Raw output: {entities_str[:200]}")
            # Try to extract entities using regex as fallback
            return self._fallback_entity_extraction(entities_str)

    def _parse_relationships(self, relationships_str: str) -> List[Dict[str, Any]]:
        """Parse relationships from DSPy JSON output with validation."""
        try:
            # Try to parse as JSON
            relationships = json.loads(relationships_str)

            # Validate structure
            validated_relationships = []
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue

                # Ensure required fields
                if "source" not in rel or "target" not in rel or "type" not in rel:
                    continue

                # Ensure confidence field
                if "confidence" not in rel:
                    rel["confidence"] = 0.7  # Default confidence for relationships

                validated_relationships.append(rel)

            return validated_relationships

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationships JSON: {e}. Raw output: {relationships_str[:200]}")
            return []  # No fallback for relationships - require proper JSON

    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback entity extraction using regex patterns when DSPy JSON parsing fails.
        This should rarely be needed if DSPy is properly configured.
        """
        import re
        entities = []

        # Extract TrakCare product names
        products = re.findall(r'\b(TrakCare|IRIS|Cache|HealthShare|Ensemble)\b', text, re.IGNORECASE)
        for product in set(products):
            entities.append({
                "text": product,
                "type": "PRODUCT",
                "confidence": 0.9
            })

        # Extract common modules
        modules = re.findall(
            r'\b(appointment|lab|patient|pharmacy|orders|admission|discharge|clinical)\b\s*module',
            text,
            re.IGNORECASE
        )
        for module in set(modules):
            entities.append({
                "text": module,
                "type": "MODULE",
                "confidence": 0.8
            })

        # Extract error patterns
        errors = re.findall(r'error\s*[:\-]\s*([^.]+)', text, re.IGNORECASE)
        for error in errors[:3]:  # Limit to first 3 errors
            entities.append({
                "text": error.strip(),
                "type": "ERROR",
                "confidence": 0.7
            })

        logger.info(f"Fallback extraction produced {len(entities)} entities")
        return entities


def configure_dspy_for_ollama(model_name: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
    """
    Configure DSPy to use Ollama for LLM inference.

    Args:
        model_name: Ollama model name (default: qwen2.5:7b - fast and accurate)
        base_url: Ollama API base URL
    """
    try:
        import dspy

        # Configure DSPy with Ollama using the correct API for DSPy 2.6.27
        # Try dspy.LM first (modern API), fallback to older APIs if needed
        try:
            # Modern DSPy 2.5+ API with ollama/ prefix
            ollama_lm = dspy.LM(
                model=f"ollama/{model_name}",
                api_base=base_url,
                max_tokens=2000,
                temperature=0.1
            )
            logger.info(f"Using dspy.LM with ollama/{model_name}")
        except Exception as e:
            logger.warning(f"dspy.LM failed: {e}, trying fallback...")
            # Fallback: try direct Ollama integration
            from dspy import OLlama  # Note: Capital O, then L
            ollama_lm = OLlama(
                model=model_name,
                base_url=base_url,
                max_tokens=2000,
                temperature=0.1
            )
            logger.info(f"Using dspy.OLlama with {model_name}")

        dspy.configure(lm=ollama_lm)
        logger.info(f"✅ DSPy configured with Ollama model: {model_name}")

    except Exception as e:
        logger.error(f"Failed to configure DSPy with Ollama: {e}")
        raise
