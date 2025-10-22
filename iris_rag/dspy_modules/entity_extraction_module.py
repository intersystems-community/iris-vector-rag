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


def register_custom_models():
    """
    DEPRECATED: No longer needed with DirectOpenAILM approach.

    Previously attempted to register custom models with LiteLLM, but LiteLLM
    fundamentally strips provider prefixes in OpenAI provider. Now using
    DirectOpenAILM class which bypasses LiteLLM entirely.
    """
    pass  # No-op for backward compatibility


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


class DirectOpenAILM(dspy.BaseLM):
    """
    Custom DSPy LM that makes direct HTTP requests to OpenAI-compatible endpoints.

    This bypasses LiteLLM entirely to preserve full model names (e.g., "openai/gpt-oss-120b")
    which is required for NVIDIA NIM endpoints that include provider prefix in model ID.

    Fixes Bug #6: LiteLLM strips provider prefix from model names.
    """

    def __init__(self, model: str, api_base: str, api_key: str, max_tokens: int = 2000, temperature: float = 0.1):
        """Initialize direct OpenAI-compatible LM."""
        super().__init__(model=model)  # Call BaseLM constructor

        self.model = model  # Full model name preserved (e.g., "openai/gpt-oss-120b")
        self.api_base = api_base.rstrip('/')  # Remove trailing slash
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Set provider to openai for DSPy compatibility
        self.provider = "openai"

        # DSPy requires kwargs attribute
        self.kwargs = {
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        logger.info(f"DirectOpenAILM initialized: model={model}, api_base={api_base}")

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Make a direct HTTP request to OpenAI-compatible endpoint.

        This method is called by DSPy when executing predictions.
        """
        import requests
        import json

        # Build messages array
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]

        # Build request payload
        payload = {
            "model": self.model,  # ✅ Full model name preserved!
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add response_format if requested (though NIM doesn't enforce it)
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]

        # Make direct HTTP request
        try:
            endpoint = f"{self.api_base}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            logger.debug(f"DirectOpenAILM request: POST {endpoint} with model={self.model}")

            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()

            # Extract response content
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")

                logger.debug(f"DirectOpenAILM response (first 200 chars): {content[:200]}")

                return [content]  # DSPy expects list of responses
            else:
                logger.error(f"Unexpected response format: {result}")
                return [""]

        except requests.exceptions.RequestException as e:
            logger.error(f"DirectOpenAILM request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise

    def basic_request(self, prompt: str, **kwargs):
        """Basic request interface for DSPy compatibility."""
        return self(prompt=prompt, **kwargs)


def configure_dspy(llm_config: dict):
    """
    Configure DSPy to use any LLM provider (Ollama, OpenAI-compatible, etc.).

    Respects configuration flags like supports_response_format and use_json_mode
    to ensure compatibility with various LLM endpoints.

    Args:
        llm_config: LLM configuration dict containing model, api_base, api_type, etc.
    """
    try:
        import dspy

        model = llm_config.get("model", "qwen2.5:7b")
        api_base = llm_config.get("api_base", "http://localhost:11434")
        api_key = llm_config.get("api_key", "dummy")  # Extract API key from config
        api_type = llm_config.get("api_type", "ollama")
        max_tokens = llm_config.get("max_tokens", 2000)
        temperature = llm_config.get("temperature", 0.1)

        # FIX for Bug #6: Use custom DirectOpenAILM for GPT-OSS NIM
        # This bypasses LiteLLM entirely to preserve full model name "openai/gpt-oss-120b"
        if model == "openai/gpt-oss-120b" or (api_type == "openai" and "/" in model):
            logger.info(f"🔧 Using DirectOpenAILM to preserve full model name: {model}")
            lm = DirectOpenAILM(
                model=model,  # Full name preserved: "openai/gpt-oss-120b"
                api_base=api_base,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature
            )
            logger.info(f"✅ DSPy configured with DirectOpenAILM: {model}")

        # Configure based on API type
        elif api_type == "openai":
            # Standard OpenAI-compatible endpoint (without prefix stripping issue)
            lm = dspy.LM(
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.info(f"✅ DSPy configured with OpenAI-compatible model: {model}")

        else:
            # Ollama or other provider
            try:
                # Modern DSPy 2.5+ API with ollama/ prefix
                lm = dspy.LM(
                    model=f"ollama/{model}",
                    api_base=api_base,
                    api_key=api_key,  # Pass API key (may be unused by Ollama)
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                logger.info(f"✅ DSPy configured with Ollama model: {model}")
            except Exception as e:
                logger.warning(f"dspy.LM failed: {e}, trying fallback...")
                # Fallback: try direct Ollama integration
                from dspy import OLlama
                lm = OLlama(
                    model=model,
                    base_url=api_base,
                    max_tokens=max_tokens,
                    temperature=temperature
                    # Note: OLlama() doesn't accept api_key parameter
                )
                logger.info(f"✅ DSPy configured with Ollama model: {model} (fallback)")

        dspy.configure(lm=lm)

    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        raise


def configure_dspy_for_ollama(model_name: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
    """
    Configure DSPy to use Ollama for LLM inference (legacy function).

    Deprecated: Use configure_dspy() with llm_config dict instead.

    Args:
        model_name: Ollama model name (default: qwen2.5:7b - fast and accurate)
        base_url: Ollama API base URL
    """
    # Call the new generic function
    llm_config = {
        "model": model_name,
        "api_base": base_url,
        "api_type": "ollama",
        "max_tokens": 2000,
        "temperature": 0.1
    }
    configure_dspy(llm_config)
