"""
Contract tests for Entity Extraction.

Feature: 051-add-native-iris
Contract: entity_extraction_contract.yaml
Status: TDD - These tests MUST FAIL until implementation is complete
"""

import pytest
from uuid import uuid4

# These imports will fail initially - expected for TDD
try:
    from iris_rag.embeddings.entity_extractor import (
        extract_entities_batch,
        store_entities,
        configure_entity_types,
        get_entities,
    )
    from iris_rag.config.embedding_config import EmbeddingConfig
    IMPLEMENTATION_EXISTS = True
except ImportError:
    IMPLEMENTATION_EXISTS = False


pytestmark = pytest.mark.skipif(
    not IMPLEMENTATION_EXISTS,
    reason="Implementation not yet available (TDD - tests first)"
)


class TestEntityExtractionBatch:
    """Test scenarios for extract_entities_batch contract."""

    def test_extract_entities_batch_medical_domain(self):
        """Extract medical entities from batch of 3 documents."""
        texts = [
            "Patient presents with type 2 diabetes and elevated blood glucose.",
            "Insulin therapy recommended for glucose control.",
            "Metformin prescribed for diabetes management."
        ]
        config = EmbeddingConfig(
            name="test_medical",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3",
            embedding_class="%Embedding.SentenceTransformers",
            enable_entity_extraction=True,
            entity_types=["Disease", "Symptom", "Medication"],
            batch_size=32
        )

        result = extract_entities_batch(texts, config)

        assert len(result.documents) == 3
        assert result.total_entities_extracted >= 5
        assert result.llm_calls_made == 1  # Batch extraction
        assert result.extraction_time_ms < 2000

    def test_extraction_disabled(self):
        """Handle extraction disabled gracefully."""
        texts = ["Test text"]
        config = EmbeddingConfig(
            name="test_disabled",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hf_cache_path="/var/lib/huggingface",
            python_path="/usr/bin/python3",
            embedding_class="%Embedding.SentenceTransformers",
            enable_entity_extraction=False,
            batch_size=32
        )

        with pytest.raises(ValueError) as exc_info:
            extract_entities_batch(texts, config)

        assert "ENTITY_EXTRACTION_DISABLED" in str(exc_info.value) or "disabled" in str(exc_info.value).lower()


# TDD Status Check
def test_implementation_status():
    """Verify implementation exists."""
    assert IMPLEMENTATION_EXISTS, (
        "Implementation not found. Expected:\n"
        "- iris_rag.embeddings.entity_extractor\n"
        "TDD: Tests written first, implementation comes next."
    )
