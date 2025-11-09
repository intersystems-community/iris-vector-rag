"""
Integration test for entity extraction logging (Bug Fix v0.3.3).

This test verifies that EntityExtractionService produces comprehensive
INFO-level logging during batch entity extraction, addressing the issue
where HippoRAG2 users saw zero logging output during 75+ minute runs.

Related: https://github.com/tdyar/hipporag2-pipeline/BUG_REPORT_IRIS_VECTOR_RAG.md
"""
import logging
import pytest
from unittest.mock import MagicMock, patch
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document
from iris_rag.services.entity_extraction import EntityExtractionService


@pytest.fixture
def config_with_openai():
    """Create config manager with OpenAI LLM configuration."""
    config_manager = ConfigurationManager()
    config_manager._config = {
        "llm": {
            "provider": "openai",
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_base": "https://api.openai.com/v1",
            "api_key": "test-key",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
        "entity_extraction": {
            "method": "llm_basic",
            "confidence_threshold": 0.7,
            "entity_types": ["PRODUCT", "USER", "MODULE", "ERROR", "ACTION"],
            "batch_processing": {
                "enabled": True,
            },
        },
    }
    return config_manager


@pytest.fixture
def sample_documents():
    """Create sample documents for extraction."""
    return [
        Document(
            id="doc1",
            page_content="User cannot access TrakCare appointment module. Error: AUTH_FAILED.",
            metadata={"source": "ticket-001"},
        ),
        Document(
            id="doc2",
            page_content="IRIS database connection timeout in Lab module. Version 2024.1.",
            metadata={"source": "ticket-002"},
        ),
        Document(
            id="doc3",
            page_content="Admin user needs to configure HealthShare module permissions.",
            metadata={"source": "ticket-003"},
        ),
    ]


def test_entity_extraction_service_logs_llm_config(config_with_openai, caplog):
    """
    Test that EntityExtractionService logs LLM configuration on initialization.

    Expected output (INFO level):
    ======================================================================
    🤖 Entity Extraction Service - LLM Configuration
    ======================================================================
      Provider:    openai
      API Type:    openai
      Model:       gpt-4o-mini
      API Base:    https://api.openai.com/v1
      Method:      llm_basic
    ======================================================================
    """
    with caplog.at_level(logging.INFO):
        service = EntityExtractionService(
            config_manager=config_with_openai,
            connection_manager=None,
            embedding_manager=None,
        )

    # Verify LLM configuration banner appears
    log_text = caplog.text
    assert "🤖 Entity Extraction Service - LLM Configuration" in log_text
    assert "Provider:    openai" in log_text
    assert "Model:       gpt-4o-mini" in log_text
    assert "API Base:    https://api.openai.com/v1" in log_text
    assert "Method:      llm_basic" in log_text


def test_batch_extraction_logs_progress(
    config_with_openai, sample_documents, caplog
):
    """
    Test that extract_batch_with_dspy logs batch processing progress.

    Expected output (INFO level):
    📦 Processing batch of 3 documents...
    ✅ Batch complete: 3 documents → 9 entities in 2.3s (avg: 3.0 entities/doc, 0.8s/doc)
    """
    service = EntityExtractionService(
        config_manager=config_with_openai,
        connection_manager=None,
        embedding_manager=None,
    )

    # Set up the _batch_dspy_module attribute (normally created lazily)
    service._batch_dspy_module = MagicMock()

    # Mock the batch DSPy module to avoid actual LLM calls
    with patch.object(service, '_batch_dspy_module', service._batch_dspy_module) as mock_module:
        # Mock batch extraction results
        mock_module.forward.return_value = [
            {
                "ticket_id": "doc1",
                "entities": [
                    {"text": "TrakCare", "type": "PRODUCT", "confidence": 0.95},
                    {"text": "appointment module", "type": "MODULE", "confidence": 0.90},
                    {"text": "User", "type": "USER", "confidence": 0.85},
                ],
            },
            {
                "ticket_id": "doc2",
                "entities": [
                    {"text": "IRIS", "type": "PRODUCT", "confidence": 0.95},
                    {"text": "Lab module", "type": "MODULE", "confidence": 0.90},
                    {"text": "Version 2024.1", "type": "VERSION", "confidence": 0.80},
                ],
            },
            {
                "ticket_id": "doc3",
                "entities": [
                    {"text": "Admin", "type": "USER", "confidence": 0.90},
                    {"text": "HealthShare", "type": "PRODUCT", "confidence": 0.95},
                    {"text": "permissions", "type": "ACTION", "confidence": 0.75},
                ],
            },
        ]

        with caplog.at_level(logging.INFO):
            result = service.extract_batch_with_dspy(sample_documents, batch_size=5)

    # Verify batch start logging
    log_text = caplog.text
    assert "📦 Processing batch of 3 documents..." in log_text

    # Verify batch completion logging with timing and entity counts
    assert "✅ Batch complete:" in log_text
    assert "3 documents" in log_text
    assert "9 entities" in log_text
    assert "entities/doc" in log_text
    assert "s/doc" in log_text

    # Verify results structure
    assert len(result) == 3
    assert "doc1" in result
    assert "doc2" in result
    assert "doc3" in result
    assert len(result["doc1"]) == 3  # 3 entities for doc1


def test_fallback_individual_extraction_logs_progress(
    config_with_openai, sample_documents, caplog
):
    """
    Test that individual extraction (when batch disabled) logs progress.

    Expected output (INFO level):
    ⚠️  Batch processing disabled - falling back to individual extraction
    Processing 3 documents individually...
    ✅ Individual processing complete: 3 documents → 9 entities in 5.2s
    """
    # Disable batch processing
    config_with_openai._config["entity_extraction"]["batch_processing"]["enabled"] = False

    service = EntityExtractionService(
        config_manager=config_with_openai,
        connection_manager=None,
        embedding_manager=None,
    )

    # Mock process_document to avoid actual extraction
    with patch.object(service, 'process_document') as mock_process:
        mock_process.return_value = {
            "stored": True,
            "entities": [
                {"text": "entity1", "type": "PRODUCT"},
                {"text": "entity2", "type": "USER"},
                {"text": "entity3", "type": "MODULE"},
            ],
        }

        with caplog.at_level(logging.INFO):
            result = service.extract_batch_with_dspy(sample_documents, batch_size=5)

    # Verify fallback logging
    log_text = caplog.text
    assert "⚠️  Batch processing disabled - falling back to individual extraction" in log_text
    assert "Processing 3 documents individually..." in log_text
    assert "✅ Individual processing complete:" in log_text
    assert "3 documents" in log_text
    assert "entities" in log_text


def test_no_silent_failures_llm_config_warning(caplog):
    """
    Test that missing LLM config produces clear warning (not silent failure).

    This addresses the bug where users saw zero logging and had no idea
    entity extraction was failing silently.
    """
    # Create config without LLM section
    config = ConfigurationManager()
    config._config = {
        "entity_extraction": {
            "method": "llm_basic",
        }
    }

    with caplog.at_level(logging.WARNING):
        service = EntityExtractionService(
            config_manager=config,
            connection_manager=None,
            embedding_manager=None,
        )

    # Should warn about missing LLM config
    log_text = caplog.text
    assert "⚠️  No LLM configuration found" in log_text or "⚠️  No LLM config" in log_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
