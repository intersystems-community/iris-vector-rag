"""
Integration test for entity extraction logging (Bug Fix v0.3.3).

This test verifies that EntityExtractionService produces comprehensive
INFO-level logging during batch entity extraction, addressing the issue
where HippoRAG2 users saw zero logging output during 75+ minute runs.

Related: https://github.com/tdyar/hipporag2-pipeline/BUG_REPORT_IRIS_VECTOR_RAG.md
"""

import logging
import os

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.services.entity_extraction import EntityExtractionService


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
            "api_key": os.environ["OPENAI_API_KEY"],
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
        EntityExtractionService(
            config_manager=config_with_openai,
            connection_manager=ConnectionManager(config_with_openai),
        )

    # Verify LLM configuration banner appears
    log_text = caplog.text
    assert "🤖 Entity Extraction Service - LLM Configuration" in log_text
    assert "Provider:    openai" in log_text
    assert "Model:       gpt-4o-mini" in log_text
    assert "API Base:    https://api.openai.com/v1" in log_text
    assert "Method:      llm_basic" in log_text


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
        EntityExtractionService(
            config_manager=config,
            connection_manager=ConnectionManager(config),
        )

    # Should warn about missing LLM config
    log_text = caplog.text
    assert (
        "⚠️  No LLM configuration found" in log_text or "⚠️  No LLM config" in log_text
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
