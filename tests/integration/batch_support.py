"""Real services for the batch extraction integration tests.

The LLM is the real OpenAI model named below, reached through the project's own
`get_llm_func` with the `OPENAI_API_KEY` of the environment; nothing here fakes it.
A test about retry or failure injects its fault at the service boundary and keeps
everything else real.
"""

from __future__ import annotations

from typing import List, Optional

from iris_vector_rag.common.utils import get_llm_func
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.services.entity_extraction import EntityExtractionService

LLM_MODEL = "gpt-4o-mini"

#: The entity types of the support-ticket documents these tests extract from.
TICKET_ENTITY_TYPES = [
    "PRODUCT",
    "ERROR",
    "MODULE",
    "USER",
    "VERSION",
    "ACTION",
    "ORGANIZATION",
]


def real_service(entity_types: Optional[List[str]] = None) -> EntityExtractionService:
    """An `EntityExtractionService` over the real LLM (and the default config)."""
    config_manager = ConfigurationManager()
    config_manager._config["entity_extraction"]["entity_types"] = list(
        entity_types or TICKET_ENTITY_TYPES
    )
    connection_manager = ConnectionManager(config_manager)
    return EntityExtractionService(
        config_manager,
        connection_manager,
        llm_func=get_llm_func(provider="openai", model_name=LLM_MODEL),
    )
