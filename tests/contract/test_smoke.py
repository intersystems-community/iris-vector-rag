"""
Smoke test: Minimal IRIS ingest+query validation for PR-integration gate.

Tests verify that real IRIS integration (ingest, embedding, retrieval) works end-to-end.
This test gates PRs to main via .github/workflows/pr-integration.yml.

Requirements: FR-004
Success Criteria: SC-004
"""

import pytest

from iris_vector_rag import create_validated_pipeline
from iris_vector_rag.core.models import Document


@pytest.mark.smoke
@pytest.mark.requires_database
class TestSmoke:
    """Smoke tests for IRIS integration."""

    def test_basic_ingest_and_query(self):
        """
        Smoke test: ingest one document, query it, assert non-empty retrieval.

        Given: No preconditions (fresh IRIS schema will be created)
        When: Pipeline loads 1 document and queries for it
        Then: Retrieved documents include the ingested content
        """
        # Create pipeline with validated factory (auto-setup)
        pipeline = create_validated_pipeline(
            pipeline_type="basic",
            auto_setup=True,
            validate_requirements=True,
        )

        # Ingest 1 document
        doc = Document(
            page_content="IRIS vector database stores embeddings for RAG.",
            metadata={"source": "smoke_test"},
        )
        load_result = pipeline.load_documents([doc])

        # Assert ingestion succeeded
        assert load_result["documents_loaded"] >= 1, "Smoke test: ingest failed"
        assert (
            load_result["documents_failed"] == 0
        ), "Smoke test: document ingestion should not have failures"

        # Query
        query_result = pipeline.query(
            "What does IRIS do?", top_k=1, generate_answer=False
        )

        # Assert retrieval succeeded
        assert query_result.get("error") is None, (
            "Smoke test: query returned error: "
            f"{query_result.get('error', 'unknown')}"
        )
        assert (
            len(query_result["retrieved_documents"]) >= 1
        ), "Smoke test: no docs retrieved"
        assert (
            len(query_result.get("contexts", [])) >= 1
        ), "Smoke test: no contexts returned"
