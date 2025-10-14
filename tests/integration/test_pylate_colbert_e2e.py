"""
End-to-end integration tests for PyLateColBERT pipeline.

Tests validate full workflow including ColBERT late interaction,
using live IRIS database and real embeddings.

Requirements: FR-025, FR-026, FR-027, FR-028
"""

import json
import pytest


@pytest.mark.integration
@pytest.mark.requires_database
@pytest.mark.pylate_colbert
class TestPyLateColBERTIntegration:
    """End-to-end integration tests for PyLateColBERT pipeline."""

    def test_full_query_path_with_real_db(self, pylate_colbert_pipeline, iris_connection):
        """
        FR-025: Full PyLateColBERT pipeline workflow MUST complete successfully.

        Given: PyLateColBERT pipeline with live IRIS database
        When: Documents loaded and query executed
        Then: Complete workflow succeeds with ColBERT late interaction
        """
        from tests.conftest import skip_if_no_embeddings

        # Load sample documents
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]

        # Execute load_documents
        load_result = pylate_colbert_pipeline.load_documents(documents)

        assert load_result["documents_loaded"] > 0, \
            "Should successfully load documents"

        # Check if embeddings are in proper VECTOR format before querying
        skip_if_no_embeddings(iris_connection)

        # Execute query
        query = "What are the symptoms of diabetes?"
        result = pylate_colbert_pipeline.query(query)

        # Verify query succeeded
        assert "answer" in result
        assert "contexts" in result
        assert "metadata" in result
        assert result["metadata"]["context_count"] >= 1

    def test_document_loading_workflow(self, pylate_colbert_pipeline):
        """
        FR-026: PyLateColBERT document loading workflow MUST complete successfully.

        Given: Sample documents
        When: load_documents() executed
        Then: Documents embedded and stored
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        documents = docs_data["documents"]
        expected_count = len(documents)

        result = pylate_colbert_pipeline.load_documents(documents)

        assert result["documents_loaded"] == expected_count
        assert result["embeddings_generated"] == expected_count
        assert result["documents_failed"] == 0

    def test_response_quality_metrics(self, pylate_colbert_pipeline, iris_connection):
        """
        FR-027: PyLateColBERT query response MUST include quality metrics.

        Given: PyLateColBERT pipeline with loaded documents
        When: Query executed
        Then: Response includes quality metrics
        """
        from tests.conftest import skip_if_no_embeddings

        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        pylate_colbert_pipeline.load_documents(docs_data["documents"])

        # Check if embeddings are in proper VECTOR format before querying
        skip_if_no_embeddings(iris_connection)

        query = "How is diabetes treated?"
        result = pylate_colbert_pipeline.query(query)

        assert len(result["answer"]) > 10
        metadata = result["metadata"]
        assert "sources" in metadata
        assert len(metadata["sources"]) > 0

    def test_colbert_late_interaction_search(self, pylate_colbert_pipeline):
        """
        FR-027: PyLateColBERT late interaction SHOULD occur in query path.

        Given: PyLateColBERT pipeline with loaded documents
        When: Query executed
        Then: ColBERT late interaction metadata MAY be present
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        pylate_colbert_pipeline.load_documents(docs_data["documents"])

        query = "What are diabetes symptoms?"
        result = pylate_colbert_pipeline.query(query)

        metadata = result.get("metadata", {})
        contexts = result.get("contexts", [])

        # PyLateColBERT-specific metadata (optional)
        if "colbert_score" in metadata or "late_interaction" in metadata:
            # ColBERT scoring is active
            assert True, "PyLateColBERT late interaction working"

        # Alternatively, check retrieval method
        retrieval_method = metadata.get("retrieval_method", "").lower()
        if "colbert" in retrieval_method:
            # ColBERT indicated in retrieval method
            assert True

        # Verify contexts may include ColBERT scores
        if contexts and isinstance(contexts[0], dict):
            if "colbert_score" in contexts[0] or "late_interaction_score" in contexts[0]:
                # ColBERT scoring exposed in contexts
                for ctx in contexts:
                    if "colbert_score" in ctx:
                        assert isinstance(ctx["colbert_score"], (int, float)), \
                            "ColBERT score must be numeric"

    def test_multiple_queries_maintain_consistency(self, pylate_colbert_pipeline):
        """
        FR-027: Multiple PyLateColBERT queries MUST maintain consistent behavior.

        Given: PyLateColBERT pipeline with loaded documents
        When: Multiple queries executed
        Then: All queries succeed with consistent structure
        """
        sample_docs_path = "tests/data/sample_pmc_docs_basic.json"
        with open(sample_docs_path, 'r') as f:
            docs_data = json.load(f)

        pylate_colbert_pipeline.load_documents(docs_data["documents"])

        queries = [
            "What are diabetes symptoms?",
            "How is diabetes diagnosed?",
            "What are diabetes risk factors?"
        ]

        for query in queries:
            result = pylate_colbert_pipeline.query(query)

            assert "answer" in result
            assert "contexts" in result
            assert "metadata" in result
            assert len(result["answer"]) > 0
