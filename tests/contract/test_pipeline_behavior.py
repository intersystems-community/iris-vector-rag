"""
Contract tests for HybridGraphRAG pipeline behavior.

These tests verify that the pipeline maintains consistent behavior
without fallback mechanisms.
"""

from unittest.mock import Mock, patch

import pytest


class TestPipelineBehavior:
    """Contract tests for pipeline behavior without fallbacks."""

    def test_pipeline_creation_requires_iris_vector_graph(self):
        """Creating HybridGraphRAG should fail if iris-vector-graph is missing."""
        # This test should FAIL until implementation is complete
        with patch(
            "iris_vector_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules"
        ) as mock_import:
            mock_import.side_effect = ImportError(
                "HybridGraphRAG requires iris-vector-graph package"
            )

            from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

            with pytest.raises(ImportError):
                HybridGraphRAGPipeline()

    def test_no_conditional_checks_in_pipeline(self):
        """Pipeline should not have conditional availability checks."""
        # This test should FAIL until implementation is complete
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        # Mock successful import
        with patch(
            "iris_vector_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules"
        ) as mock_import:
            mock_import.return_value = {
                "IRISGraphEngine": Mock(),
                "HybridSearchFusion": Mock(),
                "TextSearchEngine": Mock(),
                "VectorOptimizer": Mock(),
            }

            pipeline = HybridGraphRAGPipeline()

            # These attributes should always be initialized, never None
            assert pipeline.iris_engine is not None
            assert pipeline.retrieval_methods is not None

            # Status methods should always return True
            assert pipeline.is_hybrid_enabled() is True

            status = pipeline.get_hybrid_status()
            assert status["hybrid_enabled"] is True
            assert status["iris_engine_available"] is True
            assert status["fusion_engine_available"] is True

    @pytest.mark.parametrize(
        "method, leg",
        [
            ("hybrid", "_retrieve_via_hybrid_fusion"),
            ("rrf", "_retrieve_via_rrf"),
            ("text", "_retrieve_via_enhanced_text"),
            ("vector", "_retrieve_via_hnsw_vector"),
        ],
    )
    def test_each_retrieval_method_runs_its_own_leg(self, method, leg):
        """`method=` picks the iris_vector_graph leg of that name, not a fallback.

        This used to assert that each method was in the list it iterated over.
        """
        from iris_vector_rag.core.models import Document
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        with patch(
            "iris_vector_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules"
        ) as mock_import:
            mock_import.return_value = {
                "IRISGraphEngine": Mock(),
                "HybridSearchFusion": Mock(),
                "TextSearchEngine": Mock(),
                "VectorOptimizer": Mock(),
            }
            pipeline = HybridGraphRAGPipeline()

        found = [Document(id=f"{method}-1", page_content=method, metadata={})]
        with patch.object(pipeline, "_validate_knowledge_graph"), patch.object(
            pipeline, leg, return_value=(found, leg)
        ) as called, patch.object(pipeline, "_enhanced_hybrid_fallback") as fallback:
            result = pipeline.query("q", method=method, generate_answer=False)

        called.assert_called_once()
        fallback.assert_not_called()
        assert [d.id for d in result["retrieved_documents"]] == [f"{method}-1"]
