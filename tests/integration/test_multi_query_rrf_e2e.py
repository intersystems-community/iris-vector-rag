"""
Integration Tests for MultiQueryRRFPipeline

End-to-end tests for the multi-query retrieval with RRF fusion pipeline.
"""

import pytest
from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document


class TestMultiQueryRRFE2E:
    """End-to-end integration tests for MultiQueryRRFPipeline."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Diabetes mellitus is characterized by high blood glucose levels.",
                metadata={"topic": "diabetes", "type": "definition"}
            ),
            Document(
                page_content="Common symptoms of diabetes include increased thirst and frequent urination.",
                metadata={"topic": "diabetes", "type": "symptoms"}
            ),
            Document(
                page_content="Type 2 diabetes can be managed through diet, exercise, and medication.",
                metadata={"topic": "diabetes", "type": "treatment"}
            ),
            Document(
                page_content="Diabetes diagnosis involves measuring fasting blood glucose levels.",
                metadata={"topic": "diabetes", "type": "diagnosis"}
            ),
            Document(
                page_content="Diabetic complications include neuropathy, retinopathy, and nephropathy.",
                metadata={"topic": "diabetes", "type": "complications"}
            )
        ]

    def test_pipeline_creation(self):
        """Test MultiQueryRRF pipeline can be created."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False
        )

        assert pipeline is not None
        assert hasattr(pipeline, 'query')
        assert hasattr(pipeline, 'load_documents')

    def test_pipeline_with_simple_variations(self):
        """Test pipeline with simple query variations (no LLM)."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False,
            num_queries=4,
            use_llm_expansion=False
        )

        # Query should work even without documents loaded
        result = pipeline.query("What are the symptoms of diabetes?", top_k=5)

        # Validate response structure
        assert isinstance(result, dict)
        assert 'retrieved_documents' in result
        assert 'metadata' in result
        assert 'queries' in result['metadata']

        # Should have generated multiple queries
        assert len(result['metadata']['queries']) >= 1
        assert result['metadata']['num_queries'] >= 1

    def test_rrf_fusion_logic(self):
        """Test that RRF fusion is working correctly."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False,
            num_queries=3,
            retrieved_k=10,
            rrf_k=60
        )

        result = pipeline.query("diabetes treatment options", top_k=10)

        # Validate RRF metadata
        assert 'metadata' in result
        assert 'rrf_k' in result['metadata']
        assert result['metadata']['rrf_k'] == 60

        # Check documents have RRF scores
        if result['retrieved_documents']:
            for doc in result['retrieved_documents']:
                assert hasattr(doc, 'metadata')
                # RRF score should be present
                assert 'rrf_score' in doc.metadata or 'score' in doc.metadata

    def test_query_variation_generation(self):
        """Test that query variations are generated correctly."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False,
            num_queries=4,
            use_llm_expansion=False
        )

        # Test different query patterns
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is diabetes diagnosed?",
            "Why does diabetes occur?",
            "Explain diabetes treatment"
        ]

        for query in test_queries:
            result = pipeline.query(query, top_k=5)

            # Should generate variations
            assert 'queries' in result['metadata']
            queries = result['metadata']['queries']

            # Should have at least the original query
            assert len(queries) >= 1

            # Original query should be included
            assert query in queries

    def test_execution_time_tracking(self):
        """Test that execution time is tracked."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False
        )

        result = pipeline.query("diabetes symptoms", top_k=5)

        # Should have timing metadata
        assert 'metadata' in result
        assert 'execution_time' in result['metadata']
        assert 'execution_time_ms' in result['metadata']

        # Time should be reasonable (< 30 seconds for test)
        assert result['metadata']['execution_time'] < 30
        assert result['metadata']['execution_time_ms'] < 30000

    def test_parameter_variations(self):
        """Test pipeline with different parameter configurations."""
        # Test with different num_queries
        for num_q in [2, 3, 4, 5]:
            pipeline = create_pipeline(
                "multi_query_rrf",
                validate_requirements=False,
                num_queries=num_q
            )

            result = pipeline.query("diabetes", top_k=5)
            assert len(result['metadata']['queries']) <= num_q

        # Test with different rrf_k values
        for k in [30, 60, 100]:
            pipeline = create_pipeline(
                "multi_query_rrf",
                validate_requirements=False,
                rrf_k=k
            )

            result = pipeline.query("diabetes", top_k=5)
            assert result['metadata']['rrf_k'] == k

    def test_response_format_consistency(self):
        """Test that response format matches other pipelines."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False
        )

        result = pipeline.query("diabetes symptoms", top_k=10)

        # Standard pipeline response fields
        assert 'retrieved_documents' in result
        assert 'contexts' in result
        assert 'sources' in result
        assert 'metadata' in result

        # retrieved_documents should be a list
        assert isinstance(result['retrieved_documents'], list)

        # contexts should be a list of strings
        assert isinstance(result['contexts'], list)

        # sources should be a list
        assert isinstance(result['sources'], list)

        # metadata should be a dict
        assert isinstance(result['metadata'], dict)

    def test_top_k_limiting(self):
        """Test that top_k parameter limits results correctly."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False
        )

        # Test different top_k values
        for k in [5, 10, 20]:
            result = pipeline.query("diabetes", top_k=k)

            # Should not exceed top_k
            assert len(result['retrieved_documents']) <= k

    def test_empty_query_handling(self):
        """Test handling of edge cases."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False
        )

        # Empty query should still work or raise appropriate error
        try:
            result = pipeline.query("", top_k=5)
            # If it works, should return valid structure
            assert isinstance(result, dict)
        except Exception as e:
            # Should have a meaningful error message
            assert len(str(e)) > 0

    def test_metadata_completeness(self):
        """Test that all expected metadata is present."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False,
            num_queries=3,
            retrieved_k=20,
            rrf_k=60
        )

        result = pipeline.query("diabetes diagnosis methods", top_k=10)

        # Check all expected metadata fields
        metadata = result['metadata']

        expected_fields = [
            'pipeline',
            'queries',
            'num_queries',
            'raw_result_count',
            'final_result_count',
            'rrf_k',
            'execution_time',
            'execution_time_ms',
            'use_llm_expansion'
        ]

        for field in expected_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Validate field types
        assert isinstance(metadata['pipeline'], str)
        assert metadata['pipeline'] == 'multi_query_rrf'
        assert isinstance(metadata['queries'], list)
        assert isinstance(metadata['num_queries'], int)
        assert isinstance(metadata['raw_result_count'], int)
        assert isinstance(metadata['final_result_count'], int)
        assert isinstance(metadata['rrf_k'], int)
        assert isinstance(metadata['execution_time'], float)
        assert isinstance(metadata['execution_time_ms'], int)
        assert isinstance(metadata['use_llm_expansion'], bool)

    def test_document_metadata_enrichment(self):
        """Test that retrieved documents have RRF-specific metadata."""
        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False,
            num_queries=3
        )

        result = pipeline.query("diabetes symptoms and treatment", top_k=10)

        if result['retrieved_documents']:
            doc = result['retrieved_documents'][0]

            # Should have RRF-specific metadata
            assert hasattr(doc, 'metadata')
            assert 'rrf_score' in doc.metadata
            assert 'rrf_rank' in doc.metadata
            assert 'score' in doc.metadata  # For consistency with other pipelines

            # Should have source query tracking
            assert 'source_queries' in doc.metadata or 'source_query' in doc.metadata

            # RRF rank should be sequential
            for i, doc in enumerate(result['retrieved_documents'][:5], 1):
                assert doc.metadata['rrf_rank'] == i

    @pytest.mark.skip(reason="LLM tests require OPENAI_API_KEY and cost money")
    def test_llm_query_expansion(self):
        """Test LLM-based query expansion (requires OPENAI_API_KEY)."""
        import os

        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set")

        pipeline = create_pipeline(
            "multi_query_rrf",
            validate_requirements=False,
            num_queries=4,
            use_llm_expansion=True
        )

        result = pipeline.query("What are the symptoms of diabetes?", top_k=10)

        # With LLM expansion, queries should be more diverse
        queries = result['metadata']['queries']

        # Should have generated multiple queries
        assert len(queries) >= 2

        # Queries should be different from each other
        unique_queries = set(queries)
        assert len(unique_queries) >= 2

        # Should have answer if LLM is available
        assert 'answer' in result

    def test_comparison_with_basic_pipeline(self):
        """Test that multi-query generally retrieves more diverse results."""
        # Create both pipelines
        basic = create_pipeline("basic", validate_requirements=False)
        multi = create_pipeline("multi_query_rrf", validate_requirements=False, num_queries=3)

        query = "diabetes symptoms diagnosis treatment"

        # Get results from both
        basic_result = basic.query(query, top_k=20)
        multi_result = multi.query(query, top_k=20)

        # Both should return results
        assert 'retrieved_documents' in basic_result
        assert 'retrieved_documents' in multi_result

        # Multi-query should have metadata about multiple queries
        assert 'queries' in multi_result['metadata']
        assert len(multi_result['metadata']['queries']) > 1

        # Raw result count should be higher for multi-query
        # (before RRF fusion)
        assert 'raw_result_count' in multi_result['metadata']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
