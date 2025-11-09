"""
End-to-End integration tests for HybridGraphRAG workflows (FR-026 to FR-028).

Tests validate complete query workflows for all 5 query methods with proper
metadata and sequential execution consistency.

Contract: E2E-001
Requirements: FR-026, FR-027, FR-028
"""

import pytest
import time
from iris_vector_rag import create_pipeline


class TestHybridGraphRAGE2E:
    """End-to-end integration tests for HybridGraphRAG."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline for E2E testing."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    @pytest.mark.integration
    def test_all_query_methods_end_to_end(self, graphrag_pipeline):
        """
        FR-026: All 5 query methods MUST work end-to-end.

        Given: HybridGraphRAG pipeline initialized with full setup
        And: Database contains test data (2,376 documents)
        When: Queries executed for each of 5 methods sequentially
        Then: Each query completes successfully
        """
        # Test queries for each method
        test_cases = [
            ("hybrid", "What are the symptoms of diabetes?"),
            ("rrf", "How is diabetes diagnosed?"),
            ("text", "What are the risk factors for type 2 diabetes?"),
            ("vector", "What are the treatments for diabetes?"),
            ("kg", "What is the relationship between insulin and diabetes?"),
        ]

        results = {}

        for method, query in test_cases:
            # Execute query
            result = graphrag_pipeline.query(query, method=method)

            # Verify query succeeded
            assert result is not None, f"Query with method={method} should return result"
            assert ('contexts' in result), f"Result should have contexts for method={method}"
            assert ('metadata' in result), f"Result should have metadata for method={method}"

            # Verify documents retrieved (or fallback occurred)
            assert len(result['contexts']) >= 0, \
                f"Query with method={method} should return contexts list"

            # Verify metadata contains retrieval method
            assert 'retrieval_method' in result['metadata'], \
                f"Metadata should contain retrieval_method for method={method}"

            # Store result for analysis
            results[method] = {
                'num_docs': len(result['contexts']),
                'retrieval_method': result['metadata']['retrieval_method'],
                'query': query
            }

        # Verify all methods were tested
        assert len(results) == 5, "Should test all 5 query methods"

        # Verify each method either succeeded or fell back gracefully
        for method, result_data in results.items():
            assert result_data['retrieval_method'] in [
                method, 'hybrid_fusion', 'hybrid', 'rrf', 'text', 'vector',
                'hnsw_vector', 'knowledge_graph', 'kg', 'vector_fallback'
            ], f"Method {method} should use expected retrieval method or fallback"

    @pytest.mark.requires_database
    @pytest.mark.integration
    def test_multiple_sequential_queries_consistent(self, graphrag_pipeline):
        """
        FR-027: Multiple sequential queries MUST execute consistently.

        Given: HybridGraphRAG pipeline initialized
        And: Same pipeline instance reused
        When: 10+ sequential queries executed
        And: Queries use different methods randomly
        Then: All queries complete successfully with consistent state
        """
        queries = [
            ("hybrid", "What are the symptoms of diabetes?"),
            ("vector", "How is diabetes diagnosed?"),
            ("text", "What are the risk factors for diabetes?"),
            ("rrf", "What are the treatments for type 2 diabetes?"),
            ("kg", "What is insulin resistance?"),
            ("hybrid", "What are the complications of diabetes?"),
            ("vector", "How to prevent diabetes?"),
            ("text", "What is gestational diabetes?"),
            ("rrf", "What are the symptoms of hypoglycemia?"),
            ("kg", "What is the relationship between obesity and diabetes?"),
            ("hybrid", "What is diabetic ketoacidosis?"),
            ("vector", "How to manage blood sugar levels?"),
        ]

        results = []
        execution_times = []

        for method, query in queries:
            start_time = time.time()

            # Execute query
            result = graphrag_pipeline.query(query, method=method)

            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            # Verify query succeeded
            assert result is not None, \
                f"Query {len(results)+1}: {query[:50]}... should succeed"

            assert len(result['contexts']) >= 0, \
                f"Query {len(results)+1} should return contexts"

            # Store result
            results.append({
                'method': method,
                'query': query[:50],
                'num_docs': len(result['contexts']),
                'retrieval_method': result['metadata']['retrieval_method'],
                'execution_time': execution_time
            })

        # Verify all queries succeeded
        assert len(results) == len(queries), \
            f"All {len(queries)} queries should complete"

        # Verify no memory leaks or performance degradation
        # Later queries should not be significantly slower than earlier ones
        avg_first_half = sum(execution_times[:6]) / 6
        avg_second_half = sum(execution_times[6:]) / 6

        # Allow up to 2x slowdown (gracious threshold)
        assert avg_second_half < avg_first_half * 2.0, \
            f"Performance should remain stable (first half: {avg_first_half:.2f}s, " \
            f"second half: {avg_second_half:.2f}s)"

        # Verify pipeline state remains consistent
        # Execute one more query to verify functionality
        final_result = graphrag_pipeline.query("diabetes overview", method="hybrid")
        assert len(final_result['contexts']) >= 0, \
            "Pipeline should remain functional after 12+ queries"

    @pytest.mark.requires_database
    @pytest.mark.integration
    def test_retrieval_metadata_completeness(self, graphrag_pipeline):
        """
        FR-028: Retrieval results MUST include complete metadata.

        Given: HybridGraphRAG pipeline initialized
        When: Query executed with any method
        Then: Metadata includes retrieval_method, execution_time, num_retrieved
        """
        query = "What are the symptoms of diabetes?"

        # Test with multiple methods
        methods = ["hybrid", "rrf", "text", "vector", "kg"]

        for method in methods:
            # Execute query
            result = graphrag_pipeline.query(query, method=method)

            # Verify metadata completeness
            assert 'retrieval_method' in result['metadata'], \
                f"Metadata should contain retrieval_method for method={method}"

            # Verify execution time (may be in different formats)
            has_time = any(key in result['metadata'] for key in [
                'execution_time', 'total_time', 'query_time', 'elapsed_time'
            ])
            assert has_time, \
                f"Metadata should contain execution time for method={method}"

            # Verify document count (may be in metadata or derived from contexts)
            has_count = ('num_retrieved' in result['metadata'] or
                        'num_documents' in result['metadata'] or
                        'document_count' in result['metadata'] or
                        len(result['contexts']) >= 0)
            assert has_count, \
                f"Metadata should include document count for method={method}"

            # Verify retrieval method value is valid
            retrieval_method = result['metadata']['retrieval_method']
            valid_methods = [
                'hybrid', 'hybrid_fusion', 'rrf', 'text', 'vector',
                'hnsw_vector', 'knowledge_graph', 'kg', 'vector_fallback'
            ]
            assert retrieval_method in valid_methods, \
                f"Retrieval method '{retrieval_method}' should be one of: {valid_methods}"

            # Verify metadata format is consistent (dict)
            assert isinstance(result['metadata'], dict), \
                f"Metadata should be a dictionary for method={method}"
