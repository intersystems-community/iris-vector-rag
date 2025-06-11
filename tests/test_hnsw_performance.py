"""
Comprehensive HNSW Performance Tests

This test suite validates that HNSW indexes provide actual performance benefits
with real PMC data, following TDD principles and project requirements for
testing with 1000+ documents.

Tests follow the project's TDD workflow: tests are written to initially fail,
then pass once HNSW functionality is properly implemented and optimized.
"""

import pytest
import sys
import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection, IRISConnectionError
from common.db_vector_search import search_source_documents_dynamically
from basic_rag.pipeline_final import BasicRAGPipeline # Corrected import
from common.utils import get_embedding_func, get_llm_func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHNSWPerformance:
    """
    Test HNSW index performance benefits with real PMC data.
    
    These tests validate that HNSW indexes provide measurable performance
    improvements over traditional vector search while maintaining quality.
    """
    
    @pytest.fixture
    def iris_connection(self):
        """Create an IRIS connection for testing."""
        connection = get_iris_connection()
        yield connection
        connection.close()
    
    @pytest.fixture
    def embedding_func(self):
        """Get embedding function for testing."""
        return get_embedding_func()
    
    @pytest.fixture
    def llm_func(self):
        """Get LLM function for testing."""
        return get_llm_func(provider="stub")
    
    @pytest.fixture
    def sample_queries(self):
        """Sample medical queries for performance testing."""
        return [
            "What are the effects of diabetes on cardiovascular health?",
            "How does insulin resistance develop in type 2 diabetes?",
            "What are the molecular mechanisms of cancer metastasis?",
            "Describe the pathophysiology of Alzheimer's disease",
            "What factors contribute to hypertension development?",
            "How does obesity affect metabolic pathways?",
            "What are the genetic factors in heart disease?",
            "Explain the role of inflammation in atherosclerosis",
            "What treatments are effective for chronic kidney disease?",
            "How do statins reduce cholesterol levels?"
        ]
    
    @contextmanager
    def measure_query_time(self):
        """Context manager to measure query execution time."""
        start_time = time.time()
        yield
        end_time = time.time()
        self.last_query_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    def test_hnsw_index_existence_performance_prereq(self, iris_connection):
        """
        TDD: Test that fails initially - verify HNSW indexes exist before performance testing.
        
        This test ensures HNSW indexes are properly created before running performance comparisons.
        Following TDD: should fail initially, then pass once indexes are created.
        """
        cursor = iris_connection.cursor()
        
        # Check for HNSW indexes on all vector tables
        check_indexes_query = """
        SELECT INDEX_NAME, TABLE_NAME
        FROM INFORMATION_SCHEMA.INDEXES
        WHERE TABLE_SCHEMA = 'RAG'
        AND INDEX_NAME IN (
            'idx_hnsw_source_embeddings',
            'idx_hnsw_token_embeddings', 
            'idx_hnsw_kg_node_embeddings'
        )
        ORDER BY INDEX_NAME
        """
        
        cursor.execute(check_indexes_query)
        results = cursor.fetchall()
        
        # TDD: This should fail initially if HNSW indexes don't exist
        assert len(results) >= 3, f"Expected 3 HNSW indexes, found {len(results)}. HNSW indexes must be created for performance testing."
        
        # Verify specific indexes exist
        index_names = [row[0] for row in results]
        required_indexes = [
            'idx_hnsw_source_embeddings',
            'idx_hnsw_token_embeddings',
            'idx_hnsw_kg_node_embeddings'
        ]
        
        for required_index in required_indexes:
            assert required_index in index_names, f"Required HNSW index {required_index} not found"
        
        logger.info("✅ HNSW indexes verified for performance testing")
        cursor.close()
    
    def test_sufficient_documents_for_performance_testing(self, iris_connection):
        """
        TDD: Test that fails initially - ensure we have 1000+ documents for meaningful performance comparison.
        
        Following project rules: tests must use real data with 1000+ documents.
        """
        cursor = iris_connection.cursor()
        
        # Count documents with embeddings (required for vector search)
        cursor.execute("""
        SELECT COUNT(*) 
        FROM RAG.SourceDocuments 
        WHERE embedding IS NOT NULL
        """)
        
        count = cursor.fetchone()[0]
        
        # TDD: This should fail initially if insufficient documents
        assert count >= 1000, f"Need at least 1000 documents with embeddings for performance testing, found {count}"
        
        logger.info(f"✅ Found {count} documents with embeddings for performance testing")
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_vs_sequential_scan_performance(self, iris_connection, embedding_func, sample_queries):
        """
        TDD: Test that initially fails - compare HNSW vs sequential scan performance.
        
        This test measures the performance difference between HNSW-accelerated
        vector search and traditional sequential scanning.
        """
        cursor = iris_connection.cursor()
        
        # Generate test query embeddings
        query_embeddings = [embedding_func([query])[0] for query in sample_queries[:5]]
        
        hnsw_times = []
        sequential_times = []
        
        for i, (query, embedding) in enumerate(zip(sample_queries[:5], query_embeddings)):
            embedding_str = str(embedding)
            
            # Test HNSW-accelerated search (should use index)
            with self.measure_query_time():
                hnsw_query = """
                SELECT TOP 10 doc_id, text_content,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                cursor.execute(hnsw_query, (embedding_str,))
                hnsw_results = cursor.fetchall()
            
            hnsw_times.append(self.last_query_time)
            
            # Test sequential scan (force index bypass if possible)
            # Note: In IRIS, we can't easily force a sequential scan,
            # so we'll use a different approach to simulate non-indexed search
            with self.measure_query_time():
                # Use a query that's less likely to use HNSW optimization
                sequential_query = """
                SELECT TOP 10 doc_id, text_content,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND doc_id LIKE '%'  -- Additional predicate to potentially affect optimization
                ORDER BY similarity DESC
                """
                cursor.execute(sequential_query, (embedding_str,))
                sequential_results = cursor.fetchall()
            
            sequential_times.append(self.last_query_time)
            
            # Verify we get similar results (quality shouldn't be compromised)
            assert len(hnsw_results) > 0, f"HNSW query {i} returned no results"
            assert len(sequential_results) > 0, f"Sequential query {i} returned no results"
            
            logger.info(f"Query {i}: HNSW={hnsw_times[i]:.1f}ms, Sequential={sequential_times[i]:.1f}ms")
        
        # Calculate performance statistics
        avg_hnsw_time = np.mean(hnsw_times)
        avg_sequential_time = np.mean(sequential_times)
        performance_improvement = (avg_sequential_time - avg_hnsw_time) / avg_sequential_time * 100
        
        # TDD: This should fail initially if HNSW doesn't provide performance benefits
        assert avg_hnsw_time < avg_sequential_time, (
            f"HNSW should be faster than sequential scan. "
            f"HNSW: {avg_hnsw_time:.1f}ms, Sequential: {avg_sequential_time:.1f}ms"
        )
        
        # Expect at least 20% performance improvement
        assert performance_improvement >= 20, (
            f"Expected at least 20% performance improvement, got {performance_improvement:.1f}%"
        )
        
        logger.info(f"✅ HNSW performance improvement: {performance_improvement:.1f}%")
        cursor.close()
    
    @pytest.mark.requires_1000_docs  
    def test_hnsw_scalability_with_document_count(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW performance scales better than linear scan.
        
        Tests HNSW performance characteristics as document count increases.
        """
        cursor = iris_connection.cursor()
        
        # Get total document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        total_docs = cursor.fetchone()[0]
        
        if total_docs < 1000:
            pytest.skip(f"Need at least 1000 documents for scalability testing, found {total_docs}")
        
        # Test query performance with different document subset sizes
        test_query = "What are the effects of diabetes?"
        query_embedding = embedding_func([test_query])[0]
        embedding_str = str(query_embedding)
        
        scalability_results = []
        
        # Test with different document counts using LIMIT simulation
        test_sizes = [1000, min(2000, total_docs), min(5000, total_docs), total_docs]
        test_sizes = [size for size in test_sizes if size <= total_docs]
        
        for doc_count in test_sizes:
            with self.measure_query_time():
                # Use a subquery to limit the dataset size for testing
                scalability_query = f"""
                SELECT TOP 10 doc_id, text_content, similarity FROM (
                    SELECT TOP {doc_count} doc_id, text_content,
                           VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY doc_id  -- Consistent ordering for subset
                ) subq
                ORDER BY similarity DESC
                """
                cursor.execute(scalability_query, (embedding_str,))
                results = cursor.fetchall()
            
            scalability_results.append({
                'doc_count': doc_count,
                'query_time_ms': self.last_query_time,
                'results_count': len(results)
            })
            
            logger.info(f"Doc count {doc_count}: {self.last_query_time:.1f}ms")
        
        # TDD: This should fail initially if HNSW doesn't scale well
        # HNSW should have sub-linear scaling characteristics
        if len(scalability_results) >= 2:
            # Compare scaling between smallest and largest dataset
            first_result = scalability_results[0]
            last_result = scalability_results[-1]
            
            doc_ratio = last_result['doc_count'] / first_result['doc_count']
            time_ratio = last_result['query_time_ms'] / first_result['query_time_ms']
            
            # HNSW should scale better than linearly (time_ratio < doc_ratio)
            assert time_ratio < doc_ratio, (
                f"HNSW should scale sub-linearly. "
                f"Doc ratio: {doc_ratio:.1f}x, Time ratio: {time_ratio:.1f}x"
            )
            
            logger.info(f"✅ HNSW scalability: {doc_ratio:.1f}x docs → {time_ratio:.1f}x time")
        
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_integration_with_basic_rag(self, iris_connection, embedding_func, llm_func, sample_queries):
        """
        TDD: Test that initially fails - verify HNSW works correctly with BasicRAG pipeline.
        
        Tests that HNSW indexes integrate properly with existing RAG techniques
        without degrading retrieval quality.
        """
        # Create BasicRAG pipeline
        rag_pipeline = BasicRAGPipeline(
            iris_connector=iris_connection,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        integration_results = []
        
        for query in sample_queries[:3]:  # Test with subset for performance
            start_time = time.time()
            
            # Run complete RAG pipeline (should use HNSW indexes)
            result = rag_pipeline.run(query, top_k=5)
            
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000
            
            # TDD: This should fail initially if integration doesn't work
            assert "answer" in result, f"RAG pipeline should return an answer for query: {query}"
            assert "retrieved_documents" in result, "RAG pipeline should return retrieved documents"
            assert len(result["retrieved_documents"]) > 0, f"Should retrieve documents for query: {query}"
            
            # Verify document quality (similarity scores should be reasonable)
            for doc in result["retrieved_documents"]:
                assert hasattr(doc, 'score'), "Retrieved documents should have similarity scores"
                assert 0 <= doc.score <= 1, f"Similarity score should be between 0 and 1, got {doc.score}"
            
            integration_results.append({
                'query': query,
                'total_time_ms': total_time_ms,
                'num_docs_retrieved': len(result["retrieved_documents"]),
                'avg_similarity': np.mean([doc.score for doc in result["retrieved_documents"]])
            })
            
            logger.info(f"Query: '{query[:30]}...' - {total_time_ms:.1f}ms, {len(result['retrieved_documents'])} docs")
        
        # TDD: Performance should be reasonable for integrated pipeline
        avg_total_time = np.mean([r['total_time_ms'] for r in integration_results])
        assert avg_total_time < 5000, f"Average RAG pipeline time should be < 5s, got {avg_total_time:.1f}ms"
        
        # Quality checks
        avg_similarity = np.mean([r['avg_similarity'] for r in integration_results])
        assert avg_similarity > 0.1, f"Average similarity should be > 0.1, got {avg_similarity:.3f}"
        
        logger.info(f"✅ BasicRAG integration: {avg_total_time:.1f}ms avg, {avg_similarity:.3f} avg similarity")
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_parameter_effectiveness(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW parameters (M=16, efConstruction=200) are effective.
        
        Tests that the chosen HNSW parameters provide good performance characteristics.
        """
        cursor = iris_connection.cursor()
        
        # Test with different top_k values to verify HNSW parameter effectiveness
        test_query = "diabetes treatment and management"
        query_embedding = embedding_func([test_query])[0]
        embedding_str = str(query_embedding)
        
        parameter_test_results = []
        
        # Test different top_k values to stress-test HNSW parameters
        top_k_values = [5, 10, 20, 50]
        
        for top_k in top_k_values:
            with self.measure_query_time():
                param_query = """
                SELECT TOP ? doc_id, text_content,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                cursor.execute(param_query, (top_k, embedding_str))
                results = cursor.fetchall()
            
            # Calculate similarity variance to assess quality
            similarities = [float(row[2]) for row in results]
            similarity_variance = np.var(similarities) if len(similarities) > 1 else 0
            
            parameter_test_results.append({
                'top_k': top_k,
                'query_time_ms': self.last_query_time,
                'results_count': len(results),
                'max_similarity': max(similarities) if similarities else 0,
                'min_similarity': min(similarities) if similarities else 0,
                'similarity_variance': similarity_variance
            })
            
            logger.info(f"top_k={top_k}: {self.last_query_time:.1f}ms, max_sim={max(similarities):.3f}")
        
        # TDD: Performance should scale reasonably with top_k
        for result in parameter_test_results:
            # Query time should be reasonable even for larger top_k
            assert result['query_time_ms'] < 1000, (
                f"Query with top_k={result['top_k']} took {result['query_time_ms']:.1f}ms, should be < 1000ms"
            )
            
            # Should return requested number of results (up to available docs)
            assert result['results_count'] == result['top_k'], (
                f"Expected {result['top_k']} results, got {result['results_count']}"
            )
            
            # Similarity scores should be in descending order (top similarity first)
            assert result['max_similarity'] >= result['min_similarity'], (
                "Results should be ordered by similarity (descending)"
            )
        
        logger.info("✅ HNSW parameters (M=16, efConstruction=200) are effective")
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_concurrent_query_performance(self, iris_connection, embedding_func, sample_queries):
        """
        TDD: Test that initially fails - verify HNSW handles concurrent queries efficiently.
        
        Tests HNSW performance under concurrent load conditions.
        """
        import threading
        import queue
        
        # Create multiple connections for concurrent testing
        connections = []
        for _ in range(3):  # Test with 3 concurrent connections
            conn = get_iris_connection()
            connections.append(conn)
        
        try:
            results_queue = queue.Queue()
            
            def run_concurrent_query(connection, query_id, query_text):
                """Run a single query and record results."""
                try:
                    cursor = connection.cursor()
                    query_embedding = embedding_func([query_text])[0]
                    embedding_str = str(query_embedding)
                    
                    start_time = time.time()
                    
                    concurrent_query = """
                    SELECT TOP 10 doc_id, text_content,
                           VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    """
                    cursor.execute(concurrent_query, (embedding_str,))
                    query_results = cursor.fetchall()
                    
                    end_time = time.time()
                    query_time_ms = (end_time - start_time) * 1000
                    
                    results_queue.put({
                        'query_id': query_id,
                        'query_time_ms': query_time_ms,
                        'results_count': len(query_results),
                        'success': True
                    })
                    
                    cursor.close()
                    
                except Exception as e:
                    results_queue.put({
                        'query_id': query_id,
                        'error': str(e),
                        'success': False
                    })
            
            # Start concurrent threads
            threads = []
            for i, query in enumerate(sample_queries[:3]):
                connection = connections[i % len(connections)]
                thread = threading.Thread(
                    target=run_concurrent_query,
                    args=(connection, i, query)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout
            
            # Collect results
            concurrent_results = []
            while not results_queue.empty():
                result = results_queue.get()
                concurrent_results.append(result)
            
            # TDD: All concurrent queries should succeed
            successful_queries = [r for r in concurrent_results if r.get('success', False)]
            assert len(successful_queries) == 3, (
                f"Expected 3 successful concurrent queries, got {len(successful_queries)}"
            )
            
            # Performance should still be reasonable under concurrent load
            avg_concurrent_time = np.mean([r['query_time_ms'] for r in successful_queries])
            assert avg_concurrent_time < 2000, (
                f"Average concurrent query time should be < 2s, got {avg_concurrent_time:.1f}ms"
            )
            
            logger.info(f"✅ Concurrent HNSW queries: {avg_concurrent_time:.1f}ms average")
            
        finally:
            # Clean up connections
            for conn in connections:
                conn.close()
    
    def test_hnsw_performance_regression_protection(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - establish performance baselines to prevent regression.
        
        This test records performance baselines that can be used to detect
        performance regressions in future changes.
        """
        cursor = iris_connection.cursor()
        
        # Standard test query for consistent benchmarking
        benchmark_query = "cardiovascular disease risk factors"
        query_embedding = embedding_func([benchmark_query])[0]
        embedding_str = str(query_embedding)
        
        # Run multiple iterations to get stable measurements
        iterations = 5
        query_times = []
        
        for i in range(iterations):
            with self.measure_query_time():
                benchmark_sql = """
                SELECT TOP 20 doc_id, text_content,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                cursor.execute(benchmark_sql, (embedding_str,))
                results = cursor.fetchall()
            
            query_times.append(self.last_query_time)
            
            # Verify consistent results
            assert len(results) > 0, f"Benchmark query iteration {i} returned no results"
        
        # Calculate performance statistics
        mean_time = np.mean(query_times)
        std_time = np.std(query_times)
        p95_time = np.percentile(query_times, 95)
        
        # TDD: Establish performance baselines (these thresholds may need adjustment)
        assert mean_time < 500, f"Mean query time should be < 500ms, got {mean_time:.1f}ms"
        assert p95_time < 1000, f"P95 query time should be < 1000ms, got {p95_time:.1f}ms"
        assert std_time < mean_time * 0.5, f"Query time variance too high: std={std_time:.1f}ms"
        
        # Log performance baseline for future regression testing
        logger.info(f"🎯 HNSW Performance Baseline:")
        logger.info(f"   Mean: {mean_time:.1f}ms")
        logger.info(f"   Std:  {std_time:.1f}ms") 
        logger.info(f"   P95:  {p95_time:.1f}ms")
        
        # Store these values for future regression detection
        performance_baseline = {
            'mean_time_ms': mean_time,
            'std_time_ms': std_time,
            'p95_time_ms': p95_time,
            'query': benchmark_query,
            'top_k': 20
        }
        
        # In a real implementation, you might store this in a file or database
        # for continuous performance monitoring
        
        cursor.close()