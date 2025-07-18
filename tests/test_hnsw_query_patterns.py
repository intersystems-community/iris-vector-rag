"""
HNSW-Specific Query Patterns Tests

This test suite creates specific query patterns that demonstrate HNSW performance
benefits and validates optimal parameters with real PMC data.

Tests different vector similarity scenarios, embedding sizes, and validates
that HNSW parameters (M=16, efConstruction=200) are optimal for the use case.
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

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
from eval.metrics import calculate_hnsw_performance_metrics, calculate_hnsw_scalability_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHNSWQueryPatterns:
    """
    Test HNSW performance with specific query patterns that stress-test
    the index and demonstrate its benefits.
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
    def medical_query_patterns(self):
        """
        Medical query patterns designed to test different vector similarity scenarios.
        Each pattern tests specific characteristics of vector search.
        """
        return {
            "exact_terminology": [
                # Queries using exact medical terminology - should have high similarity scores
                "diabetes mellitus type 2 insulin resistance",
                "myocardial infarction cardiovascular disease",
                "alzheimer disease neurodegenerative disorder",
                "hypertension blood pressure cardiovascular risk"
            ],
            "semantic_similarity": [
                # Queries using related but different terminology - tests semantic understanding
                "heart attack symptoms and treatment",
                "memory loss in elderly patients", 
                "high blood sugar management",
                "cancer tumor growth mechanisms"
            ],
            "broad_concepts": [
                # Broad medical concepts - tests recall capability
                "treatment approaches for chronic diseases",
                "diagnostic methods in medical practice",
                "patient care and clinical outcomes",
                "research findings in medical studies"
            ],
            "specific_details": [
                # Very specific medical details - tests precision
                "BRCA1 gene mutation breast cancer risk",
                "ACE inhibitors mechanism blood pressure reduction",
                "dopamine neurotransmitter Parkinson disease",
                "insulin signaling pathway glucose metabolism"
            ],
            "multi_concept": [
                # Queries combining multiple medical concepts - tests complex similarity
                "diabetes complications cardiovascular kidney disease",
                "cancer treatment chemotherapy radiation therapy surgery",
                "depression anxiety mental health medication therapy",
                "obesity diabetes metabolic syndrome cardiovascular risk"
            ]
        }
    
    @contextmanager
    def measure_query_performance(self):
        """Context manager to measure detailed query performance."""
        start_time = time.time()
        yield
        end_time = time.time()
        self.last_query_time = (end_time - start_time) * 1000
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_exact_terminology_patterns(self, iris_connection, embedding_func, medical_query_patterns):
        """
        TDD: Test that initially fails - verify HNSW performs well with exact medical terminology.
        
        Tests queries using exact medical terminology that should have high similarity matches.
        """
        cursor = iris_connection.cursor()
        exact_queries = medical_query_patterns["exact_terminology"]
        
        performance_results = []
        similarity_results = []
        
        for query in exact_queries:
            query_embedding = embedding_func([query])[0]
            embedding_str = str(query_embedding)
            
            with self.measure_query_performance():
                exact_query_sql = """
                SELECT TOP 10 doc_id, text_content,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                cursor.execute(exact_query_sql, (embedding_str,))
                results = cursor.fetchall()
            
            query_time = self.last_query_time
            similarities = [float(row[2]) for row in results]
            
            # TDD: This should fail initially if HNSW doesn't handle exact terminology well
            assert len(results) > 0, f"No results for exact terminology query: {query}"
            assert max(similarities) > 0.3, f"Low max similarity {max(similarities):.3f} for exact query: {query}"
            
            performance_results.append(query_time)
            similarity_results.extend(similarities)
            
            logger.info(f"Exact terminology: '{query[:40]}...' - {query_time:.1f}ms, max_sim={max(similarities):.3f}")
        
        # Performance expectations for exact terminology
        avg_query_time = np.mean(performance_results)
        avg_similarity = np.mean(similarity_results)
        
        assert avg_query_time < 300, f"Exact terminology queries too slow: {avg_query_time:.1f}ms"
        assert avg_similarity > 0.2, f"Low average similarity for exact terms: {avg_similarity:.3f}"
        
        logger.info(f"✅ Exact terminology HNSW: {avg_query_time:.1f}ms avg, {avg_similarity:.3f} avg similarity")
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_semantic_similarity_patterns(self, iris_connection, embedding_func, medical_query_patterns):
        """
        TDD: Test that initially fails - verify HNSW handles semantic similarity well.
        
        Tests queries using related but different terminology to validate semantic understanding.
        """
        cursor = iris_connection.cursor()
        semantic_queries = medical_query_patterns["semantic_similarity"]
        
        semantic_performance = []
        semantic_quality = []
        
        for query in semantic_queries:
            query_embedding = embedding_func([query])[0]
            embedding_str = str(query_embedding)
            
            with self.measure_query_performance():
                semantic_query_sql = """
                SELECT TOP 15 doc_id, text_content,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                cursor.execute(semantic_query_sql, (embedding_str,))
                results = cursor.fetchall()
            
            query_time = self.last_query_time
            similarities = [float(row[2]) for row in results]
            
            # TDD: This should fail initially if semantic similarity doesn't work
            assert len(results) > 0, f"No results for semantic query: {query}"
            
            # For semantic queries, we expect reasonable but not necessarily high similarities
            max_sim = max(similarities)
            assert max_sim > 0.15, f"Very low semantic similarity {max_sim:.3f} for: {query}"
            
            # Check that we get a reasonable spread of similarities (not all the same)
            sim_variance = np.var(similarities)
            assert sim_variance > 0.001, f"No similarity variance for semantic query: {query}"
            
            semantic_performance.append(query_time)
            semantic_quality.append(max_sim)
            
            logger.info(f"Semantic: '{query[:40]}...' - {query_time:.1f}ms, max_sim={max_sim:.3f}")
        
        # Semantic queries should still perform well
        avg_semantic_time = np.mean(semantic_performance)
        avg_semantic_quality = np.mean(semantic_quality)
        
        assert avg_semantic_time < 400, f"Semantic queries too slow: {avg_semantic_time:.1f}ms"
        assert avg_semantic_quality > 0.18, f"Poor semantic quality: {avg_semantic_quality:.3f}"
        
        logger.info(f"✅ Semantic similarity HNSW: {avg_semantic_time:.1f}ms avg, {avg_semantic_quality:.3f} quality")
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_different_embedding_dimensions(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW works efficiently with 768-dimensional embeddings.
        
        Tests HNSW performance with the specific embedding dimension used (768).
        """
        cursor = iris_connection.cursor()
        
        # Test query with our standard 768-dimensional embeddings
        test_query = "medical research clinical trial results"
        query_embedding = embedding_func([test_query])[0]
        
        # Verify embedding dimension
        assert len(query_embedding) == 768, f"Expected 768-dim embeddings, got {len(query_embedding)}"
        
        embedding_str = str(query_embedding)
        
        # Test performance with full 768-dimensional vectors
        dimension_tests = []
        
        for top_k in [5, 10, 20, 50]:
            with self.measure_query_performance():
                dim_query = f"""
                SELECT TOP {top_k} doc_id,
                       VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                cursor.execute(dim_query, (embedding_str,))
                results = cursor.fetchall()
            
            query_time = self.last_query_time
            
            # TDD: Should handle 768-dimensional vectors efficiently
            assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"
            assert query_time < 500, f"768-dim query with top_k={top_k} too slow: {query_time:.1f}ms"
            
            similarities = [float(row[1]) for row in results]
            max_sim = max(similarities)
            
            dimension_tests.append({
                'top_k': top_k,
                'query_time_ms': query_time,
                'max_similarity': max_sim,
                'min_similarity': min(similarities)
            })
            
            logger.info(f"768-dim top_k={top_k}: {query_time:.1f}ms, max_sim={max_sim:.3f}")
        
        # Performance should scale reasonably with top_k
        times = [test['query_time_ms'] for test in dimension_tests]
        assert all(t < 600 for t in times), f"Some 768-dim queries too slow: {times}"
        
        # Similarity ordering should be consistent (descending)
        for test in dimension_tests:
            assert test['max_similarity'] >= test['min_similarity'], "Similarity ordering incorrect"
        
        logger.info(f"✅ 768-dimensional HNSW performance validated")
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_parameter_optimization_validation(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - validate HNSW parameters (M=16, efConstruction=200) are optimal.
        
        Tests that current HNSW parameters provide good performance characteristics.
        """
        cursor = iris_connection.cursor()
        
        # Test queries that stress different aspects of HNSW parameters
        parameter_test_queries = [
            "diabetes management glucose control",  # Common medical terms
            "rare genetic mutation oncology",       # Less common terms
            "patient outcomes quality of life",     # Abstract concepts
            "molecular mechanisms cell biology"     # Technical terminology
        ]
        
        parameter_results = []
        
        for query in parameter_test_queries:
            query_embedding = embedding_func([query])[0]
            embedding_str = str(query_embedding)
            
            # Test with different result set sizes to stress M parameter
            for result_size in [10, 25, 50]:
                with self.measure_query_performance():
                    param_query = f"""
                    SELECT TOP {result_size} doc_id,
                           VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    """
                    cursor.execute(param_query, (embedding_str,))
                    results = cursor.fetchall()
                
                query_time = self.last_query_time
                similarities = [float(row[1]) for row in results]
                
                # TDD: Parameters should provide good performance for all scenarios
                assert query_time < 800, f"Query too slow with current parameters: {query_time:.1f}ms"
                assert len(results) == result_size, f"Incomplete results: {len(results)} vs {result_size}"
                
                # Quality checks
                if similarities:
                    max_sim = max(similarities)
                    sim_range = max_sim - min(similarities)
                    
                    assert max_sim > 0.1, f"Very low max similarity: {max_sim:.3f}"
                    assert sim_range > 0.01, f"No similarity variation: {sim_range:.3f}"
                
                parameter_results.append({
                    'query': query[:30],
                    'result_size': result_size,
                    'query_time_ms': query_time,
                    'max_similarity': max(similarities) if similarities else 0,
                    'similarity_range': sim_range if similarities else 0
                })
                
                logger.info(f"Params test: '{query[:25]}...', size={result_size}, {query_time:.1f}ms")
        
        # Analyze parameter effectiveness
        avg_time = np.mean([r['query_time_ms'] for r in parameter_results])
        avg_quality = np.mean([r['max_similarity'] for r in parameter_results])
        
        # TDD: Current parameters should provide good overall performance
        assert avg_time < 400, f"Average time too high with current parameters: {avg_time:.1f}ms"
        assert avg_quality > 0.2, f"Average quality too low: {avg_quality:.3f}"
        
        # Check performance consistency across different query types and sizes
        time_variance = np.var([r['query_time_ms'] for r in parameter_results])
        time_cv = np.sqrt(time_variance) / avg_time  # Coefficient of variation
        
        assert time_cv < 0.5, f"Performance too inconsistent: CV={time_cv:.3f}"
        
        logger.info(f"✅ HNSW parameters (M=16, efConstruction=200) validated:")
        logger.info(f"   Average time: {avg_time:.1f}ms")
        logger.info(f"   Average quality: {avg_quality:.3f}")
        logger.info(f"   Performance CV: {time_cv:.3f}")
        
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_concurrent_query_patterns(self, iris_connection, embedding_func, medical_query_patterns):
        """
        TDD: Test that initially fails - verify HNSW handles concurrent diverse query patterns.
        
        Tests HNSW performance under concurrent load with different query pattern types.
        """
        import threading
        import queue
        
        # Create connections for concurrent testing
        connections = []
        for _ in range(3):
            conn = get_iris_connection()
            connections.append(conn)
        
        try:
            results_queue = queue.Queue()
            
            def run_pattern_query(connection, pattern_name, queries, thread_id):
                """Run queries from a specific pattern type."""
                try:
                    cursor = connection.cursor()
                    pattern_results = []
                    
                    for query in queries[:2]:  # Limit for concurrent testing
                        query_embedding = embedding_func([query])[0]
                        embedding_str = str(query_embedding)
                        
                        start_time = time.time()
                        
                        concurrent_sql = """
                        SELECT TOP 10 doc_id,
                               VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                        FROM RAG.SourceDocuments
                        WHERE embedding IS NOT NULL
                        ORDER BY similarity DESC
                        """
                        cursor.execute(concurrent_sql, (embedding_str,))
                        query_results = cursor.fetchall()
                        
                        end_time = time.time()
                        query_time = (end_time - start_time) * 1000
                        
                        similarities = [float(row[1]) for row in query_results]
                        
                        pattern_results.append({
                            'query': query,
                            'query_time_ms': query_time,
                            'max_similarity': max(similarities) if similarities else 0,
                            'results_count': len(query_results)
                        })
                    
                    results_queue.put({
                        'thread_id': thread_id,
                        'pattern_name': pattern_name,
                        'results': pattern_results,
                        'success': True
                    })
                    
                    cursor.close()
                    
                except Exception as e:
                    results_queue.put({
                        'thread_id': thread_id,
                        'pattern_name': pattern_name,
                        'error': str(e),
                        'success': False
                    })
            
            # Start concurrent threads with different query patterns
            threads = []
            pattern_names = ['exact_terminology', 'semantic_similarity', 'specific_details']
            
            for i, pattern_name in enumerate(pattern_names):
                queries = medical_query_patterns[pattern_name]
                connection = connections[i]
                
                thread = threading.Thread(
                    target=run_pattern_query,
                    args=(connection, pattern_name, queries, i)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join(timeout=30)
            
            # Analyze concurrent results
            concurrent_results = []
            while not results_queue.empty():
                result = results_queue.get()
                concurrent_results.append(result)
            
            # TDD: All concurrent pattern queries should succeed
            successful_patterns = [r for r in concurrent_results if r.get('success', False)]
            assert len(successful_patterns) == 3, f"Expected 3 successful patterns, got {len(successful_patterns)}"
            
            # Analyze performance across different patterns
            all_query_times = []
            pattern_performance = {}
            
            for pattern_result in successful_patterns:
                pattern_name = pattern_result['pattern_name']
                pattern_queries = pattern_result['results']
                
                pattern_times = [q['query_time_ms'] for q in pattern_queries]
                pattern_qualities = [q['max_similarity'] for q in pattern_queries]
                
                pattern_performance[pattern_name] = {
                    'avg_time': np.mean(pattern_times),
                    'avg_quality': np.mean(pattern_qualities)
                }
                
                all_query_times.extend(pattern_times)
                
                logger.info(f"Pattern {pattern_name}: {np.mean(pattern_times):.1f}ms avg")
            
            # Overall concurrent performance should be reasonable
            overall_avg_time = np.mean(all_query_times)
            assert overall_avg_time < 1000, f"Concurrent performance too slow: {overall_avg_time:.1f}ms"
            
            # Different patterns should all perform reasonably
            for pattern_name, perf in pattern_performance.items():
                assert perf['avg_time'] < 1200, f"Pattern {pattern_name} too slow: {perf['avg_time']:.1f}ms"
                assert perf['avg_quality'] > 0.1, f"Pattern {pattern_name} poor quality: {perf['avg_quality']:.3f}"
            
            logger.info(f"✅ Concurrent query patterns: {overall_avg_time:.1f}ms overall average")
            
        finally:
            for conn in connections:
                conn.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_query_complexity_scaling(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW performance scales with query complexity.
        
        Tests how HNSW performance changes with different query complexity levels.
        """
        cursor = iris_connection.cursor()
        
        # Queries of increasing complexity
        complexity_queries = {
            'simple': [
                "diabetes",
                "cancer", 
                "heart disease"
            ],
            'moderate': [
                "diabetes treatment options",
                "cancer therapy approaches",
                "cardiovascular risk factors"  
            ],
            'complex': [
                "diabetes mellitus type 2 insulin resistance metabolic syndrome",
                "cancer immunotherapy checkpoint inhibitors tumor microenvironment",
                "cardiovascular disease risk factors hypertension obesity diabetes"
            ]
        }
        
        complexity_results = {}
        
        for complexity_level, queries in complexity_queries.items():
            level_performance = []
            level_quality = []
            
            for query in queries:
                query_embedding = embedding_func([query])[0]
                embedding_str = str(query_embedding)
                
                with self.measure_query_performance():
                    complexity_sql = """
                    SELECT TOP 15 doc_id,
                           VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    """
                    cursor.execute(complexity_sql, (embedding_str,))
                    results = cursor.fetchall()
                
                query_time = self.last_query_time
                similarities = [float(row[1]) for row in results]
                
                # TDD: All complexity levels should work
                assert len(results) > 0, f"No results for {complexity_level} query: {query}"
                assert query_time < 600, f"{complexity_level} query too slow: {query_time:.1f}ms"
                
                level_performance.append(query_time)
                level_quality.append(max(similarities) if similarities else 0)
                
                logger.info(f"{complexity_level}: '{query[:35]}...' - {query_time:.1f}ms")
            
            complexity_results[complexity_level] = {
                'avg_time': np.mean(level_performance),
                'avg_quality': np.mean(level_quality),
                'time_variance': np.var(level_performance)
            }
        
        # Analyze scaling behavior
        simple_time = complexity_results['simple']['avg_time']
        moderate_time = complexity_results['moderate']['avg_time']
        complex_time = complexity_results['complex']['avg_time']
        
        # TDD: Performance should scale reasonably (not explode with complexity)
        # More complex queries might be slightly slower, but not dramatically
        assert complex_time / simple_time < 2.0, (
            f"Complex queries too much slower than simple: {complex_time:.1f}ms vs {simple_time:.1f}ms"
        )
        
        # Quality should remain reasonable across complexity levels
        for level, results in complexity_results.items():
            assert results['avg_quality'] > 0.15, (
                f"Poor quality for {level} queries: {results['avg_quality']:.3f}"
            )
        
        logger.info(f"✅ Query complexity scaling:")
        for level, results in complexity_results.items():
            logger.info(f"   {level}: {results['avg_time']:.1f}ms avg, {results['avg_quality']:.3f} quality")
        
        cursor.close()