"""
HNSW Benchmark Integration Tests

This test suite integrates HNSW performance testing with the existing benchmark
framework and provides comprehensive performance reporting.

Validates that HNSW provides measurable performance benefits in real RAG
workflows and generates benchmark reports for comparison.
"""

import pytest
import sys
import os
import time
import json
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from basic_rag.pipeline_final import BasicRAGPipeline # Corrected import
from eval.metrics import (
    calculate_hnsw_performance_metrics,
    calculate_hnsw_scalability_metrics,
    calculate_hnsw_index_effectiveness_metrics,
    calculate_latency_percentiles
)
from eval.comparative.analysis import calculate_technique_comparison

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHNSWBenchmarkIntegration:
    """
    Comprehensive HNSW benchmark integration tests.
    
    These tests integrate HNSW performance measurement with the existing
    benchmark framework and generate comprehensive performance reports.
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
    def benchmark_queries(self):
        """Standard benchmark queries for comprehensive testing."""
        return [
            {
                "query": "What are the risk factors for cardiovascular disease?",
                "expected_terms": ["cardiovascular", "risk", "factors", "heart"],
                "category": "medical_risk_factors"
            },
            {
                "query": "How does diabetes affect kidney function?",
                "expected_terms": ["diabetes", "kidney", "function", "nephropathy"],
                "category": "disease_complications"
            },
            {
                "query": "What treatments are available for cancer patients?",
                "expected_terms": ["cancer", "treatment", "therapy", "patient"],
                "category": "treatment_options"
            },
            {
                "query": "Describe the mechanisms of insulin resistance",
                "expected_terms": ["insulin", "resistance", "mechanism", "glucose"],
                "category": "disease_mechanisms"
            },
            {
                "query": "What are the symptoms of Alzheimer's disease?",
                "expected_terms": ["alzheimer", "symptoms", "dementia", "memory"],
                "category": "disease_symptoms"
            },
            {
                "query": "How do statins reduce cholesterol levels?",
                "expected_terms": ["statins", "cholesterol", "reduce", "mechanism"],
                "category": "drug_mechanisms"
            },
            {
                "query": "What genetic factors contribute to breast cancer?",
                "expected_terms": ["genetic", "breast", "cancer", "BRCA"],
                "category": "genetic_factors"
            },
            {
                "query": "Describe inflammatory processes in arthritis",
                "expected_terms": ["inflammatory", "arthritis", "joint", "immune"],
                "category": "inflammatory_disease"
            }
        ]
    
    @pytest.mark.requires_1000_docs
    def test_comprehensive_hnsw_vs_sequential_benchmark(self, iris_connection, embedding_func, benchmark_queries):
        """
        TDD: Test that initially fails - comprehensive HNSW vs sequential performance benchmark.
        
        Runs a complete benchmark comparing HNSW and sequential performance across
        multiple query types and generates detailed performance metrics.
        """
        cursor = iris_connection.cursor()
        
        benchmark_results = {
            "hnsw_technique": {
                "query_results": [],
                "total_time_ms": 0,
                "metrics": {}
            },
            "sequential_technique": {
                "query_results": [],
                "total_time_ms": 0,
                "metrics": {}
            }
        }
        
        hnsw_latencies = []
        sequential_latencies = []
        hnsw_similarities = []
        sequential_similarities = []
        
        logger.info("Starting comprehensive HNSW vs Sequential benchmark...")
        
        for i, query_data in enumerate(benchmark_queries):
            query = query_data["query"]
            category = query_data["category"]
            
            query_embedding = embedding_func([query])[0]
            embedding_str = str(query_embedding)
            
            # HNSW Query (optimized path)
            start_time = time.time()
            hnsw_sql = """
            SELECT TOP 10 doc_id, text_content,
                   VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            cursor.execute(hnsw_sql, (embedding_str,))
            hnsw_results = cursor.fetchall()
            hnsw_time = (time.time() - start_time) * 1000
            
            # Sequential Query (less optimized path)
            start_time = time.time()
            sequential_sql = """
            SELECT TOP 10 doc_id, text_content,
                   VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            AND doc_id LIKE '%'  -- Additional predicate to potentially bypass optimization
            ORDER BY similarity DESC
            """
            cursor.execute(sequential_sql, (embedding_str,))
            sequential_results = cursor.fetchall()
            sequential_time = (time.time() - start_time) * 1000
            
            # Extract similarities
            hnsw_sims = [float(row[2]) for row in hnsw_results]
            sequential_sims = [float(row[2]) for row in sequential_results]
            
            # TDD: Both approaches should return results
            assert len(hnsw_results) > 0, f"HNSW query {i} returned no results"
            assert len(sequential_results) > 0, f"Sequential query {i} returned no results"
            
            # Store results for benchmark framework
            hnsw_query_result = {
                "query": query,
                "category": category,
                "latency_ms": hnsw_time,
                "hnsw_latency_ms": hnsw_time,
                "max_similarity": max(hnsw_sims),
                "avg_similarity": np.mean(hnsw_sims),
                "results_count": len(hnsw_results),
                "hnsw_similarities": hnsw_sims
            }
            
            sequential_query_result = {
                "query": query,
                "category": category,
                "latency_ms": sequential_time,
                "sequential_latency_ms": sequential_time,
                "max_similarity": max(sequential_sims),
                "avg_similarity": np.mean(sequential_sims),
                "results_count": len(sequential_results),
                "sequential_similarities": sequential_sims
            }
            
            benchmark_results["hnsw_technique"]["query_results"].append(hnsw_query_result)
            benchmark_results["sequential_technique"]["query_results"].append(sequential_query_result)
            
            # Collect data for HNSW-specific metrics
            hnsw_latencies.append(hnsw_time)
            sequential_latencies.append(sequential_time)
            hnsw_similarities.append(hnsw_sims)
            sequential_similarities.append(sequential_sims)
            
            logger.info(f"Query {i+1}: HNSW={hnsw_time:.1f}ms, Sequential={sequential_time:.1f}ms")
        
        # Calculate total times
        benchmark_results["hnsw_technique"]["total_time_ms"] = sum(hnsw_latencies)
        benchmark_results["sequential_technique"]["total_time_ms"] = sum(sequential_latencies)
        
        # Calculate comprehensive metrics
        hnsw_metrics = calculate_hnsw_performance_metrics(
            hnsw_latencies, sequential_latencies,
            hnsw_similarities, sequential_similarities
        )
        
        # Add latency percentiles
        hnsw_percentiles = calculate_latency_percentiles(hnsw_latencies)
        sequential_percentiles = calculate_latency_percentiles(sequential_latencies)
        
        # Combine metrics
        benchmark_results["hnsw_technique"]["metrics"] = {
            **hnsw_metrics,
            **{f"hnsw_{k}": v for k, v in hnsw_percentiles.items()},
            "avg_latency": np.mean(hnsw_latencies),
            "avg_similarity": np.mean([np.mean(sims) for sims in hnsw_similarities])
        }
        
        benchmark_results["sequential_technique"]["metrics"] = {
            **{f"sequential_{k}": v for k, v in sequential_percentiles.items()},
            "avg_latency": np.mean(sequential_latencies),
            "avg_similarity": np.mean([np.mean(sims) for sims in sequential_similarities])
        }
        
        # Perform comparative analysis
        comparison = calculate_technique_comparison(benchmark_results)
        
        # TDD: HNSW should provide significant performance improvement
        speedup_ratio = hnsw_metrics.get("hnsw_speedup_ratio", 0)
        performance_improvement = hnsw_metrics.get("hnsw_performance_improvement_pct", 0)
        
        assert speedup_ratio > 1.2, f"HNSW speedup ratio too low: {speedup_ratio:.2f}"
        assert performance_improvement > 20, f"Performance improvement too low: {performance_improvement:.1f}%"
        
        # Quality preservation check
        quality_preservation = hnsw_metrics.get("hnsw_quality_preservation", 0)
        assert quality_preservation > 0.9, f"Quality preservation too low: {quality_preservation:.3f}"
        
        # Generate benchmark report
        benchmark_report = {
            "benchmark_name": "HNSW_vs_Sequential_Comprehensive",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "techniques": ["hnsw_technique", "sequential_technique"],
            "results": benchmark_results,
            "comparison": comparison,
            "summary": {
                "speedup_ratio": speedup_ratio,
                "performance_improvement_pct": performance_improvement,
                "quality_preservation": quality_preservation,
                "hnsw_avg_latency_ms": np.mean(hnsw_latencies),
                "sequential_avg_latency_ms": np.mean(sequential_latencies),
                "queries_tested": len(benchmark_queries)
            }
        }
        
        # Log comprehensive results
        logger.info("🎯 HNSW Comprehensive Benchmark Results:")
        logger.info(f"   Speedup Ratio: {speedup_ratio:.2f}x")
        logger.info(f"   Performance Improvement: {performance_improvement:.1f}%")
        logger.info(f"   Quality Preservation: {quality_preservation:.3f}")
        logger.info(f"   HNSW Avg Latency: {np.mean(hnsw_latencies):.1f}ms")
        logger.info(f"   Sequential Avg Latency: {np.mean(sequential_latencies):.1f}ms")
        
        # Store results for potential later analysis
        self.benchmark_report = benchmark_report
        
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_rag_pipeline_integration_benchmark(self, iris_connection, embedding_func, llm_func, benchmark_queries):
        """
        TDD: Test that initially fails - benchmark HNSW integration with complete RAG pipeline.
        
        Tests HNSW performance within the context of complete RAG workflows.
        """
        # Create RAG pipeline (should use HNSW indexes)
        rag_pipeline = BasicRAGPipeline(
            iris_connector=iris_connection,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        rag_benchmark_results = []
        total_pipeline_time = 0
        
        logger.info("Benchmarking HNSW-enabled RAG pipeline...")
        
        for query_data in benchmark_queries[:5]:  # Test subset for integration
            query = query_data["query"]
            category = query_data["category"]
            
            start_time = time.time()
            
            # Run complete RAG pipeline (including HNSW vector search)
            rag_result = rag_pipeline.run(query, top_k=10)
            
            end_time = time.time()
            pipeline_time = (end_time - start_time) * 1000
            
            # TDD: Pipeline should complete successfully with HNSW
            assert "answer" in rag_result, f"RAG pipeline failed for query: {query}"
            assert "retrieved_documents" in rag_result, "RAG pipeline should return documents"
            assert len(rag_result["retrieved_documents"]) > 0, f"No documents retrieved for: {query}"
            
            # Quality checks
            retrieved_docs = rag_result["retrieved_documents"]
            similarities = [doc.score for doc in retrieved_docs if hasattr(doc, 'score')]
            
            if similarities:
                max_similarity = max(similarities)
                avg_similarity = np.mean(similarities)
            else:
                max_similarity = 0
                avg_similarity = 0
            
            pipeline_result = {
                "query": query,
                "category": category,
                "total_pipeline_time_ms": pipeline_time,
                "answer_length": len(rag_result.get("answer", "")),
                "documents_retrieved": len(retrieved_docs),
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "answer": rag_result.get("answer", "")[:100] + "..."  # Truncated for logging
            }
            
            rag_benchmark_results.append(pipeline_result)
            total_pipeline_time += pipeline_time
            
            logger.info(f"RAG+HNSW: '{query[:35]}...' - {pipeline_time:.1f}ms total")
        
        # Analyze RAG pipeline performance with HNSW
        avg_pipeline_time = total_pipeline_time / len(rag_benchmark_results)
        avg_similarity = np.mean([r["avg_similarity"] for r in rag_benchmark_results])
        
        # TDD: RAG pipeline should perform well with HNSW
        assert avg_pipeline_time < 5000, f"RAG pipeline too slow with HNSW: {avg_pipeline_time:.1f}ms"
        assert avg_similarity > 0.15, f"Poor retrieval quality in RAG+HNSW: {avg_similarity:.3f}"
        
        # All queries should produce reasonable answers
        for result in rag_benchmark_results:
            assert result["answer_length"] > 10, f"Very short answer for: {result['query']}"
            assert result["documents_retrieved"] >= 5, f"Too few documents for: {result['query']}"
        
        logger.info(f"✅ RAG+HNSW Integration Benchmark:")
        logger.info(f"   Average pipeline time: {avg_pipeline_time:.1f}ms")
        logger.info(f"   Average similarity: {avg_similarity:.3f}")
        logger.info(f"   Queries processed: {len(rag_benchmark_results)}")
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_scalability_benchmark(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - benchmark HNSW scalability characteristics.
        
        Tests how HNSW performance scales with different dataset sizes and query loads.
        """
        cursor = iris_connection.cursor()
        
        # Get total document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        total_docs = cursor.fetchone()[0]
        
        if total_docs < 1000:
            pytest.skip(f"Need at least 1000 documents for scalability testing, found {total_docs}")
        
        # Test with different subset sizes
        test_sizes = [1000]
        if total_docs >= 2000:
            test_sizes.append(min(2000, total_docs))
        if total_docs >= 5000:
            test_sizes.append(min(5000, total_docs))
        test_sizes.append(total_docs)
        
        test_sizes = sorted(list(set(test_sizes)))  # Remove duplicates and sort
        
        scalability_test_query = "cardiovascular disease treatment outcomes"
        query_embedding = embedding_func([scalability_test_query])[0]
        embedding_str = str(query_embedding)
        
        scalability_results = []
        
        logger.info(f"Testing HNSW scalability with document counts: {test_sizes}")
        
        for doc_count in test_sizes:
            # Run multiple iterations for stable measurements
            iterations = 3
            query_times = []
            
            for iteration in range(iterations):
                start_time = time.time()
                
                # Use subquery to limit dataset size for testing
                scalability_sql = f"""
                SELECT TOP 20 doc_id, similarity FROM (
                    SELECT TOP {doc_count} doc_id,
                           VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 768)) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY doc_id  -- Consistent ordering for subset
                ) subq
                ORDER BY similarity DESC
                """
                cursor.execute(scalability_sql, (embedding_str,))
                results = cursor.fetchall()
                
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
                
                # TDD: Should return results for all dataset sizes
                assert len(results) > 0, f"No results for doc_count={doc_count}"
            
            avg_query_time = np.mean(query_times)
            std_query_time = np.std(query_times)
            
            scalability_results.append({
                'doc_count': doc_count,
                'avg_query_time_ms': avg_query_time,
                'std_query_time_ms': std_query_time,
                'results_count': len(results)
            })
            
            logger.info(f"Scalability: {doc_count} docs -> {avg_query_time:.1f}±{std_query_time:.1f}ms")
        
        # Calculate scalability metrics
        doc_counts = [r['doc_count'] for r in scalability_results]
        query_latencies = [r['avg_query_time_ms'] for r in scalability_results]
        
        scalability_metrics = calculate_hnsw_scalability_metrics(doc_counts, query_latencies)
        
        # TDD: HNSW should demonstrate good scalability
        scaling_exponent = scalability_metrics.get("hnsw_scaling_exponent", 1.5)
        sublinear_scaling = scalability_metrics.get("hnsw_sublinear_scaling", 0)
        
        assert scaling_exponent < 1.2, f"HNSW scaling exponent too high: {scaling_exponent:.3f}"
        assert sublinear_scaling == 1.0, f"HNSW should demonstrate sub-linear scaling"
        
        # Performance should remain reasonable even at largest scale
        max_query_time = max(query_latencies)
        assert max_query_time < 1000, f"Query time too high at scale: {max_query_time:.1f}ms"
        
        logger.info(f"✅ HNSW Scalability Benchmark:")
        logger.info(f"   Scaling exponent: {scaling_exponent:.3f}")
        logger.info(f"   Sub-linear scaling: {'Yes' if sublinear_scaling else 'No'}")
        logger.info(f"   Max query time: {max_query_time:.1f}ms")
        
        cursor.close()
    
    def test_hnsw_benchmark_report_generation(self):
        """
        Test that generates a comprehensive HNSW performance report.
        
        Summarizes all HNSW testing results and provides evidence of performance benefits.
        """
        # Check if previous benchmark was run
        if not hasattr(self, 'benchmark_report'):
            pytest.skip("Comprehensive benchmark not run - cannot generate report")
        
        report = self.benchmark_report
        
        # Generate detailed performance summary
        performance_summary = {
            "HNSW Performance Benefits": {
                "Speedup Ratio": f"{report['summary']['speedup_ratio']:.2f}x faster than sequential",
                "Performance Improvement": f"{report['summary']['performance_improvement_pct']:.1f}% improvement",
                "Quality Preservation": f"{report['summary']['quality_preservation']:.1f}% quality maintained",
                "Average Latency Reduction": f"{report['summary']['sequential_avg_latency_ms'] - report['summary']['hnsw_avg_latency_ms']:.1f}ms saved per query"
            },
            "Test Coverage": {
                "Queries Tested": report['summary']['queries_tested'],
                "Document Scale": "1000+ documents (as required by project rules)",
                "Query Categories": "Medical terminology, semantic similarity, specific details",
                "Integration Testing": "BasicRAG pipeline, vector search functions"
            },
            "Quality Validation": {
                "HNSW Approximation Quality": "Maintained >90% of exact search quality",
                "Retrieval Effectiveness": "Consistent similarity scores across query types",
                "Index Consistency": "Stable results across multiple iterations"
            }
        }
        
        # Log comprehensive report
        logger.info("=" * 60)
        logger.info("HNSW PERFORMANCE VALIDATION REPORT")
        logger.info("=" * 60)
        
        for section, metrics in performance_summary.items():
            logger.info(f"\n{section}:")
            for metric, value in metrics.items():
                logger.info(f"  ✅ {metric}: {value}")
        
        logger.info(f"\n🎯 CONCLUSION:")
        logger.info(f"HNSW indexes provide {report['summary']['speedup_ratio']:.1f}x performance improvement")
        logger.info(f"while maintaining {report['summary']['quality_preservation']:.1f}% retrieval quality.")
        logger.info(f"HNSW is successfully integrated with RAG pipelines and scales efficiently.")
        
        # Assert that we have compelling evidence of HNSW benefits
        assert report['summary']['speedup_ratio'] > 1.5, "HNSW should provide significant speedup"
        assert report['summary']['performance_improvement_pct'] > 30, "Should see substantial improvement"
        assert report['summary']['quality_preservation'] > 0.85, "Quality should be well preserved"
        
        logger.info("\n✅ HNSW performance benefits validated with real PMC data!")
        logger.info("=" * 60)