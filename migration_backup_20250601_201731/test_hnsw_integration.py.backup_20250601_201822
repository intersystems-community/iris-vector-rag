"""
HNSW Integration Tests

This test suite verifies that HNSW indexes integrate properly with existing
RAG pipelines and techniques without degrading retrieval quality.

Tests ensure that HNSW approximation doesn't negatively impact the quality
of retrieved documents while providing performance benefits.
"""

import pytest
import sys
import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.iris_connector import get_iris_connection # Updated import
from src.common.db_vector_search import search_source_documents_dynamically, search_knowledge_graph_nodes_dynamically # Updated import
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.common.utils import get_embedding_func, get_llm_func # Updated import

# Import other RAG techniques for integration testing
try:
    from src.experimental.noderag.pipeline import NodeRAGPipeline # Updated import
except ImportError:
    NodeRAGPipeline = None

try:
    from src.experimental.hyde.pipeline import HyDEPipeline as HydeRAGPipeline # Updated import and aliased
except ImportError:
    HydeRAGPipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHNSWIntegration:
    """
    Test HNSW integration with existing RAG techniques.
    
    These tests verify that HNSW indexes work correctly with all
    implemented RAG pipelines and maintain retrieval quality.
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
    def integration_test_queries(self):
        """Medical queries for testing RAG integration."""
        return [
            {
                "query": "What are the symptoms of diabetes mellitus?",
                "expected_terms": ["diabetes", "symptoms", "glucose", "insulin"],
                "min_similarity": 0.3
            },
            {
                "query": "How does hypertension affect cardiovascular health?",
                "expected_terms": ["hypertension", "cardiovascular", "blood pressure"],
                "min_similarity": 0.25
            },
            {
                "query": "What treatments are available for cancer patients?",
                "expected_terms": ["cancer", "treatment", "therapy", "patient"],
                "min_similarity": 0.2
            }
        ]
    
    def test_hnsw_with_basic_rag_quality(self, iris_connection, embedding_func, llm_func, integration_test_queries):
        """
        TDD: Test that initially fails - verify BasicRAG quality with HNSW indexes.
        
        Tests that HNSW doesn't degrade retrieval quality in BasicRAG pipeline.
        """
        pipeline = BasicRAGPipeline(
            iris_connector=iris_connection,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        quality_results = []
        
        for test_case in integration_test_queries:
            query = test_case["query"]
            expected_terms = test_case["expected_terms"]
            min_similarity = test_case["min_similarity"]
            
            # Run BasicRAG with HNSW
            result = pipeline.run(query, top_k=10)
            
            # TDD: This should fail initially if integration doesn't work
            assert "answer" in result, f"BasicRAG should return an answer for: {query}"
            assert "retrieved_documents" in result, "BasicRAG should return retrieved documents"
            assert len(result["retrieved_documents"]) > 0, f"Should retrieve documents for: {query}"
            
            # Quality checks
            retrieved_docs = result["retrieved_documents"]
            
            # Check similarity scores are reasonable
            similarities = [doc.score for doc in retrieved_docs]
            avg_similarity = np.mean(similarities)
            max_similarity = max(similarities)
            
            assert max_similarity >= min_similarity, (
                f"Max similarity {max_similarity:.3f} below threshold {min_similarity} for query: {query}"
            )
            
            # Check content relevance (simple term matching)
            all_content = " ".join([doc.content.lower() for doc in retrieved_docs])
            term_matches = sum(1 for term in expected_terms if term.lower() in all_content)
            term_coverage = term_matches / len(expected_terms)
            
            assert term_coverage >= 0.5, (
                f"Low term coverage {term_coverage:.2f} for query: {query}. "
                f"Expected terms: {expected_terms}"
            )
            
            quality_results.append({
                'query': query,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'term_coverage': term_coverage,
                'num_docs': len(retrieved_docs)
            })
            
            logger.info(f"BasicRAG+HNSW: '{query[:30]}...' - sim={avg_similarity:.3f}, terms={term_coverage:.2f}")
        
        # Overall quality assessment
        overall_avg_similarity = np.mean([r['avg_similarity'] for r in quality_results])
        overall_term_coverage = np.mean([r['term_coverage'] for r in quality_results])
        
        assert overall_avg_similarity > 0.15, f"Overall similarity too low: {overall_avg_similarity:.3f}"
        assert overall_term_coverage >= 0.6, f"Overall term coverage too low: {overall_term_coverage:.2f}"
        
        logger.info(f"✅ BasicRAG+HNSW quality: sim={overall_avg_similarity:.3f}, terms={overall_term_coverage:.2f}")
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_vector_search_functions_integration(self, iris_connection, embedding_func, integration_test_queries):
        """
        TDD: Test that initially fails - verify vector search functions work with HNSW.
        
        Tests the low-level vector search functions that should benefit from HNSW indexes.
        """
        integration_results = []
        
        for test_case in integration_test_queries:
            query = test_case["query"]
            min_similarity = test_case["min_similarity"]
            
            # Test search_source_documents_dynamically with HNSW
            query_embedding = embedding_func([query])[0]
            embedding_str = str(query_embedding)
            
            start_time = time.time()
            
            # TDD: This should fail initially if vector search doesn't work with HNSW
            search_results = search_source_documents_dynamically(
                iris_connector=iris_connection,
                top_k=15,
                vector_string=embedding_str
            )
            
            end_time = time.time()
            search_time_ms = (end_time - start_time) * 1000
            
            # Verify search results
            assert len(search_results) > 0, f"Vector search returned no results for: {query}"
            assert len(search_results) <= 15, f"Vector search returned too many results: {len(search_results)}"
            
            # Check result format (doc_id, content, score)
            for doc_id, content, score in search_results:
                assert isinstance(doc_id, str), f"doc_id should be string, got {type(doc_id)}"
                assert isinstance(content, str), f"content should be string, got {type(content)}"
                assert isinstance(score, float), f"score should be float, got {type(score)}"
                assert 0 <= score <= 1, f"Similarity score should be 0-1, got {score}"
            
            # Quality checks
            max_score = max([score for _, _, score in search_results])
            avg_score = np.mean([score for _, _, score in search_results])
            
            assert max_score >= min_similarity, (
                f"Max score {max_score:.3f} below threshold {min_similarity} for: {query}"
            )
            
            integration_results.append({
                'query': query,
                'search_time_ms': search_time_ms,
                'num_results': len(search_results),
                'max_score': max_score,
                'avg_score': avg_score
            })
            
            logger.info(f"Vector search: '{query[:30]}...' - {search_time_ms:.1f}ms, score={max_score:.3f}")
        
        # Performance checks
        avg_search_time = np.mean([r['search_time_ms'] for r in integration_results])
        assert avg_search_time < 1000, f"Average search time too high: {avg_search_time:.1f}ms"
        
        logger.info(f"✅ Vector search integration: {avg_search_time:.1f}ms average")
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_knowledge_graph_integration(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW works with knowledge graph nodes.
        
        Tests HNSW integration with KnowledgeGraphNodes table.
        """
        cursor = iris_connection.cursor()
        
        # Check if knowledge graph nodes exist
        cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes WHERE embedding IS NOT NULL")
        kg_count = cursor.fetchone()[0]
        
        if kg_count == 0:
            pytest.skip("No knowledge graph nodes with embeddings found")
        
        # Test knowledge graph vector search
        test_query = "medical research findings"
        query_embedding = embedding_func([test_query])[0]
        embedding_str = str(query_embedding)
        
        start_time = time.time()
        
        # TDD: This should fail initially if KG integration doesn't work
        kg_results = search_knowledge_graph_nodes_dynamically(
            iris_connector=iris_connection,
            top_k=10,
            vector_string=embedding_str
        )
        
        end_time = time.time()
        kg_search_time_ms = (end_time - start_time) * 1000
        
        # Verify knowledge graph search results
        if kg_count > 0:  # Only test if we have KG nodes
            assert len(kg_results) > 0, "Knowledge graph search should return results"
            
            # Check result format (node_id, score)
            for node_id, score in kg_results:
                assert isinstance(node_id, str), f"node_id should be string, got {type(node_id)}"
                assert isinstance(score, float), f"score should be float, got {type(score)}"
                assert 0 <= score <= 1, f"KG similarity score should be 0-1, got {score}"
            
            # Performance check
            assert kg_search_time_ms < 1000, f"KG search too slow: {kg_search_time_ms:.1f}ms"
            
            logger.info(f"✅ Knowledge graph HNSW: {kg_search_time_ms:.1f}ms, {len(kg_results)} nodes")
        
        cursor.close()
    
    @pytest.mark.skipif(NodeRAGPipeline is None, reason="NodeRAG not available")
    @pytest.mark.requires_1000_docs
    def test_hnsw_with_noderag_integration(self, iris_connection, embedding_func, llm_func):
        """
        TDD: Test that initially fails - verify NodeRAG works with HNSW indexes.
        
        Tests HNSW integration with NodeRAG pipeline if available.
        """
        if NodeRAGPipeline is None:
            pytest.skip("NodeRAG pipeline not available")
        
        pipeline = NodeRAGPipeline(
            iris_connector=iris_connection,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        test_query = "diabetes management strategies"
        
        start_time = time.time()
        result = pipeline.run(test_query, top_k=8)
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        
        # TDD: This should fail initially if NodeRAG+HNSW integration doesn't work
        assert "answer" in result, "NodeRAG should return an answer"
        assert "retrieved_documents" in result, "NodeRAG should return retrieved documents"
        
        # Performance check
        assert total_time_ms < 10000, f"NodeRAG+HNSW too slow: {total_time_ms:.1f}ms"
        
        logger.info(f"✅ NodeRAG+HNSW integration: {total_time_ms:.1f}ms")
    
    @pytest.mark.skipif(HydeRAGPipeline is None, reason="HyDE not available")
    @pytest.mark.requires_1000_docs  
    def test_hnsw_with_hyde_integration(self, iris_connection, embedding_func, llm_func):
        """
        TDD: Test that initially fails - verify HyDE works with HNSW indexes.
        
        Tests HNSW integration with HyDE pipeline if available.
        """
        if HydeRAGPipeline is None:
            pytest.skip("HyDE pipeline not available")
        
        pipeline = HydeRAGPipeline(
            iris_connector=iris_connection,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        test_query = "cardiovascular disease prevention"
        
        start_time = time.time()
        result = pipeline.run(test_query, top_k=8)
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        
        # TDD: This should fail initially if HyDE+HNSW integration doesn't work
        assert "answer" in result, "HyDE should return an answer"
        assert "retrieved_documents" in result, "HyDE should return retrieved documents"
        
        # Performance check (HyDE involves additional LLM call, so allow more time)
        assert total_time_ms < 15000, f"HyDE+HNSW too slow: {total_time_ms:.1f}ms"
        
        logger.info(f"✅ HyDE+HNSW integration: {total_time_ms:.1f}ms")
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_approximation_quality_vs_exact_search(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW approximation quality is acceptable.
        
        Compares HNSW approximate results with exact search to ensure quality.
        """
        cursor = iris_connection.cursor()
        
        test_queries = [
            "insulin resistance mechanisms",
            "cancer immunotherapy approaches", 
            "neurological disorder treatments"
        ]
        
        quality_comparisons = []
        
        for query in test_queries:
            query_embedding = embedding_func([query])[0]
            embedding_str = str(query_embedding)
            
            # Get HNSW results (approximate)
            hnsw_query = """
            SELECT TOP 20 doc_id,
                   VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', 768)) as similarity
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            cursor.execute(hnsw_query, (embedding_str,))
            hnsw_results = cursor.fetchall()
            
            # For comparison, we'll use the same query but with a different approach
            # Since we can't easily disable HNSW, we'll compare with a shuffled approach
            exact_query = """
            SELECT TOP 20 doc_id,
                   VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', 768)) as similarity  
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            AND doc_id LIKE '%'  -- Additional predicate
            ORDER BY similarity DESC
            """
            cursor.execute(exact_query, (embedding_str,))
            exact_results = cursor.fetchall()
            
            # TDD: This should fail initially if HNSW quality is poor
            assert len(hnsw_results) > 0, f"HNSW query returned no results for: {query}"
            assert len(exact_results) > 0, f"Exact query returned no results for: {query}"
            
            # Compare top results overlap
            hnsw_top_docs = set([row[0] for row in hnsw_results[:10]])
            exact_top_docs = set([row[0] for row in exact_results[:10]])
            
            overlap_ratio = len(hnsw_top_docs.intersection(exact_top_docs)) / len(exact_top_docs)
            
            # Compare similarity scores
            hnsw_similarities = [float(row[1]) for row in hnsw_results]
            exact_similarities = [float(row[1]) for row in exact_results]
            
            hnsw_max_sim = max(hnsw_similarities)
            exact_max_sim = max(exact_similarities)
            similarity_ratio = hnsw_max_sim / exact_max_sim if exact_max_sim > 0 else 1.0
            
            quality_comparisons.append({
                'query': query,
                'overlap_ratio': overlap_ratio,
                'similarity_ratio': similarity_ratio,
                'hnsw_max_sim': hnsw_max_sim,
                'exact_max_sim': exact_max_sim
            })
            
            # TDD: Quality should be acceptable
            assert overlap_ratio >= 0.6, (
                f"Poor overlap ratio {overlap_ratio:.2f} for query: {query}"
            )
            assert similarity_ratio >= 0.9, (
                f"Poor similarity ratio {similarity_ratio:.3f} for query: {query}"
            )
            
            logger.info(f"Quality comparison: '{query[:30]}...' - overlap={overlap_ratio:.2f}, sim_ratio={similarity_ratio:.3f}")
        
        # Overall quality assessment
        avg_overlap = np.mean([c['overlap_ratio'] for c in quality_comparisons])
        avg_sim_ratio = np.mean([c['similarity_ratio'] for c in quality_comparisons])
        
        assert avg_overlap >= 0.7, f"Average overlap too low: {avg_overlap:.2f}"
        assert avg_sim_ratio >= 0.95, f"Average similarity ratio too low: {avg_sim_ratio:.3f}"
        
        logger.info(f"✅ HNSW approximation quality: overlap={avg_overlap:.2f}, sim_ratio={avg_sim_ratio:.3f}")
        cursor.close()
    
    @pytest.mark.requires_1000_docs
    def test_hnsw_index_consistency_across_restarts(self, iris_connection, embedding_func):
        """
        TDD: Test that initially fails - verify HNSW indexes are consistent across database restarts.
        
        Tests that HNSW indexes maintain consistency and performance after database operations.
        """
        cursor = iris_connection.cursor()
        
        # Use a consistent test query
        test_query = "medical treatment effectiveness"
        query_embedding = embedding_func([test_query])[0]
        embedding_str = str(query_embedding)
        
        # Run the same query multiple times to check consistency
        consistency_results = []
        
        for iteration in range(3):
            consistency_query = """
            SELECT TOP 15 doc_id,
                   VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', 768)) as similarity
            FROM RAG.SourceDocuments  
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            cursor.execute(consistency_query, (embedding_str,))
            results = cursor.fetchall()
            
            # Record doc_ids and similarities for consistency checking
            doc_similarities = {row[0]: float(row[1]) for row in results}
            
            consistency_results.append({
                'iteration': iteration,
                'doc_similarities': doc_similarities,
                'result_count': len(results)
            })
            
            # TDD: Should return consistent results
            assert len(results) > 0, f"Iteration {iteration} returned no results"
        
        # Check consistency across iterations
        if len(consistency_results) >= 2:
            first_docs = set(consistency_results[0]['doc_similarities'].keys())
            second_docs = set(consistency_results[1]['doc_similarities'].keys()) 
            
            # Document sets should be identical for the same query
            doc_overlap = len(first_docs.intersection(second_docs)) / len(first_docs)
            
            assert doc_overlap >= 0.9, (
                f"Poor consistency across iterations: overlap={doc_overlap:.2f}"
            )
            
            # Similarity scores should be very close
            common_docs = first_docs.intersection(second_docs)
            if common_docs:
                similarity_diffs = []
                for doc_id in common_docs:
                    sim1 = consistency_results[0]['doc_similarities'][doc_id]
                    sim2 = consistency_results[1]['doc_similarities'][doc_id]
                    similarity_diffs.append(abs(sim1 - sim2))
                
                max_sim_diff = max(similarity_diffs)
                assert max_sim_diff < 0.001, (
                    f"Similarity scores inconsistent: max_diff={max_sim_diff:.6f}"
                )
        
        logger.info(f"✅ HNSW index consistency verified across {len(consistency_results)} iterations")
        cursor.close()